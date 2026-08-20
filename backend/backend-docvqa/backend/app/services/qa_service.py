"""Accounting QA service backed by the team's Qwen2.5 checkpoint when configured."""
import json
from difflib import SequenceMatcher
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import settings

from app.services.ocr_service import OCRToken
from app.services.kie_service import FieldResult


@dataclass
class QAResult:
    answer: str
    evidence_bbox: Optional[List[float]]
    confidence: float


@lru_cache(maxsize=1)
def _get_qa_runtime() -> Tuple[object, object, object]:
    """Load the local Qwen checkpoint lazily, keeping API startup lightweight."""
    if not settings.qa_model_path:
        raise RuntimeError("QA_MODEL_PATH chưa được cấu hình.")
    checkpoint = Path(settings.qa_model_path)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Không tìm thấy QA checkpoint: {checkpoint}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Qwen runtime chưa được cài trong backend environment.") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype="auto",
        device_map="auto" if settings.device.startswith("cuda") else None,
    )
    if settings.qa_adapter_path:
        adapter = Path(settings.qa_adapter_path)
        if not adapter.is_dir():
            raise RuntimeError(f"Không tìm thấy QLoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    if not settings.device.startswith("cuda"):
        model.to(settings.device)
    model.eval()
    return model, tokenizer, torch


def _build_invoice_context(fields: List[FieldResult]) -> Dict[str, str]:
    """Give Stage 3 only structured KIE data, preserving the no-raw-image design."""
    return {field.key: field.value for field in fields}


def _model_answer(fields: List[FieldResult], question: str) -> str:
    model, tokenizer, torch = _get_qa_runtime()
    invoice_json = json.dumps(_build_invoice_context(fields), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý kế toán. Chỉ trả lời dựa trên JSON hoá đơn được cung cấp. "
                "Nếu dữ liệu không có câu trả lời, hãy nói rõ không tìm thấy thông tin."
            ),
        },
        {"role": "user", "content": f"Dữ liệu hoá đơn: {invoice_json}\n\nCâu hỏi: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    return tokenizer.decode(output[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()


def _best_evidence(
    answer: str, ocr_tokens: List[OCRToken], extracted_fields: List[FieldResult]
) -> Tuple[Optional[List[float]], float]:
    """Map generated text back to an existing field/OCR span without inventing evidence."""
    candidates = [(field.value, field.bbox, field.confidence) for field in extracted_fields]
    candidates.extend((token.text, token.bbox, token.confidence) for token in ocr_tokens)
    answer_normalized = answer.lower().strip()
    best = (None, 0.0)
    for text, bbox, confidence in candidates:
        if not bbox or not text:
            continue
        score = 1.0 if text.lower() in answer_normalized else SequenceMatcher(None, text.lower(), answer_normalized).ratio()
        if score > best[1]:
            best = (bbox, score * confidence)
    return best if best[1] >= 0.6 else (None, 0.0)


def answer_question(
    image_path: str,
    ocr_tokens: List[OCRToken],
    extracted_fields: List[FieldResult],
    question: str,
) -> QAResult:
    """Answer with Qwen2.5 or use deterministic KIE lookup when no QA checkpoint exists."""
    if settings.qa_model_path:
        answer = _model_answer(extracted_fields, question)
        evidence_bbox, confidence = _best_evidence(answer, ocr_tokens, extracted_fields)
        return QAResult(answer=answer, evidence_bbox=evidence_bbox, confidence=confidence)

    if not settings.allow_rule_based_fallback:
        raise RuntimeError("QA_MODEL_PATH chưa được cấu hình và fallback đã bị tắt.")

    q_lower = question.lower()

    keyword_map = {
        "total_amount": ["tổng", "bao nhiêu tiền", "total", "amount"],
        "invoice_date": ["ngày", "date"],
        "store_name": ["cửa hàng", "tên cửa hàng", "store", "shop"],
    }

    for field in extracted_fields:
        keywords = keyword_map.get(field.key, [])
        if any(kw in q_lower for kw in keywords):
            return QAResult(
                answer=field.value,
                evidence_bbox=field.bbox,
                confidence=field.confidence,
            )

    return QAResult(
        answer="Không tìm thấy thông tin phù hợp trong hoá đơn để trả lời câu hỏi này.",
        evidence_bbox=None,
        confidence=0.0,
    )
