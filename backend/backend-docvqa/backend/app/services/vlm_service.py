"""End-to-end Qwen2-VL adapter for extraction and document question answering."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings


EXTRACTION_PROMPT = """Đọc kỹ ảnh tài liệu / hóa đơn và trả về DUY NHẤT một JSON hợp lệ (không markdown, không giải thích):
{
  "SELLER": "Tên đơn vị / người bán hàng, hoặc null",
  "INVOICE_NUMBER": "Số thứ tự hóa đơn (chỉ lấy dãy số ở phần Số/No., không lấy ký hiệu/mẫu số), hoặc null",
  "TAX_CODE": "Mã số thuế của bên bán (chuỗi 10 hoặc 13 chữ số), hoặc null",
  "TIMESTAMP": "Ngày tháng năm lập chứng từ, hoặc null",
  "TOTAL_COST": "Tổng cộng tiền thanh toán cuối cùng đã bao gồm thuế, hoặc null"
}
Chỉ trích xuất đúng thông tin có trên tài liệu."""

FIELD_ALIASES = {
    "SELLER": "store_name",
    "STORE_NAME": "store_name",
    "COMPANY": "store_name",
    "INVOICE_NO": "invoice_number",
    "INVOICE_NUMBER": "invoice_number",
    "RECEIPT_NO": "invoice_number",
    "TAX_CODE": "tax_code",
    "TAX_ID": "tax_code",
    "MST": "tax_code",
    "DATE": "invoice_date",
    "TIMESTAMP": "invoice_date",
    "INVOICE_DATE": "invoice_date",
    "TOTAL": "total_amount",
    "TOTAL_COST": "total_amount",
    "TOTAL_AMOUNT": "total_amount",
    "ADDRESS": "address",
    "ITEM_NAME": "items",
    "ITEMS": "items",
}

FIELD_QUESTIONS = (
    ("store_name", "Tên cửa hàng / bên bán trên hóa đơn là gì?"),
    ("invoice_number", "Số hóa đơn là bao nhiêu? Chỉ trả về số hoặc ký hiệu hóa đơn."),
    ("tax_code", "Mã số thuế của bên bán là gì?"),
    ("invoice_date", "Ngày giờ lập hóa đơn là khi nào?"),
    ("total_amount", "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"),
)


@dataclass(frozen=True)
class VLMField:
    key: str
    value: str
    confidence: float | None = None


class VLMRuntimeError(RuntimeError):
    """Raised when the local Qwen2-VL runtime or checkpoints are unavailable."""


def _complete_adapter_path() -> str | None:
    adapter_path = settings.vlm_adapter_path or (
        settings.model_dir / "stage1_vlm" / "output" / "lora_adapters"
    )
    adapter_dir = Path(adapter_path)
    if (
        adapter_dir.is_dir()
        and (adapter_dir / "adapter_config.json").is_file()
        and any(adapter_dir.glob("adapter_model.*"))
    ):
        return str(adapter_dir)
    return None


@lru_cache(maxsize=1)
def _get_engine():
    """Load the team's fine-tuned VLM once per FastAPI worker."""
    project_root = str(settings.model_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    adapter_path = _complete_adapter_path()
    if not adapter_path and not settings.vlm_allow_base_model:
        raise VLMRuntimeError(
            "Không tìm thấy trọng số adapter_model.* trong VLM_ADAPTER_PATH. "
            "Hãy mount LoRA adapter hoặc đặt VLM_ALLOW_BASE_MODEL=true."
        )
    try:
        from model.stage1_vlm.src.inference import VQAEngine

        return VQAEngine(adapter_dir=adapter_path, base_model=settings.vlm_base_model)
    except ImportError as exc:
        raise VLMRuntimeError(
            "Thiếu VLM dependencies. Cài backend requirements trước khi chạy Qwen2-VL."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - keep model tracebacks out of API responses
        raise VLMRuntimeError(f"Không thể tải Qwen2-VL: {exc}") from exc


def answer_question(
    image_path: str,
    question: str,
    max_new_tokens: int = 256,
    use_adapter: bool = True,
) -> str:
    """Run the VLM on a preprocessed document image and a natural-language question."""
    if not Path(image_path).is_file():
        raise VLMRuntimeError("Không tìm thấy ảnh đã tiền xử lý cho document này.")
    response = _get_engine().extract_and_answer(
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
        use_adapter=use_adapter,
    )
    if not response or not response.strip():
        raise VLMRuntimeError("Qwen2-VL trả về nội dung rỗng.")
    return response.strip()


def _parse_json_response(response: str) -> dict[str, Any] | None:
    """Accept bare JSON or fenced JSON emitted by a generative VLM."""
    candidate = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _single_field_value(response: str, expected_key: str) -> str | None:
    """Normalize a short VQA answer that may still be wrapped in a one-key JSON."""
    parsed = _parse_json_response(response)
    value: Any = response.strip()
    if parsed:
        normalized = {
            FIELD_ALIASES.get(
                re.sub(r"[^A-Z0-9]+", "_", str(key).upper()).strip("_"),
                re.sub(r"[^A-Z0-9]+", "_", str(key).upper()).strip("_").lower(),
            ): item
            for key, item in parsed.items()
        }
        value = normalized.get(expected_key)
        if value is None and len(normalized) == 1:
            value = next(iter(normalized.values()))
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        text_value = json.dumps(value, ensure_ascii=False)
    else:
        text_value = str(value).strip()
    invalid_values = {"", "null", "none", "n/a", expected_key, expected_key.upper()}
    invalid_values.update(alias for alias, canonical in FIELD_ALIASES.items() if canonical == expected_key)
    if text_value.lower() in {item.lower() for item in invalid_values}:
        return None
    return _normalize_field_value(expected_key, text_value)


def _normalize_field_value(key: str, value: str) -> str:
    """Normalize conservative display-only formats without inventing information."""
    if key == "total_amount" and re.fullmatch(
        r"\s*\d[\d.,\s]*(?:vnd|vnđ|đ|đồng)?\s*", value, flags=re.IGNORECASE
    ):
        digits = re.sub(r"\D", "", value)
        if digits:
            return f"{int(digits):,}".replace(",", ".")
    return value


def extract_fields(image_path: str) -> tuple[list[VLMField], str]:
    """Ask Qwen2-VL for structured fields and preserve raw output for auditability."""
    if settings.vlm_extraction_mode == "multi":
        fields: list[VLMField] = []
        raw_answers: dict[str, str] = {}
        for key, question in FIELD_QUESTIONS:
            response = answer_question(image_path, question, max_new_tokens=96)
            raw_answers[key] = response
            value = _single_field_value(response, key)
            if value:
                fields.append(VLMField(key=key, value=value))
        raw_response = json.dumps(raw_answers, ensure_ascii=False)
        return fields or [VLMField(key="vlm_response", value=raw_response)], raw_response

    response = answer_question(
        image_path,
        EXTRACTION_PROMPT,
        max_new_tokens=192,
        use_adapter=settings.vlm_extraction_mode != "base",
    )
    parsed = _parse_json_response(response)
    if not parsed:
        return [VLMField(key="vlm_response", value=response)], response

    fields: list[VLMField] = []
    seen_keys: set[str] = set()
    for raw_key, value in parsed.items():
        if value is None:
            continue
        normalized_key = re.sub(r"[^A-Z0-9]+", "_", str(raw_key).upper()).strip("_")
        key = FIELD_ALIASES.get(normalized_key, normalized_key.lower())
        if not key or key in seen_keys:
            continue
        if isinstance(value, (dict, list)):
            text_value = json.dumps(value, ensure_ascii=False)
        else:
            text_value = str(value).strip()
        if text_value:
            fields.append(VLMField(key=key, value=_normalize_field_value(key, text_value)))
            seen_keys.add(key)
    return fields or [VLMField(key="vlm_response", value=response)], response
