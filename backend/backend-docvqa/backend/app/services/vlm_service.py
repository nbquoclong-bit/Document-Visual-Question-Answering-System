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


EXTRACTION_PROMPT = """Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON.
Đọc kỹ ảnh tài liệu hóa đơn và điền thông tin thực tế vào các trường sau:
- SELLER: Tên cửa hàng / công ty / đơn vị bán hàng
- INVOICE_NUMBER: Số hóa đơn (nếu có)
- TAX_CODE: Mã số thuế người bán (nếu có)
- TIMESTAMP: Ngày tháng năm lập hóa đơn
- TOTAL_COST: Tổng cộng tiền thanh toán cuối cùng

Quy tắc bắt buộc:
1. Điền NỘI DUNG THỰC TẾ đọc được từ ảnh vào giá trị. Tuyệt đối KHÔNG chép lại văn bản hướng dẫn.
2. Nếu trường nào không xuất hiện trên hóa đơn, bắt buộc đặt giá trị là null.
3. Chỉ trả về một đối tượng JSON hợp lệ duy nhất:
{"SELLER": null, "INVOICE_NUMBER": null, "TAX_CODE": null, "TIMESTAMP": null, "TOTAL_COST": null}"""

def _is_invalid_or_placeholder(value: str) -> bool:
    """Check if value is empty, null-like, or an echoed prompt instruction."""
    if not value or not str(value).strip():
        return True
    val_lower = str(value).strip().lower()
    if val_lower in {"", "null", "none", "n/a", "undefined", "không có", "chưa rõ", "unknown"}:
        return True
    placeholder_indicators = [
        "hoặc null", "hoac null", "người bán", "nguoi ban",
        "chỉ lấy dãy số", "chi lay day so", "chuỗi 10", "chuoi 10",
        "lập chứng từ", "lap chung tu", "đã bao gồm thuế", "da bao gom thue",
        "ký hiệu", "ky hieu", "mẫu số", "mau so", "tên đơn vị", "ten don vi",
        "thứ tự hóa đơn", "thu tu hoa don", "thực tế", "thuc te", "hướng dẫn", "huong dan"
    ]
    if any(ind in val_lower for ind in placeholder_indicators):
        return True
    return False


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


def _calculate_format_confidence(key: str, value: str) -> float:
    """Evaluate format and business sanity for a field, returning score in [0.0, 1.0]."""
    if not value or not value.strip():
        return 0.20
    val = value.strip()
    val_lower = val.lower()

    # Nhận diện các câu trả lời thể hiện thông tin không tồn tại hoặc mập mờ
    UNAVAILABLE_PHRASES = ["không có", "không tìm thấy", "không rõ", "chưa xác định", "không đề cập", "chưa rõ", "n/a", "null"]
    if any(phrase in val_lower for phrase in UNAVAILABLE_PHRASES):
        return 0.25

    if key == "total_amount":
        digits = re.sub(r"\D", "", val)
        if digits and len(digits) >= 4:
            return 0.95
        elif digits and len(digits) >= 3:
            return 0.70
        return 0.20  # Không trích xuất được số tiền hợp lệ -> Phạt nặng

    if key == "tax_code":
        digits = re.sub(r"\D", "", val)
        if len(digits) in (10, 13) or (len(digits) == 14 and "-" in val):
            return 0.98
        elif len(digits) >= 8:
            return 0.65
        return 0.25  # Mã số thuế sai chuẩn (quá ngắn hoặc lẫn chữ) -> Phạt nặng

    if key == "invoice_date":
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", val):
            return 0.95
        elif any(c.isdigit() for c in val):
            return 0.60
        return 0.30  # Không có cấu trúc ngày tháng hợp lệ

    if key in ("store_name", "seller"):
        if len(val) >= 4 and any(c.isalpha() for c in val):
            return 0.95
        return 0.35  # Tên cửa hàng quá ngắn hoặc vô nghĩa

    if key == "invoice_number":
        digits = re.sub(r"\D", "", val)
        if digits and len(digits) >= 1:
            return 0.92
        return 0.35

    if len(val) >= 2:
        return 0.85
    return 0.40


def answer_question(
    image_path: str,
    question: str,
    max_new_tokens: int = 256,
    use_adapter: bool = True,
    return_confidence: bool = False,
):
    """Run the VLM on a preprocessed document image and a natural-language question."""
    if not Path(image_path).is_file():
        raise VLMRuntimeError("Không tìm thấy ảnh đã tiền xử lý cho document này.")
    result = _get_engine().extract_and_answer(
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
        use_adapter=use_adapter,
        return_confidence=True,
    )
    if isinstance(result, tuple):
        response, confidence = result
    else:
        response = result
        confidence = 0.88

    if not response or not response.strip():
        raise VLMRuntimeError("Qwen2-VL trả về nội dung rỗng.")

    clean_res = response.strip()
    UNAVAILABLE_PHRASES = ["không có", "không tìm thấy", "không rõ", "chưa xác định", "không đề cập", "chưa rõ", "n/a", "null"]
    if any(phrase in clean_res.lower() for phrase in UNAVAILABLE_PHRASES):
        confidence = min(float(confidence), 0.38)

    if return_confidence:
        return clean_res, float(confidence)
    return clean_res


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
            res = answer_question(image_path, question, max_new_tokens=96, return_confidence=True)
            if isinstance(res, tuple):
                response, vlm_conf = res
            else:
                response, vlm_conf = res, 0.88
            raw_answers[key] = response
            value = _single_field_value(response, key)
            if value:
                fmt_conf = _calculate_format_confidence(key, value)
                composite_conf = round(0.70 * vlm_conf + 0.30 * fmt_conf, 2)
                fields.append(VLMField(key=key, value=value, confidence=composite_conf))
        raw_response = json.dumps(raw_answers, ensure_ascii=False)
        return fields or [VLMField(key="vlm_response", value=raw_response, confidence=0.80)], raw_response

    res = answer_question(
        image_path,
        EXTRACTION_PROMPT,
        max_new_tokens=110,
        use_adapter=True,
        return_confidence=True,
    )
    if isinstance(res, tuple):
        response, vlm_conf = res
    else:
        response, vlm_conf = res, 0.88

    parsed = _parse_json_response(response)
    if not parsed:
        return [VLMField(key="vlm_response", value=response, confidence=round(vlm_conf * 0.7, 2))], response

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
        if text_value and not _is_invalid_or_placeholder(text_value):
            norm_val = _normalize_field_value(key, text_value)
            fmt_conf = _calculate_format_confidence(key, norm_val)
            composite_conf = round(0.70 * vlm_conf + 0.30 * fmt_conf, 2)
            fields.append(VLMField(key=key, value=norm_val, confidence=composite_conf))
            seen_keys.add(key)

    if len(fields) < 2:
        # Fallback: Nếu trích xuất JSON chưa đủ các trường cốt lõi, hỏi trực diện VLM
        CORE_QUESTIONS = (
            ("store_name", "Tên cửa hàng / bên bán trên hóa đơn là gì?"),
            ("total_amount", "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"),
            ("invoice_date", "Ngày giờ lập hóa đơn là khi nào?"),
        )
        for c_key, c_q in CORE_QUESTIONS:
            if c_key in seen_keys:
                continue
            res_c = answer_question(image_path, c_q, max_new_tokens=32, return_confidence=True)
            if isinstance(res_c, tuple):
                ans, c_vlm_conf = res_c
            else:
                ans, c_vlm_conf = res_c, 0.88
            val = _single_field_value(ans, c_key)
            if val and not _is_invalid_or_placeholder(val):
                fmt_conf = _calculate_format_confidence(c_key, val)
                composite_conf = round(0.70 * c_vlm_conf + 0.30 * fmt_conf, 2)
                fields.append(VLMField(key=c_key, value=val, confidence=composite_conf))
                seen_keys.add(c_key)

    return fields or [VLMField(key="vlm_response", value=response, confidence=vlm_conf)], response
