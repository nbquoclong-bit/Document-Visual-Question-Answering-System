"""KIE service: LayoutLMv3 inference when a fine-tuned checkpoint is configured."""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict, List, Optional

from app.config import settings
from app.services.ocr_service import OCRToken


@dataclass
class FieldResult:
    key: str
    value: str
    bbox: Optional[List[float]]
    confidence: float


# Các field chuẩn mà frontend/báo cáo mong đợi — nhóm có thể mở rộng thêm.
FIELD_KEYS = ("store_name", "invoice_number", "tax_code", "invoice_date", "total_amount")


@lru_cache(maxsize=1)
def _get_kie_config():
    """Load the team's LayoutLMv3 checkpoint only when the first document arrives."""
    if not settings.kie_model_path:
        return None

    checkpoint = Path(settings.kie_model_path)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Không tìm thấy KIE checkpoint: {checkpoint}")

    project_root = str(settings.model_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from model.stage2_kie.src.kie_engine import KIEConfig
    except ImportError as exc:
        raise RuntimeError(
            "Không thể nạp LayoutLMv3 runtime. Kiểm tra model dependencies và MODEL_DIR."
        ) from exc
    return KIEConfig(model_dir=str(checkpoint))


def _model_fields(image_path: str, ocr_tokens: List[OCRToken]) -> List[FieldResult]:
    """Translate normalized OCR boxes to LayoutLMv3 and back to the API contract."""
    config = _get_kie_config()
    if config is None:
        return []

    project_root = str(settings.model_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from model.stage2_kie.src.kie_engine import predict
    except ImportError as exc:  # pragma: no cover - defensive against broken deployments
        raise RuntimeError("Không thể import bộ suy luận LayoutLMv3.") from exc

    words = [token.text for token in ocr_tokens]
    boxes_1000 = [
        [round(coordinate * 1000) for coordinate in token.bbox]
        for token in ocr_tokens
    ]
    entities: Dict[str, Dict[str, object]] = predict(config, image_path, words, boxes_1000)
    results: List[FieldResult] = []
    for key, entity in entities.items():
        value = str(entity.get("value", "")).strip()
        raw_box = entity.get("bbox") or None
        bbox = [float(value) / 1000 for value in raw_box] if raw_box else None
        if value:
            results.append(FieldResult(key=key.lower(), value=value, bbox=bbox, confidence=1.0))
    return results


def _fallback_fields(ocr_tokens: List[OCRToken]) -> List[FieldResult]:
    """Provide an auditable baseline when a trained KIE checkpoint is unavailable."""
    results: List[FieldResult] = []


    date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    amount_pattern = r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:\s?(?:VND|đ|₫))?"
    for token in ocr_tokens:
        text_lower = token.text.lower()

        if any(keyword in text_lower for keyword in ("ngày", "date")):
            match = re.search(date_pattern, token.text)
            if match:
                results.append(FieldResult("invoice_date", match.group(), token.bbox, token.confidence))

        elif any(keyword in text_lower for keyword in ("tổng", "thanh toán", "total", "amount")):
            matches = re.findall(amount_pattern, token.text, flags=re.IGNORECASE)
            match = matches[-1] if matches else None
            if match:
                results.append(FieldResult("total_amount", match, token.bbox, token.confidence))

    # Giả định token đầu tiên là tên cửa hàng (mock)
    if ocr_tokens:
        first = ocr_tokens[0]
        results.append(FieldResult("store_name", first.text, first.bbox, first.confidence))

    return results


def extract_fields(image_path: str, ocr_tokens: List[OCRToken]) -> List[FieldResult]:
    """Extract fields using LayoutLMv3, with an explicit OCR-only baseline fallback."""
    if settings.kie_model_path:
        return _model_fields(image_path, ocr_tokens)
    if settings.allow_rule_based_fallback:
        return _fallback_fields(ocr_tokens)
    raise RuntimeError("KIE_MODEL_PATH chưa được cấu hình và fallback đã bị tắt.")
