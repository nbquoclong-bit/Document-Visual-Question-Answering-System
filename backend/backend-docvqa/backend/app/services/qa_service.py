from dataclasses import dataclass
from typing import Optional

from app.services import vlm_service, ocr_service


@dataclass
class QAResult:
    answer: str
    evidence_bbox: Optional[list[float]]
    confidence: Optional[float]


def answer_question(
    processed_image_path: str,
    question: str,
    ocr_tokens: Optional[list[dict]] = None,
) -> QAResult:
    """Answer directly from the image with Qwen2-VL, then locate evidence bbox from OCR tokens."""
    answer = vlm_service.answer_question(processed_image_path, question)
    
    # Định vị toạ độ bằng chứng từ OCR tokens
    evidence_bbox = None
    if ocr_tokens:
        evidence_bbox = ocr_service.locate_evidence_bbox(ocr_tokens, answer)

    return QAResult(answer=answer, evidence_bbox=evidence_bbox, confidence=None)
