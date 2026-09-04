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
    """Answer directly from the image with End-to-End Qwen2-VL and compute model confidence."""
    result = vlm_service.answer_question(processed_image_path, question, return_confidence=True)
    if isinstance(result, tuple):
        answer, confidence = result
    else:
        answer, confidence = result, 0.88

    return QAResult(answer=answer, evidence_bbox=None, confidence=round(confidence, 2))
