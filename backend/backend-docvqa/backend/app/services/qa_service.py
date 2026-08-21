from dataclasses import dataclass
from typing import Optional

from app.services import vlm_service


@dataclass
class QAResult:
    answer: str
    evidence_bbox: Optional[list[float]]
    confidence: Optional[float]


def answer_question(processed_image_path: str, question: str) -> QAResult:
    """Answer directly from the image with Qwen2-VL; this model has no bbox head."""
    answer = vlm_service.answer_question(processed_image_path, question)
    return QAResult(answer=answer, evidence_bbox=None, confidence=None)
