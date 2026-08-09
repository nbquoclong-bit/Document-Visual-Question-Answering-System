"""
QA SERVICE (Visual Question Answering)
=======================================
Nhiệm vụ: nhận câu hỏi tự nhiên của người dùng + ảnh hoá đơn (+ token OCR đã có),
trả về câu trả lời và vùng ảnh làm bằng chứng (evidence bbox) để highlight.

>>> ĐÂY LÀ ĐIỂM TÍCH HỢP SẢN PHẨM CỦA NHÓM <<<
Theo đề cương: Qwen2-VL 2B (đọc ảnh trực tiếp) hoặc Qwen2.5-3B (xử lý text sau OCR).
Người phụ trách: Model Lead (Lê Minh Sang).

Hàm `answer_question()` hiện dùng logic tra cứu đơn giản trên các field đã trích
xuất (mock), đủ để test toàn bộ luồng end-to-end mà không cần load model nặng.

Khi tích hợp model thật, giữ nguyên chữ ký hàm:
    Input : image_path, ocr_tokens, extracted_fields, question
    Output: QAResult(answer, evidence_bbox, confidence)
"""
from dataclasses import dataclass
from typing import List, Optional

from app.services.ocr_service import OCRToken
from app.services.kie_service import FieldResult


@dataclass
class QAResult:
    answer: str
    evidence_bbox: Optional[List[float]]
    confidence: float


def answer_question(
    image_path: str,
    ocr_tokens: List[OCRToken],
    extracted_fields: List[FieldResult],
    question: str,
) -> QAResult:
    """
    TODO(Model Lead): thay nội dung hàm này bằng lời gọi Qwen2-VL / Qwen2.5 thật.

    Ví dụ khung tích hợp thật (tham khảo):

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        _model = Qwen2VLForConditionalGeneration.from_pretrained(settings.qa_model_path)
        _processor = AutoProcessor.from_pretrained(settings.qa_model_path)

        def answer_question(image_path, ocr_tokens, extracted_fields, question):
            # build prompt từ question + (tuỳ chọn) text OCR làm context
            # chạy generate(), parse câu trả lời
            # ánh xạ câu trả lời về bbox của token/field liên quan để làm evidence
            ...

    Input :
        image_path: đường dẫn ảnh gốc
        ocr_tokens: toàn bộ token OCR (context bổ sung, có thể model không cần dùng
                    nếu là Qwen2-VL đọc ảnh trực tiếp)
        extracted_fields: các field đã có từ kie_service (mock tra cứu dùng cái này)
        question: câu hỏi của người dùng, tiếng Việt hoặc tiếng Anh
    Output:
        QAResult
    """
    # --- MOCK: tra cứu đơn giản theo từ khoá trong field đã trích xuất ---
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
