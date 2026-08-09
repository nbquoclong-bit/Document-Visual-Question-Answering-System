"""
KIE SERVICE (Key Information Extraction)
=========================================
Nhiệm vụ: từ các token OCR (text + bbox), suy ra các trường thông tin có cấu trúc
của hoá đơn: tên cửa hàng, ngày, tổng tiền, v.v.

>>> ĐÂY LÀ ĐIỂM TÍCH HỢP SẢN PHẨM CỦA NHÓM <<<
Theo đề cương: model dự kiến là LayoutLMv3 / LayoutXLM / LiLT (multimodal, kết hợp
text + layout + ảnh). Người phụ trách: Model Lead (Lê Minh Sang).

Hàm `extract_fields()` hiện dùng RULE-BASED đơn giản (regex/keyword) làm mock,
đủ để pipeline chạy end-to-end. Khi có model thật:

    1. Giữ nguyên chữ ký hàm: input (image_path, ocr_tokens) -> output List[FieldResult]
    2. Model thật cần cả ảnh gốc (image_path) lẫn token OCR vì đây là multimodal model
       (dùng cả layout + hình ảnh, không chỉ text thuần).
    3. `bbox` trong FieldResult nên là bbox của token/vùng chứa value đó, để backend
       có thể highlight bằng chứng trên frontend.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from app.services.ocr_service import OCRToken


@dataclass
class FieldResult:
    key: str
    value: str
    bbox: Optional[List[float]]
    confidence: float


# Các field chuẩn mà frontend/báo cáo mong đợi — nhóm có thể mở rộng thêm.
FIELD_KEYS = ("store_name", "invoice_date", "total_amount")


def extract_fields(image_path: str, ocr_tokens: List[OCRToken]) -> List[FieldResult]:
    """
    TODO(Model Lead): thay nội dung hàm này bằng lời gọi model KIE thật
    (LayoutLMv3 / LayoutXLM), nhận vào ảnh + token OCR, trả về các cặp key-value.

    Input :
        image_path: đường dẫn ảnh gốc (model multimodal cần ảnh, không chỉ text)
        ocr_tokens: kết quả từ ocr_service.run_ocr()
    Output:
        List[FieldResult] — mỗi field có key chuẩn hoá (xem FIELD_KEYS), value dạng
        text, bbox của vùng chứa value (để highlight), và confidence.
    """
    # --- MOCK rule-based: xoá khối này khi tích hợp model thật ---
    results: List[FieldResult] = []

    for token in ocr_tokens:
        text_lower = token.text.lower()

        if "ngày" in text_lower:
            match = re.search(r"\d{2}/\d{2}/\d{4}", token.text)
            if match:
                results.append(FieldResult("invoice_date", match.group(), token.bbox, token.confidence))

        elif "tổng" in text_lower:
            match = re.search(r"[\d.,]+\s*VND", token.text)
            if match:
                results.append(FieldResult("total_amount", match.group(), token.bbox, token.confidence))

    # Giả định token đầu tiên là tên cửa hàng (mock)
    if ocr_tokens:
        first = ocr_tokens[0]
        results.append(FieldResult("store_name", first.text, first.bbox, first.confidence))

    return results
