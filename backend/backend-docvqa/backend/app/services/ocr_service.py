"""
OCR SERVICE
===========
Nhiệm vụ: nhận đường dẫn ảnh hoá đơn, trả về danh sách token văn bản kèm bounding box.

>>> ĐÂY LÀ ĐIỂM TÍCH HỢP SẢN PHẨM CỦA NHÓM <<<
Theo đề cương, model dùng ở bước này là PaddleOCR (PP-OCRv4). Người phụ trách theo
phân công là Model Lead (Lê Minh Sang) / Data Engineer (Nguyễn Văn Nhật Nam).

Hàm `run_ocr()` bên dưới hiện đang trả về dữ liệu MOCK (giả lập) để toàn bộ API chạy
được ngay hôm nay, không phải chờ model thật. Khi nhóm có checkpoint PaddleOCR:

    1. Xoá phần mock bên trong `run_ocr`.
    2. Load model thật (gợi ý dùng `settings.ocr_model_path` trong app/config.py để
       không hard-code đường dẫn).
    3. Giữ nguyên chữ ký hàm (input: str, output: List[OCRToken]) — phần còn lại của
       backend (pipeline_service, routers) sẽ không cần sửa gì thêm.

Contract (hợp đồng dữ liệu) — PHẢI giữ nguyên khi thay bằng model thật:
    Input : image_path: str — đường dẫn ảnh đã lưu trên disk
    Output: List[OCRToken] — mỗi token gồm text, bbox [x1,y1,x2,y2] (pixel, gốc trên-trái),
            và confidence trong [0,1]
"""
from dataclasses import dataclass
from typing import List


@dataclass
class OCRToken:
    text: str
    bbox: List[float]        # [x1, y1, x2, y2]
    confidence: float


def run_ocr(image_path: str) -> List[OCRToken]:
    """
    TODO(Model Lead / Data Engineer): thay nội dung hàm này bằng lời gọi PaddleOCR thật.

    Ví dụ tích hợp thật (tham khảo, chưa chạy được vì thiếu model đã tải):

        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="vi")

        def run_ocr(image_path: str) -> List[OCRToken]:
            result = _ocr_engine.ocr(image_path, cls=True)
            tokens = []
            for line in result[0]:
                bbox_points, (text, confidence) = line
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                tokens.append(OCRToken(
                    text=text,
                    bbox=[min(xs), min(ys), max(xs), max(ys)],
                    confidence=float(confidence),
                ))
            return tokens
    """
    # --- MOCK: xoá khối này khi tích hợp model thật ---
    return [
        OCRToken(text="CỬA HÀNG ABC", bbox=[50, 20, 300, 45], confidence=0.98),
        OCRToken(text="Ngày: 01/08/2026", bbox=[50, 60, 220, 82], confidence=0.95),
        OCRToken(text="Tổng cộng: 150,000 VND", bbox=[50, 400, 320, 425], confidence=0.93),
    ]
