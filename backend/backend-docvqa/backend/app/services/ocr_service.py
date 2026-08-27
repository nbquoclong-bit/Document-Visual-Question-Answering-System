"""
OCR Service — Module trích xuất toạ độ văn bản (Token Bounding Boxes) và so khớp bằng chứng (Evidence Grounding).
"""
import re
import unicodedata
from typing import Optional

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        # Chạy CPU siêu nhẹ với tiếng Việt và tiếng Anh
        _reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    return _reader


def extract_tokens(image_path: str) -> list[dict]:
    """
    Quét toàn bộ ảnh để lấy danh sách từ/cụm từ kèm toạ độ hộp chữ nhật.
    Trả về danh sách dạng: [{'text': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}]
    """
    try:
        reader = _get_reader()
        results = reader.readtext(image_path)
        tokens = []
        for bbox, text, prob in results:
            if not text or not text.strip():
                continue
            # bbox từ easyocr là 4 điểm: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            tokens.append({
                "text": text.strip(),
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(prob), 4),
            })
        return tokens
    except Exception as e:
        print(f"[ocr_service] Error during OCR token extraction: {e}")
        return []


def _normalize_str(text: str) -> str:
    """Chuẩn hoá chuỗi để so khớp không phụ thuộc dấu cách, hoa/thường."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def locate_evidence_bbox(tokens: list[dict], answer_text: str) -> Optional[list[float]]:
    """
    Tìm kiếm toạ độ Bounding Box chính xác chứa câu trả lời của VLM trên tài liệu.
    """
    if not tokens or not answer_text:
        return None

    # Làm sạch chuỗi trả lời
    clean_ans = re.sub(r"[*_`#]", "", answer_text).strip()
    norm_ans = _normalize_str(clean_ans)

    # 1. Trích xuất các cụm số hoặc từ khóa quan trọng trong câu trả lời (MST, Tiền, Số hóa đơn, Ngày)
    candidates = []
    
    # Tìm các cụm số tiền hoặc mã số (vd: 3.404.009, 0309489281, 0000006, 28/07/2023)
    numbers = re.findall(r"\b\d[\d.,/A-Za-z-]{2,}\b", clean_ans)
    for num in numbers:
        candidates.append(num)

    # Thêm cả toàn bộ câu trả lời nếu ngắn gọn
    if len(clean_ans.split()) <= 6:
        candidates.append(clean_ans)

    # 2. Tìm kiếm khớp trực tiếp với từng token
    for cand in candidates:
        cand_norm = _normalize_str(cand)
        cand_digits = re.sub(r"\D", "", cand)
        
        for token in tokens:
            t_norm = _normalize_str(token["text"])
            t_digits = re.sub(r"\D", "", token["text"])
            
            # Khớp chính xác hoặc là chuỗi con
            if cand_norm and (cand_norm == t_norm or cand_norm in t_norm or t_norm in cand_norm):
                return token["bbox"]
                
            # Khớp dãy số (đặc biệt hữu ích cho Tiền hoặc Mã số thuế)
            if cand_digits and len(cand_digits) >= 4 and cand_digits in t_digits:
                return token["bbox"]

    # 3. Tìm kiếm dãy nhiều token liên tiếp khớp với câu trả lời (ví dụ tên công ty dài)
    ans_words = norm_ans.split()
    if len(ans_words) >= 2:
        for i in range(len(tokens)):
            matched_boxes = []
            for j in range(i, min(i + len(ans_words) + 2, len(tokens))):
                tok_text = _normalize_str(tokens[j]["text"])
                if any(w in tok_text for w in ans_words if len(w) >= 3):
                    matched_boxes.append(tokens[j]["bbox"])
            if len(matched_boxes) >= 2:
                # Gộp bounding box của các token khớp
                x1 = min(b[0] for b in matched_boxes)
                y1 = min(b[1] for b in matched_boxes)
                x2 = max(b[2] for b in matched_boxes)
                y2 = max(b[3] for b in matched_boxes)
                return [x1, y1, x2, y2]

    return None
