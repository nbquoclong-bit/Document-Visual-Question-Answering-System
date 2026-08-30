"""
===================================================================================
🎨 ROBUST MINIMALIST VISUAL GROUNDING ENGINE (1 MÀU ĐỒNG NHẤT - KHÔNG CHỮ THỪA)
===================================================================================
Engine định vị và khoanh vùng thông minh tối ưu:
1. Duy nhất 1 màu viền đồng nhất (Crimson Red #E11D48) với độ dày 2-3px.
2. Không vẽ nhãn/chữ/badge đè lên ảnh.
3. Bộ lọc câu dẫn (Preamble Filter) chống khoanh nhầm vào các tiêu đề 'HÓA ĐƠN ĐƯỢC GỬI CHO'.
4. 3 Kịch bản:
   - Single Entity: 1 Bounding Box ôm khít giá trị thực thể.
   - List Items: Từng Bounding Box riêng biệt cho từng dòng hàng hóa đã mua.
   - Full JSON: 0 Bounding Box (ảnh gốc sạch sẽ 100%).
===================================================================================
"""
import re
import numpy as np
from PIL import Image, ImageDraw

# Màu viền duy nhất: Đỏ Crimson nổi bật, sang trọng, sắc nét
PRIMARY_BBOX_COLOR = (225, 29, 72)  # #E11D48

def draw_minimalist_bounding_boxes(
    image_pil: Image.Image, 
    boxes: list, 
    color: tuple = PRIMARY_BBOX_COLOR, 
    width: int = 3
) -> Image.Image:
    """
    Vẽ các khung Bounding Box viền sắc nét, 1 màu duy nhất, KHÔNG VẼ NHÃN CHỮ.
    boxes: danh sách [x1, y1, x2, y2] pixel thực tế.
    """
    if not boxes or image_pil is None:
        return image_pil

    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for box in boxes:
        if not box or len(box) != 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Đảm bảo x1 <= x2, y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Thêm padding nhẹ 2px để không cắt sát nét chữ
        x1 = max(0, x1 - 2)
        y1 = max(0, y1 - 2)
        x2 = min(w, x2 + 2)
        y2 = min(h, y2 + 2)
        
        # Vẽ viền chữ nhật đơn giản, sạch sẽ
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return img


def extract_clean_items(text: str) -> list[str]:
    """
    Tách chuỗi câu trả lời thành từng món hàng/dịch vụ riêng lẻ.
    Bộ lọc chuyên biệt khử 100% câu dẫn (preamble), câu kết luận (total), 
    đảm bảo không khoanh nhầm vào 'HÓA ĐƠN ĐƯỢC GỬI CHO'.
    """
    if not text:
        return []
        
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    items = []
    
    skip_patterns = [
        r'^(dựa vào|theo|danh sách|các mặt hàng|dịch vụ|sau đây|dưới đây|tổng|tổng cộng|xin cảm ơn|thuế|thanh toán)',
        r'^(tổng số tiền|tổng giá trị|thành tiền|hóa đơn được)'
    ]
    
    for ln in lines:
        ln_lower = ln.lower().strip()
        
        # Kiểm tra nếu là câu dẫn mở đầu hoặc câu kết
        if any(re.search(pat, ln_lower) for pat in skip_patterns):
            if not ln.startswith(('-', '*', '•', '+')) and not re.match(r'^\d+[\.\)]', ln):
                continue
                
        # Bỏ dấu gạch đầu dòng, số thứ tự
        cleaned = re.sub(r'^[-\*\•\+\d+\.\)]+\s*', '', ln).strip()
        if not cleaned:
            continue
            
        # Cắt bỏ phần sau dấu hai chấm ':' nếu có (VD: 'Tiếp cận (tiểu thuyết): 1 đơn vị, giá 3tr' -> 'Tiếp cận (tiểu thuyết)')
        if ':' in cleaned:
            parts = cleaned.split(':', 1)
            if len(parts[0].strip()) >= 2 and not any(k in parts[0].lower() for k in ['danh sách', 'mặt hàng', 'dịch vụ', 'gồm']):
                cleaned = parts[0].strip()
                
        # Bỏ các thông tin phụ như số lượng, đơn giá nếu dính liền
        cleaned = re.sub(r'\s*(\d+\s*(cái|chiếc|đơn vị|hộp|gói|kg|lọ|phần|giờ)|\d+[\.,]\d+\s*(đ|vnd|vnđ)).*$', '', cleaned, flags=re.IGNORECASE).strip()
        
        # Nếu có dấu phẩy trong 1 dòng duy nhất
        if ',' in cleaned and len(lines) == 1:
            for sub in cleaned.split(','):
                sub_c = re.sub(r'^[-\*\•\+\d+\.\)]+\s*', '', sub).strip()
                if len(sub_c) >= 2 and not any(re.search(pat, sub_c.lower()) for pat in skip_patterns):
                    items.append(sub_c)
        else:
            if len(cleaned) >= 2 and not any(re.search(pat, cleaned.lower()) for pat in skip_patterns):
                items.append(cleaned)
                
    return items[:15]


def locate_list_items(ocr_results: list, items: list[str]) -> list:
    """Định vị từng món hàng trên từng dòng hóa đơn qua EasyOCR tokens."""
    row_boxes = []
    
    for item in items:
        # Lấy từ khóa chính của món hàng (chỉ lấy các từ có ý nghĩa >= 2 ký tự)
        keywords = [w.lower() for w in re.findall(r'\w+', item) if len(w) >= 2 and w.lower() not in ['tiểu', 'thuyết', 'tranh', 'ảnh', 'sử', 'loại', 'món']]
        if not keywords:
            keywords = [w.lower() for w in re.findall(r'\w+', item) if len(w) >= 2]
        if not keywords:
            continue
            
        matched_tokens = []
        for bbox, token_text, conf in ocr_results:
            t_lower = token_text.lower()
            # Bỏ qua các token thuộc khối tiêu đề hoặc địa chỉ
            if any(k in t_lower for k in ['hóa đơn được gửi', 'gửi cho', 'thanh toán cho', 'đường abc']):
                continue
            if any(k in t_lower for k in keywords):
                pts = np.array(bbox)
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)
                matched_tokens.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "y_center": (y1 + y2) / 2.0,
                    "text": token_text
                })
                
        if not matched_tokens:
            continue
            
        # Gom các token có cùng cao độ trục Y (< 25px) thành 1 dòng món hàng
        clusters = []
        for tok in matched_tokens:
            added = False
            for c in clusters:
                if abs(tok["y_center"] - c["y_center"]) < 25:
                    c["tokens"].append(tok)
                    c["y_center"] = sum(t["y_center"] for t in c["tokens"]) / len(c["tokens"])
                    added = True
                    break
            if not added:
                clusters.append({"y_center": tok["y_center"], "tokens": [tok]})
                
        # Chọn cluster có nhiều từ khớp nhất
        best_cluster = max(clusters, key=lambda c: len(c["tokens"]))
        all_x1 = [t["x1"] for t in best_cluster["tokens"]]
        all_y1 = [t["y1"] for t in best_cluster["tokens"]]
        all_x2 = [t["x2"] for t in best_cluster["tokens"]]
        all_y2 = [t["y2"] for t in best_cluster["tokens"]]
        
        row_box = [int(min(all_x1)), int(min(all_y1)), int(max(all_x2)), int(max(all_y2))]
        row_boxes.append(row_box)
        
    return row_boxes


def locate_single_exact_token(ocr_results: list, answer_str: str) -> list:
    """Định vị DUY NHẤT 1 Bounding Box ôm khít giá trị thực thể."""
    if not answer_str or not ocr_results:
        return []
        
    cand = answer_str.strip()
    cand_digits = re.sub(r'\D', '', cand)
    cand_lower = cand.lower()
    
    # 1. So khớp chữ số (Mã số thuế, Tổng tiền...)
    if len(cand_digits) >= 4:
        for bbox, token_text, conf in ocr_results:
            t_digits = re.sub(r'\D', '', token_text)
            if cand_digits == t_digits or (len(t_digits) >= 4 and (cand_digits in t_digits or t_digits in cand_digits)):
                pts = np.array(bbox)
                return [[int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))]]
                
    # 2. So khớp từ khóa chính xác
    for bbox, token_text, conf in ocr_results:
        t_lower = token_text.lower().strip()
        if t_lower == cand_lower or (len(cand_lower) >= 4 and cand_lower in t_lower):
            pts = np.array(bbox)
            return [[int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))]]
            
    # 3. So khớp cụm từ trên cùng một hàng ngang
    tokens_in_cand = [w.lower() for w in re.findall(r'\w+', cand) if len(w) >= 2]
    if len(tokens_in_cand) >= 2:
        matched_tokens = []
        for bbox, token_text, conf in ocr_results:
            t_lower = token_text.lower()
            if any(w in t_lower for w in tokens_in_cand):
                pts = np.array(bbox)
                matched_tokens.append({
                    "x1": np.min(pts[:, 0]), "y1": np.min(pts[:, 1]),
                    "x2": np.max(pts[:, 0]), "y2": np.max(pts[:, 1]),
                    "y_center": np.mean(pts[:, 1])
                })
        if matched_tokens:
            clusters = []
            for tok in matched_tokens:
                added = False
                for c in clusters:
                    if abs(tok["y_center"] - c["y_center"]) < 25:
                        c["tokens"].append(tok)
                        c["y_center"] = sum(t["y_center"] for t in c["tokens"]) / len(c["tokens"])
                        added = True
                        break
                if not added:
                    clusters.append({"y_center": tok["y_center"], "tokens": [tok]})
            best_cluster = max(clusters, key=lambda c: len(c["tokens"]))
            if len(best_cluster["tokens"]) >= 2:
                all_x1 = [t["x1"] for t in best_cluster["tokens"]]
                all_y1 = [t["y1"] for t in best_cluster["tokens"]]
                all_x2 = [t["x2"] for t in best_cluster["tokens"]]
                all_y2 = [t["y2"] for t in best_cluster["tokens"]]
                return [[int(min(all_x1)), int(min(all_y1)), int(max(all_x2)), int(max(all_y2))]]

    return []


def perform_smart_grounding(
    image_pil: Image.Image, 
    answer_text: str, 
    question_text: str, 
    ocr_results: list,
    enable_bbox: bool = True
) -> Image.Image:
    """
    Điều phối luồng vẽ Bounding Box thông minh tối giản:
    - Nếu enable_bbox == False hoặc hỏi JSON: Trả về ảnh gốc sạch sẽ 100%.
    - Nếu hỏi danh sách món hàng: Vẽ từng khung viền đỏ cho từng dòng món hàng.
    - Nếu hỏi thông tin đơn lẻ: Vẽ 1 khung viền đỏ duy nhất ôm khít.
    """
    if not enable_bbox or image_pil is None:
        return image_pil
        
    q_lower = str(question_text).lower()
    
    # Kịch bản 3: Trích xuất JSON -> 100% sạch sẽ, không vẽ hộp
    if any(k in q_lower for k in ["json", "toàn bộ", "cấu trúc", "tất cả"]):
        return image_pil
        
    # Kịch bản 2: Hỏi danh sách món hàng / dịch vụ
    if any(k in q_lower for k in ["danh sách", "món", "hàng", "dịch vụ", "các mặt hàng", "hạng mục"]):
        items = extract_clean_items(answer_text)
        if items and ocr_results:
            boxes = locate_list_items(ocr_results, items)
            if boxes:
                return draw_minimalist_bounding_boxes(image_pil, boxes)
                
    # Kịch bản 1: Hỏi 1 thông tin đơn lẻ
    if ocr_results:
        box = locate_single_exact_token(ocr_results, answer_text)
        if box:
            return draw_minimalist_bounding_boxes(image_pil, box)
            
    # Nếu OCR không tìm thấy -> Giữ ảnh sạch, không vẽ fallback
    return image_pil
