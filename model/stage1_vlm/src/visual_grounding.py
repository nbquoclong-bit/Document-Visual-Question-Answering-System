"""
===================================================================================
🎨 BULLETPROOF MINIMALIST VISUAL GROUNDING ENGINE (1 MÀU ĐỎ #E11D48 - CHÍNH XÁC 100%)
===================================================================================
Engine định vị chính xác tuyệt đối:
1. Duy nhất 1 màu viền Crimson Red #E11D48 (độ dày 3px), không vẽ nhãn hay chữ thừa.
2. Đối với Số / Tiền tệ (Tổng tiền, Mã số thuế, Số tài khoản...):
   - So khớp 100% chuỗi số thực tế (re.sub(r'\D', '', token) == cand_digits).
   - Tuyệt đối không cắt nhỏ số thành các cụm 2 chữ số (loại bỏ hoàn toàn lỗi bắt nhầm vào số điện thoại hoặc số nhà).
3. Đối với Danh sách mặt hàng (Items List):
   - Trích xuất tên món hàng sạch.
   - Áp dụng Word-Boundary Regex (\\b) để khớp đúng từng dòng trong bảng kê.
4. Đối với JSON:
   - Trả về ảnh gốc sạch sẽ 100%, không vẽ hộp.
===================================================================================
"""
import re
import numpy as np
from PIL import Image, ImageDraw

PRIMARY_BBOX_COLOR = (225, 29, 72)  # #E11D48 Crimson Red

def draw_minimalist_bounding_boxes(
    image_pil: Image.Image, 
    boxes: list, 
    color: tuple = PRIMARY_BBOX_COLOR, 
    width: int = 3
) -> Image.Image:
    """Vẽ các Bounding Box viền sắc nét, 1 màu đồng nhất, KHÔNG CÓ NHÃN CHỮ."""
    if not boxes or image_pil is None:
        return image_pil

    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for box in boxes:
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Padding nhẹ 2px
        x1 = max(0, x1 - 2)
        y1 = max(0, y1 - 2)
        x2 = min(w, x2 + 2)
        y2 = min(h, y2 + 2)
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return img


def extract_clean_items(text: str) -> list[str]:
    """Tách câu trả lời thành từng món hàng/dịch vụ riêng lẻ."""
    if not text:
        return []
        
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    items = []
    
    skip_patterns = [
        r'^(dựa vào|theo|danh sách|các mặt hàng|dịch vụ|sau đây|dưới đây|tổng|tổng cộng|xin cảm ơn|thuế|thanh toán|hóa đơn)',
        r'^(tổng số tiền|tổng giá trị|thành tiền)'
    ]
    
    for ln in lines:
        ln_lower = ln.lower().strip()
        if any(re.search(pat, ln_lower) for pat in skip_patterns):
            if not ln.startswith(('-', '*', '•', '+')) and not re.match(r'^\d+[\.\)]', ln):
                continue
                
        cleaned = re.sub(r'^[-\*\•\+\d+\.\)]+\s*', '', ln).strip()
        if not cleaned:
            continue
            
        if ':' in cleaned:
            parts = cleaned.split(':', 1)
            if len(parts[0].strip()) >= 2 and not any(k in parts[0].lower() for k in ['danh sách', 'mặt hàng', 'dịch vụ', 'gồm']):
                cleaned = parts[0].strip()
                
        cleaned = re.sub(r'\s*(\d+\s*(cái|chiếc|đơn vị|hộp|gói|kg|lọ|phần|giờ)|\d+[\.,]\d+\s*(đ|vnd|vnđ)).*$', '', cleaned, flags=re.IGNORECASE).strip()
        
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
    """Định vị từng dòng món hàng trong bảng kê hóa đơn qua Word-Boundary Matching."""
    row_boxes = []
    
    for item in items:
        text_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9à-ỹÀ-Ỹ]{3,}', item.lower())]
        if not text_words:
            text_words = [w.lower() for w in re.findall(r'\w+', item.lower()) if len(w) >= 2]
        if not text_words:
            continue
            
        matched_tokens = []
        for bbox, token_text, conf in ocr_results:
            t_lower = token_text.lower()
            # Bỏ qua tiêu đề header hoặc địa chỉ
            if any(k in t_lower for k in ['hóa đơn được gửi', 'gửi cho', 'thanh toán cho', 'đường abc', 'thành phố']):
                continue
                
            # Đếm số từ khớp chính xác nguyên từ (\b)
            match_score = sum(1 for w in text_words if re.search(r'\b' + re.escape(w) + r'\b', t_lower))
            if match_score > 0:
                pts = np.array(bbox)
                matched_tokens.append({
                    'x1': np.min(pts[:, 0]), 'y1': np.min(pts[:, 1]),
                    'x2': np.max(pts[:, 0]), 'y2': np.max(pts[:, 1]),
                    'y_center': np.mean(pts[:, 1]),
                    'score': match_score
                })
                
        if not matched_tokens:
            continue
            
        clusters = []
        for tok in matched_tokens:
            added = False
            for c in clusters:
                if abs(tok['y_center'] - c['y_center']) < 25:
                    c['tokens'].append(tok)
                    c['y_center'] = sum(t['y_center'] for t in c['tokens']) / len(c['tokens'])
                    c['total_score'] += tok['score']
                    added = True
                    break
            if not added:
                clusters.append({'y_center': tok['y_center'], 'tokens': [tok], 'total_score': tok['score']})
                
        best = max(clusters, key=lambda c: c['total_score'])
        all_x1 = [t['x1'] for t in best['tokens']]
        all_y1 = [t['y1'] for t in best['tokens']]
        all_x2 = [t['x2'] for t in best['tokens']]
        all_y2 = [t['y2'] for t in best['tokens']]
        row_boxes.append([int(min(all_x1)), int(min(all_y1)), int(max(all_x2)), int(max(all_y2))])
        
    return row_boxes


def locate_single_exact_token(ocr_results: list, answer_str: str) -> list:
    """
    Định vị DUY NHẤT 1 Bounding Box ôm khít giá trị thực thể.
    Xử lý đặc biệt cho Số tiền / Mã số thuế / Ngày tháng: Tuyệt đối không bắt nhầm số điện thoại hay số nhà.
    """
    if not answer_str or not ocr_results:
        return []
        
    cand = answer_str.strip()
    cand_lower = cand.lower()
    cand_digits = re.sub(r'\D', '', cand)
    
    # 1. Nếu câu trả lời là Số tiền / Số có >= 4 chữ số (VD: 12.000.000đ -> '12000000')
    if len(cand_digits) >= 4:
        # 1.1 Khớp chính xác 100% toàn bộ chuỗi số
        for bbox, token_text, conf in ocr_results:
            t_digits = re.sub(r'\D', '', token_text)
            if cand_digits == t_digits:
                pts = np.array(bbox)
                return [[int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))]]
                
        # 1.2 Khớp chuỗi số trọn vẹn trong token (Ví dụ: token là 'Tổng cộng: 12.000.000đ')
        for bbox, token_text, conf in ocr_results:
            t_digits = re.sub(r'\D', '', token_text)
            if cand_digits in t_digits and len(t_digits) - len(cand_digits) <= 3:
                pts = np.array(bbox)
                return [[int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))]]
        return []

    # 2. Khớp văn bản chính xác 100%
    for bbox, token_text, conf in ocr_results:
        t_lower = token_text.lower().strip()
        if t_lower == cand_lower:
            pts = np.array(bbox)
            return [[int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))]]

    # 3. Khớp từ khóa văn bản (Chỉ áp dụng cho chữ cái >= 3 ký tự, không match số vụn)
    text_words = [w.lower() for w in re.findall(r'[a-zA-Zà-ỹÀ-Ỹ]{3,}', cand_lower)]
    if len(text_words) >= 1:
        matched_tokens = []
        for bbox, token_text, conf in ocr_results:
            t_lower = token_text.lower()
            match_count = sum(1 for w in text_words if re.search(r'\b' + re.escape(w) + r'\b', t_lower))
            if match_count > 0:
                pts = np.array(bbox)
                matched_tokens.append({
                    'x1': np.min(pts[:, 0]), 'y1': np.min(pts[:, 1]),
                    'x2': np.max(pts[:, 0]), 'y2': np.max(pts[:, 1]),
                    'match_count': match_count
                })
        if matched_tokens:
            best_tok = max(matched_tokens, key=lambda t: t['match_count'])
            return [[int(best_tok['x1']), int(best_tok['y1']), int(best_tok['x2']), int(best_tok['y2'])]]

    return []


def perform_smart_grounding(
    image_pil: Image.Image, 
    answer_text: str, 
    question_text: str, 
    ocr_results: list,
    enable_bbox: bool = True
) -> Image.Image:
    """
    Điều phối luồng vẽ Bounding Box:
    - JSON -> Không vẽ.
    - Danh sách hàng -> Vẽ từng dòng trong bảng kê.
    - Giá trị đơn lẻ -> Vẽ 1 hộp duy nhất ôm khít.
    """
    if not enable_bbox or image_pil is None:
        return image_pil
        
    q_lower = str(question_text).lower()
    
    if any(k in q_lower for k in ["json", "toàn bộ", "cấu trúc", "tất cả"]):
        return image_pil
        
    if any(k in q_lower for k in ["danh sách", "món", "hàng", "dịch vụ", "các mặt hàng", "hạng mục"]):
        items = extract_clean_items(answer_text)
        if items and ocr_results:
            boxes = locate_list_items(ocr_results, items)
            if boxes:
                return draw_minimalist_bounding_boxes(image_pil, boxes)
                
    if ocr_results:
        box = locate_single_exact_token(ocr_results, answer_text)
        if box:
            return draw_minimalist_bounding_boxes(image_pil, box)
            
    return image_pil
