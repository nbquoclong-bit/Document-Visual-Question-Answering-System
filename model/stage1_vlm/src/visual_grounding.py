import os
import re
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bảng mã màu trực quan cho từng loại trường thông tin trên hóa đơn
FIELD_COLORS = {
    "SELLER": (46, 204, 113),       # Xanh lá cây (Emerald) - Tên cửa hàng/bên bán
    "TOTAL_COST": (231, 76, 60),     # Đỏ nổi bật (Alizarin) - Tổng tiền thanh toán
    "TIMESTAMP": (52, 152, 219),     # Xanh dương (Peter River) - Ngày giờ
    "ADDRESS": (230, 126, 34),       # Cam (Carrot) - Địa chỉ
    "ITEM_NAME": (241, 196, 15),     # Vàng (Sun Flower) - Tên mặt hàng
    "ITEM_PRICE": (155, 89, 182),    # Tím (Amethyst) - Đơn giá
    "ITEM_QTY": (26, 188, 156),      # Xanh ngọc (Turquoise) - Số lượng
    "ITEMS_LIST": (243, 156, 18),    # Cam đậm - Danh sách món
    "TAX": (26, 188, 156),           # Xanh ngọc - Mã số thuế
    "DEFAULT": (52, 73, 94)          # Xám đậm
}

def parse_box_from_prediction(prediction_text: str):
    """
    Trích xuất tọa độ Bounding Box từ câu trả lời của mô hình nếu có.
    Hỗ trợ format: {"text": "...", "box": [ymin, xmin, ymax, xmax]}
    hoặc chuỗi [ymin, xmin, ymax, xmax] / (ymin, xmin, ymax, xmax).
    """
    try:
        data = json.loads(prediction_text.strip())
        if isinstance(data, dict) and "box" in data:
            return data["box"], data.get("text", "")
    except Exception:
        pass
    
    # Tìm kiếm mẫu regex dạng [y1, x1, y2, x2]
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str(prediction_text))
    if match:
        coords = [int(match.group(i)) for i in range(1, 5)]
        return coords, prediction_text
        
    return None, prediction_text

def draw_bounding_box(image_pil: Image.Image, box, label: str = "", text_content: str = "") -> Image.Image:
    """
    Vẽ khung Bounding Box chuyên nghiệp kèm badge phân loại lên ảnh hóa đơn.
    box: [ymin, xmin, ymax, xmax] (chuẩn hóa trên thang 1000 hoặc pixel thực tế).
    """
    img = image_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    
    ymin, xmin, ymax, xmax = box
    
    # Nếu tọa độ ở thang chuẩn hóa [0, 1000], chuyển đổi về pixel ảnh thực tế
    if max(box) <= 1000 and (w > 1000 or h > 1000):
        x1 = int(xmin * w / 1000)
        y1 = int(ymin * h / 1000)
        x2 = int(xmax * w / 1000)
        y2 = int(ymax * h / 1000)
    else:
        x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)
        
    # Đảm bảo x1 < x2 và y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Giới hạn trong kích thước ảnh
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    color = FIELD_COLORS.get(label.upper(), FIELD_COLORS["DEFAULT"])
    color_fill = color + (45,)       # Màu phủ mờ 18% alpha
    color_outline = color + (255,)   # Viền đặc 100%
    
    # 1. Vẽ vùng phủ mờ bên trong và viền nổi bật (độ dày 3px)
    draw.rectangle([x1, y1, x2, y2], fill=color_fill, outline=color_outline, width=3)
    
    # 2. Vẽ nhãn badge minh chứng ở góc trên khung bao
    clean_text = str(text_content).strip()
    display_tag = f" 📍 {label}: {clean_text[:25]} " if label and clean_text else f" 📍 {label or clean_text} "
    
    tag_h = 24
    tag_w = len(display_tag) * 8 + 14
    tag_y1 = max(0, y1 - tag_h)
    tag_y2 = y1
    tag_x2 = min(w, x1 + tag_w)
    
    draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color_outline)
    draw.text((x1 + 4, tag_y1 + 4), display_tag, fill=(255, 255, 255))
    
    return img

def highlight_prediction_on_image(image_input, prediction_text: str, field_type: str = "DEFAULT") -> Image.Image:
    """
    Tự động xác định hoặc trích xuất Bounding Box từ câu trả lời của mô hình để vẽ khung minh chứng trực quan.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")
        
    w, h = img.size
    
    # 1. Kiểm tra xem câu trả lời của mô hình có chứa tọa độ Bounding Box không
    parsed_box, clean_text = parse_box_from_prediction(prediction_text)
    if parsed_box:
        return draw_bounding_box(img, parsed_box, label=field_type.upper(), text_content=clean_text)
        
    # 2. Nếu là câu hỏi text đơn lẻ, định vị theo vị trí không gian chuẩn của trường thông tin
    field = field_type.upper()
    if "SELLER" in field or "TÊN" in field or "QUÁN" in field:
        box = [int(h * 0.04), int(w * 0.08), int(h * 0.16), int(w * 0.92)]
        lbl = "SELLER"
    elif "TOTAL" in field or "TIỀN" in field or "TỔNG" in field:
        box = [int(h * 0.75), int(w * 0.45), int(h * 0.88), int(w * 0.96)]
        lbl = "TOTAL_COST"
    elif "TIME" in field or "NGÀY" in field or "GIỜ" in field:
        box = [int(h * 0.18), int(w * 0.08), int(h * 0.28), int(w * 0.75)]
        lbl = "TIMESTAMP"
    elif "ITEM" in field or "MÓN" in field or "DỊCH VỤ" in field:
        box = [int(h * 0.32), int(w * 0.06), int(h * 0.68), int(w * 0.94)]
        lbl = "ITEMS_LIST"
    elif "ADDRESS" in field or "ĐỊA CHỈ" in field:
        box = [int(h * 0.12), int(w * 0.08), int(h * 0.22), int(w * 0.92)]
        lbl = "ADDRESS"
    else:
        box = [int(h * 0.35), int(w * 0.08), int(h * 0.65), int(w * 0.92)]
        lbl = "MINH_CHỨNG"
        
    return draw_bounding_box(img, box, label=lbl, text_content=prediction_text)
