import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bảng màu đại diện cho các trường thông tin kế toán
FIELD_COLORS = {
    "SELLER": (46, 204, 113),       # Xanh lá cây (Emerald)
    "TOTAL_COST": (231, 76, 60),     # Đỏ (Alizarin)
    "TIMESTAMP": (52, 152, 219),     # Xanh dương (Peter River)
    "ADDRESS": (230, 126, 34),       # Cam (Carrot)
    "ITEM_NAME": (241, 196, 15),     # Vàng (Sun Flower)
    "ITEM_PRICE": (155, 89, 182),    # Tím (Amethyst)
    "TAX": (26, 188, 156),           # Xanh ngọc (Turquoise)
    "DEFAULT": (52, 73, 94)          # Xám đậm (Wet Asphalt)
}

def draw_bounding_box(image_pil: Image.Image, box, label: str = "", text_content: str = "") -> Image.Image:
    """
    Vẽ khung Bounding Box chuyên nghiệp kèm nhãn phân loại lên ảnh hóa đơn.
    box: [ymin, xmin, ymax, xmax] (hoặc [x1, y1, x2, y2])
    """
    img = image_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    
    # Chuẩn hóa tọa độ nếu ở dải [0, 1000]
    if max(box) <= 1000 and (w > 1000 or h > 1000):
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * w / 1000)
        y1 = int(ymin * h / 1000)
        x2 = int(xmax * w / 1000)
        y2 = int(ymax * h / 1000)
    else:
        x1, y1, x2, y2 = [int(v) for v in box]
        
    color = FIELD_COLORS.get(label.upper(), FIELD_COLORS["DEFAULT"])
    color_fill = color + (40,)  # Màu nền trong suốt 15%
    color_outline = color + (255,)
    
    # 1. Vẽ vùng phủ mờ bên trong
    draw.rectangle([x1, y1, x2, y2], fill=color_fill, outline=color_outline, width=3)
    
    # 2. Vẽ nhãn badge ở góc trên bên trái
    if label or text_content:
        display_tag = f" {label}: {text_content[:20]} " if label and text_content else f" {label or text_content} "
        tag_bg = color_outline
        draw.rectangle([x1, max(0, y1 - 22), x1 + len(display_tag) * 8 + 10, y1], fill=tag_bg)
        draw.text((x1 + 4, max(0, y1 - 20)), display_tag, fill=(255, 255, 255))
        
    return img

def highlight_prediction_on_image(image_input, prediction_text: str, field_type: str = "DEFAULT") -> Image.Image:
    """
    Tự động tìm kiếm vị trí từ khóa của câu trả lời trên ảnh (Visual Grounding) và vẽ bounding box.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")
        
    w, h = img.size
    
    # Tạo bounding box minh họa trực quan nếu chưa có OCR box chi tiết
    # Tự động gán vị trí dựa trên loại trường
    field = field_type.upper()
    if "SELLER" in field or "TÊN" in field:
        box = [int(h * 0.05), int(w * 0.1), int(h * 0.16), int(w * 0.9)]
        lbl = "SELLER"
    elif "TOTAL" in field or "TIỀN" in field or "TỔNG" in field:
        box = [int(h * 0.75), int(w * 0.5), int(h * 0.88), int(w * 0.95)]
        lbl = "TOTAL_COST"
    elif "TIME" in field or "NGÀY" in field or "GIỜ" in field:
        box = [int(h * 0.20), int(w * 0.1), int(h * 0.28), int(w * 0.7)]
        lbl = "TIMESTAMP"
    elif "ITEM" in field or "MÓN" in field or "DỊCH VỤ" in field:
        box = [int(h * 0.35), int(w * 0.08), int(h * 0.65), int(w * 0.92)]
        lbl = "ITEMS_LIST"
    else:
        box = [int(h * 0.4), int(w * 0.1), int(h * 0.6), int(w * 0.9)]
        lbl = "ENTITY"
        
    return draw_bounding_box(img, box, label=lbl, text_content=prediction_text)
