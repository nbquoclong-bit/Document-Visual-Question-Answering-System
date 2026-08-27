import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

if not torch.cuda.is_available():
    torch.set_num_threads(min(8, os.cpu_count() or 4))

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model.stage1_vlm.src.inference import VQAEngine

def test_json_output():
    print("=" * 80)
    print("🧪 [KIỂM THỬ JSON] TEST BÓC TÁCH THÔNG TIN HÓA ĐƠN DẠNG JSON CẤU TRÚC")
    print("=" * 80)
    
    adapter_path = os.path.abspath("model/stage1_vlm/output/lora_adapters")
    print(f"📦 Đang nạp mô hình Qwen2-VL kèm LoRA Adapter tại: {adapter_path}...")
    
    t0_load = time.time()
    engine = VQAEngine(adapter_dir=adapter_path)
    print(f"✅ Nạp mô hình thành công trong: {time.time() - t0_load:.2f}s!\n")
    
    img_path = "datasets/vietnamese-receipts-v3/val/images/cafe_highlands_val_001.png"
    if not os.path.exists(img_path):
        for p in Path("datasets").rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                img_path = str(p)
                break
                
    print(f"📸 Ảnh kiểm thử: {img_path}")
    print("-" * 80)
    
    # Prompt trích xuất JSON chuẩn Schema
    json_prompt = (
        "Trích xuất toàn bộ thông tin hóa đơn và trả về DUY NHẤT một JSON hợp lệ có cấu trúc như sau:\n"
        "{\n"
        '  "store_name": "Tên cửa hàng",\n'
        '  "timestamp": "Ngày giờ lập hóa đơn",\n'
        '  "items": ["Danh sách món đã mua"],\n'
        '  "total_amount": "Tổng tiền thanh toán"\n'
        "}"
    )
    
    print("❓ PROMPT YÊU CẦU TRÍCH XUẤT:")
    print(json_prompt)
    print("-" * 80)
    print("⏳ AI đang đọc hóa đơn và sinh cấu trúc JSON...")
    
    t0_gen = time.time()
    raw_response = engine.extract_and_answer(img_path, json_prompt)
    latency = time.time() - t0_gen
    
    print("\n" + "=" * 80)
    print(f"🤖 RAW OUTPUT TỪ MÔ HÌNH (Thời gian: {latency:.2f}s):")
    print("=" * 80)
    print(raw_response)
    print("=" * 80)
    
    # Kiểm tra tính hợp lệ của JSON
    print("\n🔍 ĐÁNH GIÁ CHUẨN ĐỊNH DẠNG JSON:")
    
    # Thử bóc tách JSON nếu có kèm markdown code block
    candidate = raw_response.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json")[1].split("```")[0].strip()
    elif "```" in candidate:
        candidate = candidate.split("```")[1].split("```")[0].strip()
        
    try:
        parsed_json = json.loads(candidate)
        print("✅ KẾT QUẢ: ĐÚNG CHUẨN JSON 100%! (Valid JSON Syntax)")
        print("\n📋 DỮ LIỆU ĐÃ PARSE THÀNH CÔNG (SẴN SÀNG NẠP VÀO DATABASE / BACKEND):")
        print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
        
        print("\n📊 CHI TIẾT CÁC TRƯỜNG ĐÃ BÓC TÁCH ĐƯỢC:")
        for k, v in parsed_json.items():
            print(f"  🔹 {k:<15}: {v}")
            
    except json.JSONDecodeError as err:
        print(f"⚠️ JSON CẦN LÀM SẠCH (Lỗi parse: {err})")
        print("👉 Gợi ý: Dùng regex wrapper trong backend để tự động chuẩn hóa.")

if __name__ == "__main__":
    test_json_output()
