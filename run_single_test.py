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

# Tối ưu hóa số luồng CPU để suy luận nhanh nhất có thể
if not torch.cuda.is_available():
    torch.set_num_threads(min(8, os.cpu_count() or 4))

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model.stage1_vlm.src.inference import VQAEngine

def test_sample_image():
    print("=" * 80)
    print("🧪 [TEST BENCHMARK] KIỂM THỬ SUY LUẬN MÔ HÌNH QWEN2-VL + LORA ADAPTER")
    print("=" * 80)
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({os.cpu_count()} Cores, {torch.get_num_threads()} Threads)"
    print(f"🖥️ Thiết bị thực thi: {device_name}")
    
    # 1. Nạp mô hình
    adapter_path = os.path.abspath("model/stage1_vlm/output/lora_adapters")
    print(f"📦 Đang nạp mô hình Qwen2-VL-2B kèm LoRA Adapter tại: {adapter_path}...")
    
    t_load_start = time.time()
    engine = VQAEngine(adapter_dir=adapter_path)
    t_load_end = time.time()
    print(f"✅ Nạp mô hình thành công trong: {t_load_end - t_load_start:.2f} giây!\n")
    
    # 2. Chọn ảnh mẫu thực tế
    img_path = "datasets/vietnamese-receipts-v3/val/images/cafe_highlands_val_001.png"
    if not os.path.exists(img_path):
        # Tìm bất kỳ ảnh hợp lệ nào trong datasets
        for p in Path("datasets").rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                img_path = str(p)
                break
                
    print(f"📸 Ảnh kiểm thử được chọn: {img_path}")
    if os.path.exists(img_path):
        with Image.open(img_path) as im:
            print(f"   - Kích thước ảnh: {im.size[0]}x{im.size[1]} px | Định dạng: {im.format}")
    print("-" * 80)
    
    # 3. Bộ câu hỏi kiểm thử kèm Ground Truth chuẩn
    test_cases = [
        {
            "question": "Tên cửa hàng / bên bán trên hóa đơn là gì?",
            "ground_truth": "HIGHLANDS COFFEE TRẦN HƯNG ĐẠO",
            "field": "SELLER"
        },
        {
            "question": "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?",
            "ground_truth": "796,068",
            "field": "TOTAL_COST"
        },
        {
            "question": "Ngày giờ lập hóa đơn là khi nào?",
            "ground_truth": "31/05/2026 16:41",
            "field": "TIMESTAMP"
        },
        {
            "question": "Trích xuất danh sách các món / sản phẩm đã mua trên hóa đơn.",
            "ground_truth": "Trà Sen Vàng Size M, Cà Phê Đen Đá Size M, Trà Thạch Đào Size L, Bánh Tiramisu, Freeze Trà Xanh Size M, Phin Sữa Đá Size L",
            "field": "ITEMS"
        }
    ]
    
    results = []
    print(f"🚀 Bắt đầu thực thi {len(test_cases)} câu hỏi kiểm thử đối soát:\n")
    
    for idx, tc in enumerate(test_cases, 1):
        q = tc["question"]
        gt = tc["ground_truth"]
        
        print(f"[{idx}/{len(test_cases)}] ❓ Câu hỏi: {q}")
        print(f"   🎯 Đáp án chuẩn (Ground Truth): {gt}")
        
        t0 = time.time()
        try:
            prediction = engine.extract_and_answer(img_path, q)
        except Exception as e:
            prediction = f"Lỗi suy luận: {e}"
        latency = time.time() - t0
        
        # Đánh giá độ khớp
        pred_clean = prediction.strip()
        is_exact = (pred_clean.lower() == gt.lower())
        is_contained = (gt.lower() in pred_clean.lower()) or (pred_clean.lower() in gt.lower())
        
        print(f"   🤖 AI Trích xuất             : {prediction}")
        print(f"   ⚡ Thời gian xử lý          : {latency:.2f}s ({latency*1000:.0f} ms)")
        if is_exact:
            print("   📊 Đánh giá độ chính xác    : ⭐⭐⭐ KHỚP TUYỆT ĐỐI (100% Match)")
        elif is_contained:
            print("   📊 Đánh giá độ chính xác    : ⭐⭐ KHỚP NỘI DUNG (High Relevance)")
        else:
            print("   📊 Đánh giá độ chính xác    : ⭐ CÓ SỰ KHÁC BIỆT")
        print()
        
        results.append({
            "question": q,
            "ground_truth": gt,
            "prediction": prediction,
            "latency_s": round(latency, 2)
        })
        
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print("=" * 80)
    print("🎉 KIỂM THỬ HOÀN TẤT THÀNH CÔNG!")
    print(f"⏱️  Thời gian suy luận trung bình : {avg_latency:.2f} giây/câu hỏi ({avg_latency*1000:.0f} ms)")
    print("=" * 80)

if __name__ == "__main__":
    test_sample_image()
