import json
import sys

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("model/output/evaluation_report.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 75)
print("📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH SAU KHI FINETUNE TRÊN 174 CÂU HỎI:")
print("=" * 75)
print(f"- Mô hình        : {d.get('model_name')}")
print(f"- Số mẫu kiểm thử: {d.get('total_test_records')} câu hỏi")
print(f"- Điểm ANLS      : {d.get('anls_percentage')}")
print(f"- Exact Match    : {d.get('exact_match_percentage')}")
print(f"- Token F1-Score : {d.get('f1_percentage')}")
print(f"- Độ trễ trung bình: {d.get('avg_latency_seconds')} giây/câu")
print(f"- Dung lượng VRAM: {d.get('vram_allocated_gb')} GB")
print("-" * 75)
print("🔍 MỘT SỐ MẪU DỰ ĐOÁN THỰC TẾ TRÊN CÁC LOẠI HÓA ĐƠN:")
print("-" * 75)

samples_to_show = [0, 1, 3, 14, 15, 26, 38, 50, 70, 90, 110, 130, 150]
for idx in samples_to_show:
    if idx < len(d["details"]):
        item = d["details"][idx]
        print(f"[{item.get('template')}]")
        print(f"  ❓ Câu hỏi : {item.get('question')}")
        print(f"  🎯 Chuẩn GT: {item.get('ground_truth')}")
        print(f"  🤖 Model PR: {item.get('prediction')}")
        print(f"  📊 ANLS: {item.get('anls')} | EM: {item.get('exact_match')} | F1: {item.get('f1')}")
        print("-" * 75)
