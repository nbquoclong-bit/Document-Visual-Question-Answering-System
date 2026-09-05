import json
import os
import sys
import glob

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

output_dir = os.path.join(os.path.dirname(__file__), "output")

if len(sys.argv) > 1 and sys.argv[1] in ["--all", "--compare", "-c"]:
    print("=" * 95)
    print("📊 BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ CÁC PHIÊN BẢN MÔ HÌNH TRONG model/output/")
    print("=" * 95)
    header = f"{'Tên File':<45} | {'Mẫu':<4} | {'ANLS':<8} | {'Exact Match':<11} | {'Token F1':<8} | {'Độ trễ':<8}"
    print(header)
    print("-" * 95)
    
    files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    for f in files:
        fname = os.path.basename(f)
        if fname == "evaluation_report.json":
            continue  # Tránh lặp vì evaluation_report.json là alias của 04
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            n_samples = d.get("total_test_records", len(d.get("details", [])))
            anls = d.get("anls_percentage", "N/A")
            em = d.get("exact_match_percentage", "N/A")
            f1 = d.get("f1_percentage", "N/A")
            lat = f"{d.get('avg_latency_seconds', 0):.2f}s"
            print(f"{fname:<45} | {str(n_samples):<4} | {str(anls):<8} | {str(em):<11} | {str(f1):<8} | {lat:<8}")
        except Exception as e:
            print(f"{fname:<45} | Lỗi đọc file: {e}")
    print("=" * 95)
    print("💡 Mẹo: Xem tài liệu chi tiết tại model/output/README.md")
    sys.exit(0)

# Chọn file cần xem chi tiết (mặc định là evaluation_report.json)
target_file = os.path.join(output_dir, "evaluation_report.json")
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    arg_file = sys.argv[1]
    if os.path.exists(arg_file):
        target_file = arg_file
    elif os.path.exists(os.path.join(output_dir, arg_file)):
        target_file = os.path.join(output_dir, arg_file)

with open(target_file, "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 75)
print(f"📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ({os.path.basename(target_file)}):")
print("=" * 75)
print(f"- Mô hình          : {d.get('model_name')}")
print(f"- Phần cứng        : {d.get('hardware', 'GPU Tesla T4')}")
print(f"- Số mẫu kiểm thử  : {d.get('total_test_records')} câu hỏi")
print(f"- Điểm ANLS        : {d.get('anls_percentage')}")
print(f"- Exact Match      : {d.get('exact_match_percentage')}")
print(f"- Token F1-Score   : {d.get('f1_percentage')}")
print(f"- Độ trễ trung bình: {d.get('avg_latency_seconds')} giây/câu")
print(f"- Dung lượng VRAM  : {d.get('vram_allocated_gb')} GB")
print("-" * 75)
print("🔍 MỘT SỐ MẪU DỰ ĐOÁN THỰC TẾ TRÊN CÁC LOẠI HÓA ĐƠN:")
print("-" * 75)

samples_to_show = [0, 1, 3, 14, 15, 26, 38, 50, 70, 90, 110, 130, 150]
for idx in samples_to_show:
    if idx < len(d.get("details", [])):
        item = d["details"][idx]
        print(f"[{item.get('template')}]")
        print(f"  ❓ Câu hỏi : {item.get('question')}")
        print(f"  🎯 Chuẩn GT: {item.get('ground_truth')}")
        print(f"  🤖 Model PR: {item.get('prediction')}")
        print(f"  📊 ANLS: {item.get('anls')} | EM: {item.get('exact_match')} | F1: {item.get('f1')}")
        print("-" * 75)

print("\n💡 Gợi ý lệnh:")
print("  - So sánh tất cả phiên bản: python model/check_eval_results.py --compare")
print("  - Xem file cụ thể         : python model/check_eval_results.py 02_qwen2_5_vl_3b_baseline_zeroshot.json")
