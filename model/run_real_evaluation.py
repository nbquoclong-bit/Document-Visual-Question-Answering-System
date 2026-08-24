import sys
import os

# Tự động nạp thư mục gốc dự án và thư mục model vào sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import json
import time
from typing import List, Dict, Any

try:
    from stage1_vlm.src.inference import VQAEngine
except ImportError:
    from model.stage1_vlm.src.inference import VQAEngine

try:
    from evaluate_metrics import calculate_anls, calculate_exact_match, print_evaluation_summary
except ImportError:
    from model.evaluate_metrics import calculate_anls, calculate_exact_match, print_evaluation_summary

def find_adapter_dir():
    possible_paths = [
        "stage1_vlm/output/lora_adapters",
        "output/lora_adapters",
        "stage1_vlm/output"
    ]
    base_dir = os.path.dirname(__file__)
    for p in possible_paths:
        full_p = os.path.join(base_dir, p)
        if os.path.exists(os.path.join(full_p, "adapter_config.json")):
            return full_p
    return None

def run_real_model_evaluation(num_samples: int = 10):
    """
    Cho mô hình Qwen2-VL SUY LUẬN THỰC TẾ trên các file ảnh hóa đơn thực tế
    từ bộ archive.zip và tính toán chỉ số ANLS + Exact Match thật 100%.
    """
    print("=" * 60)
    print("BAT DAU CHAY SUY LUAN THUC TE TREN MO HINH QWEN2-VL VLM")
    print("=" * 60)
    
    adapter_dir = find_adapter_dir()
    print(f"[ModelLoader] Loading VQAEngine voi adapter: {adapter_dir}...")
    
    try:
        engine = VQAEngine(adapter_dir=adapter_dir)
        print("[ModelLoader] Nap VQAEngine thanh cong!")
    except Exception as e:
        print(f"[Error] Loi nap mo hinh: {e}")
        return

    unseen_path = os.path.join(os.path.dirname(__file__), "test_unseen_dataset.json")
    vlm_test_path = os.path.join(os.path.dirname(__file__), "data", "vlm_test.json")

    all_samples = []
    if os.path.exists(unseen_path):
        with open(unseen_path, "r", encoding="utf-8") as f:
            all_samples.extend(json.load(f))
            
    if os.path.exists(vlm_test_path):
        with open(vlm_test_path, "r", encoding="utf-8") as f:
            vlm_test_records = json.load(f)
            for r in vlm_test_records:
                all_samples.append({
                    "image_path": r.get("image_path", ""),
                    "question": r.get("instruction", "Trích xuất thông tin hóa đơn."),
                    "ground_truth": r.get("output", "")
                })

    if not all_samples:
        print("[Error] Không tìm thấy dữ liệu test ở cả test_unseen_dataset.json lẫn data/vlm_test.json.")
        return

    # Quét tự động thư mục datasets để lập chỉ mục tất cả các file ảnh thực tế
    datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    image_map = {}
    if os.path.exists(datasets_dir):
        for root, _, files in os.walk(datasets_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_map[f] = os.path.join(root, f)

    eval_samples = []
    for s in all_samples:
        raw_path = str(s.get("image_path", "")).replace('\\', '/')
        img_name = os.path.basename(raw_path)
        
        if img_name in image_map:
            s["image_path"] = image_map[img_name]
            eval_samples.append(s)
        elif raw_path and os.path.exists(raw_path):
            s["image_path"] = raw_path
            eval_samples.append(s)
            
        if len(eval_samples) >= num_samples:
            break
                
    if not eval_samples:
        print("[Notice] Khong tim thay file anh dinh kem tuong ung trong thu muc datasets/.")
        print("👉 Hãy đảm bảo bạn đã chạy prepare_vlm_data để tạo dữ liệu test.")
        return
        
    print(f"[Inference] Da chon {len(eval_samples)} mau hoa don thuc te de suy luan danh gia...")

    
    results = []
    total_anls = 0.0
    total_em = 0.0
    
    for idx, sample in enumerate(eval_samples):
        img = sample["image_path"]
        q = sample["question"]
        gt = sample["ground_truth"]
        
        print(f"\n[{idx+1}/{len(eval_samples)}] Dang doc anh: {os.path.basename(img)}...")
        print(f"   - Cau hoi: {q}")
        print(f"   - Dap an chuan: {gt}")
        
        start_t = time.time()
        try:
            pred = engine.extract_and_answer(img, q)
        except Exception as err:
            pred = f"Loi suy luan: {err}"
        elapsed = time.time() - start_t
        
        anls_val = calculate_anls(pred, gt)
        em_val = calculate_exact_match(pred, gt)
        
        total_anls += anls_val
        total_em += em_val
        
        print(f"   > Du doan tu Qwen2-VL: {pred} (Thoi gian: {elapsed:.2f}s)")
        print(f"   > Diem ANLS: {anls_val:.4f} | Exact Match: {int(em_val)}")
        
        results.append({
            "id": idx + 1,
            "image": os.path.basename(img),
            "question": q,
            "prediction": pred,
            "ground_truth": gt,
            "anls": round(anls_val, 4),
            "exact_match": int(em_val),
            "latency_seconds": round(elapsed, 2)
        })
        
    n = len(results)
    avg_anls = total_anls / n if n > 0 else 0.0
    avg_em = total_em / n if n > 0 else 0.0
    
    report = {
        "total_test_records": n,
        "total_evaluated": n,
        "anls_score": round(avg_anls, 4),
        "anls_percentage": f"{avg_anls * 100:.2f}%",
        "exact_match_rate": round(avg_em, 4),
        "exact_match_percentage": f"{avg_em * 100:.2f}%",
        "results": results
    }
    
    print_evaluation_summary(report)
    
    out_file = os.path.join(os.path.dirname(__file__), "output", "real_evaluation_report.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"- Da luu ket qua suy luan thuc te ra file: model/output/real_evaluation_report.json")

if __name__ == "__main__":
    run_real_model_evaluation(num_samples=10)
