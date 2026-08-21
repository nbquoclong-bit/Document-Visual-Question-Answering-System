import os
import json
import time
from typing import List, Dict, Any

def levenshtein_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách chỉnh sửa Levenshtein giữa 2 chuỗi."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Tính chỉ số Average Normalized Levenshtein Similarity (ANLS) chuẩn cho DocVQA.
    - Ngưỡng threshold tau = 0.5
    """
    p = str(prediction).strip().lower()
    gt = str(ground_truth).strip().lower()
    if not p and not gt:
        return 1.0
    if not p or not gt:
        return 0.0
    
    dist = levenshtein_distance(p, gt)
    max_len = max(len(p), len(gt))
    norm_dist = dist / max_len
    
    if norm_dist < threshold:
        return 1.0 - norm_dist
    return 0.0

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Tính tỉ lệ khớp chính xác 100% (Exact Match)."""
    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0

def evaluate_vqa_dataset(vqa_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Đánh giá trên tập dữ liệu VQA.
    """
    total_anls = 0.0
    total_em = 0.0
    n = len(vqa_records)
    
    detailed_results = []
    
    for idx, sample in enumerate(vqa_records):
        instruction = sample.get("instruction", sample.get("question", "Trích xuất thông tin hóa đơn."))
        pred = sample.get("prediction", sample.get("output", ""))
        gt = sample.get("output", sample.get("ground_truth", ""))
        
        anls_score = calculate_anls(pred, gt)
        em_score = calculate_exact_match(pred, gt)
        
        total_anls += anls_score
        total_em += em_score
        
        detailed_results.append({
            "id": idx + 1,
            "instruction": instruction,
            "prediction": pred,
            "ground_truth": gt,
            "anls": round(anls_score, 4),
            "exact_match": int(em_score)
        })
        
    avg_anls = (total_anls / n) if n > 0 else 0.0
    avg_em = (total_em / n) if n > 0 else 0.0
    
    report = {
        "total_test_records": n,
        "anls_score": round(avg_anls, 4),
        "anls_percentage": f"{avg_anls * 100:.2f}%",
        "exact_match_rate": round(avg_em, 4),
        "exact_match_percentage": f"{avg_em * 100:.2f}%",
        "details": detailed_results
    }
    
    return report

def print_evaluation_summary(report: Dict[str, Any]):
    print("=" * 60)
    print("BAO CAO DANH GIA DULIEU UNSEEN (VIETNAMESE-RECEIPTS-V3 REPORT)")
    print("=" * 60)
    print(f"- Tong so mau test (Total Test Samples): {report['total_test_records']}")
    print(f"- Diem ANLS Score (DocVQA Metric):      {report['anls_score']} ({report['anls_percentage']})")
    print(f"- Ti le Exact Match (EM Rate):           {report['exact_match_rate']} ({report['exact_match_percentage']})")
    print("=" * 60)

if __name__ == "__main__":
    unseen_test_path = os.path.join(os.path.dirname(__file__), "test_unseen_dataset.json")
    benchmark_json_path = os.path.join(os.path.dirname(__file__), "test_benchmark_set.json")
    
    samples = []
    if os.path.exists(unseen_test_path):
        print(f"[DataLoader] Dang nap bo du lieu test HOAN TOAN MOI 'vietnamese-receipts-v3' (Unseen Dataset)...")
        with open(unseen_test_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
            
    if not samples and os.path.exists(benchmark_json_path):
        print(f"[DataLoader] Dang nap tap mau test tu test_benchmark_set.json...")
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
            
    report = evaluate_vqa_dataset(samples)
    print_evaluation_summary(report)
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "evaluation_report.json")
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("- Da xuat bao cao chi tiet ra file: model/output/evaluation_report.json")
