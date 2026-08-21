import os
import json
import time
from typing import List, Dict, Any

def levenshtein_distance(s1: str, s2: str) -> int:
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
    """
    Tính tỉ lệ khớp chính xác 100% (Exact Match).
    """
    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0

def run_benchmark_evaluation(predictions: List[str], ground_truths: List[str]):
    """
    Tính toán và in ra báo cáo tổng hợp ANLS, Exact Match (EM).
    """
    total_anls = 0.0
    total_em = 0.0
    n = len(predictions)
    
    for p, gt in zip(predictions, ground_truths):
        total_anls += calculate_anls(p, gt)
        total_em += calculate_exact_match(p, gt)
        
    avg_anls = (total_anls / n) if n > 0 else 0.0
    avg_em = (total_em / n) if n > 0 else 0.0
    
    print("=" * 50)
    print("KET QUA DANH GIA CHI SO (EVALUATION REPORT)")
    print("=" * 50)
    print(f"- So luong cau hoi (Test Samples): {n}")
    print(f"- ANLS Score (DocVQA Metric):    {avg_anls:.4f} ({avg_anls * 100:.2f}%)")
    print(f"- Exact Match (EM Rate):         {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print("=" * 50)
    
    return {"ANLS": avg_anls, "Exact_Match": avg_em}

if __name__ == "__main__":
    # Test thử nghiệm với ví dụ mẫu
    sample_preds = [
        "12.000.000 VNĐ",
        "Nhà xuất bản Huyền Thoại",
        "0312345678"
    ]
    sample_gts = [
        "12.000.000 VNĐ",
        "Nhà Xuất Bản Huyền Thoại",
        "0312345678"
    ]
    run_benchmark_evaluation(sample_preds, sample_gts)
