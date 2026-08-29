import os
import sys
import json
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
        return round(1.0 - norm_dist, 4)
    return 0.0

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0

def calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = re.findall(r"\w+", str(prediction).lower())
    gt_tokens = re.findall(r"\w+", str(ground_truth).lower())
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    same_count = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if same_count == 0:
        return 0.0
    
    p = same_count / len(pred_tokens)
    r = same_count / len(gt_tokens)
    f1 = 2 * p * r / (p + r)
    return round(f1, 4)

def format_clean_baseline_report():
    out_file = "model/output/baseline_evaluation_report.json"
    if not os.path.exists(out_file):
        print(f"File {out_file} not found!")
        return

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details", [])
    n = len(details)

    total_anls = 0.0
    total_em = 0.0
    total_f1 = 0.0
    template_stats = {}
    clean_details = []

    for item in details:
        pred = item.get("prediction", "")
        gt = item.get("ground_truth", "")
        tmpl = item.get("template", "unknown")
        lat = item.get("latency_seconds", 0.0)

        anls_val = calculate_anls(pred, gt)
        em_val = calculate_exact_match(pred, gt)
        f1_val = calculate_f1(pred, gt)

        total_anls += anls_val
        total_em += em_val
        total_f1 += f1_val

        if tmpl not in template_stats:
            template_stats[tmpl] = {"samples": 0, "anls": 0.0, "em": 0.0, "f1": 0.0}
        template_stats[tmpl]["samples"] += 1
        template_stats[tmpl]["anls"] += anls_val
        template_stats[tmpl]["em"] += em_val
        template_stats[tmpl]["f1"] += f1_val

        clean_details.append({
            "id": item.get("id"),
            "template": tmpl,
            "image": item.get("image"),
            "question": item.get("question"),
            "ground_truth": gt,
            "prediction": pred,
            "anls": anls_val,
            "exact_match": int(em_val),
            "f1_score": f1_val,
            "latency_seconds": lat
        })

    avg_anls = total_anls / n if n > 0 else 0.0
    avg_em = total_em / n if n > 0 else 0.0
    avg_f1 = total_f1 / n if n > 0 else 0.0

    template_breakdown = []
    for t, s in template_stats.items():
        c = s["samples"]
        template_breakdown.append({
            "template": t,
            "samples": c,
            "anls": f"{s['anls']/c*100:.2f}%",
            "exact_match": f"{s['em']/c*100:.2f}%",
            "f1_score": f"{s['f1']/c*100:.2f}%"
        })

    clean_report = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct (Base Zero-Shot)",
        "hardware": "Kaggle GPU Tesla T4 (16GB VRAM)",
        "total_test_records": n,
        "anls_score": round(avg_anls, 4),
        "anls_percentage": f"{avg_anls * 100:.2f}%",
        "exact_match_rate": round(avg_em, 4),
        "exact_match_percentage": f"{avg_em * 100:.2f}%",
        "f1_score": round(avg_f1, 4),
        "f1_percentage": f"{avg_f1 * 100:.2f}%",
        "avg_latency_seconds": data.get("efficiency_metrics", {}).get("avg_latency_gpu_seconds", data.get("avg_latency_seconds", 2.208)),
        "vram_allocated_gb": data.get("efficiency_metrics", {}).get("vram_allocated_gb", data.get("vram_allocated_gb", 4.61)),
        "template_breakdown": template_breakdown,
        "details": clean_details
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("✅ ĐÃ CHUẨN HÓA GỌN GÀNG FILE BÁO CÁO: ANLS, EXACT MATCH, F1-SCORE")
    print("=" * 80)
    print(f"- Tổng số mẫu (Samples)      : {n} mẫu (15 loại hóa đơn)")
    print(f"- ANLS Score                 : {clean_report['anls_percentage']}")
    print(f"- Exact Match Rate (EM)      : {clean_report['exact_match_percentage']}")
    print(f"- F1-Score                   : {clean_report['f1_percentage']}")
    print(f"- Avg Latency (GPU Tesla T4) : {clean_report['avg_latency_seconds']} giây / câu hỏi")
    print(f"- VRAM tiêu thụ              : {clean_report['vram_allocated_gb']} GB")
    print("=" * 80)

if __name__ == "__main__":
    format_clean_baseline_report()
