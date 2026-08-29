import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

# UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def prepare_and_run_baseline():
    kernel_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cấu hình metadata Kaggle
    metadata = {
        "id": "lminhsang241/qwen2-vl-receipt-vqa-golden",
        "title": "qwen2-vl-receipt-vqa-golden",
        "code_file": "qwen2_vl_lora_training.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/docvqa-benchmark-dataset",
            "lminhsang241/docvqa-lora-adapters"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    
    with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Nội dung code Python chạy trên GPU Tesla T4 của Kaggle
    notebook_code = r'''# ==============================================================================
# 🎯 KAGGLE GPU BENCHMARK: BASE MODEL (ZERO-SHOT) VS FINE-TUNED LORA
# Đánh giá độc lập trên GPU Nvidia Tesla T4 (16GB VRAM)
# ==============================================================================

# 1. CÀI ĐẶT THƯ VIỆN
print("=" * 75)
print("📦 [1/4] Cài đặt môi trường trên Kaggle...")
print("=" * 75)

!pip install -q --no-deps qwen-vl-utils==0.0.8
!pip install -q "transformers==4.46.2" "peft==0.13.2" "accelerate==0.34.2" pyyaml

import sys
for mod in list(sys.modules.keys()):
    if any(mod.startswith(k) for k in ["transformers", "peft", "accelerate"]):
        del sys.modules[mod]

import os
import gc
import json
import time
import re
import zipfile
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")
else:
    gpu_name = "CPU"
    print("⚠️ Dùng CPU.")

# 2. HÀM TÍNH METRICS CHUẨN DOCVQA
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
        return 1.0 - norm_dist
    return 0.0

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0

def clean_model_prediction(pred: str) -> str:
    if not pred:
        return ""
    text = str(pred).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        cleaned_json = match.group(1).strip()
        try:
            parsed = json.loads(cleaned_json)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return cleaned_json
    for prefix in ["Đáp án:", "Câu trả lời:", "Dưới đây là", "Thông tin:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text

# 3. QUÉT DỮ LIỆU TEST VÀ GIẢI NÉN
print("\n" + "=" * 75)
print("📊 [2/4] Chuẩn bị dữ liệu kiểm thử thực tế...")
print("=" * 75)

# Giải nén images nếu là zip
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "images.zip":
            with zipfile.ZipFile(os.path.join(root, f), 'r') as zf:
                zf.extractall("/kaggle/working/images")

image_map = {}
for root, dirs, files in os.walk("/kaggle"):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_map[f] = os.path.join(root, f)
            image_map[os.path.splitext(f)[0]] = os.path.join(root, f)

print(f"✅ Đã lập chỉ mục {len(image_map)} ảnh hóa đơn!")

# Đọc tập test questions
test_json_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "test_unseen_dataset.json" in files:
        test_json_path = os.path.join(root, "test_unseen_dataset.json")
        break

eval_samples = []
if test_json_path and os.path.exists(test_json_path):
    with open(test_json_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
        for item in raw_items:
            raw_p = str(item.get("image_path", ""))
            fname = os.path.basename(raw_p.replace('\\', '/'))
            bname = os.path.splitext(fname)[0]
            
            real_path = image_map.get(fname) or image_map.get(bname)
            if real_path and os.path.exists(real_path):
                eval_samples.append({
                    "image_path": real_path,
                    "question": item.get("question", "Trích xuất thông tin hóa đơn."),
                    "ground_truth": item.get("ground_truth", "")
                })

print(f"🎯 Đã khớp thành công {len(eval_samples)} mẫu kiểm thử có ảnh thật!")
test_subset = eval_samples[:30] if len(eval_samples) >= 30 else eval_samples

# 4. NẠP MÔ HÌNH VÀ SUY LUẬN BASE MODEL
print("\n" + "=" * 75)
print("🧠 [3/4] Nạp Base Model Qwen2-VL-2B-Instruct lên GPU...")
print("=" * 75)

base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(base_model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

def run_vqa(m_inst, img_path, question):
    image = Image.open(img_path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia trích xuất thông tin hóa đơn tiếng Việt. Hãy trả lời ngắn gọn, trực diện con số hoặc nội dung được hỏi theo tài liệu, không giải thích lan man."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": 256 * 28 * 28, "max_pixels": 768 * 28 * 28},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(m_inst.device)
    eos_ids = [processor.tokenizer.eos_token_id, 151645, 151643]
    
    with torch.no_grad():
        out = m_inst.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_ids
        )
    trim = [out[0][len(inputs.input_ids[0]):]]
    return processor.batch_decode(trim, skip_special_tokens=True)[0].strip()

# --- A. SUY LUẬN BASE MODEL (ZERO-SHOT) ---
print("\n" + "=" * 75)
print("🚀 [A] CHẠY SUY LUẬN BASE MODEL (ZERO-SHOT - CHƯA FINE-TUNE)...")
print("=" * 75)

base_results = []
base_total_anls = 0.0
base_total_em = 0.0
base_latencies = []

for idx, sample in enumerate(test_subset):
    t0 = time.time()
    raw_pred = run_vqa(model, sample["image_path"], sample["question"])
    lat = time.time() - t0
    base_latencies.append(lat)
    
    clean_pred = clean_model_prediction(raw_pred)
    anls = calculate_anls(clean_pred, sample["ground_truth"])
    em = calculate_exact_match(clean_pred, sample["ground_truth"])
    
    base_total_anls += anls
    base_total_em += em
    
    print(f"[{idx+1}/{len(test_subset)}] {os.path.basename(sample['image_path'])}")
    print(f"   ❓ Câu hỏi: {sample['question']}")
    print(f"   🎯 Ground Truth: {sample['ground_truth']}")
    print(f"   🤖 Base Model Predict: {clean_pred} (Thời gian: {lat:.2f}s | ANLS: {anls:.4f} | EM: {int(em)})")
    
    base_results.append({
        "id": idx + 1,
        "image": os.path.basename(sample["image_path"]),
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prediction": clean_pred,
        "anls": round(anls, 4),
        "exact_match": int(em),
        "latency_seconds": round(lat, 2)
    })

n = len(base_results)
base_avg_anls = base_total_anls / n if n else 0.0
base_avg_em = base_total_em / n if n else 0.0
base_avg_lat = sum(base_latencies) / len(base_latencies) if base_latencies else 0.0
allocated_vram = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

# 5. TỔNG HỢP BÁO CÁO BASELINE
print("\n" + "=" * 75)
print("📊 [4/4] BÁO CÁO KẾT QUẢ BASELINE (BASE MODEL ZERO-SHOT)")
print("=" * 75)
print(f"- Số lượng mẫu kiểm thử : {n}")
print(f"- GPU Phần cứng         : {gpu_name} (16GB VRAM)")
print(f"- GPU VRAM Tiêu thụ     : {allocated_vram:.2f} GB")
print(f"- Latency trung bình    : {base_avg_lat:.2f}s / ảnh")
print(f"- ANLS Score (Baseline) : {base_avg_anls:.4f} ({base_avg_anls * 100:.2f}%)")
print(f"- Exact Match (Baseline): {base_avg_em:.4f} ({base_avg_em * 100:.2f}%)")
print("=" * 75)

report = {
    "total_test_records": n,
    "hardware": f"{gpu_name} (16GB VRAM)",
    "vram_allocated_gb": round(allocated_vram, 2),
    "avg_latency_seconds": round(base_avg_lat, 2),
    "model_name": "Qwen2-VL-2B-Instruct (Base Zero-Shot)",
    "anls_score": round(base_avg_anls, 4),
    "anls_percentage": f"{base_avg_anls * 100:.2f}%",
    "exact_match_rate": round(base_avg_em, 4),
    "exact_match_percentage": f"{base_avg_em * 100:.2f}%",
    "details": base_results
}

with open("/kaggle/working/baseline_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

with open("/kaggle/working/evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n🎉 ĐÃ LƯU BÁO CÁO BASELINE RA FILE: baseline_evaluation_report.json THÀNH CÔNG!")
'''

    notebook_json = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in notebook_code.split("\n")]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(kernel_dir / "qwen2_vl_lora_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)
    print("✅ Đã cập nhật qwen2_vl_lora_training.ipynb để chạy Base Model Evaluation!")

    # 3. Đẩy lên Kaggle và theo dõi
    api = KaggleApi()
    for auth_try in range(5):
        try:
            api.authenticate()
            break
        except Exception as e:
            print(f"  ⚠️ Xác thực Kaggle lần {auth_try+1} bị lỗi ({e}). Thử lại sau 3s...")
            time.sleep(3)
    kernel_slug = "lminhsang241/qwen2-vl-receipt-vqa-golden"
    
    print("🚀 [Kaggle Agent] Đang đẩy notebook lên GPU Kaggle Tesla T4...")
    pushed = False
    for attempt in range(5):
        try:
            api.kernels_push(str(kernel_dir))
            pushed = True
            break
        except Exception as err:
            print(f"  ⚠️ Thử lại lần {attempt+1}/5 ({err})...")
            time.sleep(3)
            
    if not pushed:
        print("❌ Lỗi đẩy kernel lên Kaggle.")
        return False
        
    print(f"✅ Đã đẩy kernel thành công: {kernel_slug}")
    print("⏳ Đang theo dõi tiến trình chạy thực tế trên GPU Tesla T4...")
    
    time.sleep(15)
    start_time = time.time()
    while True:
        status_info = api.kernels_status(kernel_slug)
        status = getattr(status_info, "status", str(status_info))
        status_str = status.name if hasattr(status, "name") else str(status).upper()
        elapsed = time.time() - start_time
        print(f"  ⏱️ [{int(elapsed)}s] Trạng thái Kernel: {status_str}")
        
        if "COMPLETE" in status_str and elapsed > 25:
            print("\n🎉 BASE MODEL EVALUATION ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN GPU KAGGLE!")
            break
        elif "ERROR" in status_str or "CANCEL" in status_str:
            print(f"\n❌ Kernel kết thúc với trạng thái: {status_str}")
            return False
            
        time.sleep(15)
        
    # 4. Tải file kết quả báo cáo về local
    print("\n📦 Đang tải file báo cáo baseline về máy...")
    target_output_dir = Path("d:/STUDY/MLIoT/project/model/output")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    with api.build_kaggle_client() as kaggle:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "lminhsang241"
        req.kernel_slug = "qwen2-vl-receipt-vqa-golden"
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        
        for item in response.files or []:
            if "baseline" in item.file_name or "evaluation" in item.file_name:
                out_path = target_output_dir / os.path.basename(item.file_name)
                print(f"⬇️ Đang tải {item.file_name} -> {out_path}...")
                resp = requests.get(item.url)
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Đã tải và cập nhật: {out_path}")
                
    return True

if __name__ == "__main__":
    prepare_and_run_baseline()
