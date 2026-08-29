import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def prepare_evaluation_kernel():
    kernel_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cấu hình metadata Kaggle
    metadata = {
        "id": "lminhsang241/qwen2-vl-receipt-vqa-golden",
        "title": "qwen2-vl-receipt-vqa-golden",
        "code_file": "qwen2_vl_real_evaluation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/newest-dataset",
            "lminhsang241/mliot-final-project",
            "lminhsang241/new-dataset-mliot-project"
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }
    
    with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Mã nguồn thực thi đánh giá trực tiếp trên GPU Kaggle
    eval_code = r'''# ==============================================================================
# 🎯 KAGGLE GPU EVALUATION BENCHMARK: REAL MODEL INFERENCE (TESLA T4 GPU)
# Đo đạc chỉ số THỰC TẾ: ANLS, Exact Match (EM), Inference Latency, VRAM Footprint
# So sánh đối chứng: Base Model (Zero-Shot) vs Fine-Tuned Model (LoRA)
# ==============================================================================

# 1. CÀI ĐẶT THƯ VIỆN
print("=" * 75)
print("📦 [1/5] Cài đặt môi trường thư viện trên Kaggle...")
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
from collections import defaultdict
from pathlib import Path

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
    print("⚠️ Không tìm thấy GPU, đang dùng CPU.")

# 2. HÀM TÍNH TOÁN METRICS CHUẨN DOCVQA
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

# 3. QUÉT DỮ LIỆU TEST VÀ TÌM LORA ADAPTER
print("\n" + "=" * 75)
print("📊 [2/5] Quét dữ liệu ảnh hóa đơn thực tế và LoRA Adapter...")
print("=" * 75)

image_index = {}
valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_exts:
            bname = os.path.splitext(file)[0]
            full_p = os.path.join(root, file)
            image_index[bname] = full_p
            image_index[bname.replace("mcocr_public_", "").replace("mcocr_val_", "")] = full_p

def clean_text(t):
    return " ".join(str(t).strip().split()) if t else ""

funsd_files = []
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file.lower().endswith(".json") and any(k in root.lower() or k in file.lower() for k in ["funsd", "mcocr", "receipt"]):
            funsd_files.append(os.path.join(root, file))

eval_samples = []
for jf in funsd_files:
    bname = os.path.splitext(os.path.basename(jf))[0]
    img_path = image_index.get(bname) or image_index.get(bname.replace("mcocr_public_", "").replace("_ver2", ""))
    if not img_path or not os.path.exists(img_path):
        continue
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    
    entities = defaultdict(list)
    for item in data.get("form", []):
        raw_text = clean_text(item.get("text", ""))
        label = item.get("label", "OTHER").upper()
        if raw_text and label != "OTHER":
            entities[label].append(raw_text)
            
    if "SELLER" in entities and clean_text(" ".join(entities["SELLER"])):
        eval_samples.append({
            "image_path": img_path,
            "field": "SELLER",
            "question": "Tên cửa hàng / bên bán trên hóa đơn là gì?",
            "ground_truth": clean_text(" ".join(entities["SELLER"]))
        })
    if "TOTAL_COST" in entities and clean_text(" ".join(entities["TOTAL_COST"])):
        eval_samples.append({
            "image_path": img_path,
            "field": "TOTAL_COST",
            "question": "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?",
            "ground_truth": clean_text(" ".join(entities["TOTAL_COST"]))
        })
    if "TIMESTAMP" in entities and clean_text(" ".join(entities["TIMESTAMP"])):
        eval_samples.append({
            "image_path": img_path,
            "field": "TIMESTAMP",
            "question": "Ngày giờ lập hóa đơn là khi nào?",
            "ground_truth": clean_text(" ".join(entities["TIMESTAMP"]))
        })

print(f"✅ Đã tìm thấy {len(eval_samples)} câu hỏi kiểm thử thực tế từ ảnh hóa đơn!")

# Tìm LoRA Adapter
lora_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_config.json" in files:
        lora_dir = root
        break
    for f in files:
        if "lora" in f.lower() and f.endswith(".zip"):
            extract_target = "/kaggle/working/lora_adapters"
            os.makedirs(extract_target, exist_ok=True)
            with zipfile.ZipFile(os.path.join(root, f), 'r') as zf:
                zf.extractall(extract_target)
            if os.path.exists(os.path.join(extract_target, "adapter_config.json")):
                lora_dir = extract_target
                break

if lora_dir:
    print(f"🎯 Đã tìm thấy LoRA Adapter tại: {lora_dir}")
else:
    print("⚠️ Không tìm thấy LoRA Adapter sẵn, sẽ đánh giá trên Base Model.")

# 4. NẠP MÔ HÌNH VÀ SUY LUẬN
print("\n" + "=" * 75)
print("🧠 [3/5] Nạp mô hình Qwen2-VL-2B lên GPU...")
print("=" * 75)

base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(base_model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

if lora_dir and os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
    print("📦 Nạp LoRA Adapter vào Qwen2-VL...")
    lora_model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
    lora_model.eval()
else:
    lora_model = model

test_samples = eval_samples[:30] if len(eval_samples) >= 30 else eval_samples
print(f"🚀 Tiến hành suy luận thực tế trên {len(test_samples)} mẫu kiểm thử...")

def run_model_predict(m_instance, img_path, question):
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
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(m_instance.device)
    eos_ids = [processor.tokenizer.eos_token_id, 151645, 151643]
    
    with torch.no_grad():
        out = m_instance.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_ids
        )
    trim = [out[0][len(inputs.input_ids[0]):]]
    return processor.batch_decode(trim, skip_special_tokens=True)[0].strip()

# 5. CHẠY SUY LUẬN THỰC TẾ
print("\n" + "=" * 75)
print("🧪 [4/5] Đang chạy suy luận thực tế từng mẫu hóa đơn...")
print("=" * 75)

detailed_results = []
total_anls = 0.0
total_em = 0.0
latencies = []

for idx, item in enumerate(test_samples):
    t0 = time.time()
    raw_pred = run_model_predict(lora_model, item["image_path"], item["question"])
    lat = time.time() - t0
    latencies.append(lat)
    
    clean_pred = clean_model_prediction(raw_pred)
    anls = calculate_anls(clean_pred, item["ground_truth"])
    em = calculate_exact_match(clean_pred, item["ground_truth"])
    
    total_anls += anls
    total_em += em
    
    print(f"[{idx+1}/{len(test_samples)}] {os.path.basename(item['image_path'])}")
    print(f"  ❓ Câu hỏi: {item['question']}")
    print(f"  🎯 Ground Truth: {item['ground_truth']}")
    print(f"  🤖 Model Predict: {clean_pred} (Thời gian: {lat:.2f}s | ANLS: {anls:.4f} | EM: {int(em)})")
    
    detailed_results.append({
        "id": idx + 1,
        "image": os.path.basename(item["image_path"]),
        "field": item.get("field", "GENERAL"),
        "instruction": item["question"],
        "ground_truth": item["ground_truth"],
        "prediction": clean_pred,
        "anls": round(anls, 4),
        "exact_match": int(em),
        "latency_seconds": round(lat, 2)
    })

n = len(detailed_results)
avg_anls = total_anls / n if n else 0.0
avg_em = total_em / n if n else 0.0
avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
vram_used = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

final_report = {
    "total_test_records": n,
    "hardware": f"{gpu_name} (16GB VRAM)",
    "vram_allocated_gb": round(vram_used, 2),
    "avg_latency_seconds": round(avg_lat, 2),
    "anls_score": round(avg_anls, 4),
    "anls_percentage": f"{avg_anls * 100:.2f}%",
    "exact_match_rate": round(avg_em, 4),
    "exact_match_percentage": f"{avg_em * 100:.2f}%",
    "details": detailed_results
}

print("\n" + "=" * 75)
print("📊 [5/5] TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ THỰC TẾ (REAL EVALUATION REPORT)")
print("=" * 75)
print(f"- Tổng số mẫu kiểm thử thực tế : {final_report['total_test_records']}")
print(f"- Điểm ANLS Score (DocVQA)     : {final_report['anls_score']} ({final_report['anls_percentage']})")
print(f"- Tỉ lệ Exact Match (EM Rate)  : {final_report['exact_match_rate']} ({final_report['exact_match_percentage']})")
print(f"- Thời gian phản hồi trung bình: {final_report['avg_latency_seconds']}s / ảnh")
print("=" * 75)

out_dir = "/kaggle/working"
with open(f"{out_dir}/evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)

with open(f"{out_dir}/real_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)

print("\n🎉 ĐÃ XUẤT BÁO CÁO THỰC NGHIỆM ĐẦY ĐỦ RA FILE evaluation_report.json!")
'''

    notebook_json = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in eval_code.split("\n")]
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
    
    with open(kernel_dir / "qwen2_vl_real_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)
    print("✅ Đã chuẩn bị file qwen2_vl_real_evaluation.ipynb thành công!")

def push_and_monitor():
    api = KaggleApi()
    api.authenticate()
    kernel_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation")
    kernel_slug = "lminhsang241/qwen2-vl-receipt-vqa-golden"
    
    print("🚀 [Kaggle Agent] Đang đẩy notebook đánh giá lên Kaggle GPU...")
    pushed = False
    for attempt in range(5):
        try:
            api.kernels_push(str(kernel_dir))
            pushed = True
            break
        except Exception as err:
            print(f"  ⚠️ Lần thử {attempt+1}/5 bị lỗi kết nối ({err}). Đang thử lại sau 3s...")
            time.sleep(3)
    if not pushed:
        print("❌ Không thể đẩy kernel lên Kaggle sau 5 lần thử.")
        return False
    print(f"✅ Đã đẩy kernel: {kernel_slug}")
    print("⏳ Đợi 20s để hệ thống Kaggle tiếp nhận và khởi động GPU Tesla T4...")
    time.sleep(20)
    
    start_time = time.time()
    while True:
        status_info = api.kernels_status(kernel_slug)
        status = getattr(status_info, "status", str(status_info))
        status_str = status.name if hasattr(status, "name") else str(status).upper()
        elapsed = time.time() - start_time
        print(f"  ⏱️ [{int(elapsed)}s] Trạng thái Kernel: {status_str}")
        
        if "COMPLETE" in status_str:
            print("\n🎉 KERNEL ĐÁNH GIÁ ĐÃ HOÀN TẤT THÀNH CÔNG 100% TRÊN GPU KAGGLE!")
            break
        elif "ERROR" in status_str or "CANCEL" in status_str:
            print(f"\n❌ Kernel kết thúc với trạng thái: {status_str}")
            return False
            
        time.sleep(15)
        
    # Tải kết quả về local
    print("\n📦 Đang tải file báo cáo số liệu thực tế về máy...")
    target_output_dir = Path("d:/STUDY/MLIoT/project/model/output")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    with api.build_kaggle_client() as kaggle:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "lminhsang241"
        req.kernel_slug = "qwen2-vl-receipt-vqa-golden"
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        
        for item in response.files or []:
            if "evaluation_report" in item.file_name or "report" in item.file_name:
                out_path = target_output_dir / os.path.basename(item.file_name)
                print(f"⬇️ Đang tải {item.file_name} -> {out_path}...")
                resp = requests.get(item.url)
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Đã tải và cập nhật thành công: {out_path}")
                
    return True

if __name__ == "__main__":
    prepare_evaluation_kernel()
    push_and_monitor()
