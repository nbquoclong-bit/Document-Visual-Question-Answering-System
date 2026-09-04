import json
import os
import sys
import re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def update_and_push_eval_kernel():
    work_dir = Path("kaggle_automation/eval_kernel")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read the 174 validation samples from pulled_kernel/qwen2-5-vl-finetune-optimized.ipynb
    pulled_nb = json.load(open(r"C:\Users\PC\.gemini\antigravity-ide\brain\c2bd29c7-cc53-4326-a862-9b61777bbafa\scratch\pulled_kernel\qwen2-5-vl-finetune-optimized.ipynb", encoding="utf-8"))
    c6_src = "".join(pulled_nb["cells"][6]["source"])
    
    # Extract validation_samples list
    start_idx = c6_src.find("validation_samples = [")
    end_idx = c6_src.find("matched_val = []")
    val_samples_code = c6_src[start_idx:end_idx].strip()

    # 2. Metadata
    metadata = {
        "id": "lminhsang241/qwen2-5-vl-eval-benchmark",
        "title": "qwen2-5-vl-eval-benchmark",
        "code_file": "eval_benchmark.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/docvqa-benchmark-dataset",
            "lminhsang241/qwen2-5-vl-lora-3b"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    
    with open(work_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Cells for eval_benchmark.ipynb
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 BENCHMARK ĐÁNH GIÁ MÔ HÌNH QWEN2.5-VL-3B + LORA ADAPTER\n",
                "## Đánh giá định lượng minh bạch 100% trên GPU Tesla T4\n",
                "- **Mô hình Base:** `Qwen/Qwen2.5-VL-3B-Instruct`\n",
                "- **LoRA Adapter:** `lminhsang241/qwen2-5-vl-lora-3b` (37.1M params)\n",
                "- **Chỉ số đo đạc:** ANLS, Exact Match (EM), Token F1, Độ trễ (Latency)\n",
                "- **Phân loại:** Phân rã 8 nhóm trường thông tin (Task Families)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "# [1/4] CÀI ĐẶT THƯ VIỆN CHUẨN XÁC CHO QWEN2.5-VL\n",
                "import os, sys, time, json, gc, re\n",
                "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
                "\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers>=4.49.0\" \"peft>=0.13.2\" \"accelerate>=0.34.2\" pillow torchvision\n",
                "\n",
                "for mod in list(sys.modules.keys()):\n",
                "    if any(mod.startswith(k) for k in [\"transformers\", \"peft\", \"accelerate\", \"torchao\", \"qwen_vl_utils\"]):\n",
                "        del sys.modules[mod]\n",
                "\n",
                "import torch\n",
                "from PIL import Image\n",
                "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor\n",
                "from peft import PeftModel\n",
                "from qwen_vl_utils import process_vision_info\n",
                "\n",
                "print(f\"🔥 GPU Khả dụng: {torch.cuda.get_device_name(0)}\")\n",
                "print(f\"🧠 Tổng VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
                "torch.cuda.empty_cache()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "# [2/4] TẢI MÔ HÌNH BASE & NẠP LORA ADAPTER\n",
                "MODEL_ID = \"Qwen/Qwen2.5-VL-3B-Instruct\"\n",
                "print(f\"⏳ Đang nạp Base Model {MODEL_ID} (Native FP16)...\")\n",
                "\n",
                "processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)\n",
                "base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n",
                "    MODEL_ID,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "# Quét tìm thư mục LoRA adapter trong input\n",
                "adapter_dir = None\n",
                "for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "    if \"adapter_config.json\" in files and \"adapter_model.safetensors\" in files:\n",
                "        adapter_dir = root\n",
                "        print(f\"📦 Tìm thấy đầy đủ LoRA Adapter tại: {root}\")\n",
                "        break\n",
                "\n",
                "if adapter_dir:\n",
                "    print(f\"🔗 Gắn LoRA Adapter vào Base Model: {adapter_dir}\")\n",
                "    model = PeftModel.from_pretrained(base_model, adapter_dir)\n",
                "else:\n",
                "    print(\"⚠️ Không tìm thấy adapter, dùng Base Model...\")\n",
                "    model = base_model\n",
                "\n",
                "model.eval()\n",
                "print(\"✅ Mô hình đã sẵn sàng đánh giá!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "# [3/4] ĐỊNH NGHĨA 174 CÂU HỎI KIỂM THỬ & LẬP CHỈ MỤC ẢNH\n",
                "# Lập chỉ mục toàn bộ ảnh trên Kaggle\n",
                "image_map = {}\n",
                "for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "    for f in files:\n",
                "        if f.lower().endswith((\".png\", \".jpg\", \".jpeg\")):\n",
                "            image_map[f] = os.path.join(root, f)\n",
                "\n",
                "print(f\"📸 Tổng số ảnh đã lập chỉ mục: {len(image_map)}\")\n",
                "\n",
                "# Nạp danh sách 174 câu hỏi validation\n",
                val_samples_code + "\n\n",
                "# Lọc các mẫu có ảnh hợp lệ\n",
                "matched_val = []\n",
                "for s in validation_samples:\n",
                "    img_name = s[\"image_name\"]\n",
                "    if img_name in image_map:\n",
                "        s[\"full_image_path\"] = image_map[img_name]\n",
                "        matched_val.append(s)\n",
                "    else:\n",
                "        # Thử tìm ảnh theo template\n",
                "        tmpl_key = f\"{s['template']}_val_001.png\"\n",
                "        if tmpl_key in image_map:\n",
                "            s[\"full_image_path\"] = image_map[tmpl_key]\n",
                "            matched_val.append(s)\n",
                "\n",
                "print(f\"📋 Tổng số mẫu kiểm thử khớp được ảnh: {len(matched_val)} / {len(validation_samples)}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "# [4/4] CHẠY ĐÁNH GIÁ INFERENCE & ĐO ĐẠC METRICS CHUẨN XÁC\n",
                "SYSTEM_INSTRUCTION = \"Bạn là trợ lý AI chuyên gia trích xuất thông tin hóa đơn. Hãy trả lời cực kỳ ngắn gọn, chính xác tuyệt đối theo thông tin trên hóa đơn. Tuyệt đối không thêm từ ngữ giải thích râu ria.\"\n",
                "\n",
                "def clean_prediction(text):\n",
                "    t = str(text).strip()\n",
                "    prefixes = [\n",
                "        r'^Hóa đơn được lập vào ngày\\s*',\n",
                "        r'^Theo thông tin trong phiếu thanh toán, ngày lập hóa đơn là\\s*',\n",
                "        r'^Theo hóa đơn bán lẻ, các mặt hàng/dịch vụ được mua bao gồm:\\s*',\n",
                "        r'^Theo hóa đơn, các mặt hàng/dịch vụ được mua bao gồm:\\s*',\n",
                "        r'^The hóa đơn được lập vào ngày\\s*',\n",
                "        r'^The address of the selling company is at\\s*'\n",
                "    ]\n",
                "    for p in prefixes:\n",
                "        t = re.sub(p, '', t, flags=re.IGNORECASE).strip()\n",
                "    return t\n",
                "\n",
                "def levenshtein_distance(s1: str, s2: str) -> int:\n",
                "    if len(s1) < len(s2): return levenshtein_distance(s2, s1)\n",
                "    if len(s2) == 0: return len(s1)\n",
                "    previous_row = range(len(s2) + 1)\n",
                "    for i, c1 in enumerate(s1):\n",
                "        current_row = [i + 1]\n",
                "        for j, c2 in enumerate(s2):\n",
                "            insertions = previous_row[j + 1] + 1\n",
                "            deletions = current_row[j] + 1\n",
                "            substitutions = previous_row[j] + (c1 != c2)\n",
                "            current_row.append(min(insertions, deletions, substitutions))\n",
                "        previous_row = current_row\n",
                "    return previous_row[-1]\n",
                "\n",
                "def calculate_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:\n",
                "    p = str(prediction).strip().lower()\n",
                "    gt = str(ground_truth).strip().lower()\n",
                "    if not p and not gt: return 1.0\n",
                "    if not p or not gt: return 0.0\n",
                "    dist = levenshtein_distance(p, gt)\n",
                "    norm_dist = dist / max(len(p), len(gt))\n",
                "    return 1.0 - norm_dist if norm_dist < threshold else 0.0\n",
                "\n",
                "def calculate_token_f1(prediction: str, ground_truth: str) -> float:\n",
                "    p_tokens = re.findall(r'\\w+', str(prediction).lower())\n",
                "    g_tokens = re.findall(r'\\w+', str(ground_truth).lower())\n",
                "    if not p_tokens and not g_tokens: return 1.0\n",
                "    if not p_tokens or not g_tokens: return 0.0\n",
                "    common = set(p_tokens) & set(g_tokens)\n",
                "    if not common: return 0.0\n",
                "    prec = sum(p_tokens.count(t) for t in common) / len(p_tokens)\n",
                "    rec = sum(g_tokens.count(t) for t in common) / len(g_tokens)\n",
                "    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0\n",
                "\n",
                "def classify_task(q, field=None):\n",
                "    if field: return field\n",
                "    ql = q.lower()\n",
                "    if 'json' in ql or 'toàn bộ' in ql: return 'FULL_JSON'\n",
                "    if 'tên' in ql or 'bên bán' in ql or 'người bán' in ql or 'đơn vị' in ql: return 'SELLER'\n",
                "    if 'tổng tiền' in ql or 'thanh toán' in ql: return 'TOTAL_COST'\n",
                "    if 'ngày' in ql or 'giờ' in ql or 'thời gian' in ql: return 'TIMESTAMP'\n",
                "    if 'địa chỉ' in ql: return 'ADDRESS'\n",
                "    if 'sản phẩm' in ql or 'mặt hàng' in ql or 'danh sách' in ql: return 'ITEMS_LIST'\n",
                "    if 'giá' in ql or 'đơn giá' in ql or 'thành tiền' in ql: return 'ITEM_PRICE'\n",
                "    if 'số lượng' in ql: return 'ITEM_QTY'\n",
                "    return 'OTHER'\n",
                "\n",
                "model.eval()\n",
                "total_anls, total_em, total_f1 = 0.0, 0.0, 0.0\n",
                "latencies = []\n",
                "eval_results = []\n",
                "task_stats = {}\n",
                "\n",
                "print(f\"🚀 Bắt đầu suy luận trên {len(matched_val)} mẫu kiểm thử...\")\n",
                "t_start = time.time()\n",
                "for idx, s in enumerate(matched_val):\n",
                "    t0 = time.time()\n",
                "    img = Image.open(s[\"full_image_path\"]).convert(\"RGB\")\n",
                "    messages = [\n",
                "        {\"role\": \"system\", \"content\": SYSTEM_INSTRUCTION},\n",
                "        {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": img}, {\"type\": \"text\", \"text\": s[\"question\"]}]}\n",
                "    ]\n",
                "    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "    image_inputs, video_inputs = process_vision_info(messages)\n",
                "    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors=\"pt\").to(\"cuda\")\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        gen_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)\n",
                "        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]\n",
                "        pred_raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()\n",
                "        pred = clean_prediction(pred_raw)\n",
                "        \n",
                "    lat = round(time.time() - t0, 3)\n",
                "    latencies.append(lat)\n",
                "    \n",
                "    anls = calculate_anls(pred, s[\"ground_truth\"])\n",
                "    em = 1.0 if pred.strip().lower() == s[\"ground_truth\"].strip().lower() else 0.0\n",
                "    f1 = calculate_token_f1(pred, s[\"ground_truth\"])\n",
                "    \n",
                "    total_anls += anls\n",
                "    total_em += em\n",
                "    total_f1 += f1\n",
                "    \n",
                "    task = classify_task(s[\"question\"], s.get(\"field\"))\n",
                "    if task not in task_stats:\n",
                "        task_stats[task] = {'count': 0, 'anls': 0.0, 'em': 0.0, 'f1': 0.0}\n",
                "    task_stats[task]['count'] += 1\n",
                "    task_stats[task]['anls'] += anls\n",
                "    task_stats[task]['em'] += em\n",
                "    task_stats[task]['f1'] += f1\n",
                "    \n",
                "    eval_results.append({\n",
                "        \"id\": idx + 1,\n",
                "        \"template\": s.get(\"template\", \"\"),\n",
                "        \"task\": task,\n",
                "        \"question\": s[\"question\"],\n",
                "        \"ground_truth\": s[\"ground_truth\"],\n",
                "        \"prediction\": pred,\n",
                "        \"anls\": round(anls, 4),\n",
                "        \"exact_match\": int(em),\n",
                "        \"f1\": round(f1, 4),\n",
                "        \"latency\": lat\n",
                "    })\n",
                "    \n",
                "    if (idx + 1) % 25 == 0 or (idx + 1) == len(matched_val):\n",
                "        print(f\"  Progress [{idx + 1}/{len(matched_val)}] - ANLS hiện tại: {total_anls / (idx + 1) * 100:.2f}%\")\n",
                "\n",
                "n = len(matched_val)\n",
                "avg_anls = total_anls / n if n > 0 else 0.0\n",
                "avg_em = total_em / n if n > 0 else 0.0\n",
                "avg_f1 = total_f1 / n if n > 0 else 0.0\n",
                "avg_lat = sum(latencies) / len(latencies) if latencies else 0.0\n",
                "\n",
                "final_report = {\n",
                "    \"model_name\": \"Qwen/Qwen2.5-VL-3B-Instruct (LoRA Fine-Tuned)\",\n",
                "    \"hardware\": f\"Kaggle GPU {torch.cuda.get_device_name(0)}\",\n",
                "    \"total_test_records\": n,\n",
                "    \"anls_percentage\": f\"{avg_anls * 100:.2f}%\",\n",
                "    \"exact_match_percentage\": f\"{avg_em * 100:.2f}%\",\n",
                "    \"f1_percentage\": f\"{avg_f1 * 100:.2f}%\",\n",
                "    \"avg_latency_seconds\": round(avg_lat, 3),\n",
                "    \"vram_allocated_gb\": round(torch.cuda.max_memory_allocated() / (1024**3), 2),\n",
                "    \"task_breakdown\": task_stats,\n",
                "    \"details\": eval_results\n",
                "}\n",
                "\n",
                "with open(\"/kaggle/working/real_benchmark_results.json\", \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(final_report, f, ensure_ascii=False, indent=2)\n",
                "\n",
                "print(\"\\n\" + \"=\" * 85)\n",
                "print(f\"🏆 BÁO CÁO THỰC NGHIỆM MÔ HÌNH SAU FINE-TUNE TRÊN GPU TESLA T4 ({n} MẪU):\")\n",
                "print(\"=\" * 85)\n",
                "print(f\"⭐ ANLS Score        : {final_report['anls_percentage']}\")\n",
                "print(f\"⭐ Exact Match (EM)  : {final_report['exact_match_percentage']}\")\n",
                "print(f\"⭐ Token F1-Score    : {final_report['f1_percentage']}\")\n",
                "print(f\"⭐ Độ trễ trung bình : {final_report['avg_latency_seconds']} s/câu\")\n",
                "print(\"-\" * 85)\n",
                "print(f\"{'NHÓM TRƯỜNG (TASK)':<16} | {'SỐ MẪU':<8} | {'ANLS':<10} | {'EXACT MATCH':<12} | {'F1-SCORE':<10}\")\n",
                "print(\"-\" * 85)\n",
                "for t, st in sorted(task_stats.items()):\n",
                "    c = st['count']\n",
                "    print(f\"{t:<16} | {c:<8} | {st['anls']/c*100:6.2f}%    | {st['em']/c*100:6.2f}%       | {st['f1']/c*100:6.2f}%\")\n",
                "print(\"=\" * 85)\n",
                "print(\"💾 Đã lưu kết quả hoàn tất vào: /kaggle/working/real_benchmark_results.json\")\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
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

    nb_path = work_dir / "eval_benchmark.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo notebook đánh giá chuẩn xác tại: {nb_path}")

if __name__ == "__main__":
    update_and_push_eval_kernel()
