import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

# Cấu hình UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def prepare_and_run_optimized_training():
    print("=" * 85)
    print("🚀 [KAGGLE GPU TRAINING] HUẤN LUYỆN TỐI ƯU HÓA TOÀN DIỆN QWEN2.5-VL (SYSTEM PROMPT + DATA SCALE)")
    print("=" * 85)

    api = KaggleApi()
    api.authenticate()
    print("✅ Xác thực thành công tài khoản Kaggle: lminhsang241")

    kernel_slug = "qwen2-5-vl-finetune-optimized"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    train_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/train_kernel_optimized")
    train_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-5-vl-finetune-optimized",
        "code_file": "qwen2_5_vl_optimized.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/docvqa-benchmark-dataset"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(train_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    val_path = Path("d:/STUDY/MLIoT/project/datasets/val_benchmark_upload/multitemplate_validation_questions.json")
    with open(val_path, "r", encoding="utf-8") as f:
        multitemplate_validation_questions = json.load(f)

    # Đọc 2,400 mẫu đa dạng từ vlm_train_master phủ trọn 15 template và 8 nhóm trường
    train_master_path = Path("d:/STUDY/MLIoT/project/model/data/vlm_train_master.json")
    train_samples = []
    if train_master_path.exists():
        with open(train_master_path, "r", encoding="utf-8") as f:
            all_train = json.load(f)
            template_counts = {}
            for item in all_train:
                t = item.get("template", "unknown")
                if template_counts.get(t, 0) < 160:
                    train_samples.append({
                        "image_name": os.path.basename(item.get("image_path", "")),
                        "template": t,
                        "field": item.get("field", ""),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("answer", "")
                    })
                    template_counts[t] = template_counts.get(t, 0) + 1

    print(f"📊 Đã chuẩn bị {len(train_samples)} mẫu huấn luyện nâng cao (2,400 samples) trên 15 loại hóa đơn.")

    SYSTEM_PROMPT = "Bạn là trợ lý AI kế toán chuyên trích xuất hóa đơn. Hãy đọc ảnh và trả lời câu hỏi thật ngắn gọn, chính xác tuyệt đối giá trị thực thể, không thêm bất kỳ lời chào hay giải thích nào."

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 FINE-TUNING TỐI ƯU HÓA TOÀN DIỆN QWEN2.5-VL-3B\n",
                "- **System Prompt Optimization:** Ép câu trả lời súc tích tuyệt đối.\n",
                "- **Data Scale:** 2,400 mẫu cân bằng trên 15 template.\n",
                "- **PEFT LoRA:** r=16, alpha=32, Gradient Checkpointing (VRAM < 6GB)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Cài đặt môi trường chuẩn xác\n",
                "import os\n",
                "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
                "\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers>=4.49.0\" \"peft>=0.13.2\" \"accelerate>=0.34.2\" pillow torchvision\n",
                "\n",
                "import sys\n",
                "for mod in list(sys.modules.keys()):\n",
                "    if any(mod.startswith(k) for k in [\"transformers\", \"peft\", \"accelerate\", \"torchao\", \"qwen_vl_utils\"]):\n",
                "        del sys.modules[mod]\n",
                "\n",
                "import time\n",
                "import json\n",
                "import re\n",
                "import zipfile\n",
                "import torch\n",
                "from PIL import Image\n",
                "from torch.utils.data import Dataset\n",
                "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor\n",
                "from peft import LoraConfig, get_peft_model, TaskType\n",
                "from qwen_vl_utils import process_vision_info\n",
                "\n",
                "print(f\"🔥 GPU: {torch.cuda.get_device_name(0)}\")\n",
                "print(f\"🧠 Tổng VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Giải nén và lập chỉ mục ảnh\n",
                "extract_dir = \"/kaggle/working/extracted_images\"\n",
                "os.makedirs(extract_dir, exist_ok=True)\n",
                "\n",
                "for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "    for f in files:\n",
                "        if f == \"images.zip\":\n",
                "            print(f\"📦 Đang giải nén {f}...\")\n",
                "            with zipfile.ZipFile(os.path.join(root, f), 'r') as zf:\n",
                "                zf.extractall(extract_dir)\n",
                "\n",
                "image_map = {}\n",
                "for root, dirs, files in os.walk(\"/kaggle\"):\n",
                "    for f in files:\n",
                "        if f.lower().endswith((\".png\", \".jpg\", \".jpeg\")):\n",
                "            image_map[f] = os.path.join(root, f)\n",
                "\n",
                "print(f\"📸 Tổng số ảnh đã lập chỉ mục trên Kaggle: {len(image_map)}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Khởi tạo Qwen2.5-VL-3B với Gradient Checkpointing\n",
                "model_name = \"Qwen/Qwen2.5-VL-3B-Instruct\"\n",
                "print(f\"⏳ Đang nạp Base Model: {model_name} (Native FP16)... \")\n",
                "processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28)\n",
                "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n",
                "    model_name,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "model.gradient_checkpointing_enable()\n",
                "model.enable_input_require_grads()\n",
                "\n",
                "lora_config = LoraConfig(\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.05,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "    bias=\"none\",\n",
                "    task_type=TaskType.CAUSAL_LM\n",
                ")\n",
                "\n",
                "model = get_peft_model(model, lora_config)\n",
                "model.print_trainable_parameters()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Dataset với Strict System Prompt & Target-Only Masking\n",
                f"raw_train_data = {json.dumps(train_samples, ensure_ascii=False, indent=2)}\n",
                f"SYSTEM_INSTRUCTION = \"{SYSTEM_PROMPT}\"\n",
                "\n",
                "train_items = []\n",
                "for item in raw_train_data:\n",
                "    img_name = item.get(\"image_name\", \"\")\n",
                "    if img_name in image_map:\n",
                "        item[\"full_image_path\"] = image_map[img_name]\n",
                "        train_items.append(item)\n",
                "\n",
                "print(f\"🎯 Khớp thành công {len(train_items)} mẫu huấn luyện nâng cao!\")\n",
                "\n",
                "class OptimizedDocVQADataset(Dataset):\n",
                "    def __init__(self, items, processor):\n",
                "        self.items = items\n",
                "        self.processor = processor\n",
                "    \n",
                "    def __len__(self):\n",
                "        return len(self.items)\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        item = self.items[idx]\n",
                "        image = Image.open(item[\"full_image_path\"]).convert(\"RGB\")\n",
                "        messages = [\n",
                "            {\"role\": \"system\", \"content\": SYSTEM_INSTRUCTION},\n",
                "            {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": image}, {\"type\": \"text\", \"text\": item[\"question\"]}]},\n",
                "            {\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": item[\"ground_truth\"]}]}\n",
                "        ]\n",
                "        prompt_only = [\n",
                "            {\"role\": \"system\", \"content\": SYSTEM_INSTRUCTION},\n",
                "            {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": image}, {\"type\": \"text\", \"text\": item[\"question\"]}]}\n",
                "        ]\n",
                "        \n",
                "        full_text = self.processor.apply_chat_template(messages, tokenize=False)\n",
                "        prompt_text = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)\n",
                "        \n",
                "        image_inputs, video_inputs = process_vision_info(messages)\n",
                "        inputs = self.processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=False, return_tensors=\"pt\")\n",
                "        prompt_inputs = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=False, return_tensors=\"pt\")\n",
                "        \n",
                "        input_ids = inputs.input_ids[0]\n",
                "        prompt_len = prompt_inputs.input_ids.shape[1]\n",
                "        \n",
                "        labels = input_ids.clone()\n",
                "        labels[:prompt_len] = -100\n",
                "        \n",
                "        return {\n",
                "            \"input_ids\": input_ids,\n",
                "            \"attention_mask\": inputs.attention_mask[0],\n",
                "            \"pixel_values\": inputs.pixel_values if \"pixel_values\" in inputs else None,\n",
                "            \"image_grid_thw\": inputs.image_grid_thw if \"image_grid_thw\" in inputs else None,\n",
                "            \"labels\": labels\n",
                "        }\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Vòng lặp Huấn luyện Tối ưu hóa\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)\n",
                "dataset = OptimizedDocVQADataset(train_items, processor)\n",
                "\n",
                "num_epochs = 3\n",
                "grad_accum_steps = 16\n",
                "batch_size = 1\n",
                "total_steps = (len(dataset) // (batch_size * grad_accum_steps)) * num_epochs\n",
                "lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))\n",
                "\n",
                "print(f\"🚀 BẮT ĐẦU HUẤN LUYỆN NÂNG CAO ({len(dataset)} MẪU)...\")\n",
                "\n",
                "model.train()\n",
                "step_count = 0\n",
                "\n",
                "for epoch in range(num_epochs):\n",
                "    t_start = time.time()\n",
                "    running_loss = 0.0\n",
                "    optimizer.zero_grad()\n",
                "    \n",
                "    for idx in range(len(dataset)):\n",
                "        item = dataset[idx]\n",
                "        input_ids = item[\"input_ids\"].unsqueeze(0).to(\"cuda\")\n",
                "        attention_mask = item[\"attention_mask\"].unsqueeze(0).to(\"cuda\")\n",
                "        labels = item[\"labels\"].unsqueeze(0).to(\"cuda\")\n",
                "        \n",
                "        kwargs = {\"input_ids\": input_ids, \"attention_mask\": attention_mask, \"labels\": labels}\n",
                "        if item[\"pixel_values\"] is not None:\n",
                "            kwargs[\"pixel_values\"] = item[\"pixel_values\"].to(\"cuda\")\n",
                "        if item[\"image_grid_thw\"] is not None:\n",
                "            kwargs[\"image_grid_thw\"] = item[\"image_grid_thw\"].to(\"cuda\")\n",
                "            \n",
                "        outputs = model(**kwargs)\n",
                "        loss = outputs.loss / grad_accum_steps\n",
                "        loss.backward()\n",
                "        running_loss += outputs.loss.item()\n",
                "        \n",
                "        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(dataset):\n",
                "            optimizer.step()\n",
                "            lr_scheduler.step()\n",
                "            optimizer.zero_grad()\n",
                "            step_count += 1\n",
                "            \n",
                "            if step_count % 15 == 0:\n",
                "                cur_lr = optimizer.param_groups[0][\"lr\"]\n",
                "                vram_gb = torch.cuda.max_memory_allocated() / (1024**3)\n",
                "                print(f\"Epoch [{epoch+1}/{num_epochs}] Step {step_count} | Loss: {outputs.loss.item():.4f} | LR: {cur_lr:.2e} | VRAM: {vram_gb:.2f} GB\")\n",
                "                \n",
                "    avg_loss = running_loss / len(dataset)\n",
                "    print(f\"✅ Hoàn thành Epoch {epoch+1}/{num_epochs} trong {time.time() - t_start:.1f}s | Avg Loss: {avg_loss:.4f}\")\n",
                "\n",
                "output_adapter_dir = \"/kaggle/working/qwen2_5_vl_lora_adapters\"\n",
                "model.save_pretrained(output_adapter_dir)\n",
                "processor.save_pretrained(output_adapter_dir)\n",
                "!cd /kaggle/working && zip -r qwen2_5_vl_lora_adapters.zip qwen2_5_vl_lora_adapters\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Đánh giá Tối ưu hóa kèm Bộ lọc Hậu xử lý (Post-Processing)\n",
                "print(\"=\" * 85)\n",
                "print(\"🚀 BẮT ĐẦU ĐÁNH GIÁ CHUẨN ĐỊNH LƯỢNG TỐI ƯU HÓA TRÊN 174 CÂU HỎI...\")\n",
                "print(\"=\" * 85)\n",
                "\n",
                f"validation_samples = {json.dumps(multitemplate_validation_questions, ensure_ascii=False, indent=2)}\n",
                "\n",
                "matched_val = []\n",
                "for s in validation_samples:\n",
                "    img_name = s[\"image_name\"]\n",
                "    if img_name in image_map:\n",
                "        s[\"full_image_path\"] = image_map[img_name]\n",
                "        matched_val.append(s)\n",
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
                "model.eval()\n",
                "total_anls, total_em, total_f1 = 0.0, 0.0, 0.0\n",
                "latencies = []\n",
                "eval_results = []\n",
                "\n",
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
                "        gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)\n",
                "        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]\n",
                "        pred_raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()\n",
                "        pred = clean_prediction(pred_raw)\n",
                "        \n",
                "    lat = time.time() - t0\n",
                "    latencies.append(lat)\n",
                "    anls = calculate_anls(pred, s[\"ground_truth\"])\n",
                "    em = 1.0 if pred.strip().lower() == s[\"ground_truth\"].strip().lower() else 0.0\n",
                "    f1 = calculate_token_f1(pred, s[\"ground_truth\"])\n",
                "    \n",
                "    total_anls += anls\n",
                "    total_em += em\n",
                "    total_f1 += f1\n",
                "    \n",
                "    eval_results.append({\n",
                "        \"id\": idx + 1,\n",
                "        \"template\": s.get(\"template\", \"\"),\n",
                "        \"question\": s[\"question\"],\n",
                "        \"ground_truth\": s[\"ground_truth\"],\n",
                "        \"prediction\": pred,\n",
                "        \"anls\": round(anls, 4),\n",
                "        \"exact_match\": int(em),\n",
                "        \"f1\": round(f1, 4)\n",
                "    })\n",
                "\n",
                "n = len(matched_val)\n",
                "final_report = {\n",
                "    \"model_name\": \"Qwen/Qwen2.5-VL-3B-Instruct (LoRA Advanced Optimized)\",\n",
                "    \"hardware\": f\"Kaggle GPU {torch.cuda.get_device_name(0)}\",\n",
                "    \"total_test_records\": n,\n",
                "    \"anls_percentage\": f\"{total_anls / n * 100:.2f}%\",\n",
                "    \"exact_match_percentage\": f\"{total_em / n * 100:.2f}%\",\n",
                "    \"f1_percentage\": f\"{total_f1 / n * 100:.2f}%\",\n",
                "    \"avg_latency_seconds\": round(sum(latencies)/len(latencies), 3),\n",
                "    \"vram_allocated_gb\": round(torch.cuda.max_memory_allocated() / (1024**3), 2),\n",
                "    \"details\": eval_results\n",
                "}\n",
                "\n",
                "with open(\"/kaggle/working/evaluation_report.json\", \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(final_report, f, ensure_ascii=False, indent=2)\n",
                "\n",
                "print(\"\\n\" + \"=\" * 85)\n",
                "print(\"📊 TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TỐI ƯU HÓA TOÀN DIỆN:\")\n",
                "print(\"=\" * 85)\n",
                "print(f\"- Điểm ANLS Score        : {final_report['anls_percentage']}\")\n",
                "print(f\"- Tỉ lệ Exact Match (EM) : {final_report['exact_match_percentage']}\")\n",
                "print(f\"- Điểm Token F1-Score    : {final_report['f1_percentage']}\")\n",
                "print(f\"- Độ trễ trung bình      : {final_report['avg_latency_seconds']} s/câu\")\n",
                "print(\"=\" * 85)\n"
            ]
        }
    ]

    notebook_content = {
        "cells": notebook_cells,
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

    nb_path = train_dir / "qwen2_5_vl_optimized.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"📦 Đã tạo Notebook Tối Ưu Hóa tại: {nb_path}")
    print("📤 Đang đẩy Kernel Tối Ưu Hóa lên Kaggle GPU...")
    api.kernels_push(str(train_dir))
    print(f"🚀 Đã kích hoạt Kaggle Kernel Tối Ưu Hóa: https://www.kaggle.com/code/{kernel_id}")
    print("-" * 85)

if __name__ == "__main__":
    prepare_and_run_optimized_training()
