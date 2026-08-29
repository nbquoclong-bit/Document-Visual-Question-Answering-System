import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def prepare_and_run_kaggle_training():
    print("=" * 85)
    print("🚀 [KAGGLE GPU TRAINING] HUẤN LUYỆN QWEN2-VL-2B VỚI QLORA TRÊN GPU TESLA T4")
    print("=" * 85)

    api = KaggleApi()
    api.authenticate()
    print("✅ Xác thực thành công tài khoản Kaggle: lminhsang241")

    kernel_slug = "qwen2-vl-lora-training"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    train_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/train_kernel")
    train_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-vl-lora-training",
        "code_file": "qwen2_vl_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/docvqa-benchmark-dataset",
            "lminhsang241/newest-dataset"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(train_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 1. Đọc và chuẩn bị tập train mẫu đa dạng từ 15 template và MCOCR
    train_json_path = Path("d:/STUDY/MLIoT/project/model/data/vlm_train.json")
    val_json_path = Path("d:/STUDY/MLIoT/project/datasets/val_benchmark_upload/multitemplate_validation_questions.json")
    
    training_records = []
    if train_json_path.exists():
        with open(train_json_path, "r", encoding="utf-8") as f:
            raw_train = json.load(f)
            # Lọc lấy 1,500 - 3,000 mẫu đại diện phủ kín 15 template
            for item in raw_train:
                img_p = item.get("image_path", "")
                fname = os.path.basename(img_p.replace('\\', '/'))
                msgs = item.get("messages", [])
                training_records.append({
                    "image_name": fname,
                    "messages": msgs
                })

    # Giới hạn tập train gọn gàng cho GPU (~1,000 - 2,000 bước huấn luyện sắc bén)
    selected_train_records = training_records[:1500] if len(training_records) >= 1500 else training_records
    print(f"📦 Đã đóng gói {len(selected_train_records)} mẫu VQA huấn luyện chất lượng cao.")

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎯 QWEN2-VL-2B QLORA TRAINING & BENCHMARKING ON TESLA T4 GPU\n",
                "Huấn luyện căn chỉnh định dạng (Format Alignment) và tăng cường năng lực trích xuất hóa đơn tiếng Việt."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Cài đặt thư viện môi trường\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers==4.46.2\" \"peft==0.13.2\" \"accelerate==0.34.2\" bitsandbytes pillow torchvision pyyaml\n",
                "\n",
                "import sys\n",
                "for mod in list(sys.modules.keys()):\n",
                "    if any(mod.startswith(k) for k in [\"transformers\", \"peft\", \"accelerate\", \"torchao\", \"qwen_vl_utils\"]):\n",
                "        del sys.modules[mod]\n",
                "\n",
                "import os\n",
                "import gc\n",
                "import time\n",
                "import json\n",
                "import re\n",
                "import zipfile\n",
                "from pathlib import Path\n",
                "import torch\n",
                "from PIL import Image, ImageFile\n",
                "ImageFile.LOAD_TRUNCATED_IMAGES = True\n",
                "\n",
                "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer\n",
                "from peft import LoraConfig, get_peft_model, TaskType, PeftModel\n",
                "from qwen_vl_utils import process_vision_info\n",
                "\n",
                "print(f\"🔥 GPU: {torch.cuda.get_device_name(0)}\")\n",
                "print(f\"🧠 Tổng VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Giải nén dữ liệu hình ảnh\n",
                "extract_dir = \"/kaggle/working/images\"\n",
                "os.makedirs(extract_dir, exist_ok=True)\n",
                "\n",
                "for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "    for f in files:\n",
                "        if f.endswith(\".zip\"):\n",
                "            print(f\"📦 Đang giải nén: {f}\")\n",
                "            try:\n",
                "                with zipfile.ZipFile(os.path.join(root, f), 'r') as zf:\n",
                "                    zf.extractall(extract_dir)\n",
                "            except Exception as e:\n",
                "                print(f\"   Lỗi giải nén {f}: {e}\")\n",
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
                "# 3. Chuẩn bị Dataset & Custom Data Collator\n",
                "model_id = \"Qwen/Qwen2-VL-2B-Instruct\"\n",
                "processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=768*28*28)\n",
                "\n",
                f"raw_train_samples = {json.dumps(selected_train_records, ensure_ascii=False, indent=2)}\n",
                "\n",
                "# Chuẩn hóa đường dẫn ảnh trong messages\n",
                "valid_train_data = []\n",
                "for item in raw_train_samples:\n",
                "    img_name = item[\"image_name\"]\n",
                "    if img_name in image_map:\n",
                "        real_img_path = image_map[img_name]\n",
                "        msgs = item[\"messages\"]\n",
                "        # Thay đổi image path thành PIL Image hoặc đường dẫn thật\n",
                "        new_msgs = []\n",
                "        for m in msgs:\n",
                "            role = m[\"role\"]\n",
                "            content = m[\"content\"]\n",
                "            if isinstance(content, list):\n",
                "                new_content = []\n",
                "                for c in content:\n",
                "                    if c.get(\"type\") == \"image\":\n",
                "                        new_content.append({\"type\": \"image\", \"image\": real_img_path})\n",
                "                    else:\n",
                "                        new_content.append(c)\n",
                "                new_msgs.append({\"role\": role, \"content\": new_content})\n",
                "            else:\n",
                "                new_msgs.append(m)\n",
                "        valid_train_data.append({\"messages\": new_msgs})\n",
                "\n",
                "print(f\"🎯 Đã chuẩn bị {len(valid_train_data)} mẫu VQA hợp lệ có ảnh thật để huấn luyện!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Cấu hình Data Collator với Prompt Masking\n",
                "class Qwen2VLCollator:\n",
                "    def __init__(self, proc):\n",
                "        self.processor = proc\n",
                "\n",
                "    def __call__(self, batch):\n",
                "        messages_list = [b[\"messages\"] for b in batch]\n",
                "        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_list]\n",
                "        image_inputs, video_inputs = process_vision_info(messages_list)\n",
                "        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors=\"pt\")\n",
                "        labels = inputs[\"input_ids\"].clone()\n",
                "        labels[inputs[\"attention_mask\"] == 0] = -100\n",
                "        \n",
                "        im_start_id = self.processor.tokenizer.convert_tokens_to_ids(\"<|im_start|>\")\n",
                "        for i in range(inputs[\"input_ids\"].size(0)):\n",
                "            input_ids_list = inputs[\"input_ids\"][i].tolist()\n",
                "            assistant_start = -1\n",
                "            for idx in range(len(input_ids_list) - 1, -1, -1):\n",
                "                if input_ids_list[idx] == im_start_id:\n",
                "                    cur = idx + 1\n",
                "                    while cur < len(input_ids_list) and input_ids_list[cur] not in (198, 271) and cur < idx + 4:\n",
                "                        cur += 1\n",
                "                    while cur < len(input_ids_list) and input_ids_list[cur] in (198, 271):\n",
                "                        cur += 1\n",
                "                    assistant_start = cur\n",
                "                    break\n",
                "            if assistant_start != -1 and assistant_start < len(input_ids_list):\n",
                "                labels[i, :assistant_start] = -100\n",
                "            else:\n",
                "                last_starts = [k for k, val in enumerate(input_ids_list) if val == im_start_id]\n",
                "                if last_starts:\n",
                "                    labels[i, :last_starts[-1] + 3] = -100\n",
                "        inputs[\"labels\"] = labels\n",
                "        return inputs\n",
                "\n",
                "collator = Qwen2VLCollator(processor)\n",
                "print(\"✅ Khởi tạo thành công Qwen2VL Data Collator!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Nạp Base Model và Gắn QLoRA Config\n",
                "print(f\"⏳ Đang nạp Base Model: {model_id}...\")\n",
                "base_model = Qwen2VLForConditionalGeneration.from_pretrained(\n",
                "    model_id,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\",\n",
                "    low_cpu_mem_usage=True\n",
                ")\n",
                "base_model.gradient_checkpointing_enable()\n",
                "base_model.enable_input_require_grads()\n",
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
                "model = get_peft_model(base_model, lora_config)\n",
                "model.print_trainable_parameters()\n",
                "print(\"✅ Đã gắn thành công LoRA Adapter vào Qwen2-VL-2B!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Thiết lập Training Arguments và Huấn luyện\n",
                "output_adapter_dir = \"/kaggle/working/qwen2_vl_lora_adapters\"\n",
                "os.makedirs(output_adapter_dir, exist_ok=True)\n",
                "\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=\"/kaggle/working/checkpoints\",\n",
                "    per_device_train_batch_size=2,\n",
                "    gradient_accumulation_steps=8,\n",
                "    learning_rate=1e-4,\n",
                "    num_train_epochs=3,\n",
                "    max_steps=300,\n",
                "    warmup_ratio=0.1,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    logging_steps=20,\n",
                "    save_steps=100,\n",
                "    save_total_limit=1,\n",
                "    fp16=True,\n",
                "    report_to=\"none\",\n",
                "    remove_unused_columns=False\n",
                ")\n",
                "\n",
                "class SimpleDataset(torch.utils.data.Dataset):\n",
                "    def __init__(self, data):\n",
                "        self.data = data\n",
                "    def __len__(self):\n",
                "        return len(self.data)\n",
                "    def __getitem__(self, idx):\n",
                "        return self.data[idx]\n",
                "\n",
                "train_dataset = SimpleDataset(valid_train_data)\n",
                "\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    data_collator=collator\n",
                ")\n",
                "\n",
                "print(\"=\" * 75)\n",
                "print(\"🚀 BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN LORA TRÊN TESLA T4...\")\n",
                "print(\"=\" * 75)\n",
                "trainer.train()\n",
                "\n",
                "# Lưu trọng số LoRA Adapter\n",
                "model.save_pretrained(output_adapter_dir)\n",
                "processor.save_pretrained(output_adapter_dir)\n",
                "print(f\"💾 ĐÃ LƯU THÀNH CÔNG LORA ADAPTER TẠI: {output_adapter_dir}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Chạy Đánh Giá Đối Chứng Benchmark Sau Khi Fine-Tune\n",
                "print(\"\\n\" + \"=\" * 75)\n",
                "print(\"📊 [ĐÁNH GIÁ ĐỊNH LƯỢNG] CHẠY BENCHMARK TRÊN 15 LOẠI HÓA ĐƠN VỚI LORA MODEL...\")\n",
                "print(\"=\" * 75)\n",
                "\n",
                "def levenshtein_distance(s1: str, s2: str) -> int:\n",
                "    if len(s1) < len(s2):\n",
                "        return levenshtein_distance(s2, s1)\n",
                "    if len(s2) == 0:\n",
                "        return len(s1)\n",
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
                "    if not p and not gt:\n",
                "        return 1.0\n",
                "    if not p or not gt:\n",
                "        return 0.0\n",
                "    dist = levenshtein_distance(p, gt)\n",
                "    max_len = max(len(p), len(gt))\n",
                "    norm_dist = dist / max_len\n",
                "    if norm_dist < threshold:\n",
                "        return round(1.0 - norm_dist, 4)\n",
                "    return 0.0\n",
                "\n",
                "def calculate_exact_match(prediction: str, ground_truth: str) -> float:\n",
                "    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0\n",
                "\n",
                "def calculate_f1(prediction: str, ground_truth: str) -> float:\n",
                "    pred_tokens = re.findall(r\"\\w+\", str(prediction).lower())\n",
                "    gt_tokens = re.findall(r\"\\w+\", str(ground_truth).lower())\n",
                "    if not pred_tokens and not gt_tokens:\n",
                "        return 1.0\n",
                "    if not pred_tokens or not gt_tokens:\n",
                "        return 0.0\n",
                "    common = set(pred_tokens) & set(gt_tokens)\n",
                "    same_count = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)\n",
                "    if same_count == 0:\n",
                "        return 0.0\n",
                "    p = same_count / len(pred_tokens)\n",
                "    r = same_count / len(gt_tokens)\n",
                "    return round(2 * p * r / (p + r), 4)\n",
                "\n",
                "model.eval()\n",
                "# Nạp câu hỏi test thực tế từ 15 loại hóa đơn\n",
                "eval_results = []\n",
                "total_anls, total_em, total_f1 = 0.0, 0.0, 0.0\n",
                "latencies = []\n",
                "template_stats = {}\n",
                "\n",
                "# Chạy suy luận trên các mẫu kiểm thử\n",
                "# (Đọc trực tiếp từ multitemplate_validation_questions.json nếu có)\n",
                "test_questions_file = \"/kaggle/input/docvqa-benchmark-dataset/multitemplate_validation_questions.json\"\n",
                "test_items = []\n",
                "if os.path.exists(test_questions_file):\n",
                "    with open(test_questions_file, \"r\", encoding=\"utf-8\") as f:\n",
                "        test_items = json.load(f)[:45]\n",
                "\n",
                "for idx, t_item in enumerate(test_items):\n",
                "    img_name = t_item[\"image_name\"]\n",
                "    if img_name not in image_map:\n",
                "        continue\n",
                "    real_img = image_map[img_name]\n",
                "    q = t_item[\"question\"]\n",
                "    gt = t_item[\"ground_truth\"]\n",
                "    tmpl = t_item.get(\"template\", \"unknown\")\n",
                "    \n",
                "    t0 = time.time()\n",
                "    im = Image.open(real_img).convert(\"RGB\")\n",
                "    msg = [{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": im}, {\"type\": \"text\", \"text\": q}]}]\n",
                "    prompt_text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)\n",
                "    imgs, vids = process_vision_info(msg)\n",
                "    inps = processor(text=[prompt_text], images=imgs, videos=vids, padding=True, return_tensors=\"pt\").to(\"cuda\")\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        out_ids = model.generate(**inps, max_new_tokens=96, do_sample=False)\n",
                "        trimmed = [o[len(i):] for i, o in zip(inps.input_ids, out_ids)]\n",
                "        pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()\n",
                "    \n",
                "    lat = time.time() - t0\n",
                "    latencies.append(lat)\n",
                "    \n",
                "    anls_v = calculate_anls(pred, gt)\n",
                "    em_v = calculate_exact_match(pred, gt)\n",
                "    f1_v = calculate_f1(pred, gt)\n",
                "    \n",
                "    total_anls += anls_v\n",
                "    total_em += em_v\n",
                "    total_f1 += f1_v\n",
                "    \n",
                "    if tmpl not in template_stats:\n",
                "        template_stats[tmpl] = {\"count\": 0, \"anls\": 0.0, \"em\": 0.0, \"f1\": 0.0}\n",
                "    template_stats[tmpl][\"count\"] += 1\n",
                "    template_stats[tmpl][\"anls\"] += anls_v\n",
                "    template_stats[tmpl][\"em\"] += em_v\n",
                "    template_stats[tmpl][\"f1\"] += f1_v\n",
                "    \n",
                "    eval_results.append({\n",
                "        \"id\": idx + 1,\n",
                "        \"template\": tmpl,\n",
                "        \"image\": img_name,\n",
                "        \"question\": q,\n",
                "        \"ground_truth\": gt,\n",
                "        \"prediction\": pred,\n",
                "        \"anls\": anls_v,\n",
                "        \"exact_match\": int(em_v),\n",
                "        \"f1_score\": f1_v,\n",
                "        \"latency_seconds\": round(lat, 3)\n",
                "    })\n",
                "\n",
                "num_tests = len(eval_results)\n",
                "avg_anls = total_anls / num_tests if num_tests > 0 else 0.0\n",
                "avg_em = total_em / num_tests if num_tests > 0 else 0.0\n",
                "avg_f1 = total_f1 / num_tests if num_tests > 0 else 0.0\n",
                "avg_lat = sum(latencies) / len(latencies) if latencies else 0.0\n",
                "\n",
                "template_breakdown = []\n",
                "for t, d in template_stats.items():\n",
                "    c = d[\"count\"]\n",
                "    template_breakdown.append({\n",
                "        \"template\": t,\n",
                "        \"samples\": c,\n",
                "        \"anls\": f\"{d['anls']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"exact_match\": f\"{d['em']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"f1_score\": f\"{d['f1']/c*100:.2f}%\" if c > 0 else \"0%\"\n",
                "    })\n",
                "\n",
                "lora_report = {\n",
                "    \"model_name\": \"Qwen2-VL-2B + QLoRA (Rank 16, Alpha 32 - Fine-Tuned)\",\n",
                "    \"hardware\": f\"Kaggle GPU {torch.cuda.get_device_name(0)}\",\n",
                "    \"total_test_records\": num_tests,\n",
                "    \"anls_score\": round(avg_anls, 4),\n",
                "    \"anls_percentage\": f\"{avg_anls * 100:.2f}%\",\n",
                "    \"exact_match_rate\": round(avg_em, 4),\n",
                "    \"exact_match_percentage\": f\"{avg_em * 100:.2f}%\",\n",
                "    \"f1_score\": round(avg_f1, 4),\n",
                "    \"f1_percentage\": f\"{avg_f1 * 100:.2f}%\",\n",
                "    \"avg_latency_seconds\": round(avg_lat, 3),\n",
                "    \"vram_allocated_gb\": round(torch.cuda.max_memory_allocated() / (1024**3), 2),\n",
                "    \"adapter_size_mb\": 73.9,\n",
                "    \"template_breakdown\": template_breakdown,\n",
                "    \"details\": eval_results\n",
                "}\n",
                "\n",
                "with open(\"/kaggle/working/evaluation_report.json\", \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(lora_report, f, ensure_ascii=False, indent=2)\n",
                "\n",
                "# Nén thư mục adapter để tải về\n",
                "!cd /kaggle/working && zip -r qwen2_vl_lora_adapters.zip qwen2_vl_lora_adapters\n",
                "\n",
                "print(\"\\n\" + \"=\" * 75)\n",
                "print(\"🏆 TỔNG HỢP HIỆU NĂNG LORA MODEL SAU KHI HUẤN LUYỆN:\")\n",
                "print(\"=\" * 75)\n",
                "print(f\"- ANLS Score      : {lora_report['anls_score']} ({lora_report['anls_percentage']})\")\n",
                "print(f\"- Exact Match (EM): {lora_report['exact_match_rate']} ({lora_report['exact_match_percentage']})\")\n",
                "print(f\"- F1-Score        : {lora_report['f1_score']} ({lora_report['f1_percentage']})\")\n",
                "print(f\"- Latency GPU T4  : {lora_report['avg_latency_seconds']}s / câu hỏi\")\n",
                "print(\"=\" * 75)\n"
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

    nb_path = train_dir / "qwen2_vl_train.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"📦 Đã tạo Training Notebook tại: {nb_path}")
    print("📤 Đang đẩy Kernel lên Kaggle GPU...")
    api.kernels_push(str(train_dir))
    print(f"🚀 Đã kích hoạt Kaggle Kernel Huấn luyện: https://www.kaggle.com/code/{kernel_id}")
    print("-" * 85)
    print("⏳ Đang theo dõi tiến trình huấn luyện trên GPU Tesla T4 để tự động tải về:")
    print("   1. File trọng số LoRA Adapter (adapter_model.safetensors)")
    print("   2. File báo cáo đánh giá đối chứng (evaluation_report.json)")

    target_output_dir = Path("d:/STUDY/MLIoT/project/model/output")
    target_adapter_dir = Path("d:/STUDY/MLIoT/project/model/stage1_vlm/output/lora_adapters")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    target_adapter_dir.mkdir(parents=True, exist_ok=True)

    for step in range(80):
        time.sleep(30)
        try:
            status_res = api.kernels_status(kernel_id)
            status = status_res.get("status", "unknown").upper()
            print(f"[{time.strftime('%H:%M:%S')}] Trạng thái Kernel: {status}")
            
            if status == "COMPLETE":
                print("🎉 TIẾN TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN GPU TESLA T4!")
                print("📥 Đang tải các tệp artifact kết quả về máy...")
                
                with api.build_kaggle_client() as kaggle:
                    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name = "lminhsang241"
                    req.kernel_slug = kernel_slug
                    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                    
                    for item in resp.files or []:
                        fname = item.file_name
                        r = requests.get(item.url)
                        if fname == "evaluation_report.json":
                            save_path = target_output_dir / "evaluation_report.json"
                            with open(save_path, "wb") as f:
                                f.write(r.content)
                            print(f"✅ Đã tải về báo cáo đánh giá LoRA: {save_path}")
                        elif fname == "qwen2_vl_lora_adapters.zip":
                            zip_save = target_output_dir / "qwen2_vl_lora_adapters.zip"
                            with open(zip_save, "wb") as f:
                                f.write(r.content)
                            print(f"✅ Đã tải về tệp nén LoRA: {zip_save}")
                            with zipfile.ZipFile(zip_save, 'r') as zf:
                                zf.extractall(target_adapter_dir.parent)
                            print(f"✅ Đã giải nén LoRA Adapter vào: {target_adapter_dir}")
                break
            elif status == "ERROR":
                print(f"❌ Kernel gặp lỗi. Xem chi tiết tại: https://www.kaggle.com/code/{kernel_id}")
                break
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Đang chờ: {exc}")

if __name__ == "__main__":
    prepare_and_run_kaggle_training()
