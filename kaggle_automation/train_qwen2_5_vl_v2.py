"""
===================================================================================
🚀 KAGGLE AUTOMATION: HUẤN LUYỆN QWEN2.5-VL-3B V2 (OPTIMIZED VRAM & NO-OOM)
===================================================================================
Pipeline huấn luyện toàn diện phiên bản V2:
- Tối ưu hóa VRAM chống OOM: Gradient Checkpointing, Batch Size 1, Grad Accum 8
- Tích hợp Native Bounding Box Generation (Học tọa độ trực tiếp qua 2D M-RoPE)
- Multi-Task Curriculum (Direct Text QA + Hierarchical JSON + Coordinate Grounding)
- Đóng gói checkpoint tự động: qwen2_5_vl_lora_v2_grounding.zip
===================================================================================
"""
import os
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from kaggle.api.kaggle_api_extended import KaggleApi

def launch_training_v2():
    print("=" * 85)
    print("🚀 [QWEN2.5-VL V2 TRAINING] KHỞI TẠO HUẤN LUYỆN TRÊN KAGGLE GPU TESLA T4")
    print("=" * 85)

    api = KaggleApi()
    api.authenticate()

    kernel_slug = "qwen2-5-vl-v2-grounding-training"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    train_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/train_kernel_v2")
    train_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-5-vl-v2-grounding-training",
        "code_file": "train_qwen2_5_vl_v2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "lminhsang241/docvqa-benchmark-dataset",
            "lminhsang241/newest-dataset",
            "lminhsang241/nh-ha-n"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(train_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 HUẤN LUYỆN QWEN2.5-VL-3B V2: DOCVQA & NATIVE VISUAL GROUNDING\n",
                "### 🎯 Mục tiêu: Đạt ANLS >= 95%, F1 >= 93% và Tự động sinh Bounding Box [ymin, xmin, ymax, xmax] trực tiếp (VRAM Optimized)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Gỡ lỗi xung đột torchao & Cài đặt môi trường huấn luyện chuẩn\n",
                "import os, sys, time, json, re, random, glob, zipfile, torch, numpy as np\n",
                "from PIL import Image\n",
                "from collections import defaultdict\n",
                "\n",
                "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
                "\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers>=4.49.0\" \"peft>=0.13.2\" \"accelerate>=0.34.2\" easyocr\n",
                "\n",
                "import easyocr\n",
                "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor\n",
                "from qwen_vl_utils import process_vision_info\n",
                "from peft import LoraConfig, get_peft_model, TaskType\n",
                "\n",
                "print(f\"🔥 GPU: {torch.cuda.get_device_name(0)}\")\n",
                "print(f\"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Chuẩn Bị & Quét Toàn Diện Dữ Liệu Multi-Task Đa Nhiệm (DocVQA + Grounding)\n",
                "print(\"📦 Đang chuẩn bị tập dữ liệu Multi-Task Curriculum...\")\n",
                "\n",
                "image_map = {}\n",
                "for root, dirs, files in os.walk('/kaggle'):\n",
                "    for f in files:\n",
                "        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not f.startswith('.'):\n",
                "            p = os.path.join(root, f)\n",
                "            bname = os.path.splitext(f)[0]\n",
                "            image_map[f] = p\n",
                "            image_map[bname] = p\n",
                "            image_map[f.lower()] = p\n",
                "            image_map[bname.lower()] = p\n",
                "\n",
                "print(f\"📸 Tổng số ảnh đã lập chỉ mục trên Kaggle: {len(image_map)}\")\n",
                "\n",
                "train_records = []\n",
                "\n",
                "# Nạp từ vlm_train_master.json nếu có\n",
                "for root, dirs, files in os.walk('/kaggle'):\n",
                "    for file in files:\n",
                "        if file == 'vlm_train_master.json':\n",
                "            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:\n",
                "                raw_samples = json.load(f)\n",
                "                for s in raw_samples:\n",
                "                    bname = os.path.basename(s['image_path'])\n",
                "                    no_ext = os.path.splitext(bname)[0]\n",
                "                    real_p = image_map.get(bname) or image_map.get(no_ext) or image_map.get(bname.lower()) or image_map.get(no_ext.lower())\n",
                "                    if real_p and os.path.exists(real_p):\n",
                "                        train_records.append({\n",
                "                            'image_path': real_p,\n",
                "                            'question': s['question'],\n",
                "                            'answer': s['answer'],\n",
                "                            'field': s.get('field', 'GENERAL')\n",
                "                        })\n",
                "\n",
                "# Nếu chưa nạp đủ, quét thêm từ các tệp annotation gốc của Vietnamese Receipts\n",
                "if len(train_records) < 500:\n",
                "    QUESTION_TEMPLATES = {\n",
                "        'SELLER': ['Tên đơn vị / người bán hàng trên hóa đơn là gì?', 'Hóa đơn này do công ty / cửa hàng nào phát hành?'],\n",
                "        'TOTAL_COST': ['Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?', 'Khách hàng phải thanh toán tổng cộng bao nhiêu tiền?'],\n",
                "        'TIMESTAMP': ['Ngày giờ lập hóa đơn là khi nào?', 'Hóa đơn này được xuất vào ngày tháng năm nào?'],\n",
                "        'ADDRESS': ['Địa chỉ của đơn vị bán hàng là ở đâu?', 'Cửa hàng phát hành hóa đơn nằm ở địa chỉ nào?'],\n",
                "        'GROUNDING_TOTAL': ['Tìm và định vị vùng chứa tổng tiền thanh toán trên hóa đơn?'],\n",
                "        'GROUNDING_SELLER': ['Tìm và định vị vùng chứa tên đơn vị bán hàng trên hóa đơn?']\n",
                "    }\n",
                "    for root, dirs, files in os.walk('/kaggle'):\n",
                "        for file in files:\n",
                "            if file.lower().endswith('.json') and not file.startswith('.'):\n",
                "                try:\n",
                "                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:\n",
                "                        d = json.load(f)\n",
                "                except Exception:\n",
                "                    continue\n",
                "                if isinstance(d, dict) and 'annotations' in d:\n",
                "                    fname = d.get('file_name', '')\n",
                "                    bname = os.path.splitext(fname)[0]\n",
                "                    img_p = image_map.get(fname) or image_map.get(bname) or image_map.get(fname.lower()) or image_map.get(bname.lower())\n",
                "                    if not img_p or not os.path.exists(img_p): continue\n",
                "                    seller, total, timestamp, address = '', '', '', ''\n",
                "                    s_box, t_box = None, None\n",
                "                    for a in d.get('annotations', []):\n",
                "                        lbl = a.get('label', '').upper()\n",
                "                        txt = str(a.get('text', '')).strip()\n",
                "                        box = a.get('box')\n",
                "                        if lbl == 'SELLER' and not seller:\n",
                "                            seller = txt; s_box = box\n",
                "                        elif lbl == 'TOTAL_COST' and not total:\n",
                "                            total = txt; t_box = box\n",
                "                        elif lbl == 'TIMESTAMP' and not timestamp:\n",
                "                            timestamp = txt\n",
                "                        elif lbl == 'ADDRESS' and not address:\n",
                "                            address = txt\n",
                "                    if seller:\n",
                "                        train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['SELLER'][0], 'answer': seller, 'field': 'SELLER'})\n",
                "                        if s_box:\n",
                "                            train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['GROUNDING_SELLER'][0], 'answer': json.dumps({'text': seller, 'box': s_box}, ensure_ascii=False), 'field': 'GROUNDING_SELLER'})\n",
                "                    if total:\n",
                "                        train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['TOTAL_COST'][0], 'answer': total, 'field': 'TOTAL_COST'})\n",
                "                        if t_box:\n",
                "                            train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['GROUNDING_TOTAL'][0], 'answer': json.dumps({'text': total, 'box': t_box}, ensure_ascii=False), 'field': 'GROUNDING_TOTAL'})\n",
                "                    if timestamp:\n",
                "                        train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['TIMESTAMP'][0], 'answer': timestamp, 'field': 'TIMESTAMP'})\n",
                "                    if address:\n",
                "                        train_records.append({'image_path': img_p, 'question': QUESTION_TEMPLATES['ADDRESS'][0], 'answer': address, 'field': 'ADDRESS'})\n",
                "\n",
                "print(f\"🎯 TỔNG SỐ MẪU HUẤN LUYỆN ĐÃ TẠO: {len(train_records)} MẪU!\")\n",
                "assert len(train_records) > 0, '⚠️ Không tìm thấy mẫu huấn luyện nào!'\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Khởi Tạo Qwen2.5-VL-3B & Tối Ưu Hóa Bộ Nhớ VRAM (Gradient Checkpointing)\n",
                "model_name = \"Qwen/Qwen2.5-VL-3B-Instruct\"\n",
                "processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28)\n",
                "base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n",
                "    model_name,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "base_model.gradient_checkpointing_enable()\n",
                "base_model.enable_input_require_grads()\n",
                "\n",
                "peft_config = LoraConfig(\n",
                "    task_type=TaskType.CAUSAL_LM,\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.05,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "    bias=\"none\"\n",
                ")\n",
                "\n",
                "model = get_peft_model(base_model, peft_config)\n",
                "model.print_trainable_parameters()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Vòng Lặp Huấn Luyện VRAM-Safe (Batch Size 1, Grad Accum 8, Cosine Scheduler)\n",
                "SYSTEM_PROMPT = \"Bạn là chuyên gia AI kế toán chuyên đọc và bóc tách hóa đơn, chứng từ tài chính tiếng Việt. Hãy đọc ảnh và trả lời câu hỏi trực tiếp, chính xác, ngắn gọn theo đúng nội dung trên tài liệu, không giải thích lan man.\"\n",
                "\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)\n",
                "epochs = 3\n",
                "batch_size = 1  # 1 ảnh / forward pass chống OOM\n",
                "grad_accum_steps = 8 # Effective Batch Size = 8\n",
                "\n",
                "random.shuffle(train_records)\n",
                "train_subset = train_records[:min(len(train_records), 800)]\n",
                "\n",
                "steps_per_epoch = max(1, len(train_subset) // (batch_size * grad_accum_steps))\n",
                "total_steps = steps_per_epoch * epochs\n",
                "lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)\n",
                "\n",
                "print(f\"🏋️ Bắt đầu huấn luyện V2 {epochs} Epochs ({len(train_subset)} mẫu, {total_steps} Optimizer Steps)...\")\n",
                "model.train()\n",
                "\n",
                "loss_history = []\n",
                "global_step = 0\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    epoch_loss = 0.0\n",
                "    step_in_epoch = 0\n",
                "    optimizer.zero_grad()\n",
                "    t0 = time.time()\n",
                "    \n",
                "    for idx in range(0, len(train_subset), batch_size):\n",
                "        batch_items = train_subset[idx:idx + batch_size]\n",
                "        if not batch_items: continue\n",
                "        \n",
                "        messages_batch = []\n",
                "        for it in batch_items:\n",
                "            try:\n",
                "                im = Image.open(it['image_path']).convert('RGB')\n",
                "                messages_batch.append([\n",
                "                    {\"role\": \"system\", \"content\": SYSTEM_PROMPT},\n",
                "                    {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": im}, {\"type\": \"text\", \"text\": it['question']}]},\n",
                "                    {\"role\": \"assistant\", \"content\": str(it['answer'])}\n",
                "                ])\n",
                "            except Exception:\n",
                "                continue\n",
                "                \n",
                "        if not messages_batch: continue\n",
                "        \n",
                "        try:\n",
                "            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_batch]\n",
                "            image_inputs, video_inputs = process_vision_info(messages_batch)\n",
                "            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors=\"pt\").to(\"cuda\")\n",
                "            \n",
                "            labels = inputs.input_ids.clone()\n",
                "            labels[labels == processor.tokenizer.pad_token_id] = -100\n",
                "            \n",
                "            outputs = model(**inputs, labels=labels)\n",
                "            loss = outputs.loss / grad_accum_steps\n",
                "            loss.backward()\n",
                "            \n",
                "            epoch_loss += loss.item() * grad_accum_steps\n",
                "            step_in_epoch += 1\n",
                "            \n",
                "            if step_in_epoch % grad_accum_steps == 0:\n",
                "                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
                "                optimizer.step()\n",
                "                lr_scheduler.step()\n",
                "                optimizer.zero_grad()\n",
                "                torch.cuda.empty_cache()\n",
                "                global_step += 1\n",
                "        except Exception as e:\n",
                "            torch.cuda.empty_cache()\n",
                "            continue\n",
                "            \n",
                "    avg_loss = epoch_loss / max(1, step_in_epoch)\n",
                "    dur = time.time() - t0\n",
                "    loss_history.append({'epoch': epoch + 1, 'loss': round(avg_loss, 4), 'time': round(dur, 1)})\n",
                "    print(f\" Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Thời gian: {dur:.1f}s | LR: {lr_scheduler.get_last_lr()[0]:.2e}\")\n",
                "\n",
                "# Lưu Adapter Checkpoint V2\n",
                "save_dir = \"/kaggle/working/qwen2_5_vl_lora_v2_grounding\"\n",
                "model.save_pretrained(save_dir)\n",
                "processor.save_pretrained(save_dir)\n",
                "!cd /kaggle/working && zip -r qwen2_5_vl_lora_v2_grounding.zip qwen2_5_vl_lora_v2_grounding\n",
                "print(f\"💾 Đã lưu thành công bộ trọng số LoRA V2 vào: {save_dir}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Đánh Giá Đối Chứng Toàn Diện Sau Khi Fine-tune (ANLS, Exact Match, F1)\n",
                "print(\"=\" * 85)\n",
                "print(\"📊 ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH V2 TRÊN BENCHMARK ĐỘC LẬP...\")\n",
                "print(\"=\" * 85)\n",
                "\n",
                "def levenshtein_distance(s1, s2):\n",
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
                "def calculate_anls(pred, gt, threshold=0.5):\n",
                "    p, g = str(pred).strip().lower(), str(gt).strip().lower()\n",
                "    if not p and not g: return 1.0\n",
                "    if not p or not g: return 0.0\n",
                "    dist = levenshtein_distance(p, g)\n",
                "    norm_dist = dist / max(len(p), len(g))\n",
                "    return round(1.0 - norm_dist, 4) if norm_dist < threshold else 0.0\n",
                "\n",
                "def calculate_f1(pred, gt):\n",
                "    p_tok = re.findall(r'\\w+', str(pred).lower())\n",
                "    g_tok = re.findall(r'\\w+', str(gt).lower())\n",
                "    if not p_tok and not g_tok: return 1.0\n",
                "    if not p_tok or not g_tok: return 0.0\n",
                "    common = set(p_tok) & set(g_tok)\n",
                "    same = sum(min(p_tok.count(t), g_tok.count(t)) for t in common)\n",
                "    if same == 0: return 0.0\n",
                "    p, r = same / len(p_tok), same / len(g_tok)\n",
                "    return round(2 * p * r / (p + r), 4)\n",
                "\n",
                "model.eval()\n",
                "test_samples = []\n",
                "for root, dirs, files in os.walk('/kaggle'):\n",
                "    for f in files:\n",
                "        if f == 'multitemplate_validation_questions.json':\n",
                "            with open(os.path.join(root, f), 'r', encoding='utf-8') as fl:\n",
                "                test_samples = json.load(fl)\n",
                "            break\n",
                "\n",
                "if not test_samples:\n",
                "    test_samples = [\n",
                "        {\"id\": 1, \"template\": \"einvoice_viettel\", \"image_name\": \"einvoice_viettel_val_001.png\", \"question\": \"Tên đơn vị / người bán hàng trên hóa đơn là gì?\", \"ground_truth\": \"CÔNG TY CỔ PHẦN ĐẦU TƯ & PHÁT TRIỂN HƯNG PHÁT\"},\n",
                "        {\"id\": 2, \"template\": \"einvoice_viettel\", \"image_name\": \"einvoice_viettel_val_001.png\", \"question\": \"Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?\", \"ground_truth\": \"24,389,200đ\"},\n",
                "        {\"id\": 3, \"template\": \"einvoice_viettel\", \"image_name\": \"einvoice_viettel_val_001.png\", \"question\": \"Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?\", \"ground_truth\": \"Bút bi Thiên Long FO-03, Giấy Double A A4 70gsm, Dịch vụ Bảo trì Hệ thống mạng\"}\n",
                "    ]\n",
                "\n",
                "eval_results = []\n",
                "tot_anls, tot_em, tot_f1 = 0.0, 0.0, 0.0\n",
                "\n",
                "for s in test_samples[:60]:\n",
                "    img_name = s['image_name']\n",
                "    bname = os.path.splitext(img_name)[0]\n",
                "    real_img = image_map.get(img_name) or image_map.get(bname) or image_map.get(img_name.lower()) or image_map.get(bname.lower())\n",
                "    if not real_img or not os.path.exists(real_img): continue\n",
                "    \n",
                "    im = Image.open(real_img).convert('RGB')\n",
                "    msgs = [\n",
                "        {\"role\": \"system\", \"content\": SYSTEM_PROMPT},\n",
                "        {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": im}, {\"type\": \"text\", \"text\": s['question']}]}\n",
                "    ]\n",
                "    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\n",
                "    img_inps, vid_inps = process_vision_info(msgs)\n",
                "    inputs = processor(text=[text], images=img_inps, videos=vid_inps, padding=True, return_tensors=\"pt\").to(\"cuda\")\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)\n",
                "        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]\n",
                "        pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()\n",
                "        \n",
                "    gt = s['ground_truth']\n",
                "    anls_v = calculate_anls(pred, gt)\n",
                "    em_v = 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0\n",
                "    f1_v = calculate_f1(pred, gt)\n",
                "    \n",
                "    tot_anls += anls_v; tot_em += em_v; tot_f1 += f1_v\n",
                "    eval_results.append({'question': s['question'], 'ground_truth': gt, 'prediction': pred, 'anls': anls_v, 'em': em_v, 'f1': f1_v})\n",
                "\n",
                "n = len(eval_results)\n",
                "final_anls = tot_anls / n if n > 0 else 0.0\n",
                "final_em = tot_em / n if n > 0 else 0.0\n",
                "final_f1 = tot_f1 / n if n > 0 else 0.0\n",
                "\n",
                "report_v2 = {\n",
                "    \"model\": \"Qwen2.5-VL-3B + LoRA V2 (DocVQA + Native Grounding)\",\n",
                "    \"anls_score\": f\"{final_anls * 100:.2f}%\",\n",
                "    \"exact_match\": f\"{final_em * 100:.2f}%\",\n",
                "    \"f1_score\": f\"{final_f1 * 100:.2f}%\",\n",
                "    \"loss_history\": loss_history,\n",
                "    \"details\": eval_results\n",
                "}\n",
                "\n",
                "with open('/kaggle/working/evaluation_report_v2.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(report_v2, f, ensure_ascii=False, indent=2)\n",
                "\n",
                "print(\"=\" * 85)\n",
                "print(f\"🎉 KẾT QUẢ KIỂM ĐỊNH MÔ HÌNH V2:\")\n",
                "print(f\"- ANLS Score      : {report_v2['anls_score']}\")\n",
                "print(f\"- Exact Match (EM): {report_v2['exact_match']}\")\n",
                "print(f\"- F1-Score        : {report_v2['f1_score']}\")\n",
                "print(\"=\" * 85)\n"
            ]
        }
    ]

    notebook_content = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = train_dir / "train_qwen2_5_vl_v2.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"📦 Đã tạo Notebook Huấn Luyện V2 tại: {nb_path}")
    print("📤 Đang đẩy Kernel Huấn Luyện V2 lên Kaggle GPU...")
    api.kernels_push(str(train_dir))
    print(f"🚀 Đã kích hoạt Kaggle Kernel: https://www.kaggle.com/code/{kernel_id}")
    print("-" * 85)

if __name__ == "__main__":
    launch_training_v2()
