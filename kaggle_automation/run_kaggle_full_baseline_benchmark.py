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

def prepare_and_run_full_baseline_evaluation():
    print("=" * 85)
    print("🚀 [KAGGLE GPU EVALUATION] ĐÁNH GIÁ CHUẨN ĐỊNH LƯỢNG BASE MODEL TRÊN 15 LOẠI HÓA ĐƠN")
    print("=" * 85)

    api = KaggleApi()
    api.authenticate()
    print("✅ Xác thực thành công tài khoản Kaggle: lminhsang241")

    kernel_slug = "qwen2-vl-base-model-eval"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    eval_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/base_eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-vl-base-model-eval",
        "code_file": "qwen2_vl_base_eval.ipynb",
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

    with open(eval_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Đọc toàn bộ 174 câu hỏi validation đại diện chuẩn mực của 15 loại hóa đơn
    questions_path = Path("d:/STUDY/MLIoT/project/datasets/val_benchmark_upload/multitemplate_validation_questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        multitemplate_questions = json.load(f)

    print(f"📋 Tổng số câu hỏi kiểm định (Validation Benchmark Set): {len(multitemplate_questions)} câu hỏi")
    
    template_counts = {}
    field_counts = {}
    for q in multitemplate_questions:
        t = q.get("template", "unknown")
        fld = q.get("field", "unknown")
        template_counts[t] = template_counts.get(t, 0) + 1
        field_counts[fld] = field_counts.get(fld, 0) + 1

    print(f"📊 Phân bố theo 15 loại hóa đơn:")
    for t, c in sorted(template_counts.items()):
        print(f"   • {t:<28}: {c} câu hỏi")
    print(f"📊 Phân bố theo các trường thông tin (Fields):")
    for fld, c in sorted(field_counts.items()):
        print(f"   • {fld:<20}: {c} câu hỏi")

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎯 KAGGLE GPU BENCHMARK: BASE MODEL QWEN2-VL-2B-INSTRUCT (ZERO-SHOT)\n",
                "Đánh giá học thuật toàn diện trên 174 câu hỏi thực tế thuộc 15 loại hóa đơn tiếng Việt trên GPU NVIDIA Tesla T4 (16GB VRAM).\n",
                "Các chỉ số đo lường:\n",
                "- **ANLS (Average Normalized Levenshtein Similarity)**: Tiêu chuẩn quốc tế DocVQA / TextVQA.\n",
                "- **Exact Match (EM %)**: Tỷ lệ khớp chính xác từng ký tự.\n",
                "- **Token F1-Score**: Độ đo bao phủ thực thể mức từ (Token Precision, Recall, F1).\n",
                "- **Inference Latency & GPU VRAM**: Tốc độ xử lý thực tế và mức tiêu hao tài nguyên phần cứng."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Cài đặt môi trường & thư viện\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers==4.46.2\" \"peft==0.13.2\" \"accelerate==0.34.2\" pillow torchvision\n",
                "\n",
                "import sys\n",
                "for mod in list(sys.modules.keys()):\n",
                "    if any(mod.startswith(k) for k in [\"transformers\", \"peft\", \"accelerate\", \"torchao\", \"qwen_vl_utils\"]):\n",
                "        del sys.modules[mod]\n",
                "\n",
                "import os\n",
                "import time\n",
                "import json\n",
                "import re\n",
                "import zipfile\n",
                "import torch\n",
                "from PIL import Image\n",
                "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor\n",
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
                "# 2. Xây dựng các hàm tính toán Metrics học thuật chuẩn xác\n",
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
                "        return 1.0 - norm_dist\n",
                "    return 0.0\n",
                "\n",
                "def calculate_exact_match(prediction: str, ground_truth: str) -> float:\n",
                "    return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0\n",
                "\n",
                "def calculate_token_f1(prediction: str, ground_truth: str) -> float:\n",
                "    def tokenize(text: str):\n",
                "        return re.findall(r'\\w+', str(text).lower())\n",
                "    p_tokens = tokenize(prediction)\n",
                "    g_tokens = tokenize(ground_truth)\n",
                "    if not p_tokens and not g_tokens:\n",
                "        return 1.0\n",
                "    if not p_tokens or not g_tokens:\n",
                "        return 0.0\n",
                "    common = set(p_tokens) & set(g_tokens)\n",
                "    if not common:\n",
                "        return 0.0\n",
                "    p_count = sum(p_tokens.count(t) for t in common)\n",
                "    g_count = sum(g_tokens.count(t) for t in common)\n",
                "    prec = p_count / len(p_tokens)\n",
                "    rec = g_count / len(g_tokens)\n",
                "    if prec + rec == 0:\n",
                "        return 0.0\n",
                "    return (2 * prec * rec) / (prec + rec)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Giải nén và lập chỉ mục 15 loại ảnh hóa đơn\n",
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
                "print(f\"📸 Đã lập chỉ mục {len(image_map)} ảnh hóa đơn trong hệ thống!\")\n",
                "\n",
                "# Nạp toàn bộ 174 câu hỏi kiểm định đại diện\n",
                f"validation_samples = {json.dumps(multitemplate_questions, ensure_ascii=False, indent=2)}\n",
                "\n",
                "# Khớp đường dẫn ảnh thật\n",
                "matched_samples = []\n",
                "for s in validation_samples:\n",
                "    img_name = s[\"image_name\"]\n",
                "    if img_name in image_map:\n",
                "        s[\"full_image_path\"] = image_map[img_name]\n",
                "        matched_samples.append(s)\n",
                "\n",
                "print(f\"🎯 Khớp thành công {len(matched_samples)} / {len(validation_samples)} mẫu kiểm thử có ảnh thật!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Nạp Base Model Qwen2-VL-2B-Instruct vào GPU\n",
                "model_name = \"Qwen/Qwen2-VL-2B-Instruct\"\n",
                "print(f\"⏳ Đang nạp Base Model: {model_name} (Native FP16)... \")\n",
                "processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1024*28*28)\n",
                "model = Qwen2VLForConditionalGeneration.from_pretrained(\n",
                "    model_name,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "model.eval()\n",
                "print(\"✅ Nạp thành công Base Model vào GPU Tesla T4!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Thực thi suy luận toàn diện trên toàn bộ 174 câu hỏi\n",
                "print(\"=\" * 85)\n",
                "print(\"🚀 BẮT ĐẦU CHẠY SUY LUẬN BASE MODEL TRÊN 174 CÂU HỎI BENCHMARK...\")\n",
                "print(\"=\" * 85)\n",
                "\n",
                "results = []\n",
                "total_anls = 0.0\n",
                "total_em = 0.0\n",
                "total_f1 = 0.0\n",
                "latencies = []\n",
                "template_stats = {}\n",
                "field_stats = {}\n",
                "\n",
                "for idx, sample in enumerate(matched_samples):\n",
                "    img_path = sample[\"full_image_path\"]\n",
                "    question = sample[\"question\"]\n",
                "    gt = sample[\"ground_truth\"]\n",
                "    tmpl = sample.get(\"template\", \"unknown\")\n",
                "    fld = sample.get(\"field\", \"unknown\")\n",
                "    \n",
                "    if tmpl not in template_stats:\n",
                "        template_stats[tmpl] = {\"count\": 0, \"anls\": 0.0, \"em\": 0.0, \"f1\": 0.0}\n",
                "    if fld not in field_stats:\n",
                "        field_stats[fld] = {\"count\": 0, \"anls\": 0.0, \"em\": 0.0, \"f1\": 0.0}\n",
                "    \n",
                "    t0 = time.time()\n",
                "    image = Image.open(img_path).convert(\"RGB\")\n",
                "    messages = [\n",
                "        {\n",
                "            \"role\": \"user\",\n",
                "            \"content\": [\n",
                "                {\"type\": \"image\", \"image\": image},\n",
                "                {\"type\": \"text\", \"text\": question}\n",
                "            ]\n",
                "        }\n",
                "    ]\n",
                "    \n",
                "    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "    image_inputs, video_inputs = process_vision_info(messages)\n",
                "    inputs = processor(\n",
                "        text=[text],\n",
                "        images=image_inputs,\n",
                "        videos=video_inputs,\n",
                "        padding=True,\n",
                "        return_tensors=\"pt\"\n",
                "    ).to(\"cuda\")\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        generated_ids = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=128,\n",
                "            do_sample=False\n",
                "        )\n",
                "        generated_ids_trimmed = [\n",
                "            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)\n",
                "        ]\n",
                "        prediction = processor.batch_decode(\n",
                "            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False\n",
                "        )[0].strip()\n",
                "        \n",
                "    latency = time.time() - t0\n",
                "    latencies.append(latency)\n",
                "    \n",
                "    anls_score = calculate_anls(prediction, gt)\n",
                "    em_score = calculate_exact_match(prediction, gt)\n",
                "    f1_val = calculate_token_f1(prediction, gt)\n",
                "    \n",
                "    total_anls += anls_score\n",
                "    total_em += em_score\n",
                "    total_f1 += f1_val\n",
                "    \n",
                "    template_stats[tmpl][\"count\"] += 1\n",
                "    template_stats[tmpl][\"anls\"] += anls_score\n",
                "    template_stats[tmpl][\"em\"] += em_score\n",
                "    template_stats[tmpl][\"f1\"] += f1_val\n",
                "    \n",
                "    field_stats[fld][\"count\"] += 1\n",
                "    field_stats[fld][\"anls\"] += anls_score\n",
                "    field_stats[fld][\"em\"] += em_score\n",
                "    field_stats[fld][\"f1\"] += f1_val\n",
                "    \n",
                "    results.append({\n",
                "        \"id\": idx + 1,\n",
                "        \"template\": tmpl,\n",
                "        \"field\": fld,\n",
                "        \"image\": sample[\"image_name\"],\n",
                "        \"question\": question,\n",
                "        \"ground_truth\": gt,\n",
                "        \"prediction\": prediction,\n",
                "        \"anls\": round(anls_score, 4),\n",
                "        \"exact_match\": int(em_score),\n",
                "        \"token_f1\": round(f1_val, 4),\n",
                "        \"latency_seconds\": round(latency, 3)\n",
                "    })\n",
                "    \n",
                "    if (idx + 1) % 10 == 0 or (idx + 1) == len(matched_samples):\n",
                "        print(f\"[{idx+1:03d}/{len(matched_samples)}] ({tmpl}) Latency: {latency:.2f}s | ANLS: {anls_score:.2f} | EM: {int(em_score)} | F1: {f1_val:.2f}\")\n",
                "        print(f\"   ❓ Q:  {question}\")\n",
                "        print(f\"   🎯 GT: {gt}\")\n",
                "        print(f\"   🤖 PR: {prediction[:100]}...\")\n",
                "        print(\"-\" * 85)\n",
                "\n",
                "num_tests = len(matched_samples)\n",
                "avg_anls = total_anls / num_tests if num_tests > 0 else 0.0\n",
                "avg_em = total_em / num_tests if num_tests > 0 else 0.0\n",
                "avg_f1 = total_f1 / num_tests if num_tests > 0 else 0.0\n",
                "avg_lat = sum(latencies) / len(latencies) if latencies else 0.0\n",
                "\n",
                "# Thống kê theo loại hóa đơn\n",
                "template_breakdown = []\n",
                "for t, d in sorted(template_stats.items()):\n",
                "    c = d[\"count\"]\n",
                "    template_breakdown.append({\n",
                "        \"template\": t,\n",
                "        \"samples\": c,\n",
                "        \"anls\": f\"{d['anls']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"exact_match\": f\"{d['em']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"f1_score\": f\"{d['f1']/c*100:.2f}%\" if c > 0 else \"0%\"\n",
                "    })\n",
                "\n",
                "# Thống kê theo trường dữ liệu\n",
                "field_breakdown = []\n",
                "for fld, d in sorted(field_stats.items()):\n",
                "    c = d[\"count\"]\n",
                "    field_breakdown.append({\n",
                "        \"field\": fld,\n",
                "        \"samples\": c,\n",
                "        \"anls\": f\"{d['anls']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"exact_match\": f\"{d['em']/c*100:.2f}%\" if c > 0 else \"0%\",\n",
                "        \"f1_score\": f\"{d['f1']/c*100:.2f}%\" if c > 0 else \"0%\"\n",
                "    })\n",
                "\n",
                "# Xuất báo cáo JSON ra thư mục /kaggle/working\n",
                "final_report = {\n",
                "    \"model_name\": \"Qwen/Qwen2-VL-2B-Instruct (Base Zero-Shot)\",\n",
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
                "    \"field_breakdown\": field_breakdown,\n",
                "    \"template_breakdown\": template_breakdown,\n",
                "    \"details\": results\n",
                "}\n",
                "\n",
                "with open(\"/kaggle/working/baseline_evaluation_report.json\", \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(final_report, f, ensure_ascii=False, indent=2)\n",
                "\n",
                "print(\"\\n\" + \"=\" * 85)\n",
                "print(\"📊 TỔNG HỢP KẾT QUẢ BASE MODEL TRÊN 174 MẪU HÓA ĐƠN BENCHMARK:\")\n",
                "print(\"=\" * 85)\n",
                "print(f\"- Tổng số mẫu kiểm định (Validation Samples) : {num_tests}\")\n",
                "print(f\"- Điểm ANLS Score (DocVQA Metric)            : {final_report['anls_score']} ({final_report['anls_percentage']})\")\n",
                "print(f\"- Tỉ lệ Exact Match (EM Rate)                 : {final_report['exact_match_rate']} ({final_report['exact_match_percentage']})\")\n",
                "print(f\"- Điểm Token F1-Score                         : {final_report['f1_score']} ({final_report['f1_percentage']})\")\n",
                "print(f\"- Thời gian suy luận trung bình (Avg Latency): {final_report['avg_latency_seconds']} giây / câu hỏi\")\n",
                "print(f\"- Dung lượng VRAM tiêu thụ                   : {final_report['vram_allocated_gb']} GB\")\n",
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

    nb_path = eval_dir / "qwen2_vl_base_eval.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"📦 Đã tạo Notebook tại: {nb_path}")
    print("📤 Đang đẩy Kernel lên Kaggle GPU...")
    api.kernels_push(str(eval_dir))
    print(f"🚀 Đã kích hoạt Kaggle Kernel Đánh giá Base Model: https://www.kaggle.com/code/{kernel_id}")
    print("-" * 85)
    print("⏳ Bắt đầu giám sát tiến trình đánh giá trên GPU Tesla T4...")

    target_output_dir = Path("d:/STUDY/MLIoT/project/model/output")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    for step in range(60):
        time.sleep(25)
        try:
            status_res = api.kernels_status(kernel_id)
            status = status_res.get("status", "unknown").upper()
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] (Đã chạy: {elapsed/60:.1f} phút) Trạng thái Kernel: {status}")
            
            if status == "COMPLETE":
                print("\n🎉 KERNEL ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN GPU TESLA T4!")
                print("📥 Đang tải file báo cáo gốc baseline_evaluation_report.json...")
                
                with api.build_kaggle_client() as kaggle:
                    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name = "lminhsang241"
                    req.kernel_slug = kernel_slug
                    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                    
                    for item in resp.files or []:
                        if item.file_name == "baseline_evaluation_report.json":
                            save_path = target_output_dir / "baseline_evaluation_report.json"
                            r = requests.get(item.url)
                            with open(save_path, "wb") as f:
                                f.write(r.content)
                            print(f"✅ Đã tải và lưu báo cáo chính thức tại: {save_path}")
                            break
                break
            elif status == "ERROR":
                print(f"❌ Kernel gặp lỗi. Xem chi tiết tại: https://www.kaggle.com/code/{kernel_id}")
                break
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Chờ kết quả: {exc}")

if __name__ == "__main__":
    prepare_and_run_full_baseline_evaluation()
