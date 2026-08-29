import os
import sys
import json
import time
import re
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def setup_and_launch_kaggle_demo():
    print("=" * 80)
    print("🚀 [KAGGLE GPU DEMO] KHỞI TẠO & ĐẨY PIPELINE DEMO LÊN KAGGLE GPU T4")
    print("=" * 80)

    api = KaggleApi()
    api.authenticate()
    print("✅ Xác thực thành công tài khoản Kaggle: lminhsang241")

    kernel_slug = "qwen2-vl-docvqa-live-demo"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    demo_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/live_demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-vl-docvqa-live-demo",
        "code_file": "qwen2_vl_docvqa_demo.ipynb",
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

    with open(demo_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Tạo nội dung Notebook cho Kaggle
    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📄 DOCUMENT VQA LIVE DEMO (QWEN2-VL-2B + LORA) TRÊN GPU TESLA T4\n",
                "Hệ thống hỏi đáp và trích xuất thông tin hóa đơn tiếng Việt chạy hoàn toàn trên GPU Nvidia Tesla T4."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Cài đặt các thư viện cần thiết\n",
                "!pip install -q --no-deps qwen-vl-utils==0.0.8\n",
                "!pip install -q \"transformers==4.46.2\" \"peft==0.13.2\" \"accelerate==0.34.2\" gradio>=4.0.0 pillow torchvision\n",
                "\n",
                "import os\n",
                "import sys\n",
                "import time\n",
                "import torch\n",
                "from PIL import Image\n",
                "import gradio as gr\n",
                "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor\n",
                "from qwen_vl_utils import process_vision_info\n",
                "from peft import PeftModel\n",
                "\n",
                "print(f\"🔥 Đang sử dụng GPU: {torch.cuda.get_device_name(0)}\")\n",
                "print(f\"🧠 Tổng VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Tìm kiếm và nạp LoRA Adapter\n",
                "adapter_dir = None\n",
                "search_paths = [\n",
                "    \"/kaggle/input/docvqa-lora-adapters\",\n",
                "    \"/kaggle/input/docvqa-lora-adapters/lora_adapters\",\n",
                "    \"/kaggle/input\"\n",
                "]\n",
                "\n",
                "for sp in search_paths:\n",
                "    if os.path.exists(sp):\n",
                "        for root, dirs, files in os.walk(sp):\n",
                "            if \"adapter_config.json\" in files:\n",
                "                adapter_dir = root\n",
                "                break\n",
                "    if adapter_dir:\n",
                "        break\n",
                "\n",
                "print(f\"📍 LoRA Adapter Path: {adapter_dir}\")\n",
                "\n",
                "# 3. Nạp Base Model & LoRA weights vào GPU\n",
                "model_name = \"Qwen/Qwen2-VL-2B-Instruct\"\n",
                "print(f\"⏳ Đang nạp Processor từ {model_name}...\")\n",
                "processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)\n",
                "\n",
                "print(\"⏳ Đang nạp Base Model vào VRAM (FP16)... \")\n",
                "base_model = Qwen2VLForConditionalGeneration.from_pretrained(\n",
                "    model_name,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "if adapter_dir and os.path.exists(os.path.join(adapter_dir, \"adapter_config.json\")):\n",
                "    print(f\"🚀 Đang gắn LoRA Adapter từ {adapter_dir}...\")\n",
                "    model = PeftModel.from_pretrained(base_model, adapter_dir)\n",
                "    model.eval()\n",
                "    print(\"🎉 NẠP THÀNH CÔNG FINE-TUNED MODEL (QWEN2-VL + LORA)!\")\n",
                "else:\n",
                "    print(\"⚠️ Không tìm thấy adapter, chạy trực tiếp Base Model.\")\n",
                "    model = base_model.eval()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Hàm suy luận VQA (Inference function)\n",
                "def predict_vqa(image, question):\n",
                "    if image is None:\n",
                "        return \"⚠️ Vui lòng tải lên ảnh hóa đơn hoặc chứng từ.\"\n",
                "    if not question or not question.strip():\n",
                "        question = \"Trích xuất các trường thông tin: Tên người bán, Mã số thuế, Ngày lập, Tổng tiền thanh toán.\"\n",
                "    \n",
                "    t0 = time.time()\n",
                "    messages = [\n",
                "        {\n",
                "            \"role\": \"user\",\n",
                "            \"content\": [\n",
                "                {\"type\": \"image\", \"image\": image},\n",
                "                {\"type\": \"text\", \"text\": question.strip()}\n",
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
                "            max_new_tokens=256,\n",
                "            do_sample=False\n",
                "        )\n",
                "        generated_ids_trimmed = [\n",
                "            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)\n",
                "        ]\n",
                "        response = processor.batch_decode(\n",
                "            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False\n",
                "        )[0].strip()\n",
                "        \n",
                "    latency = time.time() - t0\n",
                "    return f\"{response}\\n\\n⏱️ Thời gian xử lý: {latency:.2f} giây (GPU Tesla T4)\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Tìm các ảnh mẫu sẵn có trong dataset để làm Examples\n",
                "sample_images = []\n",
                "benchmark_dir = \"/kaggle/input/docvqa-benchmark-dataset\"\n",
                "if os.path.exists(benchmark_dir):\n",
                "    for root, dirs, files in os.walk(benchmark_dir):\n",
                "        for f in files:\n",
                "            if f.lower().endswith((\".png\", \".jpg\", \".jpeg\")):\n",
                "                sample_images.append(os.path.join(root, f))\n",
                "                if len(sample_images) >= 5:\n",
                "                    break\n",
                "\n",
                "print(f\"📸 Tìm thấy {len(sample_images)} ảnh mẫu kiểm thử.\")\n",
                "\n",
                "# 6. Xây dựng giao diện Gradio và khởi chạy\n",
                "with gr.Blocks(title=\"Document VQA - Qwen2-VL-2B (GPU Tesla T4)\", theme=gr.themes.Soft()) as demo:\n",
                "    gr.Markdown(\"# 📄 Hệ Thống Document Visual Question Answering (DocVQA)\")\n",
                "    gr.Markdown(\"💡 Trợ lý AI hỏi đáp và bóc tách thông tin hóa đơn tiếng Việt chạy trực tiếp trên **GPU NVIDIA Tesla T4 (16GB VRAM)**.\")\n",
                "    \n",
                "    with gr.Row():\n",
                "        with gr.Column(scale=1):\n",
                "            img_input = gr.Image(type=\"pil\", label=\"Tải lên ảnh Hóa đơn / Chứng từ\")\n",
                "            q_input = gr.Textbox(\n",
                "                lines=2,\n",
                "                placeholder=\"Ví dụ: Tổng tiền thanh toán trên hóa đơn là bao nhiêu?\",\n",
                "                label=\"Câu hỏi hoặc Yêu cầu trích xuất\"\n",
                "            )\n",
                "            btn_submit = gr.Button(\"🚀 Phân tích & Trả lời\", variant=\"primary\")\n",
                "            \n",
                "            gr.Examples(\n",
                "                examples=[\n",
                "                    [\"Tổng tiền thanh toán trên hóa đơn là bao nhiêu?\"],\n",
                "                    [\"Tên cửa hàng / bên bán trên hóa đơn là gì?\"],\n",
                "                    [\"Mã số thuế của bên bán là gì?\"],\n",
                "                    [\"Ngày giờ lập hóa đơn là khi nào?\"],\n",
                "                    [\"Trích xuất toàn bộ thông tin dưới dạng JSON.\"]\n",
                "                ],\n",
                "                inputs=[q_input],\n",
                "                label=\"💡 Gợi ý câu hỏi mẫu\"\n",
                "            )\n",
                "            \n",
                "        with gr.Column(scale=1):\n",
                "            txt_output = gr.Textbox(lines=12, label=\"Kết quả phản hồi từ AI (Qwen2-VL + LoRA)\")\n",
                "            \n",
                "    btn_submit.click(fn=predict_vqa, inputs=[img_input, q_input], outputs=txt_output)\n",
                "\n",
                "print(\"🌐 Đang mở Server Gradio với Public Share Link...\")\n",
                "demo.queue().launch(share=True, debug=True)\n"
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

    nb_path = demo_dir / "qwen2_vl_docvqa_demo.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"📦 Đã tạo Notebook tại: {nb_path}")
    print("📤 Đang đẩy Kernel lên Kaggle...")
    api.kernels_push(str(demo_dir))
    print(f"🚀 Đã kích hoạt Kaggle Kernel: https://www.kaggle.com/code/{kernel_id}")
    print("-" * 80)
    print("⏳ Đang theo dõi tiến trình chạy trên Kaggle GPU để lấy link Public Demo...")
    
    # Theo dõi trạng thái kernel
    for step in range(40):
        time.sleep(15)
        try:
            status_res = api.kernels_status(kernel_id)
            status = status_res.get("status", "unknown")
            print(f"[{time.strftime('%H:%M:%S')}] Trạng thái Kernel: {status.upper()}")
            
            if status in ["running", "complete"]:
                output = api.kernels_output(kernel_id)
                log_text = output.get("log", "")
                if log_text:
                    urls = re.findall(r"https://[a-zA-Z0-9-]+\.gradio\.live", log_text)
                    if urls:
                        print("=" * 80)
                        print(f"🎉 TÌM THẤY LINK DEMO GRADIO PUBLIC:")
                        print(f"👉 {urls[-1]}")
                        print("=" * 80)
                        break
            elif status == "error":
                print(f"❌ Kernel gặp lỗi. Vui lòng kiểm tra trên web: https://www.kaggle.com/code/{kernel_id}")
                break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Đang kiểm tra: {e}")

if __name__ == "__main__":
    setup_and_launch_kaggle_demo()
