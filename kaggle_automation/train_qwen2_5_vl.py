"""
===================================================================================
🚀 KAGGLE AUTOMATION: HUẤN LUYỆN QWEN2.5-VL-3B LORA FINE-TUNING (DOCVQA TỐI ƯU HÓA)
===================================================================================
Script tự động đẩy code huấn luyện LoRA lên Kaggle GPU (Tesla T4 / P100) và theo dõi tiến trình.
- Base Model: Qwen/Qwen2.5-VL-3B-Instruct
- LoRA Config: rank=16, alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
- Dataset: DocVQA Vietnamese Receipts & Invoices
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

def launch_training():
    api = KaggleApi()
    api.authenticate()

    kernel_slug = "qwen2-5-vl-finetune-optimized"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    work_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/train_kernel")
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": "qwen2-5-vl-finetune-optimized",
        "code_file": "train_qwen2_5_vl.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
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

    with open(work_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"🚀 Đang đẩy Kernel Huấn Luyện lên Kaggle: https://www.kaggle.com/code/{kernel_id}")
    api.kernels_push(str(work_dir))
    print("✅ Đã kích hoạt tiến trình huấn luyện thành công!")

if __name__ == "__main__":
    launch_training()
