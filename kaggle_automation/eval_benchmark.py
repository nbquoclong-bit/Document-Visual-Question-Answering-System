"""
===================================================================================
📊 KAGGLE AUTOMATION: ĐÁNH GIÁ BENCHMARK TOÀN DIỆN (ANLS, TOKEN F1, EXACT MATCH)
===================================================================================
Script tự động đẩy và theo dõi tiến trình đánh giá 174 mẫu hóa đơn kiểm thử (unseen).
So sánh chi tiết Base Model vs LoRA Fine-Tuned Model theo từng trường thông tin.
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

def launch_eval_benchmark():
    api = KaggleApi()
    api.authenticate()

    kernel_slug = "qwen2-5-vl-eval-benchmark"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    work_dir = Path("d:/STUDY/MLIoT/project/kaggle_automation/eval_kernel")
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel_id,
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
            "lminhsang241/docvqa-benchmark-dataset"
        ],
        "competition_sources": [],
        "kernel_sources": [
            "lminhsang241/qwen2-5-vl-finetune-optimized"
        ],
        "model_sources": []
    }

    with open(work_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"📊 Đang kích hoạt Benchmark Evaluation tại: https://www.kaggle.com/code/{kernel_id}")
    api.kernels_push(str(work_dir))
    print("✅ Đã đẩy benchmark kernel thành công!")

if __name__ == "__main__":
    launch_eval_benchmark()
