"""
Cấu hình tập trung cho backend.

Mọi tham số có thể thay đổi theo môi trường triển khai (local / Docker / HF Spaces)
được đưa hết vào đây, đọc từ biến môi trường với giá trị mặc định hợp lý cho dev.
"""
from pathlib import Path
import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        protected_namespaces=("settings_",),
    )
    # --- Thông tin chung ---
    app_name: str = "Document VQA Backend"
    api_v1_prefix: str = "/api/v1"

    # --- Đường dẫn lưu trữ ---
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "uploads"
    result_dir: Path = base_dir / "results"
    database_url: str = f"sqlite:///{base_dir / 'app.db'}"

    # --- Giới hạn upload ---
    max_upload_size_mb: int = 10
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".pdf")

    # --- Cấu hình model ---
    # Khi chạy Docker, mount source/checkpoint vào MODEL_DIR (mặc định: model/ ở root repo).
    model_dir: Path = base_dir.parent.parent.parent / "model"
    vlm_base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_adapter_path: Path | None = None
    # Repo đã có adapter hoàn chỉnh; mặc định không được âm thầm rơi về base model.
    vlm_allow_base_model: bool = False
    # Mặc định sử dụng LoRA adapter đã fine-tune cho extraction và QA
    vlm_extraction_mode: Literal["base", "multi", "single"] = "single"

    # Kept configurable so a deployed model can be forced to CPU if necessary.
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

settings = Settings()

# Đảm bảo thư mục lưu trữ luôn tồn tại khi app khởi động
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.result_dir.mkdir(parents=True, exist_ok=True)
