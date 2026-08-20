"""
Cấu hình tập trung cho backend.

Mọi tham số có thể thay đổi theo môi trường triển khai (local / Docker / HF Spaces)
được đưa hết vào đây, đọc từ biến môi trường với giá trị mặc định hợp lý cho dev.
"""
from pathlib import Path
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
    # Chỉ đọc checkpoint từ đường dẫn local để API không tự tải nhiều GB weights
    # trong lúc nhận request. Khi chạy Docker, mount thư mục checkpoint vào MODEL_DIR.
    model_dir: Path = base_dir.parent.parent.parent / "model"
    ocr_language: Literal["vi", "en"] = "vi"
    ocr_use_angle_classifier: bool = True
    ocr_min_confidence: float = 0.5
    kie_model_path: str | None = None
    qa_model_path: str | None = None
    qa_adapter_path: str | None = None
    device: str = "cpu"  # Đặt DEVICE=cuda khi môi trường có GPU.

    # Khi checkpoint KIE/QA chưa có, demo vẫn dùng rule/field lookup trên OCR
    # thật. Đặt false để môi trường production báo lỗi thay vì fallback.
    allow_rule_based_fallback: bool = True

settings = Settings()

# Đảm bảo thư mục lưu trữ luôn tồn tại khi app khởi động
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.result_dir.mkdir(parents=True, exist_ok=True)
