"""
Cấu hình tập trung cho backend.

Mọi tham số có thể thay đổi theo môi trường triển khai (local / Docker / HF Spaces)
được đưa hết vào đây, đọc từ biến môi trường với giá trị mặc định hợp lý cho dev.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    # --- Cấu hình model (placeholder, nhóm Model/DevOps sẽ điền) ---
    # Các biến này KHÔNG được dùng trong logic hiện tại, chỉ khai báo sẵn
    # để khi tích hợp model thật (Qwen2-VL, PaddleOCR...) có chỗ cấu hình
    # đường dẫn checkpoint / device mà không phải sửa code service.
    ocr_model_path: str = "models/paddleocr"          # TODO(Model Lead / DevOps): điền path hoặc HF repo id
    kie_model_path: str = "models/layoutlmv3-base"    # TODO(Model Lead): điền path hoặc HF repo id
    qa_model_path: str = "Qwen/Qwen2-VL-2B"           # TODO(Model Lead): điền path hoặc HF repo id
    device: str = "cpu"                                # "cuda" khi có GPU

    class Config:
        env_file = ".env"


settings = Settings()

# Đảm bảo thư mục lưu trữ luôn tồn tại khi app khởi động
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.result_dir.mkdir(parents=True, exist_ok=True)
