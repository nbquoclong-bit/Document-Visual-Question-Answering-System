"""
Lớp tiện ích thao tác file trên disk. Tách riêng để sau này dễ thay bằng
S3 / GCS / Hugging Face Hub storage khi deploy mà không đụng vào route logic.
"""
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, HTTPException

from app.config import settings


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file '{ext}' không được hỗ trợ. "
                   f"Chỉ chấp nhận: {', '.join(settings.allowed_extensions)}",
        )
    return ext


def save_upload(file: UploadFile) -> str:
    """Lưu file upload vào thư mục uploads/, trả về đường dẫn tuyệt đối dạng string."""
    ext = validate_extension(file.filename)
    unique_name = f"{uuid4().hex}{ext}"
    dest_path = settings.upload_dir / unique_name

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0
    try:
        with dest_path.open("wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File vượt quá giới hạn {settings.max_upload_size_mb} MB.",
                    )
                buffer.write(chunk)
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise

    return str(dest_path)


def save_result_json(document_id: str, content: str) -> Path:
    """Ghi JSON kết quả cuối cùng ra file, phục vụ endpoint export."""
    dest_path = settings.result_dir / f"{document_id}.json"
    dest_path.write_text(content, encoding="utf-8")
    return dest_path
