"""
Entry point của backend — Document Visual Question Answering System.
Chạy local:  uvicorn app.main:app --reload --port 8000
Docs tự sinh: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.routers import documents

# Tạo bảng nếu chưa có (dev only — khi lên production nên dùng Alembic migration)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend cho hệ thống Hỏi-đáp Trực quan trên Tài liệu (Document VQA), "
        "sử dụng OCR + kiến trúc đa phương thức để trích xuất thông tin hoá đơn "
        "và trả lời câu hỏi tự nhiên kèm bằng chứng (evidence highlight)."
    ),
    version="0.1.0",
)

# CORS mở cho frontend React (dev). Khi deploy, nên giới hạn lại origin cụ thể.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": settings.app_name}
