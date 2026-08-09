"""
Pydantic schemas — hợp đồng dữ liệu (contract) giữa backend và frontend (React).
Frontend chỉ cần bám theo các schema này để build UI, không phụ thuộc vào chi tiết
model bên trong.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models_db import DocumentStatus


class UploadResponse(BaseModel):
    document_id: str
    original_filename: str
    status: DocumentStatus
    message: str = "Tải ảnh thành công. Gọi /process để bắt đầu OCR + trích xuất thông tin."


class ExtractedFieldOut(BaseModel):
    key: str
    value: str
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None

    model_config = {"from_attributes": True}


class ProcessResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    fields: List[ExtractedFieldOut] = Field(default_factory=list)
    error_message: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Tổng tiền trên hoá đơn là bao nhiêu?"])


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    evidence_bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class QARecordOut(BaseModel):
    question: str
    answer: str
    evidence_bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetailOut(BaseModel):
    """Dùng cho GET /documents/{id} và cho export JSON cuối pipeline."""
    document_id: str
    original_filename: str
    status: DocumentStatus
    error_message: Optional[str] = None
    fields: List[ExtractedFieldOut] = Field(default_factory=list)
    qa_history: List[QARecordOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
