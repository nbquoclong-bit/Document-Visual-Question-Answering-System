"""
ORM models (SQLAlchemy) — ánh xạ trực tiếp với sơ đồ pipeline trong đề cương:

    Ảnh hoá đơn --(OCR)--> text/bbox --(Multimodal KIE/QA)--> field + answer + evidence

Ba bảng:
- Document: một lượt upload ảnh hoá đơn, theo dõi trạng thái xử lý pipeline.
- ExtractedField: các trường thông tin hoá đơn được trích xuất (KIE) — vd: tên cửa hàng,
  tổng tiền, ngày hoá đơn...
- QARecord: lịch sử hỏi-đáp trên một Document, kèm evidence (bbox) để highlight trên ảnh.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Float, Text, Enum as SAEnum, JSON
)
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"        # đã nhận ảnh, chưa xử lý
    PROCESSING = "processing"    # đang chạy OCR + KIE
    PROCESSED = "processed"      # đã có kết quả OCR + KIE, sẵn sàng nhận câu hỏi
    FAILED = "failed"            # pipeline lỗi


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_uuid)
    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)          # đường dẫn ảnh đã lưu trên disk
    status = Column(SAEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    error_message = Column(Text, nullable=True)            # lý do fail nếu status = FAILED

    # Kết quả OCR thô, lưu dạng JSON: List[{"text", "bbox", "confidence"}]
    ocr_raw = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fields = relationship("ExtractedField", back_populates="document", cascade="all, delete-orphan")
    qa_records = relationship("QARecord", back_populates="document", cascade="all, delete-orphan")


class ExtractedField(Base):
    """Một trường thông tin hoá đơn được trích xuất (Key Information Extraction)."""
    __tablename__ = "extracted_fields"

    id = Column(String, primary_key=True, default=gen_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)

    key = Column(String, nullable=False)        # vd: "store_name", "total_amount", "invoice_date"
    value = Column(Text, nullable=False)
    bbox = Column(JSON, nullable=True)          # [x1, y1, x2, y2] toạ độ trên ảnh gốc
    confidence = Column(Float, nullable=True)

    document = relationship("Document", back_populates="fields")


class QARecord(Base):
    """Một lượt hỏi-đáp trên tài liệu, phục vụ tính năng VQA (Visual Question Answering)."""
    __tablename__ = "qa_records"

    id = Column(String, primary_key=True, default=gen_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)

    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    evidence_bbox = Column(JSON, nullable=True)   # bbox vùng ảnh làm bằng chứng cho câu trả lời
    confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="qa_records")
