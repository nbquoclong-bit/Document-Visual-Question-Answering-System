"""
Router: /documents
Chứa toàn bộ endpoint theo đúng luồng nghiệp vụ mô tả trong đề cương:

    upload ảnh -> Stage 0 + Qwen2-VL -> ask (VQA) -> get/export JSON
"""
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models_db import Document, DocumentStatus, QARecord
from app.schemas import (
    UploadResponse, ProcessResponse, AskRequest, AskResponse, OCRTokenOut,
    DocumentDetailOut, ExtractedFieldOut, QARecordOut,
)
from app.services import pipeline_service, preprocessing_service, qa_service, vlm_service
from app.storage import save_upload, save_result_json

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_document_or_404(db: Session, document_id: str) -> Document:
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy document_id={document_id}")
    return document


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Nhận tài liệu, lưu vào disk và tạo record DB. Chưa chạy model ở bước này."""
    stored_path = save_upload(file)

    document = Document(
        original_filename=file.filename,
        stored_path=stored_path,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return UploadResponse(
        document_id=document.id,
        original_filename=document.original_filename,
        status=document.status,
    )


@router.post("/{document_id}/process", response_model=ProcessResponse)
def process_document(document_id: str, db: Session = Depends(get_db)):
    """Chạy Stage 0 + Qwen2-VL. Có thể gọi lại để xử lý lại document."""
    document = _get_document_or_404(db, document_id)
    document = pipeline_service.process_document(db, document)

    return ProcessResponse(
        document_id=document.id,
        status=document.status,
        fields=[ExtractedFieldOut.model_validate(f) for f in document.fields],
        ocr_tokens=[OCRTokenOut(**token) for token in (document.ocr_raw or [])],
        error_message=document.error_message,
    )


@router.post("/{document_id}/ask", response_model=AskResponse)
def ask_question(document_id: str, payload: AskRequest, db: Session = Depends(get_db)):
    """Hỏi-đáp trên một document đã được process (VQA)."""
    document = _get_document_or_404(db, document_id)

    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(
            status_code=409,
            detail=f"Document đang ở trạng thái '{document.status.value}'. "
                   f"Cần gọi /process trước và đợi status='processed'.",
        )

    processed_path = preprocessing_service.get_processed_image_path(document.id)
    try:
        result = qa_service.answer_question(str(processed_path), payload.question)
    except vlm_service.VLMRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    qa_record = QARecord(
        document_id=document.id,
        question=payload.question,
        answer=result.answer,
        evidence_bbox=result.evidence_bbox,
        confidence=result.confidence,
    )
    db.add(qa_record)
    db.commit()
    db.refresh(qa_record)

    return AskResponse(
        document_id=document.id,
        question=qa_record.question,
        answer=qa_record.answer,
        evidence_bbox=qa_record.evidence_bbox,
        confidence=qa_record.confidence,
        created_at=qa_record.created_at,
    )


@router.get("/{document_id}", response_model=DocumentDetailOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Lấy toàn bộ thông tin của một document: field đã trích xuất + lịch sử hỏi-đáp."""
    document = _get_document_or_404(db, document_id)

    return DocumentDetailOut(
        document_id=document.id,
        original_filename=document.original_filename,
        status=document.status,
        error_message=document.error_message,
        ocr_tokens=[OCRTokenOut(**token) for token in (document.ocr_raw or [])],
        fields=[ExtractedFieldOut.model_validate(f) for f in document.fields],
        qa_history=[QARecordOut.model_validate(q) for q in document.qa_records],
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get("/{document_id}/image")
def get_document_image(document_id: str, db: Session = Depends(get_db)):
    """Serve the original upload so the React document viewer can overlay evidence."""
    document = _get_document_or_404(db, document_id)
    image_path = Path(document.stored_path)
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file gốc của tài liệu.")

    media_type, _ = mimetypes.guess_type(image_path.name)
    return FileResponse(path=image_path, media_type=media_type or "application/octet-stream")


@router.get("/{document_id}/export")
def export_document(document_id: str, db: Session = Depends(get_db)):
    """Xuất kết quả cuối cùng ra file .json (theo đúng yêu cầu đề cương: 'xuất JSON')."""
    document = _get_document_or_404(db, document_id)

    detail = DocumentDetailOut(
        document_id=document.id,
        original_filename=document.original_filename,
        status=document.status,
        error_message=document.error_message,
        ocr_tokens=[OCRTokenOut(**token) for token in (document.ocr_raw or [])],
        fields=[ExtractedFieldOut.model_validate(f) for f in document.fields],
        qa_history=[QARecordOut.model_validate(q) for q in document.qa_records],
        created_at=document.created_at,
        updated_at=document.updated_at,
    )

    json_path = save_result_json(document.id, detail.model_dump_json(indent=2))
    return FileResponse(
        path=json_path,
        media_type="application/json",
        filename=f"{document.id}.json",
    )
