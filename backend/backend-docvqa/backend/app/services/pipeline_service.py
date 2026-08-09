"""
PIPELINE SERVICE — điều phối toàn bộ luồng xử lý một Document:

    Document (ảnh) --run_ocr--> OCRToken[] --extract_fields--> ExtractedField[]

Đây là lớp "glue code" duy nhất phụ thuộc vào cả ocr_service và kie_service.
Router (documents.py) không gọi trực tiếp 2 service đó mà luôn đi qua đây,
để logic ghi DB / xử lý lỗi chỉ nằm ở một chỗ.
"""
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.models_db import Document, ExtractedField, DocumentStatus
from app.services import ocr_service, kie_service


def process_document(db: Session, document: Document) -> Document:
    """Chạy OCR + KIE cho một document, cập nhật status và lưu field vào DB."""
    document.status = DocumentStatus.PROCESSING
    db.commit()

    try:
        ocr_tokens = ocr_service.run_ocr(document.stored_path)
        document.ocr_raw = [asdict(t) for t in ocr_tokens]

        fields = kie_service.extract_fields(document.stored_path, ocr_tokens)

        # Xoá field cũ (nếu re-process) rồi ghi field mới
        db.query(ExtractedField).filter(ExtractedField.document_id == document.id).delete()
        for f in fields:
            db.add(ExtractedField(
                document_id=document.id,
                key=f.key,
                value=f.value,
                bbox=f.bbox,
                confidence=f.confidence,
            ))

        document.status = DocumentStatus.PROCESSED
        document.error_message = None

    except Exception as exc:  # noqa: BLE001 — pipeline demo, log lỗi cụ thể để debug
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)

    db.commit()
    db.refresh(document)
    return document
