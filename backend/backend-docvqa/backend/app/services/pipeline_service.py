"""
PIPELINE SERVICE — điều phối toàn bộ luồng xử lý một Document:

    Document (ảnh/PDF) --Stage 0 preprocessing--> Qwen2-VL --> ExtractedField[]

Đây là lớp "glue code" duy nhất phụ thuộc vào Stage 0 và VLM service.
Router (documents.py) không gọi trực tiếp các service này mà luôn đi qua đây,
để logic ghi DB / xử lý lỗi chỉ nằm ở một chỗ.
"""
from sqlalchemy.orm import Session

from app.models_db import Document, ExtractedField, DocumentStatus
from app.services import preprocessing_service, vlm_service, ocr_service


def process_document(db: Session, document: Document) -> Document:
    """Run Stage 0 then Qwen2-VL extraction + OCR grounding, saving fields and bboxes."""
    document.status = DocumentStatus.PROCESSING
    db.commit()

    try:
        preprocessed = preprocessing_service.preprocess_document(document.id, document.stored_path)
        
        # 1. OCR Token Extraction (nhẹ, nhanh, dùng để vẽ token nền và so khớp Bounding Box)
        ocr_tokens = ocr_service.extract_tokens(preprocessed.image_path)
        document.ocr_raw = ocr_tokens

        # 2. VLM Semantic Field Extraction
        fields, _raw_vlm_response = vlm_service.extract_fields(preprocessed.image_path)

        # 3. Xoá field cũ (nếu re-process) rồi ghi field mới kèm Bounding Box
        db.query(ExtractedField).filter(ExtractedField.document_id == document.id).delete()
        for f in fields:
            # Tự động tìm Bounding Box cho từng trường bóc tách
            field_bbox = ocr_service.locate_evidence_bbox(ocr_tokens, f.value)
            db.add(ExtractedField(
                document_id=document.id,
                key=f.key,
                value=f.value,
                bbox=field_bbox,
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
