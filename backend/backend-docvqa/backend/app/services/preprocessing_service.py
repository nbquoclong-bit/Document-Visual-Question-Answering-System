"""Adapter between FastAPI and the team's Stage 0 preprocessing module."""
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2

from app.config import settings


@dataclass(frozen=True)
class PreprocessedDocument:
    image_path: str
    metadata: dict[str, Any]


def _import_stage0():
    project_root = str(settings.model_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.stage0_preprocessing.src.preprocessor import PreprocessingEngine

    return PreprocessingEngine


def get_processed_image_path(document_id: str) -> Path:
    """Return the stable artifact used by both VLM extraction and later QA."""
    return settings.result_dir / f"{document_id}.processed.jpg"


def preprocess_document(document_id: str, input_path: str) -> PreprocessedDocument:
    """Run Stage 0 and persist the first document page as a VLM-ready image."""
    engine_class = _import_stage0()
    kind, payload = engine_class().process(input_path)

    if kind == "pdf":
        if not payload:
            raise ValueError("PDF không có trang nào để xử lý.")
        page_number, image, metadata = payload[0]
        metadata = {**metadata, "page_number": page_number, "source_kind": "pdf"}
    else:
        image, metadata = payload
        metadata = {**metadata, "source_kind": "image"}

    output_path = get_processed_image_path(document_id)
    saved = False
    try:
        suffix = output_path.suffix or ".jpg"
        is_success, buffer = cv2.imencode(suffix, image)
        if is_success:
            with open(output_path, "wb") as f:
                f.write(buffer)
            saved = True
    except Exception:
        pass
    if not saved and not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Không thể lưu ảnh sau tiền xử lý.")
    return PreprocessedDocument(image_path=str(output_path), metadata=metadata)
