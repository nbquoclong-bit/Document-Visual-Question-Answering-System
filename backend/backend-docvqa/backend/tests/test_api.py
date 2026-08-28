"""API contract tests with injected Stage 0/VLM results; no GPU or weights required."""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.services import ocr_service, preprocessing_service, qa_service, vlm_service


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
    test_engine.dispose()


def _fake_image_bytes() -> bytes:
    # Không cần ảnh thật hợp lệ vì OCR đang là mock, chỉ cần bytes + đúng đuôi file
    return b"\xff\xd8\xff\xe0fake-jpeg-content"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_pipeline_flow(client, monkeypatch):
    monkeypatch.setattr(
        preprocessing_service,
        "preprocess_document",
        lambda _document_id, _source: preprocessing_service.PreprocessedDocument(
            image_path="preprocessed-invoice.jpg", metadata={"source_kind": "image"}
        ),
    )
    monkeypatch.setattr(
        vlm_service,
        "extract_fields",
        lambda _path: ([vlm_service.VLMField("total_amount", "150,000 VND")], '{"total_amount":"150,000 VND"}'),
    )
    monkeypatch.setattr(ocr_service, "extract_tokens", lambda _path: [])
    monkeypatch.setattr(
        qa_service,
        "answer_question",
        lambda *_args, **_kwargs: qa_service.QAResult("150,000 VND", None, None),
    )
    # 1) Upload
    files = {"file": ("invoice.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 200, resp.text
    document_id = resp.json()["document_id"]
    assert resp.json()["status"] == "uploaded"

    # 2) Process (Stage 0 + VLM extraction)
    resp = client.post(f"/api/v1/documents/{document_id}/process")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "processed"
    assert len(body["fields"]) > 0
    assert body["ocr_tokens"] == []

    # 3) Ask (VLM mock)
    resp = client.post(
        f"/api/v1/documents/{document_id}/ask",
        json={"question": "Tổng tiền trên hoá đơn là bao nhiêu?"},
    )
    assert resp.status_code == 200, resp.text
    assert "150,000" in resp.json()["answer"]

    # 4) Get detail
    resp = client.get(f"/api/v1/documents/{document_id}")
    assert resp.status_code == 200
    assert len(resp.json()["qa_history"]) == 1

    # 5) Original image URL used by the React document viewer
    resp = client.get(f"/api/v1/documents/{document_id}/image")
    assert resp.status_code == 200

    # 6) Export JSON
    resp = client.get(f"/api/v1/documents/{document_id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"


def test_ask_before_process_returns_409(client):
    files = {"file": ("invoice2.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")}
    resp = client.post("/api/v1/documents/upload", files=files)
    document_id = resp.json()["document_id"]

    resp = client.post(
        f"/api/v1/documents/{document_id}/ask",
        json={"question": "Ngày hoá đơn là khi nào?"},
    )
    assert resp.status_code == 409


def test_upload_rejects_invalid_extension(client):
    files = {"file": ("invoice.exe", io.BytesIO(b"not-an-image"), "application/octet-stream")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400


def test_upload_rejects_file_over_size_limit(client):
    oversized = b"x" * (10 * 1024 * 1024 + 1)
    files = {"file": ("large.jpg", io.BytesIO(oversized), "image/jpeg")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 413


def test_vlm_training_labels_are_mapped_to_frontend_fields(monkeypatch):
    monkeypatch.setattr(vlm_service.settings, "vlm_extraction_mode", "single")
    monkeypatch.setattr(
        vlm_service,
        "answer_question",
        lambda *_args, **_kwargs: '{"SELLER":"Cửa hàng An An","TOTAL_COST":"27500","ADDRESS":"Dĩ An"}',
    )
    fields, _raw = vlm_service.extract_fields("invoice.jpg")
    values = {field.key: field.value for field in fields}
    assert values == {
        "store_name": "Cửa hàng An An",
        "total_amount": "27.500",
        "address": "Dĩ An",
    }


def test_vlm_extraction_mode_routes_base_and_lora(monkeypatch):
    calls = []

    def fake_answer(*_args, **kwargs):
        calls.append(kwargs["use_adapter"])
        return '{"SELLER":"Cửa hàng An An"}'

    monkeypatch.setattr(vlm_service, "answer_question", fake_answer)

    monkeypatch.setattr(vlm_service.settings, "vlm_extraction_mode", "base")
    vlm_service.extract_fields("invoice.jpg")

    monkeypatch.setattr(vlm_service.settings, "vlm_extraction_mode", "single")
    vlm_service.extract_fields("invoice.jpg")

    assert calls == [False, True]
