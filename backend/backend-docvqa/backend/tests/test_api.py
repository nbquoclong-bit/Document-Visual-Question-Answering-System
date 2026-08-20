"""API contract tests with injected model outputs; no GPU or weights are required."""
import io

from fastapi.testclient import TestClient

from app.main import app
from app.services import kie_service, ocr_service, qa_service

client = TestClient(app)


def _fake_image_bytes() -> bytes:
    # Không cần ảnh thật hợp lệ vì OCR đang là mock, chỉ cần bytes + đúng đuôi file
    return b"\xff\xd8\xff\xe0fake-jpeg-content"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_pipeline_flow(monkeypatch):
    monkeypatch.setattr(
        ocr_service,
        "run_ocr",
        lambda _: [
            ocr_service.OCRToken("CỬA HÀNG ABC", [0.1, 0.05, 0.5, 0.1], 0.98),
            ocr_service.OCRToken("Tổng cộng: 150,000 VND", [0.1, 0.8, 0.7, 0.85], 0.93),
        ],
    )
    monkeypatch.setattr(
        kie_service,
        "extract_fields",
        lambda _path, _tokens: [
            kie_service.FieldResult("total_amount", "150,000 VND", [0.1, 0.8, 0.7, 0.85], 0.93)
        ],
    )
    monkeypatch.setattr(
        qa_service,
        "answer_question",
        lambda *_args: qa_service.QAResult("150,000 VND", [0.1, 0.8, 0.7, 0.85], 0.93),
    )
    # 1) Upload
    files = {"file": ("invoice.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 200, resp.text
    document_id = resp.json()["document_id"]
    assert resp.json()["status"] == "uploaded"

    # 2) Process (OCR + KIE mock)
    resp = client.post(f"/api/v1/documents/{document_id}/process")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "processed"
    assert len(body["fields"]) > 0
    assert body["ocr_tokens"][0]["bbox"] == [0.1, 0.05, 0.5, 0.1]

    # 3) Ask (VQA mock)
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


def test_ask_before_process_returns_409():
    files = {"file": ("invoice2.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")}
    resp = client.post("/api/v1/documents/upload", files=files)
    document_id = resp.json()["document_id"]

    resp = client.post(
        f"/api/v1/documents/{document_id}/ask",
        json={"question": "Ngày hoá đơn là khi nào?"},
    )
    assert resp.status_code == 409


def test_upload_rejects_invalid_extension():
    files = {"file": ("invoice.exe", io.BytesIO(b"not-an-image"), "application/octet-stream")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400
