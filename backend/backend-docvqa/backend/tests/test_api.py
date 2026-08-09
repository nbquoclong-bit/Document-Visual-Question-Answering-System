"""
Test smoke: kiểm tra toàn bộ luồng nghiệp vụ chạy được với dữ liệu mock,
KHÔNG kiểm tra độ chính xác của model (vì model thật chưa được tích hợp).
"""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _fake_image_bytes() -> bytes:
    # Không cần ảnh thật hợp lệ vì OCR đang là mock, chỉ cần bytes + đúng đuôi file
    return b"\xff\xd8\xff\xe0fake-jpeg-content"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_pipeline_flow():
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

    # 5) Export JSON
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
