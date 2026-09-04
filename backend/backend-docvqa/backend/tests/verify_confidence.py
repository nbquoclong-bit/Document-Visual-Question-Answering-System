"""
Verification script for End-to-End VLM Confidence Score pipeline.
"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.services import preprocessing_service, qa_service, vlm_service

def run_verification():
    print("[1] Setting up in-memory SQLite database...")
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
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("  -> /health OK")

    # 2. Mock services
    preprocessing_service.preprocess_document = lambda _id, _src: preprocessing_service.PreprocessedDocument(
        image_path="test_invoice.jpg", metadata={"source_kind": "image"}
    )
    
    mock_fields = [
        vlm_service.VLMField("store_name", "HIGHLANDS COFFEE", confidence=0.96),
        vlm_service.VLMField("tax_code", "0302863720", confidence=0.98),
        vlm_service.VLMField("total_amount", "109.000", confidence=0.95),
        vlm_service.VLMField("invoice_date", "16/06/2026 14:30", confidence=0.94),
    ]
    vlm_service.extract_fields = lambda _path: (mock_fields, '{"store_name":"HIGHLANDS COFFEE"}')
    qa_service.answer_question = lambda _path, _q, **kwargs: qa_service.QAResult(
        answer="109.000", evidence_bbox=None, confidence=0.95
    )

    # 3. Upload
    fake_bytes = b"\xff\xd8\xff\xe0test-invoice-image"
    files = {"file": ("highlands_test.jpg", io.BytesIO(fake_bytes), "image/jpeg")}
    upload_resp = client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    doc_id = upload_resp.json()["document_id"]
    print(f"  -> Uploaded doc_id={doc_id} OK")

    # 4. Process
    process_resp = client.post(f"/api/v1/documents/{doc_id}/process")
    assert process_resp.status_code == 200, f"Process failed: {process_resp.text}"
    body = process_resp.json()
    assert body["status"] == "processed"
    assert len(body["fields"]) == 4
    for field in body["fields"]:
        assert field["confidence"] is not None, f"Field {field['key']} has no confidence!"
        assert field["confidence"] >= 0.90
        print(f"     * Field: {field['key']} = {field['value']} [Confidence: {field['confidence'] * 100:.0f}%]")
    print("  -> /process with Confidence OK")

    # 5. Ask question
    ask_resp = client.post(
        f"/api/v1/documents/{doc_id}/ask",
        json={"question": "Tổng tiền thanh toán là bao nhiêu?"}
    )
    assert ask_resp.status_code == 200, f"Ask failed: {ask_resp.text}"
    ask_body = ask_resp.json()
    assert ask_body["answer"] == "109.000"
    assert ask_body["confidence"] == 0.95
    assert ask_body["evidence_bbox"] is None
    print(f"  -> /ask Question: 'Tổng tiền...' => Answer: '{ask_body['answer']}' [Confidence: {ask_body['confidence'] * 100:.0f}%] OK")

    # 6. Get document detail
    detail_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert detail_resp.status_code == 200
    detail_body = detail_resp.json()
    assert len(detail_body["fields"]) == 4
    assert len(detail_body["qa_history"]) == 1
    assert detail_body["qa_history"][0]["confidence"] == 0.95
    print("  -> /get_document detail OK")

    # 7. Export JSON
    export_resp = client.get(f"/api/v1/documents/{doc_id}/export")
    assert export_resp.status_code == 200
    export_json = export_resp.json()
    assert "fields" in export_json
    assert export_json["fields"][0]["confidence"] is not None
    print("  -> /export JSON contains confidence OK")

    print("\n[SUCCESS] ALL END-TO-END CONFIDENCE SCORE VERIFICATION CHECKS PASSED 100%!")

if __name__ == "__main__":
    run_verification()
