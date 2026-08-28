# Backend — Document Visual Question Answering System

Backend FastAPI cho đề tài **"Hệ thống Hỏi-đáp Trực quan trên Tài liệu sử dụng OCR
và Kiến trúc Đa phương thức"** (Nhóm 5 — Boboiboys).

## 1. Kiến trúc & luồng xử lý

```
Ảnh hoá đơn
   │  POST /documents/upload
   ▼
[Document: status=uploaded]  (lưu ảnh + tạo record SQLite)
   │  POST /documents/{id}/process
   ▼
[Stage 0 OpenCV] ──ảnh chuẩn hoá──► [Qwen2-VL] ──field key/value──► DB
                                      ▲
[EasyOCR] ──token/bbox──► đối chiếu bằng chứng và gắn bbox cho field
   │
[Document: status=processed]
   │  POST /documents/{id}/ask  {"question": "..."}
   ▼
[Qwen2-VL + LoRA QA] ──answer──► [EasyOCR grounding] ──evidence bbox──► DB
   │
   GET /documents/{id}            → xem toàn bộ kết quả
   GET /documents/{id}/export     → tải file JSON hoàn chỉnh
```

Các service model tách biệt khỏi routing/DB. `pipeline_service` điều phối Stage 0,
EasyOCR và VLM; `qa_service` gọi VLM rồi đối chiếu câu trả lời với token OCR.

## 2. Cài đặt & chạy local

```powershell
cd "D:\ML\Repo\Document-Visual-Question-Answering-System\backend\backend-docvqa\backend"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Swagger UI: http://localhost:8000/docs

Chạy test:
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py -p no:cacheprovider
```

Chạy bằng Docker:
```powershell
cd ..\..\..
docker compose up --build
```

## 3. API Endpoints

| Method | Endpoint | Chức năng |
|---|---|---|
| POST | `/api/v1/documents/upload` | Upload ảnh hoá đơn (multipart/form-data, field `file`) |
| POST | `/api/v1/documents/{id}/process` | Tiền xử lý, OCR grounding và trích xuất field bằng Qwen2-VL |
| POST | `/api/v1/documents/{id}/ask` | Đặt câu hỏi, nhận câu trả lời + evidence bbox |
| GET | `/api/v1/documents/{id}` | Lấy toàn bộ kết quả (field + lịch sử hỏi-đáp) |
| GET | `/api/v1/documents/{id}/export` | Tải file JSON kết quả cuối cùng |
| GET | `/health` | Health check |

## 4. Cấu hình model

Backend đang dùng model thật. Cấu hình chính nằm trong `.env`:

```dotenv
VLM_BASE_MODEL=Qwen/Qwen2-VL-2B-Instruct
VLM_ALLOW_BASE_MODEL=false
VLM_EXTRACTION_MODE=base
```

- `base`: base model trích xuất JSON một lượt; LoRA vẫn dùng cho hỏi đáp.
- `single`: LoRA trích xuất JSON một lượt.
- `multi`: LoRA hỏi tuần tự từng field, chậm hơn đáng kể.

Model và EasyOCR được cache trong tiến trình. Sau khi đổi checkpoint hoặc `.env`,
phải khởi động lại backend để nạp cấu hình mới.

## 5. Cấu trúc thư mục

```
backend/
├── app/
│   ├── main.py              # FastAPI app, khởi tạo DB, mount router
│   ├── config.py            # Cấu hình tập trung (đường dẫn, model path...)
│   ├── database.py          # SQLAlchemy engine/session (SQLite)
│   ├── models_db.py         # ORM models: Document, ExtractedField, QARecord
│   ├── schemas.py            # Pydantic schemas request/response
│   ├── storage.py           # Lưu/đọc file ảnh + JSON kết quả
│   ├── routers/
│   │   └── documents.py     # Toàn bộ endpoint /documents/*
│   └── services/
│       ├── preprocessing_service.py # Adapter Stage 0 OpenCV
│       ├── vlm_service.py   # Qwen2-VL/LoRA và parser field
│       ├── ocr_service.py   # EasyOCR token + evidence bbox
│       ├── qa_service.py    # Hỏi đáp VLM + OCR grounding
│       └── pipeline_service.py  # Điều phối Stage 0 → OCR/VLM → DB
├── tests/test_api.py        # Contract test; mock inference để không cần GPU
├── requirements.txt
└── Dockerfile
```

## 6. Ghi chú cho Frontend (React)

- CORS đã mở toàn bộ (`allow_origins=["*"]`) cho môi trường dev.
- `evidence_bbox` trả về dạng `[x1, y1, x2, y2]` theo pixel gốc của ảnh đã upload
  — frontend dùng để vẽ khung highlight đè lên ảnh gốc.
- Trạng thái `Document.status` (`uploaded` → `processing` → `processed`/`failed`)
  nên được frontend dùng để disable nút "Hỏi" cho tới khi `processed`.
