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
[OCR service]  ──text/bbox──►  [KIE service]  ──field key/value/bbox──►  DB
   │
[Document: status=processed]
   │  POST /documents/{id}/ask  {"question": "..."}
   ▼
[QA service]  ──answer + evidence bbox──►  DB (QARecord)
   │
   GET /documents/{id}            → xem toàn bộ kết quả
   GET /documents/{id}/export     → tải file JSON hoàn chỉnh
```

Ba service (`ocr_service`, `kie_service`, `qa_service`) là **interface cố định**,
tách biệt hoàn toàn khỏi routing/DB. Đây chính là ranh giới tích hợp model.

## 2. Cài đặt & chạy local

```bash
cd "D:\ML\Repo\Document-Visual-Question-Answering-System\backend\backend-docvqa\backend"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

or

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

Chạy test:
```bash
pytest tests/ -v
```

Chạy bằng Docker:
```bash
docker build -t docvqa-backend .
docker run -p 8000:8000 docvqa-backend
```

## 3. API Endpoints

| Method | Endpoint | Chức năng |
|---|---|---|
| POST | `/api/v1/documents/upload` | Upload ảnh hoá đơn (multipart/form-data, field `file`) |
| POST | `/api/v1/documents/{id}/process` | Chạy OCR + trích xuất field (KIE) |
| POST | `/api/v1/documents/{id}/ask` | Đặt câu hỏi, nhận câu trả lời + evidence bbox |
| GET | `/api/v1/documents/{id}` | Lấy toàn bộ kết quả (field + lịch sử hỏi-đáp) |
| GET | `/api/v1/documents/{id}/export` | Tải file JSON kết quả cuối cùng |
| GET | `/health` | Health check |

## 4. ⚠️ CÁC ĐIỂM CẦN SẢN PHẨM CỦA NHÓM (quan trọng)

Backend hiện **chạy end-to-end bằng dữ liệu MOCK** (giả lập) để cả nhóm có API
dùng ngay, test frontend, và demo luồng — không cần chờ model train xong.
Ba file sau **phải được thay bằng model thật** trước khi báo cáo:

| File | Model theo đề cương | Người phụ trách | Việc cần làm |
|---|---|---|---|
| `app/services/ocr_service.py` → `run_ocr()` | PaddleOCR (PP-OCRv4) | Model Lead / Data Engineer | Thay mock bằng load PaddleOCR thật, giữ nguyên input/output là `List[OCRToken]` |
| `app/services/kie_service.py` → `extract_fields()` | LayoutLMv3 / LayoutXLM / LiLT | Model Lead | Thay rule-based mock bằng model multimodal thật, giữ nguyên `List[FieldResult]` |
| `app/services/qa_service.py` → `answer_question()` | Qwen2-VL 2B / Qwen2.5-3B | Model Lead | Thay tra cứu mock bằng inference model thật, giữ nguyên `QAResult` |

Mỗi file đều có docstring chi tiết + ví dụ khung code tích hợp thật ngay bên
trong hàm — chỉ cần đọc theo TODO. Miễn giữ đúng **chữ ký hàm (input/output)**,
phần router/DB/pipeline không cần sửa gì thêm.

Ngoài ra `app/config.py` đã khai báo sẵn các biến `ocr_model_path`,
`kie_model_path`, `qa_model_path`, `device` để cấu hình checkpoint/device mà
không cần sửa code — DevOps (Nguyễn Bá Quốc Long) có thể set qua file `.env`.

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
│       ├── ocr_service.py   # ⚠️ interface OCR — cần model thật
│       ├── kie_service.py   # ⚠️ interface KIE — cần model thật
│       ├── qa_service.py    # ⚠️ interface QA — cần model thật
│       └── pipeline_service.py  # Điều phối OCR → KIE, ghi DB
├── tests/test_api.py        # Test smoke toàn bộ luồng (chạy trên mock)
├── requirements.txt
└── Dockerfile
```

## 6. Ghi chú cho Frontend (React)

- CORS đã mở toàn bộ (`allow_origins=["*"]`) cho môi trường dev.
- `evidence_bbox` trả về dạng `[x1, y1, x2, y2]` theo pixel gốc của ảnh đã upload
  — frontend dùng để vẽ khung highlight đè lên ảnh gốc.
- Trạng thái `Document.status` (`uploaded` → `processing` → `processed`/`failed`)
  nên được frontend dùng để disable nút "Hỏi" cho tới khi `processed`.
