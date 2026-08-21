# Chạy sản phẩm Document VQA

Tài liệu này dành cho việc chạy sản phẩm hoàn chỉnh: React dashboard → FastAPI → Stage 0 preprocessing → Qwen2-VL + LoRA adapter. Mô hình là **end-to-end VLM**, không cần khởi động một dịch vụ OCR/KIE riêng.

## 1. Chuẩn bị model

Thư mục adapter phải có **trọng số thật** cùng với các file cấu hình đã có trong repo. Đặt chúng tại:

```text
model/stage1_vlm/output/lora_adapters/
├── adapter_config.json
├── adapter_model.safetensors   # hoặc adapter_model.bin
└── tokenizer / processor files
```

Nếu chỉ có `adapter_config.json`, backend có thể chạy Qwen2-VL base khi `VLM_ALLOW_BASE_MODEL=true`, nhưng kết quả sẽ chưa mang fine-tune của nhóm và lần đầu cần tải base model khoảng vài GB từ Hugging Face.

## 2. Chạy local trên Windows

Mở hai terminal tại thư mục gốc repo.

### Terminal 1 — backend

```powershell
cd backend/backend-docvqa/backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tuỳ chọn: đặt đường dẫn model local hoặc Hugging Face cache
$env:VLM_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
$env:VLM_ADAPTER_PATH = "D:\ML\Repo\Document-Visual-Question-Answering-System\model\stage1_vlm\output\lora_adapters"
$env:VLM_ALLOW_BASE_MODEL = "true"

uvicorn app.main:app --reload --port 8000
```

API docs: <http://localhost:8000/docs>. Lần gọi xử lý đầu tiên nạp Qwen2-VL nên có thể mất vài phút, nhất là trên CPU.

### Terminal 2 — frontend

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Mở <http://localhost:5173>. Biến `VITE_API_URL` trong `frontend/.env` phải là `http://localhost:8000/api/v1` khi chạy Vite trực tiếp.

## 3. Chạy bằng Docker Compose

Đảm bảo Docker Desktop đang chạy, sau đó thực hiện từ thư mục gốc:

```powershell
docker compose up --build
```

- Dashboard: <http://localhost>
- API: <http://localhost:8000/docs>
- SQLite, file upload và ảnh sau preprocessing được giữ ở Docker volumes.

Compose mount `./model` vào `/app/model` chỉ đọc. Với GPU NVIDIA, cài NVIDIA Container Toolkit rồi thêm cấu hình GPU phù hợp cho service `backend`; nếu không, hệ thống chạy CPU nhưng suy luận Qwen2-VL sẽ chậm đáng kể.

## 4. Luồng sử dụng

1. Tải lên ảnh hóa đơn hoặc PDF.
2. Bấm **Xử lý tài liệu**. Backend tiền xử lý Stage 0, sau đó yêu cầu Qwen2-VL xuất JSON có `store_name`, `invoice_number`, `tax_code`, `invoice_date`, `total_amount`.
3. Đặt câu hỏi tự do. Qwen2-VL đọc lại ảnh đã tiền xử lý và trả lời trong khung Hỏi đáp.
4. Bấm **Xuất kết quả JSON** để tải lịch sử, trường trích xuất và câu trả lời.

Qwen2-VL hiện không có bounding-box grounding head trong adapter của nhóm, vì vậy UI hiển thị rõ khi câu trả lời không có vị trí evidence trên ảnh.

## 5. Kiểm tra nhanh

```powershell
cd backend/backend-docvqa/backend
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py -p no:cacheprovider

cd ../../../frontend
npm run lint
npm run build
```

Các API tests mock VLM output để chạy không cần GPU hoặc checkpoint; chúng kiểm tra upload, Stage 0/VLM orchestration, hỏi đáp, ảnh gốc và xuất JSON.

## 6. Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Không tìm thấy trọng số adapter_model.*` | Chép `adapter_model.safetensors` hoặc `adapter_model.bin` vào thư mục `lora_adapters`, hoặc tạm đặt `VLM_ALLOW_BASE_MODEL=true`. |
| Lỗi tải base model | Kiểm tra mạng/Hugging Face cache; có thể đặt `VLM_BASE_MODEL` thành đường dẫn local của base model. |
| Hết RAM/VRAM | Dùng GPU; đóng các chương trình nặng; hoặc dùng base/adapter nhẹ hơn. CPU chỉ phù hợp demo ít request. |
| Frontend không gọi được API local | Kiểm tra backend ở cổng 8000 và `VITE_API_URL=http://localhost:8000/api/v1`. |

## 7. Đánh giá model (tuỳ chọn)

Sau khi có checkpoint và ảnh của bộ `vietnamese-receipts-v3` trong repo, chạy:

```powershell
.\backend\backend-docvqa\backend\.venv\Scripts\python.exe -m model.run_real_evaluation
```

Lệnh này mới tạo prediction từ Qwen2-VL trước khi tính ANLS và Exact Match. Không dùng báo cáo được sinh trực tiếp từ file nhãn, vì nhãn không phải là kết quả suy luận.
