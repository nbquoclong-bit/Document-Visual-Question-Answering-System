# Chạy sản phẩm Document VQA

Hướng dẫn này dành cho bản tích hợp hoàn chỉnh trong repo:

```text
React → FastAPI → Stage 0 OpenCV → Qwen2-VL-2B → LoRA QA
                                  └→ Base model JSON extraction
```

Backend chỉ nạp model ở request xử lý đầu tiên và giữ một instance trong mỗi worker. Không cần chạy PaddleOCR, KIE hay Gradio riêng.

## 1. Yêu cầu

- Python 3.11 hoặc 3.12.
- Node.js 20+.
- Tối thiểu khoảng 16 GB RAM nếu chạy CPU.
- Khuyến nghị NVIDIA GPU từ 6 GB VRAM khi demo nhiều tài liệu.
- Khoảng 6 GB ổ đĩa trống cho môi trường và base model.

LoRA adapter đã nằm đúng chỗ trong repo:

```text
model/stage1_vlm/output/lora_adapters/
├── adapter_config.json
├── adapter_model.safetensors   # 73.9 MB
└── tokenizer / processor files
```

Base model `Qwen/Qwen2-VL-2B-Instruct` không được commit vào Git. Lần chạy đầu cần Internet để tải khoảng 4.13 GB vào Hugging Face cache.

## 2. Chạy local trên Windows

Mở PowerShell tại thư mục gốc repo.

### Backend

```powershell
cd backend\backend-docvqa\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Mở API docs tại <http://localhost:8000/docs> và health check tại <http://localhost:8000/health>.

Không dùng nhiều worker: mỗi worker sẽ nạp một bản Qwen2-VL riêng và tiêu tốn thêm RAM/VRAM. Trên CPU, lần xử lý đầu có thể mất 2–3 phút gồm thời gian nạp model; các request sau nhanh hơn vì model được cache trong tiến trình.

### Frontend

Mở PowerShell thứ hai tại thư mục gốc repo:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Mở <http://localhost:5173>. Khi chạy Vite trực tiếp, `frontend/.env` dùng:

```dotenv
VITE_API_URL=http://localhost:8000/api/v1
```

## 3. Cấu hình model

Các biến nằm trong `backend/backend-docvqa/backend/.env`:

```dotenv
VLM_BASE_MODEL=Qwen/Qwen2-VL-2B-Instruct
VLM_ALLOW_BASE_MODEL=false
VLM_EXTRACTION_MODE=base
```

`VLM_EXTRACTION_MODE` có ba giá trị:

| Giá trị | Cách chạy | Khi nên dùng |
|---|---|---|
| `base` | Base model sinh JSON một lượt; LoRA vẫn dùng cho hỏi đáp | Mặc định, cân bằng tốc độ và kết quả |
| `single` | LoRA sinh JSON một lượt | Thử nghiệm adapter |
| `multi` | LoRA trả lời tuần tự 5 câu hỏi field | Chỉ nên thử trên GPU; rất chậm trên CPU |

Để chạy hoàn toàn offline, đặt `VLM_BASE_MODEL` thành đường dẫn tuyệt đối tới snapshot đã tải, ví dụ:

```dotenv
VLM_BASE_MODEL=C:\Users\user\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\<revision>
```

Không đặt `VLM_ALLOW_BASE_MODEL=true` trong bản demo chính thức: nếu LoRA bị thiếu hoặc hỏng, backend nên báo lỗi rõ ràng thay vì âm thầm chạy model khác.

## 4. Chạy bằng Docker Compose

Đảm bảo Docker Desktop đang hoạt động rồi chạy tại root repo:

```powershell
docker compose up --build
```

- Dashboard: <http://localhost>
- API docs: <http://localhost:8000/docs>
- Upload, kết quả, SQLite và Hugging Face cache được giữ trong named volumes.
- Model/adapter trong repo được mount read-only vào `/app/model`.
- Nginx đã được cấu hình timeout 10 phút cho inference CPU.

Compose hiện chạy CPU để tương thích rộng. Muốn dùng NVIDIA GPU, cần Docker Desktop/WSL2 và NVIDIA Container Toolkit; sau đó thêm `gpus: all` vào service `backend` trước khi build lại.

Tắt dịch vụ nhưng giữ dữ liệu:

```powershell
docker compose down
```

## 5. Sử dụng sản phẩm

1. Upload ảnh JPG/JPEG/PNG hoặc PDF tối đa 10 MB.
2. Bấm **Xử lý tài liệu**. Stage 0 sửa ảnh rồi model trích xuất tên bên bán, số hóa đơn, mã số thuế, ngày và tổng tiền.
3. Sau khi trạng thái là `processed`, đặt câu hỏi tự do trong khung **Hỏi đáp**.
4. Bấm **Xuất kết quả JSON** để tải field và lịch sử hỏi đáp.

PDF nhiều trang hiện chỉ xử lý trang đầu. Adapter không có grounding head nên API chưa trả bounding box bằng chứng.

## 6. Kiểm thử

### Backend và Stage 0

```powershell
cd backend\backend-docvqa\backend
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py -p no:cacheprovider

cd ..\..\..
.\backend\backend-docvqa\backend\.venv\Scripts\python.exe -m pytest -q model\stage0_preprocessing\tests -p no:cacheprovider
```

API tests mock inference để kiểm tra nhanh contract upload/process/ask/export. Muốn test checkpoint thật trên một ảnh:

```powershell
.\backend\backend-docvqa\backend\.venv\Scripts\python.exe -m model.query_manual `
  --image "C:\duong-dan\hoa-don.png" `
  --question "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?"
```

### Frontend và Compose

```powershell
cd frontend
npm run lint
npm run build

cd ..
docker compose config --quiet
```

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không tải được Hugging Face | Kết nối Internet ở lần đầu hoặc đặt `VLM_BASE_MODEL` tới snapshot local. |
| `Không thể nạp LoRA adapter` | Kiểm tra đủ `adapter_config.json` và `adapter_model.safetensors` trong thư mục adapter. |
| Request bị chậm trên CPU | Dùng `VLM_EXTRACTION_MODE=base`, giữ `--workers 1`, hoặc chuyển sang GPU. |
| Hết RAM/VRAM | Dừng worker/model khác; không dùng `--reload` hoặc nhiều worker; dùng GPU 4-bit trên Linux/WSL2. |
| Ảnh sau preprocessing bị cắt sai | Cập nhật code Stage 0 mới nhất; pipeline hiện chỉ perspective-crop contour tứ giác chiếm ít nhất 50% ảnh. |
| Frontend không gọi được backend local | Kiểm tra cổng 8000 và `VITE_API_URL=http://localhost:8000/api/v1`. |
| Docker tải lại base model | Không xoá volume `hf-cache`; `docker compose down` không xoá volume, còn `down -v` thì có. |
