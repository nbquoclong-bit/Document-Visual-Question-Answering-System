# 🚀 HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG DOCUMENT VQA

Tài liệu hướng dẫn khởi chạy toàn bộ các thành phần của hệ thống DocVQA: từ Full-Stack Local (FastAPI + React) đến môi trường Cloud GPU (Kaggle T4).

```text
Cấu trúc thực thi:
├── [1] Full-Stack Cục Bộ (Local): React Frontend (Vite) + FastAPI Backend
├── [2] Local Model Demo: Gradio Pure VLM Server
└── [3] Cloud GPU Kaggle: Huấn luyện LoRA + Benchmark 174 mẫu + Live Demo 10h
```

---

## 💻 1. Khởi Chạy Full-Stack Cục Bộ (Local Windows / Linux / macOS)

### A. Khởi động Backend (FastAPI)
Mở cửa sổ Terminal 1 tại thư mục gốc của dự án:

```bash
cd backend/backend-docvqa/backend
python -m venv venv
# Kích hoạt môi trường ảo:
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Khởi động server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Swagger Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

---

### B. Khởi động Frontend (React 18 + Vite)
Mở cửa sổ Terminal 2 tại thư mục gốc của dự án:

```bash
cd frontend
npm install
npm run dev
```
- **Giao diện Dashboard:** `http://localhost:5173`

---

## 🧠 2. Khởi Chạy Local Demo Gradio (Pure VLM Engine)

Nếu bạn muốn kiểm tra trực tiếp khả năng hỏi đáp và trích xuất JSON của mô hình trên máy cục bộ có GPU:

```bash
cd model
python demo_gradio.py
```
- Giao diện Gradio tương tác sẽ mở tại: `http://localhost:7860` (kèm Public Share Link).

---

## ☁️ 3. Tự Động Hóa Trên Kaggle GPU (Tesla T4)

Thư mục `kaggle_automation/` cung cấp 3 script độc lập giúp bạn quản lý toàn bộ vòng đời của mô hình AI:

### A. Huấn luyện LoRA Fine-Tuning Qwen2.5-VL-3B:
```bash
python kaggle_automation/train_qwen2_5_vl.py
```
- Tự động đẩy notebook và kích hoạt session GPU Tesla T4 để huấn luyện mô hình với 7 lớp Linear.

### B. Chạy Benchmark Đánh Giá 174 Mẫu Hóa Đơn (ANLS / F1 / EM):
```bash
python kaggle_automation/eval_benchmark.py
```
- Đánh giá toàn diện trên 174 mẫu hóa đơn thực tế và xuất báo cáo `evaluation_report.json` đạt **94.94% ANLS**.

### C. Khởi chạy Live Demo Server (Freeze Time 10 Giờ):
```bash
python kaggle_automation/run_live_demo.py
```
- Mở server Gradio trực tuyến với **1024 Max Tokens cho Full JSON**, hỗ trợ test trực tiếp trên Web.

---

## ⚙️ 4. Cấu Hình Biến Môi Trường (Environment Variables)

File `.env` tại thư mục backend cấu hình các tham số:
```env
APP_ENV=development
API_V1_STR=/api/v1
PROJECT_NAME="Document VQA System"
DATABASE_URL=sqlite:///./docvqa.db
UPLOAD_DIR=./uploads
RESULTS_DIR=./results
```
