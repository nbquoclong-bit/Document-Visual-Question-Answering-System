# SmartDoc AI - Document Visual Question Answering & Information Extraction System

> **Hệ thống Hỏi đáp Trực quan trên Tài liệu ứng dụng Pipeline Fine-Tuning 3 Giai đoạn và Tiền xử lý Thích ứng (Adaptive Preprocessing) trong Tự động hóa Xử lý Hóa đơn / Chứng từ.**
> 
> 📌 **Định hướng Kỹ thuật:** Frugal AI (Chi phí 0 VNĐ) | Triple Fine-Tuning Pipeline | Adaptive Auto-Routing | Zero-Hallucination | Privacy-First.

---

## 📌 1. Tổng quan Dự án (Project Overview)

Trong quá trình chuyển đổi số doanh nghiệp, việc tự động hóa nhập liệu và trích xuất thông tin từ hóa đơn, chứng từ tài chính là nhu cầu vô cùng cấp thiết. Việc sử dụng trực tiếp các mô hình Multimodal LLM thương mại khổng lồ (như ChatGPT, Gemini Vision) gặp rào cản lớn về chi phí API đắt đỏ, nguy cơ rò rỉ dữ liệu tài chính nhạy cảm và hiện tượng "ảo giác" (tự bịa con số).

**SmartDoc AI** giải quyết triệt để bài toán này với kiến trúc tinh gọn:

* **Đầu vào (Input):** File PDF digital hoặc ảnh chụp hóa đơn/chứng từ tiếng Việt.


* **Đầu ra (Output):** Dữ liệu hóa đơn trích xuất chuẩn cấu trúc JSON, tọa độ Bounding Box khoanh vùng bằng chứng trực tiếp trên ảnh gốc, và Chatbot AI giải đáp các thắc mắc tính toán logic (VD: *"Thuế tính đúng không?"*, *"Tổng cộng bao nhiêu sản phẩm?"*).


* **Hạ tầng 0 VNĐ:** Tối ưu hóa để huấn luyện hoàn toàn miễn phí trên Kaggle / Google Colab GPU T4 và triển khai nhẹ nhàng trên CPU nội bộ hoặc Hugging Face Spaces.



---

## 🧠 2. Kiến trúc Hệ thống: Pipeline Fine-Tuning 3 Giai đoạn & Tiền xử lý Thích ứng

Hệ thống tách biệt hoàn toàn tác vụ **"Trích xuất dữ liệu"** (Vision + Layout) và **"Lập luận logic"** (Text LLM) nhằm triệt tiêu hoàn toàn rủi ro bịa số liệu, đồng thời ứng dụng cơ chế phân nhánh tiền xử lý tự động để tối ưu hiệu năng.

### 2.1. Sơ đồ Luồng Dữ liệu (End-to-End Pipeline)

```text
[File Đầu vào (PDF / JPG / PNG)]
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 0: Adaptive Preprocessing Engine (OpenCV)          │
│ - Quality Assessment & Smart Auto-Routing                │
└──────────────────────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
[Luồng A: File Chuẩn] [Luồng B: Ảnh Kém Chất Lượng]
(Digital PDF / Scan)  (Nghiêng / Mờ / Dính góc bàn)
      │                 │
      │                 ├─ Perspective Crop (Cắt phông nền)
      │                 ├─ Deskew (Quay thẳng góc)
      │                 └─ CLAHE (Tăng tương phản)
      │                 │
      └────────┬────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Fine-Tuned PaddleOCR (Text & Box Detection)     │[cite: 3]
│ - Nhận dạng ký tự tiếng Việt trên hóa đơn mờ/in kim.     │
└──────────────────────────────────────────────────────────┘
       │ (Ký tự tiếng Việt + Tọa độ Bounding Box 2D)
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Fine-Tuned LayoutLMv3 (KIE & Visual Grounding)  │[cite: 3]
│ - Đọc hiểu cấu trúc Layout 2D (Spatial Layout).          │
│ - Trích xuất thực thể (Mã số thuế, Tổng tiền...).        │
│ - Trả về JSON sạch + Bounding Box khoanh vùng bằng chứng.│
└──────────────────────────────────────────────────────────┘
       │ (Dữ liệu JSON chuẩn - Xác minh 100%, Không ảo giác)
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: Fine-Tuned Qwen2.5-1.5B (Logical Chat)          │
│ - Đóng vai "Chuyên gia Kế toán".                         │
│ - Nhận JSON từ Stage 2 + Prompt câu hỏi người dùng.      │
│ - Thực hiện kiểm toán, tính toán logic và trả lời Chat. │
└──────────────────────────────────────────────────────────┘

```

---

### 2.2. Chi tiết Các Giai đoạn Xử lý

* **STAGE 0 — Adaptive Preprocessing Engine (OpenCV):**
* **Smart Auto-Routing:** Phân loại định dạng đầu vào. Nếu là File Digital PDF, tự động render ra ảnh độ phân giải cao ($300\text{ DPI}$) và đưa thẳng vào Stage 1.
* **Conditional Processing (Chỉ chạy khi cần thiết):**
* *Đo độ nhòe (Laplacian Variance):* Chỉ kích hoạt lọc nhiễu CLAHE/Sharpening nếu độ nhòe vượt ngưỡng.
* *Đo góc nghiêng (Hough Line Transform):* Chỉ xoay ảnh (Deskew) nếu góc lệch $> 2^\circ$.
* *Đo phông nền (Contour Detection):* Kiểm tra diện tích đường viền lớn nhất. Nếu khung hình chữ nhật chiếm $> 90\%$ diện tích (ảnh scan/chụp đè), tự động **bỏ qua Perspective Crop** để tránh cắt lẹm vào nội dung hóa đơn.




* **STAGE 1 — Fine-Tuned PaddleOCR (Text Detection & Recognition):**

* Fine-tune trên tập dữ liệu hóa đơn Việt Nam để tối ưu hóa nhận dạng chữ viết tay nhẹ, chữ in kim bị đứt nét, hóa đơn mờ nhòe.




* **STAGE 2 — Fine-Tuned LayoutLMv3 (Key Information Extraction):**

* Mô hình Multimodal kết hợp 3 luồng: Text + Visual Patch + 2D Bounding Box.


* Thực hiện gán nhãn thực thể (Token Classification) chuẩn IOB/BIO để bóc tách chính xác các trường dữ liệu (*Mã số thuế*, *Ký hiệu mẫu*, *Tổng tiền thanh toán*...) và trả về tọa độ $2D$ để vẽ Bounding Box Highlight.




* **STAGE 3 — Fine-Tuned Qwen2.5-1.5B (Logical Accounting Audit & QA):**
* Sử dụng kỹ thuật **QLoRA via Unsloth** để fine-tune mô hình ngôn ngữ siêu nhẹ `Qwen2.5-1.5B-Instruct` trên Kaggle/Colab T4.
* Mô hình không nhìn ảnh trực tiếp mà chỉ đọc dữ liệu JSON sạch từ Stage 2, đảm bảo **100% không bị ảo giác số liệu** khi giải đáp thắc mắc logic của người dùng.



---

## 📊 3. Dữ liệu Huấn luyện & Tiền xử lý (Datasets)

* **Stage 1 (OCR):** MC_OCR 2021 (~1.1k ảnh hóa đơn tiếng Việt) + Tập dữ liệu hóa đơn nội bộ.


* **Stage 2 (LayoutLMv3):** ViOCRVQA (~28k ảnh, 120k+ QA), DocVQA (~12k+ ảnh), SROIE & CORD (2,000+ ảnh hóa đơn chuẩn hóa IOB Tagging).


* **Stage 3 (LLM):** Tập dữ liệu Synthetic Accounting QA chuyên biệt cho nghiệp vụ kiểm toán hóa đơn Việt Nam.

---

## 🛠️ 4. Tech Stack & Công cụ

* **Core AI Models:** OpenCV, PaddleOCR, `microsoft/layoutlmv3-base`, `Qwen/Qwen2.5-1.5B-Instruct` (Unsloth QLoRA).


* **Backend Framework:** FastAPI, Uvicorn, LangChain.


* **Frontend UI:** React.js, Zustand State Management, HTML5 Canvas (Vẽ Highlight Bounding Box).


* **Database & Storage:** SQLite.


* **MLOps & Deployment:** Docker, Hugging Face Spaces / Private Cloud, Git Workflow.


* **Metrics Đánh giá:** Micro-F1 Score, Precision, Recall (KIE), Exact Match (EM), Inference Latency.



---

## 📑 5. Cấu trúc Dữ liệu Đầu ra (API Output Schema Example)

```json
{
  "status": "success",
  "processing_metadata": {
    "preprocessing_applied": ["Deskew", "CLAHE"],
    "latency_seconds": 1.15
  },
  "extracted_data": {
    "invoice_number": {
      "value": "HD0082391",
      "confidence": 0.985,
      "box": [450, 120, 580, 145]
    },
    "tax_code": {
      "value": "0312345678",
      "confidence": 0.991,
      "box": [120, 210, 280, 230]
    },
    "total_amount": {
      "value": "1,500,000",
      "confidence": 0.978,
      "box": [610, 820, 750, 845]
    }
  },
  "accounting_audit": {
    "is_math_valid": true,
    "audit_note": "Tiền hàng (1,363,636 VNĐ) + Thuế VAT 10% (136,364 VNĐ) khớp hoàn toàn với Tổng tiền (1,500,000 VNĐ)."
  }
}

```

---

## 🚀 6. Hướng dẫn Cài đặt & Chạy Local (Local Setup)

1. **Clone Repository:**

```bash
git clone https://github.com/nbquoclong-bit/Document-Visual-Question-Answering-System.git
cd Document-Visual-Question-Answering-System

```

2. **Cài đặt Backend (FastAPI):**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

```

3. **Cài đặt Frontend (React):**

```bash
cd ../frontend
npm install
npm run dev

```

---

## 👥 7. Phân công Nhiệm vụ Nhóm 5 (Team Boboiboys)

| Thành viên | Vai trò | Nhiệm vụ chính |
| --- | --- | --- |
| **Lê Minh Sang** | **Model Lead (Stage 2)**<br> | - Nghiên cứu pipeline Multimodal LayoutLMv3.

<br>

<br>- Xây dựng pipeline Fine-Tuning LayoutLMv3 bóc tách thực thể (KIE) & Bounding Box.

<br>

<br>- Tối ưu hóa mô hình (FP16, Quantization) & Đánh giá Metrics (F1, Precision, Recall).

 |
| **Nguyễn Văn Nhật Nam** | **Data Engineer & OCR Lead (Stage 0 & 1)**<br> | - Thu thập, gán nhãn IOB/BIO và chuẩn hóa Bounding Box 2D.

<br>

<br>- Lập trình **Adaptive Preprocessing Engine** (OpenCV).

<br>

<br>- Fine-tune PaddleOCR trên tập hóa đơn mờ/nhiễu tiếng Việt.

 |
| **Nguyễn Bá Quốc Long** | **Team Leader & LLM Lead (Stage 3 & MLOps)**<br> | - Quản lý tiến độ dự án, Git workflow & Báo cáo.

<br>

<br>- Fine-tune Qwen2.5-1.5B với Unsloth QLoRA cho tác vụ kiểm toán logic hóa đơn.<br>

<br>- Đóng gói Docker container và triển khai hệ thống.

 |
| **Trần Hoàng Minh Thiên** | **Backend Engineer**<br> | - Xây dựng RESTful API bằng FastAPI.

<br>

<br>- Kết nối 3 mô hình AI thành Pipeline hoàn chỉnh & Quản lý SQLite database.

<br>

<br>- Lập trình module Post-processing validation logic.

 |
| **Trịnh Minh Đức Hoàng** | **Frontend Engineer**<br> | - Thiết kế giao diện Web UI bằng React.

<br>

<br>- Xây dựng chức năng Upload, Canvas Highlight Bounding Box và Khung Chat VQA realtime.

<br>

<br>- Thực hiện Video Demo và tối ưu trải nghiệm người dùng (UX).

 |