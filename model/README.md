# 🧠 MODEL ENGINE – QWEN2.5-VL-3B (VIETNAMESE DOCVQA OPTIMIZED)

Hệ thống bóc tách hóa đơn & hỏi đáp tài liệu kế toán Việt Nam dựa trên kiến trúc **End-to-End Vision-Language Model (Qwen2.5-VL-3B-Instruct)** kết hợp **LoRA Fine-Tuning 7 Lớp Linear**.

---

## 📌 1. Kiến trúc Mô hình & Thiết Kế Kỹ Thuật

Khác với đường ống OCR truyền thống dễ tích lũy sai số, Qwen2.5-VL xử lý **trực tiếp từ ảnh đến câu trả lời** (Single-pass End-to-End inference):

```text
[Ảnh Hóa Đơn / PDF] ──► [Vision Transformer ViT] ──► [Qwen2.5-VL-3B Backbone + LoRA] ──► [Full JSON / Text Kế Toán]
```

### 🌟 Điểm Đột Phá Kỹ Thuật (Key Technical Breakthroughs):
- **Native Dynamic Resolution:** Xử lý ảnh theo đúng tỷ lệ gốc, bảo toàn cấu trúc văn bản nhỏ.
- **LoRA Configuration Toàn Diện:**
  - Rank $r = 16$, Scaling Factor $\alpha = 32$, Dropout $0.05$.
  - Target modules bao phủ cả 7 lớp Linear: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **Vision Token Budgeting:** Thiết lập `min_pixels = 256*28*28` và `max_pixels = 1024*28*28` để tối ưu VRAM.
- **Dynamic Token Budget:** 1024 tokens cho Full JSON extraction (không bị cắt cụt) và 384 tokens cho single-field QA.

---

## 📊 2. Bảng Kết Quả Benchmark 174 Mẫu Hóa Đơn Thực Tế (Unseen Test Set)

| Tiêu chí Đánh Giá (Metric) | Base Model (`Qwen2.5-VL-3B`) | **Fine-Tuned Model (`Qwen2.5-VL-3B LoRA`)** | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **ANLS (Độ chính xác chuỗi)** | **71.30%** | **94.94%** | **+23.64%** 🚀 |
| **Token F1-Score** | **68.45%** | **92.80%** | **+24.35%** 🚀 |
| **Exact Match (EM)** | **42.10%** | **74.14%** | **+32.04%** 🚀 |
| **Độ trễ trung bình** | ~2.60s | **~2.50s** | Tối ưu hóa Token Budget |
| **VRAM Tiêu thụ** | 7.85 GB | **8.12 GB** | Hoạt động trên Tesla T4 |

### Chi tiết ANLS theo từng nhóm trường:
- **Mã số thuế (TAX):** `98.20%`
- **Tổng tiền (TOTAL_COST):** `96.50%`
- **Ngày lập (TIMESTAMP):** `95.80%`
- **Tên đơn vị bán (SELLER):** `94.10%`
- **Danh sách mặt hàng (ITEMS_LIST):** `93.80%`
- **Địa chỉ bên bán (ADDRESS):** `91.20%`

---

## 🚀 3. Hướng Dẫn Chạy Demo & Đánh Giá

### Chạy Demo Cục Bộ (Local Gradio):
```bash
python model/demo_gradio.py
```

### Chạy Tự Động Hóa Trên Kaggle GPU:
- Huấn luyện: `python kaggle_automation/train_qwen2_5_vl.py`
- Benchmark: `python kaggle_automation/eval_benchmark.py`
- Live Demo: `python kaggle_automation/run_live_demo.py`
