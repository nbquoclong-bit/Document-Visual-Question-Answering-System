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

## 📊 2. Bảng Kết Quả Benchmark 174 Mẫu Hóa Đơn Thực Tế (Kaggle GPU Tesla T4)

| Tiêu chí Đánh Giá (Metric) | Base Zero-Shot (`02_...zeroshot.json`) | **Fine-Tuned Model (`04_...optimized_kaggle_gpu.json`)** | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **ANLS (Độ chính xác chuỗi)** | **0.68%** | **89.61%** | **+88.93%** 🚀 |
| **Token F1-Score** | **35.25%** | **89.82%** | **+54.57%** 🚀 |
| **Exact Match (EM)** | **0.00%** | **66.09%** | **+66.09%** 🚀 |
| **Độ trễ trung bình** | ~3.76s | **~3.08s** | Tối ưu hóa Greedy & FP16 |
| **VRAM Tiêu thụ** | ~4.82 GB | **~3.60 GB** | Native FP16 trên Tesla T4 |

### Chi tiết ANLS theo từng nhóm trường:
- **Tên đơn vị bán (SELLER):** ANLS `98.37%` | Exact Match `76.67%`
- **Đơn giá từng món (ITEM_PRICE):** ANLS `96.99%` | Exact Match `78.57%`
- **Tổng tiền (TOTAL_COST):** ANLS `96.77%` | Exact Match `73.33%`
- **Địa chỉ bên bán (ADDRESS):** ANLS `85.36%` | Exact Match `78.57%`
- **Ngày lập (TIMESTAMP):** ANLS `84.08%` | Exact Match `76.67%`
- **Danh mục hàng hóa (ITEMS_LIST):** ANLS `75.37%` | Exact Match `10.71%` *(Tác vụ bảng kê nhiều dòng)*

> 💡 *Chi tiết toàn bộ báo cáo và phân tích xin xem tại:* [model/output/README.md](output/README.md)

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
