# 🧠 MODEL ENGINE – QWEN2-VL-2B (VIETNAMESE DOCVQA)

Hệ thống trích xuất thông tin hóa đơn & hỏi đáp tài liệu kế toán Việt Nam dựa trên kiến trúc **End-to-End Vision-Language Model (Qwen2-VL-2B-Instruct)** kết hợp **QLoRA Fine-Tuning**.

---

## 📌 1. Kiến trúc Mô hình (Architecture Overview)

Thay vì dùng đường ống truyền thống (PaddleOCR + LayoutXLM) dễ bị tích lũy sai số qua từng công đoạn, Qwen2-VL xử lý **trực tiếp từ ảnh đến câu trả lời** (Single-pass End-to-End inference):

```text
[Ảnh Hóa Đơn / PDF] ──► [Stage 0: Preprocessing (OpenCV)] ──► [Stage 1: Qwen2-VL-2B + QLoRA] ──► [Kế Toán / JSON / Gradio]
```

### 🌟 Điểm Đột phá Kỹ thuật:
- **Naive Dynamic Resolution:** Giữ nguyên tỷ lệ khung hình ảnh hóa đơn, không nén méo chữ.
- **M-RoPE (Multimodal Rotary Position Embedding):** Tọa độ hóa vị trí 2D của ký tự trên hóa đơn (hàng, cột, khoảng cách).
- **QLoRA (Quantized Low-Rank Adaptation):** Base model 4-bit NF4, nạp trọng số LoRA ($r=16, \alpha=32$) tập trung vào `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

---

## 📊 2. So sánh Mô hình Gốc (Base) vs Mô hình Đã Fine-tune

| Đặc tính / Chỉ số | Mô hình Gốc (Base Zero-Shot) | Mô hình Đã Fine-tune (QLoRA) |
| :--- | :--- | :--- |
| **Định dạng Đầu ra** | Phản hồi tự do dạng Chatbot | Chuẩn kế toán (Markdown / JSON) |
| **Hiểu thuật ngữ VN** | Nhầm lẫn Đơn giá / Thành tiền | Bóc tách chuẩn MST, Ký hiệu mẫu, VAT |
| **Chỉ số ANLS (DocVQA)** | Chưa benchmark lại | Chưa benchmark lại bằng prediction thực |
| **Tỉ lệ Khớp Exact Match** | Chưa benchmark lại | Chưa benchmark lại bằng prediction thực |
| **Dung lượng VRAM tiêu thụ**| ~4.2 GB VRAM | **~4.2 GB VRAM** |

---

## 🚀 3. Hướng dẫn Chạy & Tích hợp (Getting Started & Integration)

### Cài đặt thư viện:
```bash
cd model
pip install -r stage1_vlm/requirements.txt
pip install gradio opencv-python pillow
```

### Đặt trọng số LoRA Adapters:
Adapter 73.9 MB đã được lưu tại:
`model/stage1_vlm/output/lora_adapters/`

### Tích hợp Backend (Python API):
```python
from model.stage1_vlm.src.inference import VQAEngine

# Khởi tạo Engine
engine = VQAEngine(adapter_dir="model/stage1_vlm/output/lora_adapters")

# Gọi hàm trích xuất
answer = engine.extract_and_answer(
    image_path="invoice.jpg", 
    question="Trích xuất thông tin hóa đơn: Mã số thuế, Số hóa đơn, Tổng tiền."
)
print(answer)
```

Xem hướng dẫn chạy toàn bộ React + FastAPI + model tại [`../RUNNING.md`](../RUNNING.md).

### Khởi chạy CLI / Web UI Demo:
```bash
# Kiểm thử CLI
python test_vlm.py

# Khởi chạy Gradio Web UI
python demo_gradio.py

# Chạy suy luận thật rồi mới đo ANLS/Exact Match
cd ..
python -m model.run_real_evaluation
```

---

## 📂 4. Cấu trúc Thư mục `model/`

```text
model/
├── stage0_preprocessing/   # Preprocessing OpenCV (Deskew, CLAHE, Sharpening)
├── stage1_vlm/             # Pipeline Qwen2-VL-2B (QLoRA)
│   ├── configs/            # train_config.yaml
│   └── src/                # trainer.py, prepare_vlm_data.py, dataset.py, model.py, inference.py
├── stage1_vlm/output/      # Thư mục lưu trọng số lora_adapters
├── test_vlm.py             # Script kiểm thử CLI
├── demo_gradio.py          # Web UI Demo bằng Gradio
├── evaluate_metrics.py     # Script đo chỉ số ANLS & Exact Match
└── METRICS_DANH_GIA.md     # Tài liệu lý thuyết bộ chỉ số đánh giá
```
