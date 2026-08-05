# Hướng dẫn chạy SmartDoc AI

Hệ thống trích xuất thông tin hóa đơn & hỏi đáp kế toán — **Pipeline 4 giai đoạn**.

---

## 1. Lưu đồ Pipeline

```
[Ảnh/PDF Hóa đơn]
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 0 – Adaptive Preprocessing                                    │
│ • Đánh giá chất lượng (mờ, xiên, tương phản)                       │
│ • Deskew + CLAHE + Sharpen (nếu cần)                                │
│ • PDF → ảnh trang (nếu cần)                                         │
│ • Output: danh sách ảnh đã xử lý + metadata                        │
└───────────────────────┬──────────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1 – PaddleOCR (fine-tuned)                                    │
│ • Nhận ảnh đã preprocess                                            │
│ • Text detection + recognition                                      │
│ • Output: [{text, confidence, bbox}, ...]                           │
└───────────────────────┬──────────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2 – LayoutLMv3 KIE                                            │
│ • Nhận OCR words + boxes                                            │
│ • Token classification (IOB/BIO tagging)                            │
│ • Output: {invoice_number, tax_code, date, total_amount}            │
└───────────────────────┬──────────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3 – Qwen2.5 QA (QLoRA)                                        │
│ • CHỈ đọc JSON từ Stage 2 — KHÔNG nhìn ảnh gốc                    │
│ • Kiểm tra tính toán (so sánh cột, VAT, tổng)                       │
│ • Output: is_math_valid + audit_note                                │
└───────────────────────┬──────────────────────────────────────────────┘
                        ▼
                  [KẾT QUẢ CUỐI CÙNG]
```

---

## 2. Yêu cầu hệ thống

### Phần cứng
- **GPU NVIDIA**: tối thiểu 4GB VRAM (Stage 3 QLoRA inference), 8GB+ để train Stage 2/3
- **RAM**: 8GB+

### Phần mềm
- Python >= 3.10
- Node.js >= 18 (chỉ khi chạy frontend)
- CUDA >= 12.1 (cho Unsloth)
- Git

> 💡 Nếu không có GPU, Stage 0 chạy được trên CPU. Stage 1 inference và Stage 2 inference cũng có thể chạy CPU nhưng chậm hơn nhiều.

---

## 3. Cài đặt môi trường

Clone repo:
```bash
git clone https://github.com/nbquoclong-bit/Document-Visual-Question-Answering-System.git
cd Document-Visual-Question-Answering-System
```

### Stage 0 — Adaptive Preprocessing

```bash
cd stage0_preprocessing
pip install -r requirements.txt
```
✅ Không cần GPU. Chỉ dùng **OpenCV, NumPy, PyMuPDF**.

### Stage 1 — PaddleOCR Fine-tuning

```bash
cd stage1_ocr
pip install -r requirements.txt
```
⚠️ Nếu dùng GPU, kiểm tra CUDA trước:
```bash
python -c "import paddle; print(paddle.version.cuda())"
```

### Stage 2 — LayoutLMv3 KIE

```bash
cd stage2_kie
pip install -r requirements.txt
```
🔧 Cần GPU để train (>= 8GB VRAM). Inference có thể CPU nhưng chậm.

### Stage 3 — Qwen QA (Qwen2.5-1.5B + QLoRA)

```bash
cd stage3_qa
pip install -r requirements.txt
```
⚠️ **Bắt buộc GPU NVIDIA + CUDA 12.x.** Unsloth chỉ hỗ trợ CUDA.

---

## 4. Chạy từng giai đoạn

---

### Stage 0 — Adaptive Preprocessing

**Chạy unit test:**
```bash
cd stage0_preprocessing
python -m tests.test_quality
python -m tests.test_preprocessing
python -m tests.test_pdf_router
```

**Sử dụng trong code:**
```python
import sys
sys.path.insert(0, ".")
from src.preprocessor import PreprocessingEngine

engine = PreprocessingEngine(config_path="src/config.yaml")
kind, result = engine.process("path/to/invoice.pdf")

# kind = "pdf" hoặc "image"
# result = list of (page_idx, processed_image, meta)
# meta = {"deskew": bool, "clahe": bool, "sharpen": bool, "perspective_crop": bool}
```

**Kiểm tra chất lượng ảnh:**
```python
from src.quality_check import QualityAssessor
import cv2

img = cv2.imread("invoice.jpg")
assessor = QualityAssessor(blur_threshold=100, skew_threshold=2.0)
report = assessor.assess(img)

print(f"Blurry: {report.is_blurry}, score: {report.blur_score:.1f}")
print(f"Skew: {report.skew_angle:.2f}°, needs_deskew: {report.needs_deskew}")
```

**Render PDF thành ảnh:**
```python
from src.pdf_router import render_pdf_pages
pages = render_pdf_pages("invoice.pdf", dpi=300, max_pages=10)
# pages = [(page_num, image_array), ...]
```

---

### Stage 1 — PaddleOCR Fine-tuning

**Chuẩn bị dữ liệu (MC_OCR 2021):**
```bash
cd stage1_ocr
python src/data_prep.py
```
Script sẽ in hướng dẫn download dataset. Sau khi download xong chạy lại.

**Fine-tune OCR:**
```bash
cd stage1_ocr
python src/train_ocr.py
```
- Config: `configs/finetune_config.yaml`
- Output: `stage1_ocr/output/best_model/`
- Early stopping: patience=10 epochs

**Inference:**
```python
from src.ocr_engine import ocr_image, ocr_file
import cv2

# Ảnh đã qua Stage 0
image = cv2.imread("invoice.jpg")
results = ocr_image(image, lang="vi", use_gpu=False)
# results = [{"text": str, "confidence": float, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]

# Tự động qua Stage 0 (PDF/ảnh đều được)
results = ocr_file("invoice.pdf", lang="vi", use_gpu=False)
# results = [{"page": int, "preprocessing": dict, "texts": list}, ...]
```

---

### Stage 2 — LayoutLMv3 KIE (Key Information Extraction)

**Chuẩn bị dữ liệu:**

Format mỗi record trong `train_records.json` / `val_records.json`:

```json
{
  "image_path": "./data/sroie_images/img001.jpg",
  "words": ["CÔNG", "TY", "ABC", ...],
  "bboxes": [[x1, y1, x2, y2], ...],
  "labels": ["O", "O", "B-INVOICE_NUMBER", ...]
}
```

- Số `labels` = số `words` (IOB tagging — mỗi word 1 label)
- Bounding box đã normalized về **0–1000** (LayoutLMv3 format)

**Danh sách entity labels:**

| Label | Ý nghĩa |
|-------|---------|
| `O` | Outside (không phải entity) |
| `B-INVOICE_NUMBER` | Bắt đầu số hóa đơn |
| `I-INVOICE_NUMBER` | Tiếp theo của số hóa đơn |
| `B-TAX_CODE` | Bắt đầu mã số thuế |
| `I-TAX_CODE` | Tiếp theo mã số thuế |
| `B-DATE` | Bắt đầu ngày |
| `I-DATE` | Tiếp theo ngày |
| `B-TOTAL_AMOUNT` | Bắt đầu tổng tiền |
| `I-TOTAL_AMOUNT` | Tiếp theo tổng tiền |

**Train model:**
```bash
cd stage2_kie
python src/trainer.py
```
- Config: `configs/train_config.yaml`
- Output: `stage2_kie/output/best_model/`
- Metrics: Micro-F1, Precision, Recall (seqeval)
- Early stopping: patience=5 theo F1

**Evaluate model:**
```bash
cd stage2_kie
python src/evaluate.py --model_dir ./output/best_model --val_records ./data/test_records.json
```
→ Xuất `eval_report.json`

**Inference:**
```python
from src.kie_engine import predict, KIEConfig

cfg = KIEConfig(model_dir="stage2_kie/output/best_model", max_length=512)

entities = predict(
    cfg,
    image_path="invoice.jpg",
    ocr_words=["HD", "0082391"],
    ocr_boxes=[[100, 200, 150, 220]]
)
# entities = {
#   "invoice_number": {"value": "HD0082391", "confidence": 0.985, "bbox": [...]},
#   "tax_code":        {"value": "0312345678", "confidence": 0.991, ...},
#   ...
# }
```

---

### Stage 3 — Qwen QA (QLoRA)

**Chuẩn bị dữ liệu:**

Tạo file JSONL tại `data/synthetic_qa_train.jsonl` (mỗi dòng 1 JSON):

```json
{"instruction": "Kiem tra tinh toan tren hoa don nay co dung khong?", "input": {"invoice_number": {"value": "HD0082391"}, ...}, "output": "Tien hang (1,363,636 VND) + Thue VAT 10% (136,364 VND) khớp với Tong tien (1,500,000 VND)."}
```

**Train:**
```bash
cd stage3_qa
python src/trainer.py
```
- Config: `configs/train_config.yaml`
- Output: `stage3_qa/output/lora_adapters/` (chỉ LoRA weights)
- Unsloth FastLanguageModel + SFTTrainer
- max_steps=500, batch_size=4, lr=2e-4

**Inference:**
```python
from src.inference import QAEngine

engine = QAEngine(
    adapter_dir="stage3_qa/output/lora_adapters",
    base_model="Qwen/Qwen2.5-1.5B-Instruct"
)

stage2_output = {
    "invoice_number": {"value": "HD0082391", "confidence": 0.985, "bbox": [450,120,580,145]},
    "tax_code":       {"value": "0312345678", "confidence": 0.991},
    "total_amount":   {"value": "1,500,000", "confidence": 0.978},
    "date":           {"value": "01/03/2024", "confidence": 0.960}
}

answer = engine.answer(stage2_output, "Kiem tra tinh toan tren hoa don nay co dung khong?")
print(answer)
```

> ⚠️ **Stage 3 CHỈ đọc JSON** output từ Stage 2 — không nhận ảnh gốc. Đây là **zero hallucination guarantee**.

---

## 5. Pipeline end-to-end

### Chạy Stage 0 → 1 → 2

File `stage3_qa/src/pipeline_stage012.py` gộp 3 giai đoạn đầu:

```python
from stage3_qa.src.pipeline_stage012 import run_pipeline

result = run_pipeline("invoice.jpg", kie_model_dir="stage2_kie/output/best_model")
print(result)
# {"status": "success", "ocr_words": 45, "entities": {...}}
```

**Flow:**
1. Stage 0: `PreprocessingEngine.process()` → ảnh đã xử lý
2. Stage 1: `ocr_image()` → PaddleOCR trả về text + bbox
3. Stage 2: `predict()` → LayoutLMv3 trả về entities

### Chạy full 4-stage:

```python
from stage3_qa.src.pipeline_stage012 import run_pipeline
from stage3_qa.src.inference import QAEngine
import json

kie_result = run_pipeline("invoice.jpg", "stage2_kie/output/best_model")

if kie_result["status"] == "success":
    engine = QAEngine(adapter_dir="stage3_qa/output/lora_adapters")
    answer = engine.answer(kie_result["entities"])

    output = {
        "status": "success",
        "processing_metadata": {"preprocessing_applied": [...]},
        "extracted_data": kie_result["entities"],
        "accounting_audit": {"is_math_valid": True, "audit_note": answer}
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
```

---

## 6. Chạy tests

```
[STAGE 0] Adaptive Preprocessing
├── test_quality.py         → chất lượng ảnh
├── test_preprocessing.py   → engine chính
└── test_pdf_router.py      → render PDF

[STAGE 1] PaddleOCR
└── test_ocr_engine.py      → OCR inference

[STAGE 2] LayoutLMv3 KIE
├── test_model.py           → model architecture
├── test_kie_config.py      → config validation
└── test_predict.py         → end-to-end predict

[STAGE 3] QA LLM
├── test_qa_engine.py       → QA inference
└── test_pipeline_stage012.py → pipeline 0→1→2
```

### Chạy từng loại:

```bash
# Stage 0
cd stage0_preprocessing
python -m pytest tests/test_quality -v
python -m pytest tests/test_preprocessing -v
python -m pytest tests/test_pdf_router -v

# Stage 1
cd stage1_ocr
python -m pytest tests/test_ocr_engine -v

# Stage 2
cd stage2_kie
python -m pytest tests/test_model -v
python -m pytest tests/test_kie_config -v
python -m pytest tests/test_predict -v

# Stage 3
cd stage3_qa
python -m pytest tests/test_qa_engine -v
python -m pytest tests/test_pipeline_stage012 -v
```

### Chạy tất cả tests:

```bash
# Từ root project
find . -name "test_*.py" -path "*/tests/*" | while read f; do
    echo "=== $f ==="
    python -m pytest "$f" -v 2>&1 | tail -5
done
```

---

## 7. Ghi chú quan trọng

- **Zero hallucination**: Stage 3 không bao giờ nhìn thấy ảnh gốc. Nó chỉ xử lý text đã được OCR + KIE trích xuất.
- **Cross-stage dependency**: Stage 2 cần output từ Stage 1, Stage 3 cần output từ Stage 2.
- **Vietnamese language**: Tất cả prompt, template, và dữ liệu training cho Stage 3 bằng tiếng Việt không dấu (tone mark removed).
- **IOB tagging**: Stage 2 sử dụng Inside-Outside-Beginning tagging scheme. Mỗi word trong OCR output được gán 1 label.
