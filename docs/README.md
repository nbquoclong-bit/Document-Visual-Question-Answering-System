# 📚 HỆ THỐNG TÀI LIỆU TOÀN DIỆN DỰ ÁN DOCUMENT VQA
## Qwen2.5-VL-3B LoRA Fine-Tuning Trên Hóa Đơn Tiếng Việt

> **Thư mục này tổng hợp toàn bộ tài liệu hướng dẫn, kế hoạch kỹ thuật, số liệu thực nghiệm và đề cương làm slide dành cho các thành viên trong nhóm.**

---

## 🗺️ Bản Đồ Tra Cứu Tài Liệu (Documentation Map)

```
project/
├── docs/                                 <-- 🎯 THƯ MỤC TÀI LIỆU CHÍNH
│   ├── README.md                         (Bản đồ tra cứu tài liệu)
│   └── DE_CUONG_LAM_SLIDE_THUYET_TRINH.md (Đề cương 12 slides + Speaker Notes cho các bạn làm slide)
│
├── model/                                <-- 🧠 THƯ MỤC MÔ HÌNH & KỸ THUẬT
│   ├── KE_HOACH_FINETUNE_QWEN2_5_VL.md   (Bản kế hoạch chi tiết, toán học LoRA & VRAM budget)
│   ├── GIAI_THICH_TOAN_DIEN_FINETUNE.md  (Tài liệu giải thích sâu kỹ thuật & kịch bản 2.5 phút)
│   ├── BAO_CAO_DANH_GIA_TOAN_DIEN.md     (Báo cáo kết quả định lượng chi tiết trên 15 loại hóa đơn)
│   ├── CAM_NANG_PHAN_BIEN_MODEL.md       (Cẩm nang 10 câu hỏi & câu trả lời phản biện Hội đồng)
│   ├── hyperparameter_tuning.py          (Module AutoML: Bayesian Optimization Optuna & LR Finder)
│   ├── evaluate_metrics.py               (Module tính ANLS, Exact Match, Token F1)
│   ├── demo_gradio.py                    (Giao diện Web Demo trực quan)
│   └── data/                             (Tập dữ liệu 114k cặp VQA trên 15 templates)
│
├── kaggle_automation/                    <-- ☁️ KỊCH BẢN CHẠY TRÊN GPU KAGGLE
│   ├── run_kaggle_qwen2_5_training.py    (Script tự động huấn luyện LoRA trên GPU Tesla T4)
│   ├── run_kaggle_qwen2_5_baseline.py    (Script tự động đánh giá Base Model Zero-Shot)
│   └── train_kernel_qwen2_5/             (Notebook huấn luyện chính thức)
│
└── data/                                 (Dữ liệu VQA Train/Test chuẩn hóa)
```

---

## 📌 Phân Công & Hướng Dẫn Sử Dụng Nhanh

| Mục đích công việc | File tài liệu cần xem | Mô tả nội dung |
| :--- | :--- | :--- |
| **Làm Slide Thuyết Trình** | [`docs/DE_CUONG_LAM_SLIDE_THUYET_TRINH.md`](file:///d:/STUDY/MLIoT/project/docs/DE_CUONG_LAM_SLIDE_THUYET_TRINH.md) | 12 Slides được soạn sẵn tiêu đề, nội dung ngắn gọn, gợi ý bố cục và lời thoại. |
| **Viết Báo Cáo / Luận Văn** | [`model/KE_HOACH_FINETUNE_QWEN2_5_VL.md`](file:///d:/STUDY/MLIoT/project/model/KE_HOACH_FINETUNE_QWEN2_5_VL.md) | Đầy đủ công thức toán học LoRA, Target-Only Loss Masking, cấu trúc VLM. |
| **Thuyết Trình & Trả Lời Phản Biện** | [`model/GIAI_THICH_TOAN_DIEN_FINETUNE.md`](file:///d:/STUDY/MLIoT/project/model/GIAI_THICH_TOAN_DIEN_FINETUNE.md) | Kịch bản nói 2.5 phút trước Hội đồng và câu trả lời cho các câu hỏi hóc búa. |
| **Lấy Số Liệu Đối Chứng** | [`model/BAO_CAO_DANH_GIA_TOAN_DIEN.md`](file:///d:/STUDY/MLIoT/project/model/BAO_CAO_DANH_GIA_TOAN_DIEN.md) | Bảng so sánh Base vs LoRA (ANLS, Exact Match, F1, Latency, VRAM). |
| **Chạy Web Demo** | [`model/demo_gradio.py`](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py) | Ứng dụng Gradio Web App tải hóa đơn và hỏi đáp trực tiếp. |
