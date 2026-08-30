# 📚 HỆ THỐNG TÀI LIỆU TOÀN DIỆN DỰ ÁN DOCUMENT VQA
## Đề Tài: Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt

> **Thư mục này tổng hợp toàn bộ tài liệu hướng dẫn, đề cương làm slide, và báo cáo dữ liệu dành cho các thành viên trong nhóm.**

---

## 🗺️ Bản Đồ Tra Cứu Tài Liệu (Documentation Map)

```
Document-Visual-Question-Answering-System/
├── 📁 docs/                                        <-- 🎯 THƯ MỤC TÀI LIỆU CHÍNH CHO NHÓM
│   ├── README.md                                   (Bản đồ tra cứu tài liệu)
│   ├── DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md          (⭐ ĐỀ CƯƠNG 12 SLIDES: Mục tiêu sản phẩm & Dữ liệu 114k mẫu)
│   ├── BAO_CAO_CHI_TIET_DATASET.md                 (Báo cáo số liệu chi tiết 15 loại mẫu & 7 tác vụ VQA)
│   └── DE_CUONG_LAM_SLIDE_THUYET_TRINH.md          (Đề cương tổng thể toàn bộ đề tài)
│
├── 📁 model/                                       <-- 🧠 THƯ MỤC MÔ HÌNH & KỸ THUẬT
│   ├── KE_HOACH_FINETUNE_QWEN2_5_VL.md             (Kế hoạch kỹ thuật & toán học LoRA VLM)
│   ├── GIAI_THICH_TOAN_DIEN_FINETUNE.md            (Tài liệu giải thích sâu phương pháp & kịch bản thuyết trình)
│   ├── BAO_CAO_DANH_GIA_TOAN_DIEN.md               (Báo cáo khung đo lường & công thức ANLS, EM, F1)
│   ├── CAM_NANG_PHAN_BIEN_MODEL.md                 (10 câu hỏi & câu trả lời phản biện Hội đồng)
│   ├── hyperparameter_tuning.py                    (Module AutoML: Optuna Bayesian TPE & LR Finder)
│   ├── demo_gradio.py                              (Giao diện Web Demo trực quan)
│   └── data/                                       (Tập dữ liệu 114k cặp VQA trên 15 templates)
│       ├── vlm_train_master.json                   (97,508 mẫu Train Master)
│       ├── vlm_val_master.json                     (17,208 mẫu Val Master)
│       └── dataset_summary.json                    (Bản tóm tắt phân bổ)
│
└── 📁 kaggle_automation/                           <-- ☁️ KỊCH BẢN TỰ ĐỘNG HÓA GPU KAGGLE
    ├── run_kaggle_qwen2_5_training.py              (Script huấn luyện LoRA tối ưu VRAM trên Tesla T4)
    ├── run_kaggle_qwen2_5_baseline.py              (Script đánh giá Base Model Zero-Shot)
    └── train_kernel_qwen2_5/                       (Notebook huấn luyện chính thức)
```

---

## 📌 Hướng Dẫn Sử Dụng Nhanh Cho Từng Công Việc

| Mục đích công việc | File tài liệu cần mở | Mô tả nội dung |
| :--- | :--- | :--- |
| **Làm Slide Mục Tiêu & Dữ Liệu** | [`docs/DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md`](file:///d:/STUDY/MLIoT/project/docs/DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md) | **12 Slides soạn sẵn** về bài toán, mục tiêu sản phẩm, quy trình và bộ dữ liệu 114k mẫu. |
| **Lấy Số Liệu Dữ Liệu Chi Tiết** | [`docs/BAO_CAO_CHI_TIET_DATASET.md`](file:///d:/STUDY/MLIoT/project/docs/BAO_CAO_CHI_TIET_DATASET.md) | Bảng thống kê 15 loại mẫu hóa đơn và 7 nhóm trường trích xuất. |
| **Lấy Công Thức & Kiến Trúc** | [`model/KE_HOACH_FINETUNE_QWEN2_5_VL.md`](file:///d:/STUDY/MLIoT/project/model/KE_HOACH_FINETUNE_QWEN2_5_VL.md) | Công thức toán học LoRA, Target-Only Loss Masking và cấu trúc VLM. |
| **Chuẩn Bị Trả Lời Phản Biện** | [`model/CAM_NANG_PHAN_BIEN_MODEL.md`](file:///d:/STUDY/MLIoT/project/model/CAM_NANG_PHAN_BIEN_MODEL.md) | 10 câu hỏi & câu trả lời chuẩn xác cho các câu hỏi của Hội đồng. |
