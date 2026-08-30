# 📚 HỆ THỐNG TÀI LIỆU TOÀN DIỆN DỰ ÁN DOCUMENT VQA
## Đề Tài: Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt

> **Thư mục này tổng hợp toàn bộ tài liệu báo cáo, kho tư liệu kỹ thuật, đề cương và số liệu chi tiết dành cho các thành viên trong nhóm.**

---

## 🗺️ Bản Đồ Tra Cứu Tài Liệu (Documentation Map)

```
Document-Visual-Question-Answering-System/
├── 📁 docs/                                        <-- 🎯 THƯ MỤC TÀI LIỆU CHÍNH CHO NHÓM
│   ├── README.md                                   (Bản đồ tra cứu tài liệu)
│   ├── TONG_HOP_MUC_TIEU_SAN_PHAM_VA_DU_LIEU.md   (⭐ KHO TƯ LIỆU TOÀN DIỆN: Mục tiêu sản phẩm & Dữ liệu 114k mẫu)
│   ├── BAO_CAO_CHI_TIET_DATASET.md                 (Báo cáo số liệu chi tiết 15 loại mẫu & 7 tác vụ VQA)
│   └── DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md          (Gợi ý khung slide mẫu tham khảo)
│
├── 📁 model/                                       <-- 🧠 MÃ NGUỒN & TRỌNG SỐ
│   ├── data/                                       (Tập dữ liệu 114,716 cặp VQA trên 15 templates)
│   │   ├── vlm_train_master.json                   (97,508 mẫu Train Master ~34MB)
│   │   ├── vlm_val_master.json                     (17,208 mẫu Val Master ~5.9MB)
│   │   └── dataset_summary.json                    (Bản tóm tắt phân bổ)
│   ├── output/                                     (Báo cáo kết quả đánh giá định lượng)
│   │   └── qwen2_5_vl_baseline_report.json         (Kết quả 174 câu hỏi Base Model trên GPU)
│   ├── hyperparameter_tuning.py                    (Module AutoML: Optuna TPE & LR Finder)
│   └── demo_gradio.py                              (Giao diện Web Demo trực quan)
│
└── 📁 kaggle_automation/                           <-- ☁️ TIẾN TRÌNH GPU KAGGLE
    ├── run_kaggle_qwen2_5_training.py              (Script huấn luyện LoRA tối ưu VRAM)
    └── train_kernel_qwen2_5/                       (Notebook huấn luyện)
```

---

## 📌 Hướng Dẫn Sử Dụng Nhanh Cho Từng Công Việc

| Mục đích công việc | File tài liệu cần mở | Mô tả nội dung |
| :--- | :--- | :--- |
| **Lấy Tư Liệu Làm Slide Mục Tiêu & Data** | [`docs/TONG_HOP_MUC_TIEU_SAN_PHAM_VA_DU_LIEU.md`](file:///d:/STUDY/MLIoT/project/docs/TONG_HOP_MUC_TIEU_SAN_PHAM_VA_DU_LIEU.md) | **Kho thông tin chi tiết nhất (8 phần đầy đủ)** về bối cảnh, giá trị sản phẩm, sơ đồ luồng, quy trình và toàn bộ 114k mẫu data. |
| **Lấy Bảng Thống Kê Số Liệu** | [`docs/BAO_CAO_CHI_TIET_DATASET.md`](file:///d:/STUDY/MLIoT/project/docs/BAO_CAO_CHI_TIET_DATASET.md) | Bảng phân bổ 15 loại mẫu hóa đơn và 7 nhóm trường trích xuất. |
| **Tham Khảo Đề Cương Slide Mẫu** | [`docs/DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md`](file:///d:/STUDY/MLIoT/project/docs/DE_CUONG_SLIDE_MUC_TIEU_VA_DATA.md) | Bản khung gợi ý bố cục slide. |
