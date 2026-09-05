# 📚 HỆ THỐNG TÀI LIỆU TOÀN DIỆN DỰ ÁN DOCUMENT VQA
## Đề Tài: Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt

> **Thư mục này tổng hợp toàn bộ tài liệu báo cáo, kho tư liệu kỹ thuật, đề cương và số liệu chi tiết dành cho các thành viên trong nhóm.**

---

## 🗺️ Bản Đồ Tra Cứu Tài Liệu (Documentation Map)

```
Document-Visual-Question-Answering-System/
├── 📁 docs/                                        <-- 🎯 THƯ MỤC TÀI LIỆU CHÍNH CHO NHÓM
│   ├── README.md                                   (Bản đồ tra cứu tài liệu)
│   ├── KE_HOACH_SLIDE.md                           (⭐ BỘ 10 SLIDE THUYẾT TRÌNH CHUẨN: 4 phần, 8 tác vụ, Case study BBox, Video Demo)
│   ├── TONG_HOP_KIEN_THUC.md                       (📚 KHO TƯ LIỆU TOÀN DIỆN: Mục tiêu sản phẩm, Kiến trúc & Dữ liệu 114k mẫu)
│   ├── BAO_CAO_CHI_TIET_DATASET.md                 (Báo cáo số liệu chi tiết 15 loại mẫu & 8 tác vụ VQA)
│   └── BAO_CAO_VAN_DE_BOUNDING_BOX_VA_HUONG_GIAI_QUYET.md (Báo cáo kỹ thuật mổ xẻ thất bại Bounding Box)
│
├── 📁 model/                                       <-- 🧠 MÃ NGUỒN & TRỌNG SỐ
│   ├── data/                                       (Tập dữ liệu 114,716 cặp VQA trên 15 templates)
│   │   ├── vlm_train_master.json                   (97,508 mẫu Train Master ~34MB)
│   │   ├── vlm_val_master.json                     (17,208 mẫu Val Master ~5.9MB)
│   │   └── dataset_summary.json                    (Bản tóm tắt phân bổ)
│   ├── output/                                     (Báo cáo kết quả đánh giá định lượng - Xem README.md chi tiết)
│   │   ├── 01_qwen2_vl_2b_baseline_legacy.json     (Thế hệ 1: Base 2B trên 45 mẫu)
│   │   ├── 02_qwen2_5_vl_3b_baseline_zeroshot.json (Thế hệ 2: Base 3B Zero-shot trên 174 mẫu - 0.68% ANLS)
│   │   ├── 03_qwen2_5_vl_3b_lora_raw_uncleaned.json (Thế hệ 3: LoRA chưa tối ưu hậu xử lý - 59.48% ANLS)
│   │   ├── 04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json (🌟 CHUẨN KAGGLE GPU: LoRA + Optimized - 89.61% ANLS)
│   │   ├── evaluation_report.json                  (Bản sao chính thức của 04 cho các script mặc định)
│   │   └── README.md                               (Bảng tổng hợp so sánh chi tiết & Task Breakdown)
│   └── stage1_vlm/                                 (Mô hình Qwen2.5-VL / Qwen2-VL kèm LoRA)
│
├── 📁 backend/                                     <-- ⚙️ BACKEND FASTAPI & ĐỘ TIN CẬY
│   └── backend-docvqa/backend/                     (FastAPI service, Confidence Score, SQLite)
│
└── 📁 frontend/                                    <-- 💻 FRONTEND REACT & WEB INTERFACE
    └── src/                                        (React 18, Vite, Giao diện Sổ Hóa Đơn & Hỏi đáp)
```

---

## 📌 Hướng Dẫn Sử Dụng Nhanh Cho Từng Công Việc

| Mục đích công việc | File tài liệu cần mở | Mô tả nội dung |
| :--- | :--- | :--- |
| **Làm Slide Thuyết Trình & Tập Nói** | [`docs/KE_HOACH_SLIDE.md`](file:///d:/STUDY/MLIoT/project/docs/KE_HOACH_SLIDE.md) | **Kế hoạch 10 slide hoàn chỉnh** bám sát cấu trúc 4 phần (Bài toán $\rightarrow$ Giải quyết $\rightarrow$ Pipeline $\rightarrow$ Hiệu quả), kèm lời thoại thuyết trình (Speaker Notes). |
| **Tra Cứu Toàn Diện Kiến Thức & Nghiệp Vụ** | [`docs/TONG_HOP_KIEN_THUC.md`](file:///d:/STUDY/MLIoT/project/docs/TONG_HOP_KIEN_THUC.md) | **Kho tư liệu đầy đủ 10 phần** về bối cảnh, giá trị sản phẩm, sơ đồ luồng, quy trình và toàn bộ 114k mẫu data. |
| **Lấy Bảng Thống Kê Số Liệu 8 Tác Vụ** | [`docs/BAO_CAO_CHI_TIET_DATASET.md`](file:///d:/STUDY/MLIoT/project/docs/BAO_CAO_CHI_TIET_DATASET.md) | Bảng phân bổ 15 loại mẫu hóa đơn và 8 nhóm trường trích xuất. |
| **Tham Khảo Case Study Bounding Box** | [`docs/BAO_CAO_VAN_DE_BOUNDING_BOX_VA_HUONG_GIAI_QUYET.md`](file:///d:/STUDY/MLIoT/project/docs/BAO_CAO_VAN_DE_BOUNDING_BOX_VA_HUONG_GIAI_QUYET.md) | Phân tích chi tiết 3 lỗi thất bại khi ghép EasyOCR và lý do chuyển sang đo Điểm Tin Cậy. |
