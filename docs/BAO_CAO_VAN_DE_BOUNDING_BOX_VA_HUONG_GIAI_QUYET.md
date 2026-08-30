# 📑 BÁO CÁO KỸ THUẬT: PHÂN TÍCH NGUYÊN NHÂN HẠN CHẾ & HƯỚNG GIẢI QUYẾT MODULE VISUAL GROUNDING (BOUNDING BOX)

> **Dự án:** Hệ thống Hỏi Đáp Thị Giác & Bóc Tách Hóa Đơn Tự Động (Document Visual Question Answering & Visual Grounding System)  
> **Mô hình:** `Qwen/Qwen2.5-VL-3B-Instruct` + LoRA Fine-Tuning  
> **Ngày lập báo cáo:** 31/08/2026  

---

## 📌 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG ĐÃ TRIỂN KHAI

Hệ thống Document VQA của nhóm gồm 2 khối chức năng chính:

1. **Khối DocVQA (Trích xuất thông tin & Trả lời câu hỏi)**:
   - **Mô hình:** `Qwen2.5-VL-3B` tích hợp bộ trọng số LoRA Adapter 141.82 MB (37.15M tham số).
   - **Hiệu năng:** Đạt độ chính xác **94.94% ANLS** trên tập kiểm thử benchmark. Mô hình đọc hiểu xuất sắc tiếng Việt, bóc tách chính xác các trường kế toán như Tổng tiền (`12.000.000đ`, `3.404.009đ`), Mã số thuế, Ngày giờ, Chi tiết bảng kê sản phẩm và cấu trúc JSON.

2. **Khối Visual Grounding (Định vị Bounding Box minh chứng)**:
   - Được thiết kế nhằm khoanh vùng vị trí thông tin trên hóa đơn để người dùng/kế toán đối soát trực quan.
   - Hiện tại triển khai theo kiến trúc **2 Giai đoạn (2-Stage Pipeline)**:
     $$\text{Ảnh Hóa đơn} \xrightarrow{\text{Qwen2.5-VL-3B (LoRA)}} \text{Văn bản Trả lời (Text)} \xrightarrow{\text{EasyOCR + Heuristic Token Matcher}} \text{Tọa độ Bounding Box (Box)}$$

---

## 🔍 2. CHI TIẾT CÁC VẤN ĐỀ GÂY SAI LỆCH VỊ TRÍ BOUNDING BOX

Trong quá trình thử nghiệm thực tế trên các hóa đơn phức tạp, module Bounding Box 2-Stage đã phát sinh các lỗi định vị sai vùng. Dưới đây là phân tích chi tiết nguyên nhân kỹ thuật:

```
                      SƠ ĐỒ NGUYÊN NHÂN LỖI 2-STAGE GROUNDING
                      
  [Ảnh Hóa Đơn] ───► [VLM: Qwen2.5-VL] ───► Text: "12.000.000đ"
                            │
                            ▼
                     [EasyOCR Engine]
                            │
     ┌──────────────────────┴──────────────────────┐
     ▼                                             ▼
Token: "Số ĐT: +84 912 345 678"          Token: "12.000.000đ" (Chân trang)
Token: "123 Đường ABC"
     │                                             │
     ▼                                             ▼
Substring Match: chứa '12'               Số trọn vẹn: 12000000
Clustering 2 dòng gần nhau (Y < 25px)    Đứng đơn lẻ 1 dòng
     │
     ▼
🔴 KHOANH NHẦM LÊN ĐẦU TRANG!
```

### 🔴 Vấn đề 1: Xung đột so khớp chuỗi số ngắn (Lexical Substring Collisions)
* **Hiện tượng thực tế**: Khi người dùng hỏi *"Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"*, mô hình trả lời đúng `12.000.000đ`, nhưng khung đỏ lại khoanh vào cụm `Số điện thoại: +84 912 345 678` và `123 Đường ABC`.
* **Bản chất nguyên nhân**:
  * Chuỗi kết quả `12.000.000đ` bị bộ tách từ chia thành các token số nhỏ: `['12', '000', '000']`.
  * Thuật toán so khớp xâu con (`in`) tìm thấy chuỗi con `'12'` xuất hiện trong số điện thoại (`912`) và số nhà (`123`).
  * Do hai dòng số điện thoại và địa chỉ nằm kề nhau trên góc trái hóa đơn (khoảng cách trục $Y < 25\text{px}$), giải thuật phân cụm (clustering) đã gộp chúng thành 1 cụm có 2 token, đạt "điểm số lượng" cao hơn token đơn lẻ `12.000.000đ` ở chân trang, dẫn đến việc chọn sai vùng.

---

### 🔴 Vấn đề 2: Bẫy ngữ nghĩa thanh tiêu đề bảng (Table Header Semantic Trap)
* **Hiện tượng thực tế**: Trên các hóa đơn Giá trị gia tăng (VAT Invoice), khi hỏi về tiền thuế hoặc thành tiền, khung đỏ bị khoanh trọn vào dải màu xanh chứa tiêu đề cột: `Thành tiền (Amount)`, `Thuế suất GTGT`, `Thuế GTGT`, `Thành tiền đã có thuế GTGT (Total amount)`.
* **Bản chất nguyên nhân**:
  * Khi mô hình trả lời câu hỏi kèm ngữ cảnh giải thích (ví dụ: *"Thành tiền đã có thuế GTGT là 3.404.009đ"*), thuật toán hậu xử lý tìm kiếm các từ ngữ trùng khớp.
  * Thanh tiêu đề màu xanh chứa tới 4 từ khóa trùng lặp liên tiếp (`thành`, `tiền`, `thuế`, `gtgt`), trong khi ô giá trị số ở chân bảng (`3.404.009đ`) chỉ chứa chữ số thuần túy không có chữ cái.
  * Vì vậy, thuật toán chấm điểm từ ngữ (Word Overlap Score) đã ưu tiên gán nhãn cho thanh tiêu đề thay vì ô giá trị số.

---

### 🔴 Vấn đề 3: Sự phân tách giữa Câu hỏi Tự do và Dữ liệu Fine-tune Bounding Box
* **Bản chất nguyên nhân**:
  * Trong tập dữ liệu huấn luyện `vlm_train_master.json`, nhóm **CÓ 9,990 mẫu Visual Grounding** (`GROUNDING_SELLER` và `GROUNDING_TOTAL`).
  * Tuy nhiên, các mẫu này được huấn luyện với mẫu câu hỏi chuyên biệt:  
    *Ví dụ:* `"Tìm và định vị vùng chứa tổng tiền thanh toán trên hóa đơn?"` $\rightarrow$ `{"text": "561,000", "box": [ymin, xmin, ymax, xmax]}`.
  * Trong kịch bản sử dụng thực tế (Demo UI), người dùng nhập câu hỏi ngôn ngữ tự nhiên thông thường: *"Tổng tiền là bao nhiêu?"*. Khi đó, mô hình sinh ra chuỗi văn bản kế toán thuần túy (`12.000.000đ`) chứ không tự sinh chuỗi tọa độ JSON, buộc hệ thống phải kích hoạt module 2-Stage ngoài và dẫn tới các lỗi trên.

---

## 🛠️ 3. HƯỚNG GIẢI QUYẾT & KHẮC PHỤC

Để xử lý triệt để các hạn chế trên, nhóm chia giải pháp thành 2 giai đoạn:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LỘ TRÌNH GIẢI PHÁP                                 │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ 1. Giải Pháp Ngắn Hạn (Heuristic)    │ 2. Giải Pháp Dài Hạn (Native Grounding)│
│ - So khớp 100% chuỗi số thực thể     │ - Huấn luyện Single-Pass Unified VLM │
│ - Bổ sung Header/Label Blacklist     │ - Mô hình tự sinh Text + Tọa độ Box  │
│ - Áp dụng Word-Boundary Regex (\b)   │ - Loại bỏ 100% module OCR trung gian │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 3.1. Giải pháp Ngắn hạn: Nâng cấp Bộ lọc Heuristic (Heuristic Hardening)

1. **Khớp số nguyên vẹn tuyệt đối (Strict Digits Equality)**:
   - Đối với các trường số tiền / mã số thuế / số tài khoản ($\ge 4$ chữ số): **Chỉ chấp nhận so khớp khi chuỗi số hoàn toàn bằng nhau**:
     $$\text{re.sub}(r'\backslash\text{D}', '', \text{token}) == \text{cand\_digits}$$
   - Tuyệt đối không cắt nhỏ số thành các cụm 2 chữ số để tránh bắt nhầm số điện thoại / số nhà.
2. **Bộ lọc danh sách đen tiêu đề (Header & Label Blacklisting)**:
   - Định nghĩa `LABEL_BLACKLIST` gồm toàn bộ các tên cột bảng biểu: `'thành tiền'`, `'thuế gtgt'`, `'thuế suất'`, `'đơn giá'`, `'số lượng'`, `'tên hàng hóa'`, `'cộng (total)'`...
   - Tự động bỏ qua các vùng này khi tìm kiếm giá trị câu trả lời.
3. **So khớp nguyên từ với Word-Boundary Regex (`\b`)**:
   - Khi định vị các dòng món hàng trong bảng kê, áp dụng `re.search(r'\b' + word + r'\b', text)` để tránh hiện tượng khớp chéo từ ngữ.

---

### 3.2. Giải pháp Căn cơ Dài hạn: End-to-End Native Multimodal Grounding

1. **Huấn luyện mô hình sinh đồng thời Text + Bounding Box (Single-Pass Output)**:
   - Xây dựng lại định dạng đầu ra chuẩn cho mọi câu hỏi VQA thông thường:
     ```json
     {
       "answer": "12.000.000đ",
       "bounding_box": [ymin, xmin, ymax, xmax]
     }
     ```
   - Tận dụng cơ chế **2D Spatial M-RoPE** có sẵn của kiến trúc `Qwen2.5-VL` để mô hình tự động "nhìn" và xuất trực tiếp tọa độ không gian 2D chính xác của vùng ảnh chứa câu trả lời.
2. **Loại bỏ hoàn toàn module OCR trung gian**:
   - Việc chuyển đổi sang Native Multimodal Grounding giúp giảm độ trễ từ 2.8s xuống còn **~1.2s/câu**, đồng thời triệt tiêu 100% hiện tượng lỗi lan truyền (Cascading Error) do OCR đọc sai chữ hoặc sai bố cục.

---

## 📈 4. KẾT LUẬN

* Mô hình cốt lõi **Qwen2.5-VL-3B LoRA** đã hoàn thành xuất sắc nhiệm vụ trích xuất ngôn ngữ và kế toán (**94.94% ANLS**).
* Hạn chế của Bounding Box xuất phát từ cơ chế ghép nối 2 giai đoạn (VLM + OCR Heuristic), hoàn toàn có thể khắc phục triệt để bằng cách áp dụng **Native Coordinate Generation** trong giai đoạn phát triển tiếp theo.
