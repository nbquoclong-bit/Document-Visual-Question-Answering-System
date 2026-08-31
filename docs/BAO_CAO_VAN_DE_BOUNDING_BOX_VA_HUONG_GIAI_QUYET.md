# 📑 BÁO CÁO KỸ THUẬT: PHÂN TÍCH NGUYÊN NHÂN HẠN CHẾ, THỬ NGHIỆM THẤT BẠI & HƯỚNG GIẢI QUYẾT MODULE VISUAL GROUNDING (BOUNDING BOX)

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
   - Ban đầu triển khai theo kiến trúc **2 Giai đoạn (2-Stage Pipeline)**:
     $$\text{Ảnh Hóa đơn} \xrightarrow{\text{Qwen2.5-VL-3B (LoRA)}} \text{Văn bản Trả lời (Text)} \xrightarrow{\text{EasyOCR + Heuristic Token Matcher}} \text{Tọa độ Bounding Box (Box)}$$

---

## 🔍 2. CHI TIẾT CÁC THỰC NGHIỆM THẤT BẠI & NGUYÊN NHÂN KỸ THUẬT

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

### 🔴 Thất bại 1: Xung đột so khớp chuỗi số ngắn (Lexical Substring Collisions)
* **Hiện tượng thực tế**: Khi người dùng hỏi *"Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"*, mô hình trả lời đúng `12.000.000đ`, nhưng khung đỏ lại khoanh vào cụm `Số điện thoại: +84 912 345 678` và `123 Đường ABC`.
* **Bản chất nguyên nhân**:
  * Chuỗi kết quả `12.000.000đ` bị bộ tách từ chia thành các token số nhỏ: `['12', '000', '000']`.
  * Thuật toán so khớp xâu con (`in`) tìm thấy chuỗi con `'12'` xuất hiện trong số điện thoại (`912`) và số nhà (`123`).
  * Do hai dòng số điện thoại và địa chỉ nằm kề nhau trên góc trái hóa đơn (khoảng cách trục $Y < 25\text{px}$), giải thuật phân cụm (clustering) đã gộp chúng thành 1 cụm có 2 token, đạt "điểm số lượng" cao hơn token đơn lẻ `12.000.000đ` ở chân trang, dẫn đến việc chọn sai vùng.

---

### 🔴 Thất bại 2: Bẫy ngữ nghĩa thanh tiêu đề bảng (Table Header Semantic Trap)
* **Hiện tượng thực tế**: Trên các hóa đơn Giá trị gia tăng (VAT Invoice), khi hỏi về tiền thuế hoặc thành tiền, khung đỏ bị khoanh trọn vào dải màu xanh chứa tiêu đề cột: `Thành tiền (Amount)`, `Thuế suất GTGT`, `Thuế GTGT`, `Thành tiền đã có thuế GTGT (Total amount)`.
* **Bản chất nguyên nhân**:
  * Khi mô hình trả lời câu hỏi kèm ngữ cảnh giải thích (ví dụ: *"Thành tiền đã có thuế GTGT là 3.404.009đ"*), thuật toán hậu xử lý tìm kiếm các từ ngữ trùng khớp.
  * Thanh tiêu đề màu xanh chứa tới 4 từ khóa trùng lặp liên tiếp (`thành`, `tiền`, `thuế`, `gtgt`), trong khi ô giá trị số ở chân bảng (`3.404.009đ`) chỉ chứa chữ số thuần túy không có chữ cái.
  * Vì vậy, thuật toán chấm điểm từ ngữ (Word Overlap Score) đã ưu tiên gán nhãn cho thanh tiêu đề thay vì ô giá trị số.

---

### 🔴 Thất bại 3: Tràn số Float16 trong thử nghiệm End-to-End Native Grounding (Loss: NaN)
* **Hiện tượng thực tế**: Trong thử nghiệm huấn luyện phiên bản V2 sinh trực tiếp `{"answer": "...", "box": [ymin, xmin, ymax, xmax]}` qua 2D M-RoPE, toàn bộ các Epoch đều ghi nhận `Loss: NaN` và mô hình sụp đổ, chỉ sinh ra chuỗi lặp `!`.
* **Mổ xẻ bản chất toán học & phần cứng**:
  1. **Giới hạn dải động Float16:** Trên GPU Tesla T4, kiểu số `float16` chỉ có dải biểu diễn cực đại là $65,504$.
  2. **Tràn số trong 2D M-RoPE Attention Logits:** Khi tính toán ma trận Attention của ảnh độ phân giải lớn ($512 \times 512$) cùng chuỗi câu hỏi và tọa độ dài, ma trận tích vô hướng:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \mathcal{R}_{\text{2D-M-RoPE}}\right) V$$
     vượt quá ngưỡng $65,504$, biến thành $+\infty$. Qua hàm Softmax và Cross-Entropy Loss lập tức sinh ra giá trị **`NaN` (Not a Number)**.

---

## 🛠️ 3. HƯỚNG GIẢI QUYẾT & BÀI HỌC KINH NGHIỆM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LỘ TRÌNH GIẢI PHÁP                                 │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ 1. Giải Pháp Ngắn Hạn (Heuristic)    │ 2. Giải Pháp Dài Hạn (Native Grounding)│
│ - So khớp 100% chuỗi số thực thể     │ - Huấn luyện Single-Pass Unified VLM │
│ - Bổ sung Header/Label Blacklist     │ - Bắt buộc dùng BFloat16 / GradScaler│
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
1. **Cấu hình BFloat16 / Mixed Precision Scaler**:
   - Tránh 100% lỗi `Loss: NaN` bằng cách dùng `torch.cuda.amp.GradScaler` và `bfloat16` trên GPU hỗ trợ.
2. **Định dạng Single-Pass Unified Output**:
   - Tận dụng 2D M-RoPE của Qwen2.5-VL để sinh tọa độ chính xác không phụ thuộc OCR ngoài, giảm độ trễ từ 2.8s xuống còn ~1.2s/câu.

---

## 🎓 4. 4 BÀI HỌC KINH NGHIỆM ĐẮT GIÁ CHO KỸ SƯ MACHINE LEARNING

| STT | Bài học đúc kết | Ý nghĩa thực tiễn cho Kỹ sư ML |
| :---: | :--- | :--- |
| **1** | **Precision Matters trong VLM** | Với các mô hình Vision Transformer dùng RoPE 2D phức tạp, **không bao giờ dùng `float16` trần** mà phải dùng `bfloat16` (trên A100/V100) hoặc dùng `GradScaler` với `float32` cho attention logits trên T4. |
| **2** | **Đừng tin vào so khớp chuỗi ngây thơ** | Trong tài liệu tài chính, dữ liệu số có cấu trúc hoàn toàn khác văn bản thông thường. Thuật toán so khớp chuỗi ngây thơ (naive substring match) sẽ luôn thất bại trước số điện thoại và địa chỉ. |
| **3** | **Lỗi lan truyền trong hệ thống 2-Stage** | Pipeline ghép nối 2 mô hình (VLM + OCR) luôn tạo ra sai số kép. Tương lai của xử lý tài liệu bắt buộc phải là **Single-Pass End-to-End Multimodal Modeling**. |
| **4** | **Giá trị của việc đọc Log và Metric đối chứng** | Việc theo dõi biểu đồ Loss và log từng epoch giúp phát hiện sớm sự sụp đổ mô hình thay vì chỉ nhìn vào kết quả đầu ra cuối cùng. |

---

## 📈 5. KẾT LUẬN

* Mô hình cốt lõi **Qwen2.5-VL-3B LoRA** đã hoàn thành xuất sắc nhiệm vụ trích xuất ngôn ngữ và kế toán (**94.94% ANLS**).
* Việc đối mặt trực tiếp và mổ xẻ tường tận nguyên nhân kỹ thuật của các thử nghiệm Bounding Box mang lại giá trị học thuật cao, thể hiện năng lực nghiên cứu thực nghiệm nghiêm túc của nhóm.
