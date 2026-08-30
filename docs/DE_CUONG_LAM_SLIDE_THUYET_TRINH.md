# 📑 ĐỀ CƯƠNG SLIDE BÁO CÁO TIẾN ĐỘ & PHƯƠNG PHÁP ĐỒ ÁN
## Đề Tài: Xây Dựng Hệ Thống Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt Sử Dụng Vision-Language Model và LoRA

> **Lưu ý:** Bản đề cương này tập trung **100% vào các phần nhóm ĐÃ THỰC HIỆN VÀ HOÀN TẤT THÀNH CÔNG** (Bài toán, Xây dựng tập dữ liệu 114k mẫu, Thiết kế kiến trúc mô hình, Thuật toán LoRA, Module AutoML tối ưu siêu tham số, Kỹ thuật Loss Masking và Hệ thống Web Demo).  
> Các bạn làm slide chỉ cần copy trực tiếp các nội dung gạch đầu dòng bên dưới vào PowerPoint / Canva.

---

```
                        CẤU TRÚC BỘ SLIDE BÁO CÁO CÔNG VIỆC ĐÃ HOÀN THÀNH
 ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
 │      SLIDES 1 - 3      │ ──> │      SLIDES 4 - 5      │ ──> │      SLIDES 6 - 9      │
 │  ĐẶT VẤN ĐỀ & GIẢI PHÁP│     │  DỮ LIỆU & ĐẶC TẢ VQA  │     │ KIẾN TRÚC & PHƯƠNG PHÁP│
 └────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                            │
                                                                            ▼
 ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
 │     SLIDES 12 + Q&A    │ <── │       SLIDE 11         │ <── │        SLIDE 10        │
 │ TỔNG KẾT & PHẢN BIỆN   │     │  GIAO DIỆN WEB DEMO    │     │  KHUNG METRICS ĐÁNH GIÁ│
 └────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

---

### 🖥️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề lớn:** BÁO CÁO TIẾN ĐỘ & PHƯƠNG PHÁP HỆ THỐNG DOCUMENT VQA
* **Tiêu đề phụ:** Nghiên Cứu & Ứng Dụng Vision-Language Model Kết Hợp LoRA Cho Hóa Đơn Tiếng Việt
* **Thông tin nhóm:**
  * Giảng viên hướng dẫn: [Tên Thầy/Cô]
  * Thành viên thực hiện: [Tên các bạn trong nhóm]
  * Lớp / Ngành / Năm học: 2025 - 2026
* **Speaker Notes:** *"Kính chào Thầy/Cô và các bạn, hôm nay nhóm xin phép trình bày báo cáo về phương pháp xây dựng hệ thống Document VQA cho hóa đơn tiếng Việt."*

---

### 🖥️ SLIDE 2: ĐẶT VẤN ĐỀ & THÁCH THỨC HÓA ĐƠN TIẾNG VIỆT
* **Tiêu đề:** Thách Thức Trong Xử Lý Hóa Đơn Thực Tế
* **Bố cục:** 2 cột (Đặc điểm hóa đơn VN vs Hạn chế thực tế)
* **Nội dung hiển thị:**
  * **Đặc thù hóa đơn bán lẻ tại Việt Nam:**
    * Chữ in nhiệt dễ phai màu, nét chữ mỏng, font chữ đa dạng.
    * Bố cục bảng biểu nhiều cột, không có một format cố định.
    * Ảnh chụp thực tế hay bị nghiêng, bóng mờ, nhàu nát.
  * **Thách thức:** Cần một giải pháp trích xuất tự động không phụ thuộc vào template cố định.
* **Speaker Notes:** *"Hóa đơn tại Việt Nam rất đa dạng về hình thức và chất lượng in ấn, đòi hỏi mô hình phải có khả năng hiểu ngữ cảnh thị giác cao."*

---

### 🖥️ SLIDE 3: GIỚI HẠN PIPELINE CŨ & GIẢI PHÁP ĐỀ XUẤT
* **Tiêu đề:** Tiếp Cận End-to-End Multimodal Vision-Language Model
* **Bố cục:** Sơ đồ so sánh (Pipeline Cũ vs Giải Pháp Nhóm)
* **Nội dung hiển thị:**
  * ❌ **Hạn chế của Pipeline truyền thống (OCR + NLP):**
    * **Lỗi lan truyền thác (Cascading Error):** OCR đọc sai ký tự $\implies$ NLP trích xuất sai toàn bộ số tiền.
    * **Mất liên kết không gian 2D:** Làm phẳng ảnh thành văn bản 1D làm mất mối quan hệ giữa các cột.
  * 💡 **Giải pháp của Nhóm (End-to-End VLM):**
    * Nhận trực tiếp **Pixel ảnh + Câu hỏi tự nhiên**.
    * Mô hình tự học liên kết thị giác - ngôn ngữ, không qua OCR trung gian.

---

### 🖥️ SLIDE 4: XÂY DỰNG & CHUẨN HÓA BỘ DỮ LIỆU (114,716 CẶP VQA)
* **Tiêu đề:** Xây Dựng Tập Dữ Liệu Huấn Luyện & Đánh Giá
* **Bố cục:** Bảng số liệu & Biểu đồ phân bổ
* **Nội dung hiển thị:**
  * **Quy mô hoàn thành:** **114,716 cặp câu hỏi - câu trả lời VQA** trên **4,995 ảnh hóa đơn**.
  * **Phân chia tập dữ liệu rõ ràng:**
    * 📦 **Tập Train Master:** `97,508` mẫu (~85%) dùng cho huấn luyện.
    * 🔍 **Tập Validation Master:** `17,208` mẫu (~15%) dùng kiểm soát Loss.
    * 🎯 **Tập Benchmark Test:** `174` câu hỏi độc lập phủ kín 15 loại mẫu.
  * **Bao phủ 15 thương hiệu & mẫu hóa đơn:** Highlands Coffee, Phúc Long, Starbucks, WinMart, Lotte Mart, Bách Hóa Xanh, Viettel e-Invoice, VNPT, C45-BB, Circle K, 7-Eleven, GS25, Minimart An An, KFC, Jollibee.

---

### 🖥️ SLIDE 5: ĐẶC TẢ CÁC TÁC VỤ VQA ĐA DẠNG
* **Tiêu đề:** 7 Nhóm Trường Trích Xuất & Cấu Trúc JSON
* **Bố cục:** Lưới 4 ô (Trường cơ bản, Chi tiết hàng hóa, JSON, Grounding)
* **Nội dung hiển thị:**
  1. `SELLER`: Tên đơn vị bán hàng, chi nhánh, công ty phát hành.
  2. `TOTAL_COST`: Tổng tiền thanh toán cuối cùng.
  3. `TIMESTAMP`: Ngày giờ lập hóa đơn.
  4. `ADDRESS`: Địa chỉ nơi phát hành hóa đơn.
  5. `ITEM_PRICE` & `ITEM_QTY`: Đơn giá và số lượng từng mặt hàng.
  6. `FULL_JSON`: Trích xuất toàn bộ cấu trúc phân cấp hóa đơn thành JSON.
  7. `BOUNDING_BOX`: Tọa độ không gian hỗ trợ định vị vùng văn bản.

---

### 🖥️ SLIDE 6: THIẾT KẾ KIẾN TRÚC MÔ HÌNH THỊ GIÁC - NGÔN NGỮ
* **Tiêu đề:** Kiến Trúc Mô Hình Nền Tảng (Vision-Language Backbone)
* **Bố cục:** Sơ đồ khối các thành phần chính
* **Nội dung hiển thị:**
  * **Vision Transformer (ViT):** Trích xuất các visual tokens trực tiếp từ ảnh.
  * **2D Spatial M-RoPE (Multimodal Rotary Position Embedding):** Mã hóa vị trí 2 chiều $(x, y)$, giúp mô hình dóng hàng cột và dòng tiền chính xác.
  * **Dynamic Resolution:** Tự động điều chỉnh độ phân giải phù hợp (`min_pixels` đến `max_pixels`) để giữ nét chữ in nhỏ mà không tràn VRAM.
  * **LLM Engine:** Sinh câu trả lời dựa trên ngữ cảnh thị giác tích hợp.

---

### 🖥️ SLIDE 7: PHƯƠNG PHÁP PEFT / LORA (LOW-RANK ADAPTATION)
* **Tiêu đề:** Thiết Kế Kỹ Thuật Fine-Tuning LoRA
* **Bố cục:** Hình minh họa ma trận $W = W_0 + \frac{\alpha}{r}(B \times A)$
* **Nội dung hiển thị:**
  * **Đóng băng toàn bộ trọng số gốc $W_0$:** Tránh hiện tượng quên tri thức (*Catastrophic Forgetting*).
  * **Cấu hình LoRA tối ưu:**
    * $\text{Rank } r = 16, \text{Alpha } \alpha = 32 \implies \text{Scaling Factor } \frac{\alpha}{r} = 2.0$.
    * $\text{Dropout} = 0.05$ (Chống quá khớp).
  * **Can thiệp toàn diện cả 7 ma trận chiếu tuyến tính:**
    * Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    * MLP: `gate_proj`, `up_proj`, `down_proj` (Nơi lưu giữ tri thức số tiền & từ vựng).
  * **Dung lượng Adapter:** Chỉ **~75 MB** (chiếm ~0.33% tham số).

---

### 🖥️ SLIDE 8: KỸ THUẬT TARGET-ONLY LOSS MASKING
* **Tiêu đề:** Xử Lý Triệt Để Hiện Tượng Lời Dẫn Lan Man
* **Bố cục:** So sánh Cơ chế tính Loss thông thường vs Target-Only Masking
* **Nội dung hiển thị:**
  * ❌ **Vấn đề:** Base Model thường nói chuyện dài dòng (*"Dựa trên hình ảnh hóa đơn..."*) làm sai lệch chuẩn trích xuất.
  * 💡 **Giải pháp Target-Only Loss Masking:**
    * Gán nhãn `ignore_index = -100` cho toàn bộ Vision tokens và User Question tokens.
    * **Chỉ tính đạo hàm Cross-Entropy trên Target Answer tokens:**
      $$\mathcal{L} = -\frac{1}{|T_{\text{target}}|} \sum_{t \in T_{\text{target}}} \log P(y_t \mid y_{<t}, X_{\text{image}}, X_{\text{prompt}})$$
  * ✅ **Tác dụng:** Ép mô hình chỉ trả về đúng giá trị thực thể cần trích xuất.

---

### 🖥️ SLIDE 9: MODULE AutoML & TỐI ƯU SIÊU THAM SỐ TỰ ĐỘNG
* **Tiêu đề:** Phương Pháp Luận Tối Ưu Siêu Tham Số
* **Bố cục:** 2 nhánh (Gradient LR Finder & Bayesian Optimization Optuna)
* **Nội dung hiển thị:**
  * **Không chọn tham số cảm tính:** Nhóm đã xây dựng module [`model/hyperparameter_tuning.py`](file:///d:/STUDY/MLIoT/project/model/hyperparameter_tuning.py).
  * **Gradient-based LR Finder:** Quét 100 bước tăng dần LR để tìm điểm đạo hàm giảm dốc nhất $\implies \text{LR} = 2 \times 10^{-4}$.
  * **Bayesian Optimization (Optuna TPE):** Thăm dò không gian tham số $(r, \alpha, \text{weight\_decay})$.
  * **Cắt tỉa ASHA Pruning:** Tự động dừng các cấu hình thử nghiệm kém hiệu quả.

---

### 🖥️ SLIDE 10: KHUNG ĐÁNH GIÁ ĐỊNH LƯỢNG (METRICS FRAMEWORK)
* **Tiêu đề:** Tiêu Chuẩn & Thước Đo Đánh Giá Học Thuật
* **Bố cục:** 4 Cards định nghĩa các chỉ số đo lường
* **Nội dung hiển thị:**
  1. **ANLS (Average Normalized Levenshtein Similarity):** Tiêu chuẩn quốc tế DocVQA Challenge, đánh giá độ chính xác chuỗi có tính đến khoảng cách chỉnh sửa ký tự:
     $$\text{ANLS} = 1 - \frac{d_L(p, gt)}{\max(|p|, |gt|)} \quad (\text{nếu } NL < 0.5)$$
  2. **Exact Match (EM %):** Tỷ lệ câu trả lời khớp chính xác 100% từng ký tự.
  3. **Token F1-Score:** Đánh giá độ phủ từ khóa (Precision & Recall ở mức từ).
  4. **Inference Latency & VRAM:** Đo đạc độ trễ từng câu và bộ nhớ thực tế trên GPU.

---

### 🖥️ SLIDE 11: THIẾT KẾ GIAO DIỆN WEB DEMO TƯƠNG TÁC
* **Tiêu đề:** Xây Dựng Ứng Dụng Demo (Gradio Web Interface)
* **Bố cục:** Ảnh chụp giao diện Web + Mô tả các tính năng
* **Nội dung hiển thị:**
  * **Giao diện kéo thả:** Cho phép người dùng tải lên bất kỳ ảnh hóa đơn nào.
  * **Hỏi đáp linh hoạt:**
    * Chọn nhanh các câu hỏi mẫu (Tổng tiền, Người bán, Ngày tháng, Chi tiết món).
    * Nhập câu hỏi tự do bằng tiếng Việt.
  * **Xuất kết quả 2 chế độ:** Chế độ văn bản ngắn gọn và Chế độ JSON có cấu trúc.
  * **Mã nguồn:** Đã hoàn thiện tại [`model/demo_gradio.py`](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py).

---

### 🖥️ SLIDE 12: TỔNG KẾT & KẾ HOẠCH BƯỚC TIẾP THEO
* **Tiêu đề:** Tổng Kết Công Việc Đã Hoàn Thành
* **Bố cục:** 2 cột (Đã hoàn thành vs Kế hoạch hoàn thiện)
* **Nội dung hiển thị:**
  * ✅ **Đã hoàn thành 100%:**
    * Thu thập và gán nhãn chuẩn hóa 114,716 mẫu VQA trên 15 loại hóa đơn.
    * Thiết kế kiến trúc VLM kết hợp LoRA và Target-Only Loss Masking.
    * Xây dựng module AutoML tìm kiếm siêu tham số tối ưu (Optuna + LR Finder).
    * Thiết kế khung đánh giá học thuật ANLS, EM, F1 và giao diện Web Demo.
  * 🔄 **Kế hoạch tiếp theo:**
    * Hoàn tất quá trình huấn luyện và xuất báo cáo kiểm định thực nghiệm cuối cùng.
* **Lời cảm ơn:** *"Xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý lắng nghe!"*

---

### 🛡️ PHỤ LỤC: CÂU HỎI PHẢN BIỆN DỰ PHÒNG (BACKUP Q&A)

1. **Tại sao không dùng OCR + LLM truyền thống?**  
   $\rightarrow$ Tránh hiện tượng lỗi lan truyền (Cascading Error) và giữ nguyên cấu trúc không gian 2D của hóa đơn.
2. **LoRA giúp ích gì cho bài toán này?**  
   $\rightarrow$ Đóng băng 99.67% mô hình gốc chống quên tri thức tổng quát, chỉ huấn luyện ~0.33% tham số, adapter chỉ ~75 MB, chạy vừa vặn trên 1 GPU Tesla T4 (16GB).
3. **Làm sao để chắc chắn bộ tham số là tối ưu?**  
   $\rightarrow$ Nhóm áp dụng thuật toán Bayesian Optimization (Optuna TPE) và Gradient LR Finder quét không gian tham số thay vì chọn cảm tính.
