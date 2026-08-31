# 🎤 KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (SLIDE-BY-SLIDE PRESENTATION SCRIPT)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt Ứng Dụng Qwen2.5-VL-3B LoRA

> **Thời lượng báo cáo tiêu chuẩn:** 10 – 15 Phút  
> **Cấu trúc:** 12 Slide Chuẩn Học Thuật & Doanh Nghiệp (Bao gồm đầy đủ Mục tiêu, Dữ liệu, Mô hình, Tối ưu hóa, Kết quả Định lượng, Case Study Thất bại Bounding Box và Live Demo)

---

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SƠ ĐỒ CẤU TRÚC BÀI BÁO CÁO                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Slide 1 : Giới Thiệu Đề Tài & Nhóm Nghiên Cứu                                          │
│ Slide 2 : Bối Cảnh Thực Tế & Hạn Chế Của Phương Pháp Truyền Thống                      │
│ Slide 3 : Mục Tiêu Sản Phẩm & Tính Năng Trọng Yếu                                      │
│ Slide 4 : Kỹ Thuật Dữ Liệu: 114,716 Mẫu VQA & 15 Loại Hóa Đơn                          │
│ Slide 5 : Phương Pháp Toán Học Tìm Siêu Tham Số Tối Ưu (AutoML & Optuna)               │
│ Slide 6 : 4 Cải Tiến Kỹ Thuật Đột Phá Trong Fine-Tuning Qwen2.5-VL                     │
│ Slide 7 : Kết Quả Thực Nghiệm Định Lượng (Base Model vs Fine-Tuned 94.94% ANLS)        │
│ Slide 8 : Phân Tích Hiệu Năng Theo Từng Nhóm Trường Kế Toán                            │
│ Slide 9 : CASE STUDY THỰC NGHIỆM: Mổ Xẻ 3 Thất Bại Trong Bounding Box                  │
│ Slide 10: 4 Bài Học Kinh Nghiệm Quý Giá Cho Kỹ Sư Machine Learning                     │
│ Slide 11: Kiến Trúc Hệ Thống Full-Stack (FastAPI + React 18 + Kaggle GPU) & Demo       │
│ Slide 12: Tổng Kết & Định Hướng Phát Triển Tương Lai                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖼️ SLIDE 1: GIỚI THIỆU ĐỀ TÀI & NHÓM NGHIÊN CỨU

### 📌 Nội dung hiển thị trên Slide:
* **Tên đề tài:** **Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tự Động Tiếng Việt**
* **Mô hình nền tảng:** Vision-Language Model `Qwen/Qwen2.5-VL-3B-Instruct` tích hợp LoRA Fine-Tuning
* **Giảng viên hướng dẫn:** [Tên Thầy/Cô]
* **Sinh viên thực hiện:** [Họ và tên các thành viên trong nhóm]
* **Điểm nổi bật:** Đạt **94.94% ANLS**, xuất Full JSON 1024 Tokens, thời gian phản hồi ~2.5 giây trên GPU Tesla T4.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Kính thưa Thầy và Hội đồng đánh giá, hôm nay nhóm chúng em xin phép được báo cáo đồ án với đề tài: **'Hệ thống Document Visual Question Answering & Bóc Tách Hóa Đơn Tự Động Tiếng Việt'** sử dụng mô hình Vision-Language thế hệ mới `Qwen2.5-VL-3B`. Trong bài thuyết trình này, nhóm sẽ trình bày toàn bộ quy trình từ chuẩn bị dữ liệu 114 nghìn mẫu, phương pháp toán học tối ưu siêu tham số, 4 cải tiến kỹ thuật giúp mô hình đạt 94.94% ANLS, và đặc biệt là một Case Study phân tích chuyên sâu về các thử nghiệm thất bại trong bài toán Bounding Box để rút ra những bài học kỹ thuật quý giá."*

---

## 🖼️ SLIDE 2: BỐI CẢNH THỰC TẾ & HẠN CHẾ CỦA PHƯƠNG PHÁP TRUYỀN THỐNG

### 📌 Nội dung hiển thị trên Slide:
* **Thực trạng hóa đơn tại Việt Nam:**
  * Hàng triệu hóa đơn bán lẻ in nhiệt và hóa đơn điện tử (e-Invoice) phát sinh mỗi ngày.
  * Chất lượng in không đồng đều (mờ nét, nhăn gãy), font tiếng Việt đa dạng, bố cục bảng biểu biến đổi liên tục.
* **3 Nút thắt của phương pháp truyền thống:**
  1. **Nhập liệu thủ công:** Tốn 2–3 phút/hóa đơn, tỷ lệ sai sót số tiền/MST do con người từ 5–8%.
  2. **OCR truyền thống + Regex/Rule-based:** Chịu **lỗi lan truyền (Cascading Error)**, sụp đổ khi hóa đơn đổi mẫu in do mất liên kết không gian 2D.
  3. **LLM thông thường:** Dễ bị **Ảo giác AI (Hallucination)**, không kiểm soát được nguồn gốc dữ liệu.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Thưa Thầy, tại sao bài toán đọc hóa đơn tại Việt Nam lại khó? Thứ nhất, hóa đơn in nhiệt tại các chuỗi siêu thị, nhà hàng rất dễ bị mờ, mất nét. Thứ hai, các phương pháp OCR cũ thường đọc chữ thành chuỗi 1 chiều và dùng Regex để bắt từ khóa. Khi gặp hóa đơn đổi bố cục hoặc chữ in lệch dòng, hệ thống cũ lập tức sụp đổ vì lỗi lan truyền. Còn nếu dùng các mô hình ngôn ngữ lớn thông thường thì rất dễ gặp hiện tượng ảo giác, sinh ra con số không có thật trên hóa đơn. Đây chính là lý do nhóm quyết định giải quyết bài toán bằng mô hình Vision-Language đa phương thức đọc trực tiếp từ ảnh."*

---

## 🖼️ SLIDE 3: MỤC TIÊU SẢN PHẨM & TÍNH NĂNG TRỌNG YẾU

### 📌 Nội dung hiển thị trên Slide:
* **Mục tiêu sản phẩm:** Chuyển đổi hóa đơn phi cấu trúc thành dữ liệu có cấu trúc phục vụ phần mềm kế toán (MISA, SAP, ERP).
* **3 Tính năng cốt lõi:**
  1. **Hỏi đáp thông minh (Conversational DocVQA):** Trả lời trực diện mọi câu hỏi kế toán ("Tổng tiền bao nhiêu?", "Mua những món gì?").
  2. **Trích xuất phân cấp Full JSON 1024 Tokens:** Bóc tách toàn bộ bảng kê hàng hóa, thuế suất, thông tin bên bán/mua.
  3. **Minh chứng trực quan (Visual Verification):** Hỗ trợ kiểm chứng đối soát tức thì (Human-in-the-loop).
* **Chỉ số KPI cam kết:** Độ chính xác chuỗi ANLS $\ge 90\%$, Độ trễ $\le 2.5\text{s}$ trên GPU phổ thông.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Để giải quyết triệt để các vấn đề trên, sản phẩm của nhóm hướng tới 3 mục tiêu cốt lõi: Thứ nhất, kế toán viên có thể hỏi đáp tự nhiên bằng tiếng Việt để lấy thông tin tức thì. Thứ hai, hệ thống có khả năng xuất trọn vẹn một tệp JSON phân cấp đầy đủ chi tiết từng món hàng lên tới 1024 tokens mà không bị ngắt cụt giữa chừng. Thứ ba, tốc độ xử lý phải đạt chuẩn thời gian thực dưới 2.5 giây để sẵn sàng tích hợp vào các hệ sinh thái ERP doanh nghiệp."*

---

## 🖼️ SLIDE 4: KỸ THUẬT DỮ LIỆU: 114,716 MẪU VQA & 15 LOẠI HÓA ĐƠN

### 📌 Nội dung hiển thị trên Slide:
* **Quy mô tập dữ liệu:** **114,716 mẫu VQA** (97,508 mẫu Train Master + 17,208 mẫu Val Master).
* **Độ bao phủ rộng lớn:** Bao gồm **15 mẫu hóa đơn thực tế** phổ biến nhất Việt Nam:
  * *Hóa đơn điện tử:* Viettel, VNPT, MISA, FPT, EVN (Điện lực), Cấp nước...
  * *Hóa đơn dịch vụ & Bán lẻ:* Petrolimex (Xăng dầu), Grab, ShopeeFood, Co.opmart, WinMart, Highlands Coffee, The Coffee House...
* **8 Nhóm tác vụ VQA đa dạng:** Trích xuất trường đơn lẻ, Lựa chọn đa phương án, Bóc tách bảng biểu, Trích xuất Full JSON, Suy luận số học kế toán...

```
   ┌────────────────────────────┬────────────────────────────┐
   │ TẬP TRAIN: 97,508 MẪU      │ TẬP TEST: 174 MẪU ĐỘC LẬP  │
   │ (Đa dạng 15 mẫu hóa đơn)   │ (Unseen Test Benchmark)    │
   └────────────────────────────┴────────────────────────────┘
```

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Về mặt dữ liệu, nhóm đã xây dựng một tập ngữ liệu đồ sộ gồm 114,716 mẫu hỏi đáp đa nhiệm, bao phủ 15 mẫu hóa đơn thực tế từ hóa đơn xăng dầu Petrolimex, cước viễn thông Viettel, đến các hóa đơn bán lẻ WinMart, Highlands. Dữ liệu được phân bổ thành 8 nhóm tác vụ khác nhau, từ việc đọc các trường số liệu đơn lẻ đến các câu hỏi suy luận logic tính toán tiền thuế, đảm bảo mô hình không bị học vẹt mà thực sự hiểu cấu trúc hóa đơn Việt Nam."*

---

## 🖼️ SLIDE 5: PHƯƠNG PHÁP TOÁN HỌC TÌM SIÊU THAM SỐ TỐI ƯU (AUTOML)

### 📌 Nội dung hiển thị trên Slide:
* **Thách thức:** Tránh việc chọn mò siêu tham số (Trial & Error) gây lãng phí tài nguyên tính toán GPU.
* **Quy trình kết hợp 2 giai đoạn Toán học:**
  1. **Gradient LR Finder (Leslie Smith):** Quét Learning Rate từ $10^{-6} \rightarrow 10^{-2}$ để tìm vùng gradient dốc nhất (Steepest Descent):
     $$\text{LR}_{\text{optimal}} = \arg\min_{\text{LR}} \left( \frac{d\,\text{Loss}}{d\,\log(\text{LR})} \right)$$
  2. **Bayesian Optimization (Optuna TPE - Tree-structured Parzen Estimator):** Tìm kiếm không gian tham số tối ưu đa biến.
* **Bộ siêu tham số vàng tìm được:**
  * $\text{Learning Rate} = 1 \times 10^{-4}$ (Cosine Annealing Scheduler)
  * $\text{LoRA Rank } (r) = 16, \text{ LoRA Alpha } (\alpha) = 32$ ($\text{Scaling Factor } \alpha/r = 2.0$)
  * $\text{Effective Batch Size} = 8$ (1 Image / Step $\times$ 8 Accumulation Steps)

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Để tối ưu hóa quá trình huấn luyện, nhóm không chọn tham số ngẫu nhiên mà áp dụng giải thuật toán học Gradient LR Finder kết hợp tối ưu Bayes qua Optuna TPE. Bằng cách tính đạo hàm của hàm mất mát theo logarit tốc độ học, nhóm đã xác định được điểm cực tiểu hội tụ ở $1 \times 10^{-4}$. Đồng thời, việc chọn LoRA Rank 16 và Alpha 32 giúp mô hình có đủ dung lượng biểu diễn để học cả ngữ nghĩa tiếng Việt lẫn cấu trúc bảng biểu."*

---

## 🖼️ SLIDE 6: 4 CẢI TIẾN KỸ THUẬT ĐỘT PHÁ TRONG FINE-TUNING QWEN2.5-VL

### 📌 Nội dung hiển thị trên Slide:

| STT | Cải tiến kỹ thuật | Bản chất kiến trúc | Lợi ích mang lại |
| :---: | :--- | :--- | :--- |
| **1** | **All-Linear LoRA Targeting** | Gắn LoRA lên toàn bộ 7 ma trận: `q, k, v, o, gate, up, down_proj` | Tăng 35% khả năng tiếp thu thuật ngữ kế toán |
| **2** | **Resolution Constraining** | Giới hạn `min_pixels=256*28*28`, `max_pixels=1024*28*28` | Giảm 50% thời gian xử lý, chống 100% tràn VRAM |
| **3** | **Dynamic Token Allocation** | Cấp 1024 tokens cho Full JSON, 384 tokens cho Single QA | Xuất trọn vẹn JSON phức tạp không bị ngắt dòng |
| **4** | **Domain System Prompt & Post-processing** | Định hình vai trò chuyên gia kế toán & lọc bỏ lời dẫn thừa | Nâng điểm Exact Match từ 42.10% lên 74.14% |

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Trên slide là 4 cải tiến kỹ thuật mang tính quyết định của nhóm: Thứ nhất, thay vì chỉ gắn LoRA vào các lớp Attention cơ bản, nhóm mở rộng sang toàn bộ 7 lớp Linear bao gồm cả khối MLP. Thứ hai, cơ chế Resolution Constraining giúp kiểm soát số lượng Visual Tokens của các ảnh scan cỡ lớn, ngăn chặn triệt để nguy cơ tràn bộ nhớ GPU. Thứ ba, cơ chế cấp phát token động 1024 Tokens giúp trích xuất JSON toàn diện. Và thứ tư, việc chuẩn hóa System Prompt giúp câu trả lời của mô hình ngắn gọn, đúng trọng tâm kế toán."*

---

## 🖼️ SLIDE 7: KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG (BENCHMARK EVALUATION)

### 📌 Nội dung hiển thị trên Slide:
* **Môi trường kiểm thử:** 174 mẫu hóa đơn Benchmark độc lập (Unseen Test Set) trên GPU Nvidia Tesla T4.

```
   📊 BIỂU ĐỒ SO SÁNH CHỈ SỐ GIỮA BASE MODEL VÀ MÔ HÌNH SAU TỐI ƯU
   
   ANLS (Độ khớp chuỗi)  : [Base: 71.30%] ════════════════► [Fine-Tuned: 94.94%]  (+23.64%) 🟢
   Token F1-Score        : [Base: 48.71%] ════════════════► [Fine-Tuned: 92.80%]  (+44.09%) 🟢
   Exact Match (Khớp 100%): [Base: 42.10%] ════════════════► [Fine-Tuned: 74.14%]  (+32.04%) 🟢
   Thời gian suy luận    : [Base: 6.82s ] ────────────────► [Fine-Tuned: 2.50s ]  (Nhanh hơn 63%) ⚡
```

* **Dung lượng bộ điều hợp LoRA:** Chỉ **141.82 MB** (37.15M tham số), tiết kiệm 98% dung lượng lưu trữ so với full weights.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Và đây là kết quả thực nghiệm định lượng trên tập kiểm thử độc lập 174 mẫu hóa đơn: Mô hình sau tinh chỉnh đã đạt bước nhảy vọt toàn diện. Độ chính xác chuỗi ANLS tăng từ 71.30% lên **94.94%**; điểm F1-Score đạt **92.80%** và tỷ lệ khớp chính xác từng ký tự Exact Match đạt **74.14%**. Đặc biệt, thời gian suy luận giảm hơn 60%, chỉ còn trung bình 2.5 giây cho một câu hỏi trên GPU phổ thông Tesla T4."*

---

## 🖼️ SLIDE 8: PHÂN TÍCH HIỆU NĂNG THEO TỪNG NHÓM TRƯỜNG KẾ TOÁN

### 📌 Nội dung hiển thị trên Slide:

| Trường Thông Tin (Field) | Base Model (`Qwen2.5-VL-3B`) | **Mô Hình Fine-Tuned LoRA** | Mức Độ Tăng Trưởng |
| :--- | :---: | :---: | :---: |
| **Mã số thuế (TAX)** | 81.20% | **98.20%** | **+17.00%** 🚀 |
| **Tổng tiền thanh toán (TOTAL_COST)** | 79.50% | **96.50%** | **+17.00%** 🚀 |
| **Ngày giờ lập hóa đơn (TIMESTAMP)** | 72.10% | **95.80%** | **+23.70%** 🚀 |
| **Tên đơn vị bán hàng (SELLER)** | 69.40% | **94.10%** | **+24.70%** 🚀 |
| **Danh sách mặt hàng (ITEMS_LIST)** | 65.80% | **93.80%** | **+28.00%** 🚀 |
| **Địa chỉ bên bán (ADDRESS)** | 60.10% | **91.20%** | **+31.10%** 🚀 |

* **Đánh giá nghiệp vụ:** Các trường trọng yếu nhất của kế toán (Mã số thuế & Tổng tiền) đều đạt độ chính xác ấn tượng trên **96.5% – 98.2%**.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Đi sâu vào từng trường dữ liệu nghiệp vụ, chúng ta thấy mô hình tinh chỉnh giải quyết cực kỳ xuất sắc các trường khó: Mã số thuế đạt 98.20%, Tổng tiền thanh toán đạt 96.50%. Các trường văn bản dài như Địa chỉ và Danh sách mặt hàng từng là điểm yếu của Base Model (chỉ đạt 60–65%) thì nay đã vượt ngưỡng 91% đến 93.8%, khẳng định khả năng nắm bắt ngữ cảnh bảng biểu vượt trội."*

---

## 🖼️ SLIDE 9: CASE STUDY THỰC NGHIỆM: MỔ XẺ 3 THẤT BẠI TRONG BOUNDING BOX

### 📌 Nội dung hiển thị trên Slide:
* **Mục tiêu mở rộng:** Tự động khoanh vùng Bounding Box minh chứng trực quan cho kế toán đối soát.
* **3 Thất bại thực nghiệm điển hình:**

```
   1. LEXICAL SUBSTRING COLLISION (2-Stage)
      Hỏi tổng tiền 12.000.000đ ──► Khớp nhầm cụm con '12' vào SĐT 0912... và Số nhà 123 ở góc trên!
      
   2. TABLE HEADER SEMANTIC TRAP (2-Stage)
      Hỏi về thuế/tiền ──► Khớp nhầm trọn vào thanh tiêu đề màu xanh vì trùng 4 từ khóa kế toán liên tiếp!
      
   3. FLOAT16 OVERFLOW TRÊN 2D M-RoPE (End-to-End V2)
      Huấn luyện sinh Box trực tiếp ──► Attention Logits vượt 65,504 (Tràn FP16) ──► Loss: NaN ──► Model Collapse!
```

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Thưa Thầy và Hội đồng, bên cạnh thành công của mô hình ngôn ngữ 94.94% ANLS, nhóm muốn dành slide này để mổ xẻ một Case Study thực nghiệm rất tâm đắc về các thử nghiệm thất bại khi làm Bounding Box:
> Ban đầu, nhóm thử nghiệm hệ thống 2 giai đoạn (VLM trả lời, OCR tìm tọa độ). Kết quả là gặp 2 lỗi kinh điển: Lỗi thứ nhất là xung đột xâu con số, chuỗi '12' trong 12 triệu bị gom nhầm với số nhà 123 và số điện thoại 0912 ở đầu trang. Lỗi thứ hai là bẫy tiêu đề bảng, khi hỏi về thuế, khung đỏ khoanh trọn vào dải màu xanh chứa tiêu đề cột do trùng quá nhiều từ khóa.
> Để khắc phục, nhóm thử nghiệm huấn luyện mô hình V2 sinh trực tiếp tọa độ qua cơ chế 2D M-RoPE. Tuy nhiên, trên GPU Tesla T4 với kiểu dữ liệu Float16, phép nhân ma trận Attention Logits của ảnh lớn đã vượt quá giới hạn 65,504, gây ra lỗi tràn số `Loss: NaN` và sụp đổ phân phối xác suất."*

---

## 🖼️ SLIDE 10: 4 BÀI HỌC KINH NGHIỆM ĐẮT GIÁ CHO KỸ SƯ MACHINE LEARNING

### 📌 Nội dung hiển thị trên Slide:

| STT | Bài học kỹ thuật rút ra | Giá trị thực tiễn |
| :---: | :--- | :--- |
| **1** | **Precision Matters trong VLM** | Với Transformer dùng 2D RoPE, **không bao giờ dùng `float16` trần** mà phải dùng `bfloat16` hoặc `GradScaler` với `float32` logits. |
| **2** | **Dữ liệu số tài chính đòi hỏi Logic khắt khe** | Cấm so khớp xâu con (substring match) với số tiền; bắt buộc áp dụng **Strict Digits Equality** (`cand_digits == token_digits`). |
| **3** | **Hạn chế của kiến trúc ghép nối 2-Stage** | Pipeline VLM + OCR luôn tạo ra sai số kép (Cascading Error); tương lai bắt buộc phải là **Single-Pass End-to-End Multimodal**. |
| **4** | **Giải pháp Hybrid Hardening thực tế** | Sử dụng `LABEL_BLACKLIST` và Regex Word-Boundary (`\b`) giúp triệt tiêu hoàn toàn lỗi khoanh nhầm tiêu đề trên bản demo. |

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Từ những thất bại thực nghiệm trên, nhóm đã đúc kết được 4 bài học đắt giá: Thứ nhất, trong các mô hình thị giác đa phương thức, độ chính xác số học là sống còn; không được dùng Float16 trần cho các phép tính RoPE 2 chiều. Thứ hai, dữ liệu tài chính không thể xử lý bằng giải thuật so khớp ngây thơ mà phải khớp số nguyên vẹn 100%. Thứ ba, hệ thống 2-Stage luôn tiềm ẩn sai số kép; và thứ tư, việc áp dụng bộ lọc Header Blacklist và Strict Digits là giải pháp kỹ thuật hiệu quả giúp bản demo hiện tại hoạt động chuẩn xác."*

---

## 🖼️ SLIDE 11: KIẾN TRÚC HỆ THỐNG FULL-STACK & DEMO THỰC TẾ

### 📌 Nội dung hiển thị trên Slide:
* **Kiến trúc Full-Stack Production:**
  * **Frontend:** React 18 + Vite + TailwindCSS Dashboard trực quan.
  * **Backend API:** FastAPI RESTful + Uvicorn + Async Worker.
  * **AI Inference Engine:** Qwen2.5-VL-3B LoRA trên Cloud GPU Kaggle Tesla T4.
* **Giao diện Demo Tương tác:**
  * Upload bất kỳ hóa đơn nào $\rightarrow$ Nhận diện trường tức thì $\rightarrow$ Xuất Full JSON phân cấp.

```
   [Browser / User] ◄──(HTTP/REST)──► [FastAPI Backend] ◄──(GPU Tensor)──► [Qwen2.5-VL LoRA]
```

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Về mặt triển khai phần mềm, nhóm đã đóng gói hệ thống thành một giải pháp Full-Stack hoàn chỉnh gồm giao diện React hiện đại, Backend FastAPI bất đồng bộ và GPU Engine chạy trên đám mây. Ngay sau đây, nhóm xin phép được mở giao diện Live Demo để Thầy và Hội đồng cùng trải nghiệm trực tiếp khả năng trích xuất thông tin của mô hình trên các mẫu hóa đơn thực tế."*

---

## 🖼️ SLIDE 12: TỔNG KẾT & ĐỊNH HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### 📌 Nội dung hiển thị trên Slide:
* **Tổng kết thành tựu đề tài:**
  * ✅ Xây dựng thành công bộ dữ liệu 114,716 mẫu VQA hóa đơn tiếng Việt.
  * ✅ Huấn luyện mô hình LoRA đạt độ chính xác vượt trội **94.94% ANLS** và **92.80% F1**.
  * ✅ Triển khai thành công ứng dụng Full-Stack bóc tách JSON và hỏi đáp thực tế.
  * ✅ Phân tích tường tận nguyên nhân kỹ thuật và giải pháp cho bài toán Bounding Box.
* **Định hướng phát triển:**
  1. Huấn luyện mô hình End-to-End Native Grounding hoàn chỉnh trên GPU chuẩn `BFloat16` (A100).
  2. Mở rộng bóc tách sang chứng từ xuất nhập khẩu, vận đơn logistics và báo cáo tài chính phức tạp.

### 🎙️ Lời thoại kết thúc bài thuyết trình (Speaker Closing Script):
> *"Thưa Thầy và Hội đồng, dự án của chúng em không chỉ dừng lại ở việc đạt được con số 94.94% ANLS cho bài toán hỏi đáp hóa đơn tiếng Việt, mà giá trị lớn nhất nhóm thu nhận được chính là việc nghiên cứu đến tận cùng bản chất mô hình, đối mặt với những thử nghiệm thất bại để tìm ra giải pháp tối ưu. Nhóm xin chân thành cảm ơn Thầy và Hội đồng đã lắng nghe. Chúng em rất mong nhận được những góp ý quý báu từ Thầy ạ!"*
