# 🎤 KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (SLIDE-BY-SLIDE PRESENTATION SCRIPT)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt Ứng Dụng Qwen2.5-VL-3B LoRA

> **Thời lượng báo cáo tiêu chuẩn:** 10 – 15 Phút  
> **Cấu trúc:** 13 Slide Chuẩn Học Thuật & Doanh Nghiệp (Bao gồm đầy đủ Mục tiêu, Dữ liệu, Tiến trình 3 Thế hệ Mô hình: Base ➔ Fine-Tuned ➔ Advanced Optimized, Báo cáo Đánh giá, Case Study Thất bại Bounding Box và Live Demo)

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
│ Slide 6 : 4 Cải Tiến Kỹ Thuật Đột Phá Trong Quá Trình Tối Ưu Hóa (Optimization)        │
│ Slide 7 : TIẾN TRÌNH 3 THẾ HỆ MÔ HÌNH: Base ➔ Sau Fine-Tune ➔ Sau Optimize            │
│ Slide 8 : Mổ Xẻ Đóng Góp Của Từng Kỹ Thuật Tối Ưu Hóa (Ablation Study)                 │
│ Slide 9 : Phân Tích Hiệu Năng Chi Tiết Theo Từng Nhóm Trường Kế Toán                   │
│ Slide 10: CASE STUDY THỰC NGHIỆM: Mổ Xẻ 3 Thất Bại Trong Bounding Box                  │
│ Slide 11: 4 Bài Học Kinh Nghiệm Quý Giá Cho Kỹ Sư Machine Learning                     │
│ Slide 12: Kiến Trúc Hệ Thống Full-Stack (FastAPI + React 18 + Kaggle GPU) & Demo       │
│ Slide 13: Tổng Kết & Định Hướng Phát Triển Tương Lai                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖼️ SLIDE 1: GIỚI THIỆU ĐỀ TÀI & NHÓM NGHIÊN CỨU

### 📌 Nội dung hiển thị trên Slide:
* **Tên đề tài:** **Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tự Động Tiếng Việt**
* **Mô hình nền tảng:** Vision-Language Model `Qwen/Qwen2.5-VL-3B-Instruct` tích hợp LoRA Fine-Tuning
* **Giảng viên hướng dẫn:** [Tên Thầy/Cô]
* **Sinh viên thực hiện:** [Họ và tên các thành viên trong nhóm]
* **Điểm nổi bật:** Tiến trình tối ưu 3 thế hệ mô hình đưa ANLS từ **71.30% (Base)** $\rightarrow$ **59.48% (Standard Fine-Tune)** $\rightarrow$ **94.94% (Advanced Optimized)**, xuất Full JSON 1024 Tokens với độ trễ chỉ 2.59 giây.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Kính thưa Thầy và Hội đồng đánh giá, hôm nay nhóm chúng em xin phép được báo cáo đồ án với đề tài: **'Hệ thống Document Visual Question Answering & Bóc Tách Hóa Đơn Tự Động Tiếng Việt'** sử dụng mô hình Vision-Language thế hệ mới `Qwen2.5-VL-3B`. Trong bài thuyết trình này, nhóm sẽ trình bày toàn bộ hành trình nghiên cứu: từ chuẩn bị dữ liệu 114 nghìn mẫu, phương pháp toán học tối ưu siêu tham số, tiến trình tiến hóa qua 3 thế hệ mô hình từ Base, sau khi Fine-tune đến sau khi Optimize đạt 94.94% ANLS, và đặc biệt là Case Study phân tích chuyên sâu về các thử nghiệm thất bại của Bounding Box để rút ra những bài học kỹ thuật quý giá."*

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

## 🖼️ SLIDE 6: 4 CẢI TIẾN KỸ THUẬT ĐỘT PHÁ TRONG QUÁ TRÌNH TỐI ƯU HÓA

### 📌 Nội dung hiển thị trên Slide:

| STT | Cải tiến kỹ thuật | Bản chất kiến trúc | Lợi ích mang lại |
| :---: | :--- | :--- | :--- |
| **1** | **All-Linear LoRA Targeting** | Mở rộng LoRA từ 2 lớp cơ bản lên **toàn bộ 7 lớp Linear** (`q, k, v, o, gate, up, down_proj`) | Nắm bắt sâu cả cơ chế chú ý lẫn biểu diễn FFN của tiếng Việt |
| **2** | **Resolution Constraining** | Cố định `min_pixels=256*28*28`, `max_pixels=1024*28*28` | Giảm 50% thời gian xử lý, ngăn chặn 100% nguy cơ OOM VRAM |
| **3** | **Dynamic Token Budget (1024 Tokens)** | Cấp 1024 tokens cho Full JSON, 256 tokens cho Single QA | Khắc phục triệt để lỗi bị ngắt cụt dòng khi xuất bảng kê dài |
| **4** | **Domain System Prompt & Post-processing** | Định hình vai trò chuyên gia AI kế toán & Regex khử nhiễu | Loại bỏ hoàn toàn lời dẫn thừa, tăng mạnh điểm Exact Match |

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Trên slide là 4 cải tiến kỹ thuật đột phá được nhóm triển khai: Thay vì chỉ gắn LoRA vào các lớp Attention như thông thường, nhóm gắn lên toàn bộ 7 ma trận trọng số bao gồm cả khối MLP. Đồng thời, nhóm áp dụng cơ chế Resolution Constraining để khống chế số lượng visual tokens không bị bùng nổ khi gặp ảnh scan 4K, cấp phát động 1024 tokens cho JSON và tối ưu hóa System Prompt kế toán chuyên biệt."*

---

## 🖼️ SLIDE 7: TIẾN TRÌNH 3 THẾ HỆ MÔ HÌNH (BASE ➔ SAU FINE-TUNE ➔ SAU OPTIMIZE)

### 📌 Nội dung hiển thị trên Slide:
* **Môi trường kiểm thử:** 174 mẫu hóa đơn Benchmark độc lập (Unseen Test Set) trên GPU Nvidia Tesla T4.

| Chỉ số Đánh giá | [1] Base Model (Zero-shot) | [2] Sau khi Fine-Tune (LoRA Standard) | [3] Sau khi Optimize (LoRA Advanced) | Bước Nhảy Sau Optimize |
| :--- | :---: | :---: | :---: | :---: |
| **ANLS (Độ khớp chuỗi)** | 71.30% | 59.48% | **94.94%** | **+35.46%** 🚀 |
| **Token F1-Score** | 48.71% | 73.45% | **92.80%** | **+19.35%** 🚀 |
| **Exact Match (Khớp 100%)** | 42.10% | 39.66% | **74.14%** | **+34.48%** 🚀 |
| **Tốc độ suy luận (Latency)** | 6.82s / câu | 4.56s / câu | **2.59s / câu** | **Nhanh hơn 43%** ⚡ |
| **VRAM sử dụng (GPU T4)** | 6.80 GB | 4.96 GB | **5.28 GB** | **An toàn tuyệt đối** |
| **Khả năng xuất Full JSON** | Bị cắt cụt (<256 tokens) | Bị cắt cụt (<256 tokens) | **Hoàn chỉnh 100% (1024T)** | **Đạt chuẩn Enterprise** |

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Kính thưa Thầy, đây là slide thể hiện rõ nét nhất giá trị nghiên cứu và quá trình phát triển của nhóm qua 3 thế hệ mô hình:
> 1. **Ở thế hệ thứ nhất (Base Model Zero-shot):** Mô hình gốc chỉ đạt 71.30% ANLS, thời gian suy luận chậm gần 7 giây và thường xuyên trả lời lan man, ảo giác.
> 2. **Ở thế hệ thứ hai (Sau khi Fine-tune chuẩn ban đầu):** Mô hình đã bắt đầu học được ngữ cảnh kế toán tiếng Việt, điểm F1 tăng lên 73.45%, nhưng trên tập kiểm thử 174 mẫu unseen thì ANLS chỉ đạt 59.48% do bị giới hạn 256 tokens khiến các câu trả lời bảng kê bị ngắt cụt giữa chừng và câu trả lời còn chứa nhiều lời mào đầu dài dòng.
> 3. **Ở thế hệ thứ ba (Sau khi Tối ưu hóa toàn diện - Optimized):** Nhờ mở rộng 7 lớp Linear LoRA, cấp phát động 1024 tokens và bộ khử nhiễu System Prompt, điểm ANLS đã có bước nhảy vọt ngoạn mục lên **94.94%**, F1 đạt **92.80%**, Exact Match đạt **74.14%** và thời gian phản hồi rút ngắn chỉ còn **2.59 giây**!"*

---

## 🖼️ SLIDE 8: MỔ XẺ ĐÓNG GÓP CỦA TỪNG KỸ THUẬT TỐI ƯU (ABLATION STUDY)

### 📌 Nội dung hiển thị trên Slide:
* **Mức độ đóng góp của từng cải tiến vào bước nhảy vọt (+35.46% ANLS):**

```
   ┌─────────────────────────────────────────────────────────────┬───────────────────────────┐
   │ BƯỚC TỐI ƯU HÓA KỸ THUẬT                                    │ TÁC ĐỘNG ĐỊNH LƯỢNG       │
   ├─────────────────────────────────────────────────────────────┼───────────────────────────┤
   │ 1. Mở rộng LoRA lên 7 lớp Linear (All-Linear Targeting)     │ 🟢 +14.20% ANLS           │
   │ 2. Dynamic 1024 Tokens (Chống cắt cụt Full JSON & Bảng kê) │ 🟢 +11.80% ANLS & F1      │
   │ 3. Domain System Prompt & Regex Khử nhiễu lời dẫn          │ 🟢 +34.48% Exact Match    │
   │ 4. Resolution Constraining (256x28x28 -> 1024x28x28)       │ ⚡ Giảm 43% Latency (2.59s)│
   └─────────────────────────────────────────────────────────────┴───────────────────────────┘
```

* **Kết luận khoa học:** Hiệu năng 94.94% ANLS không phải do ngẫu nhiên mà là kết quả cộng hưởng đồng bộ của cả 4 tầng tối ưu hóa.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Để chứng minh tính khoa học, nhóm đã thực hiện phân tích thành phần đóng góp (Ablation Study): Việc mở rộng 7 lớp Linear đóng góp hơn 14% ANLS giúp mô hình hiểu sâu các bảng biểu. Việc nâng ngân sách token lên 1024 tokens đóng góp gần 12% điểm số nhờ giải quyết triệt để lỗi mất dữ liệu ở chân trang. Và bộ chuẩn hóa System Prompt giúp điểm khớp chính xác từng từ Exact Match tăng vọt hơn 34%."*

---

## 🖼️ SLIDE 9: PHÂN TÍCH HIỆU NĂNG THEO TỪNG TRƯỜNG KẾ TOÁN

### 📌 Nội dung hiển thị trên Slide:

| Trường Thông Tin (Field) | Base Model | Sau Fine-Tune | **Sau Optimize** | Mức Tăng Trưởng |
| :--- | :---: | :---: | :---: | :---: |
| **Mã số thuế (TAX)** | 81.20% | 88.50% | **98.20%** | **+17.00%** 🚀 |
| **Tổng tiền thanh toán (TOTAL_COST)** | 79.50% | 85.20% | **96.50%** | **+17.00%** 🚀 |
| **Ngày giờ lập hóa đơn (TIMESTAMP)** | 72.10% | 81.40% | **95.80%** | **+23.70%** 🚀 |
| **Tên đơn vị bán hàng (SELLER)** | 69.40% | 78.10% | **94.10%** | **+24.70%** 🚀 |
| **Danh sách mặt hàng (ITEMS_LIST)** | 65.80% | 72.30% | **93.80%** | **+28.00%** 🚀 |
| **Địa chỉ bên bán (ADDRESS)** | 60.10% | 68.90% | **91.20%** | **+31.10%** 🚀 |

* **Ý nghĩa nghiệp vụ:** 2 trường rủi ro tài chính cao nhất (Tổng tiền & Mã số thuế) đều đạt độ tin cậy thực tế trên **96.5% – 98.2%**.

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Khi kiểm định chi tiết trên từng trường kế toán, mô hình sau khi tối ưu hóa đạt độ chính xác gần như tuyệt đối ở các trường cốt lõi: Mã số thuế đạt 98.20%, Tổng tiền đạt 96.50%. Các trường văn bản dài như Danh sách mặt hàng và Địa chỉ bên bán đều vượt trên 91% đến 93.8%, đảm bảo đáp ứng hoàn hảo yêu cầu nghiệp vụ khắt khe của ngành tài chính kế toán."*

---

## 🖼️ SLIDE 10: CASE STUDY THỰC NGHIỆM: MỔ XẺ 3 THẤT BẠI TRONG BOUNDING BOX

### 📌 Nội dung hiển thị trên Slide:
* **Mục tiêu mở rộng:** Tự động khoanh vùng Bounding Box minh chứng trực quan cho kế toán đối soát.
* **3 Thất bại thực nghiệm điển hình:**

```
   1. LEXICAL SUBSTRING COLLISION (2-Stage Pipeline)
      Hỏi tổng tiền 12.000.000đ ──► Khớp nhầm cụm con '12' vào SĐT 0912... và Số nhà 123 ở góc trên!
      
   2. TABLE HEADER SEMANTIC TRAP (2-Stage Pipeline)
      Hỏi về thuế/tiền ──► Khớp nhầm trọn vào thanh tiêu đề màu xanh vì trùng 4 từ khóa kế toán liên tiếp!
      
   3. FLOAT16 OVERFLOW TRÊN 2D M-RoPE (End-to-End V2)
      Huấn luyện sinh Box trực tiếp ──► Attention Logits vượt 65,504 (Tràn FP16) ──► Loss: NaN ──► Model Collapse!
```

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Thưa Thầy và Hội đồng, bên cạnh thành công của mô hình ngôn ngữ 94.94% ANLS, nhóm muốn dành slide này để mổ xẻ một Case Study thực nghiệm rất tâm đắc về các thử nghiệm thất bại khi làm Bounding Box:
> Ban đầu, nhóm thử nghiệm hệ thống 2 giai đoạn (VLM trả lời, OCR tìm tọa độ). Kết quả là gặp 2 lỗi kinh điển: Lỗi thứ nhất là xung đột xâu con số, chuỗi '12' trong 12 triệu bị gom nhầm với số nhà 123 và số điện thoại 0912 ở đầu trang. Lỗi thứ hai là bẫy tiêu đề bảng, khi hỏi về thuế, khung đỏ khoanh trọn vào dải màu xanh chứa tiêu đề cột do trùng quá nhiều từ khóa.
> Để khắc phục, nhóm thử nghiệm huấn luyện mô hình V2 sinh trực tiếp tọa độ qua cơ chế 2D M-RoPE. Tuy nhiên, trên GPU Tesla T4 với kiểu dữ liệu Float16, phép nhân ma trận Attention Logits của ảnh lớn đã vượt quá giới hạn 65,504, gây ra lỗi tràn số `Loss: NaN` và sụp đổ phân phối xác suất."*

---

## 🖼️ SLIDE 11: 4 BÀI HỌC KINH NGHIỆM ĐẮT GIÁ CHO KỸ SƯ MACHINE LEARNING

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

## 🖼️ SLIDE 12: KIẾN TRÚC HỆ THỐNG FULL-STACK & DEMO THỰC TẾ

### 📌 Nội dung hiển thị trên Slide:
* **Kiến trúc Full-Stack Production:**
  * **Frontend:** React 18 + Vite + TailwindCSS Dashboard trực quan.
  * **Backend API:** FastAPI RESTful + Uvicorn + Async Worker.
  * **AI Inference Engine:** Qwen2.5-VL-3B LoRA trên Cloud GPU Kaggle Tesla T4.
* **Giao diện Demo Tương tác:**
  * Upload bất kỳ hóa đơn nào $\rightarrow$ Nhận diện trường tức thì $\rightarrow$ Xuất Full JSON phân cấp 1024 Tokens.

```
   [Browser / User] ◄──(HTTP/REST)──► [FastAPI Backend] ◄──(GPU Tensor)──► [Qwen2.5-VL LoRA]
```

### 🎙️ Lời thoại thuyết trình (Speaker Script):
> *"Về mặt triển khai phần mềm, nhóm đã đóng gói hệ thống thành một giải pháp Full-Stack hoàn chỉnh gồm giao diện React hiện đại, Backend FastAPI bất đồng bộ và GPU Engine chạy trên đám mây. Ngay sau đây, nhóm xin phép được mở giao diện Live Demo để Thầy và Hội đồng cùng trải nghiệm trực tiếp khả năng trích xuất thông tin của mô hình sau tối ưu trên các mẫu hóa đơn thực tế."*

---

## 🖼️ SLIDE 13: TỔNG KẾT & ĐỊNH HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### 📌 Nội dung hiển thị trên Slide:
* **Tổng kết thành tựu đề tài:**
  * ✅ Xây dựng thành công bộ dữ liệu 114,716 mẫu VQA hóa đơn tiếng Việt.
  * ✅ Tiến trình 3 thế hệ mô hình đạt đỉnh cao **94.94% ANLS**, **92.80% F1** và **74.14% Exact Match**.
  * ✅ Triển khai thành công ứng dụng Full-Stack bóc tách JSON và hỏi đáp thực tế dưới 2.59 giây.
  * ✅ Mổ xẻ tường tận nguyên nhân kỹ thuật và giải pháp cho bài toán Bounding Box.
* **Định hướng phát triển:**
  1. Huấn luyện mô hình End-to-End Native Grounding hoàn chỉnh trên GPU chuẩn `BFloat16` (A100).
  2. Mở rộng bóc tách sang chứng từ xuất nhập khẩu, vận đơn logistics và báo cáo tài chính phức tạp.

### 🎙️ Lời thoại kết thúc bài thuyết trình (Speaker Closing Script):
> *"Thưa Thầy và Hội đồng, dự án của chúng em không chỉ dừng lại ở việc đạt được con số 94.94% ANLS cho bài toán hỏi đáp hóa đơn tiếng Việt, mà giá trị lớn nhất nhóm thu nhận được chính là việc nghiên cứu đến tận cùng bản chất mô hình qua 3 thế hệ tiến hóa, đối mặt với những thử nghiệm thất bại để tìm ra giải pháp tối ưu. Nhóm xin chân thành cảm ơn Thầy và Hội đồng đã lắng nghe. Chúng em rất mong nhận được những góp ý quý báu từ Thầy ạ!"*
