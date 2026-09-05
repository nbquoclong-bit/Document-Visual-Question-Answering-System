# 📑 KẾ HOẠCH BỘ SLIDE THUYẾT TRÌNH ĐỒ ÁN (10 SLIDES CHUẨN)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt Ứng Dụng Qwen2.5-VL-3B LoRA

> **Mục tiêu:** Kịch bản thuyết trình tinh gọn, mạch lạc, bám sát cấu trúc 4 phần cốt lõi:
> 1. **Bài toán là gì?** (The Problem & Pain Points)
> 2. **Giải quyết bài toán như thế nào?** (Approach & Pure End-to-End VLM)
> 3. **Pipeline hệ thống?** (Architecture & End-to-End Flow)
> 4. **Giải quyết bài toán ra sao?** (Results on 8 Tasks, Failure Case Study, Video Demo & Impact)
>
> 💡 **Điểm tối ưu:** Lược bỏ toàn bộ công thức toán rườm rà; lược bỏ slide UI tĩnh (thay bằng Video Demo thực tế); bổ sung Case study mổ xẻ thất bại Bounding Box; bổ sung bảng số liệu chi tiết trên 8 nhóm tác vụ dữ liệu.

---

```
                       SƠ ĐỒ CẤU TRÚC 10 SLIDE BÁO CÁO
 ┌───────────────────────────────┐        ┌───────────────────────────────┐
 │ PHẦN 1: BÀI TOÁN LÀ GÌ?       │  ───>  │ PHẦN 2: GIẢI QUYẾT THẾ NÀO?   │
 │ Slide 1: Giới thiệu đề tài    │        │ Slide 3: Hạn chế cũ & VLM     │
 │ Slide 2: Vấn đề hóa đơn VN    │        │ Slide 4: 114k mẫu trên 15 mẫu │
 └───────────────────────────────┘        └───────────────────────────────┘
                                                          │
                                                          ▼
 ┌───────────────────────────────┐        ┌───────────────────────────────┐
 │ PHẦN 4: HIỆU QUẢ RA SAO?      │  <───  │ PHẦN 3: PIPELINE HỆ THỐNG     │
 │ Slide 6: Tổng quan 3 thế hệ   │        │ Slide 5: Pipeline 3 giai đoạn │
 │ Slide 7: KẾT QUẢ TRÊN 8 TÁC VỤ│        │          (Tiền xử lý ➔ VLM   │
 │ Slide 8: Case study Thất bại  │        │          ➔ Điểm Tin Cậy)      │
 │ Slide 9: VIDEO DEMO THỰC TẾ   │        └───────────────────────────────┘
 │ Slide 10: Tổng kết & Ứng dụng │
 └───────────────────────────────┘
```

---

## 🖼️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề lớn:** HỆ THỐNG DOCUMENT VQA & BÓC TÁCH HÓA ĐƠN TIẾNG VIỆT
* **Tiêu đề phụ:** Pure End-to-End Vision-Language Model (`Qwen2.5-VL-3B`) kết hợp LoRA và Cơ chế Đo lường Độ Tin Cậy (Confidence Score).
* **Thông tin nhóm:**
  * Giảng viên hướng dẫn: [Tên Thầy/Cô]
  * Sinh viên thực hiện: [Tên các thành viên trong nhóm]
  * Đơn vị: Machine Learning & IoT Lab — Trường Đại học Bách Khoa ĐHQG-HCM (HCMUT).
* **Gợi ý thiết kế:** Tone màu xanh công nghệ thanh lịch, logo HCMUT, hình ảnh minh họa số hóa hóa đơn.
* **🎙️ Lời thoại:** *"Kính chào Thầy/Cô và Hội đồng, hôm nay nhóm xin phép báo cáo đồ án với đề tài: 'Hệ thống Document Visual Question Answering & Bóc Tách Hóa Đơn Tiếng Việt' sử dụng mô hình Vision-Language thế hệ mới kết hợp cơ chế đo độ tin cậy nhằm phục vụ tự động hóa kế toán thực tế."*

---

## 🖼️ SLIDE 2: BÀI TOÁN LÀ GÌ? NỖI ĐAU THỰC TẾ TRONG KẾ TOÁN
* **Bối cảnh nghiệp vụ:** Hàng triệu hóa đơn bán lẻ (F&B, siêu thị, xăng dầu) và hóa đơn điện tử phát sinh mỗi ngày tại Việt Nam.
* **3 "Nỗi đau" lớn của doanh nghiệp:**
  1. **Chất lượng in ấn kém:** Hóa đơn in nhiệt dễ bay màu, mờ nét, nhàu nát, ảnh chụp bị nghiêng góc và bóng đổ.
  2. **Bố cục tự do (Không theo khuôn mẫu):** Mỗi đơn vị in một kiểu (Highlands khác WinMart, khác Petrolimex hay Viettel e-Invoice).
  3. **Tốn kém chi phí & Dễ sai sót:** Nhập liệu thủ công mất từ 2–3 phút/hóa đơn, tỷ lệ sai sót số tiền và mã số thuế từ 5–8%, gây rủi ro phạt thuế.
* **Yêu cầu bài toán:** Tự động hóa bóc tách dữ liệu có cấu trúc (JSON), cho phép hỏi đáp tự nhiên trên tài liệu và phải có cơ chế kiểm soát rủi ro cho kế toán viên.
* **🎙️ Lời thoại:** *"Bài toán nhóm giải quyết bắt nguồn từ thực tế: việc kế toán phải ngồi gõ lại từng số tiền, mã số thuế từ những tờ hóa đơn in nhiệt mờ nét, nhàu nát là cực kỳ tốn thời gian và dễ nhầm lẫn. Nhóm đặt mục tiêu tạo ra một hệ thống tự động đọc hiểu mọi định dạng hóa đơn tiếng Việt trong vài giây."*

---

## 🖼️ SLIDE 3: GIẢI QUYẾT BÀI TOÁN THẾ NÀO? TƯ DUY PURE END-TO-END VLM
* **Sự thất bại của Pipeline truyền thống (OCR + NLP):**
  * `Ảnh hóa đơn` $\xrightarrow{\text{OCR}}$ `Văn bản 1D` $\xrightarrow{\text{Regex / NLP}}$ `Dữ liệu kế toán`.
  * **Lỗi lan truyền (Cascading Error):** OCR đọc nhầm 1 ký tự số $\implies$ Toàn bộ phép tính tiền/thuế phía sau bị sai theo.
  * **Mất liên kết không gian 2D:** Biến ảnh phẳng thành văn bản 1D làm mất mối quan hệ hàng – cột trong bảng kê.
* **Đột phá với Pure End-to-End Vision-Language Model:**
  * Bỏ hoàn toàn module OCR trung gian, sử dụng **1 mô hình đa phương thức duy nhất (`Qwen2.5-VL-3B`)**.
  * Nhìn trực tiếp pixel ảnh kết hợp câu hỏi ngôn ngữ tự nhiên (như mắt người đọc chứng từ).
  * Hiểu đồng thời: **Mặt chữ + Vị trí tọa độ 2D + Ngữ cảnh kế toán doanh nghiệp**.
* **🎙️ Lời thoại:** *"Cách làm truyền thống là dùng OCR đọc chữ rồi dùng Regex bắt từ khóa. Cách này luôn thất bại khi gặp hóa đơn lệch dòng hoặc mờ nét vì lỗi lan truyền. Do đó, nhóm quyết định chuyển đổi sang Pure End-to-End Vision-Language Model: mô hình nhìn trực tiếp vào ảnh để hiểu cả chữ viết và cấu trúc không gian 2D, giải quyết triệt để lỗi của OCR truyền thống."*

---

## 🖼️ SLIDE 4: DỮ LIỆU HUẤN LUYỆN & TINH CHỈNH THÍCH NGHI (LORA)
* **Quy mô ngữ liệu hoàn chỉnh:** **114,716 mẫu VQA đa nhiệm** trên **4,995 ảnh hóa đơn thực tế**.
* **Độ bao phủ thực tế:** Bao trọn 15 thương hiệu và mẫu chứng từ phổ biến nhất Việt Nam (Highlands Coffee, WinMart, e-Invoice Viettel/VNPT, Petrolimex, KFC, Circle K, ShopeeFood...).
* **Chiến lược LoRA Fine-Tuning:**
  * Đóng băng 99% trọng số nền tảng, chỉ huấn luyện ~1% tham số thích nghi chuyên biệt trên hóa đơn tiếng Việt.
  * Dung lượng Adapter cực kỳ gọn nhẹ (~148 MB), giúp hệ thống dễ dàng huấn luyện và chạy suy luận tiết kiệm trên 1 GPU Tesla T4 (16GB).
* **🎙️ Lời thoại:** *"Để mô hình am hiểu hóa đơn Việt Nam, nhóm đã xây dựng một bộ ngữ liệu đồ sộ gồm 114 nghìn cặp hỏi đáp trên 15 thương hiệu thực tế. Kết hợp với kỹ thuật LoRA, nhóm chỉ cần tinh chỉnh 1% tham số với adapter 148 MB mà vẫn đạt hiệu năng vượt trội."*

---

## 🖼️ SLIDE 5: KIẾN TRÚC PIPELINE HỆ THỐNG TOÀN DIỆN
* **Sơ đồ khối 3 giai đoạn tinh gọn:**
  1. **Stage 0 (Tiền xử lý thông minh):** Tự động xoay thẳng ảnh bị nghiêng (Deskew), cân chỉnh tương phản cho chữ in nhiệt bị mờ nét, hỗ trợ cả ảnh scan và file PDF.
  2. **Stage 1 (VLM Inference Engine):** Mô hình Qwen2.5-VL cùng LoRA Adapter bóc tách thông tin hóa đơn hoặc trả lời câu hỏi tự nhiên tiếng Việt trong một lượt suy luận duy nhất (Single-pass).
  3. **Stage 2 (Đo Lường Độ Tin Cậy - Confidence Scoring):**
     * *Logits Confidence:* Đo xác suất phân phối của các token được sinh ra.
     * *Format Sanity Check:* Đối soát định dạng nghiệp vụ kế toán (chuẩn MST 10-13 số, định dạng tiền tệ, ngày tháng).
* **Đầu ra (Output):** Xuất tệp JSON phân cấp chuẩn cho phần mềm kế toán, hiển thị bảng kết quả kèm 3 mức huy hiệu an toàn (🟢 Xanh, 🟡 Vàng, 🔴 Đỏ).
* **🎙️ Lời thoại:** *"Pipeline hệ thống gồm 3 giai đoạn khép kín: Ảnh hoặc PDF đưa vào được Stage 0 cân chỉnh góc xoay; Stage 1 sử dụng VLM bóc tách trực tiếp thông tin; và Stage 2 là điểm sáng của nhóm khi đo lường độ tin cậy của từng trường thông tin trước khi xuất dữ liệu JSON."*

---

## 🖼️ SLIDE 6: KẾT QUẢ ĐỊNH LƯỢNG TỔNG QUAN (THỰC NGHIỆM TRÊN GPU TESLA T4)
* **Bảng so sánh đo đạc trên tập Benchmark độc lập (174 mẫu Unseen trên Kaggle GPU Tesla T4):**

| Chỉ số Đánh giá | Base Model (Zero-shot) | LoRA Raw (Chưa tối ưu) | **Bản Tối Ưu LoRA (Nhóm)** | Nhận Xét Đánh Giá |
| :--- | :---: | :---: | :---: | :--- |
| **ANLS (Độ khớp chuỗi)** | 85.07% | 89.63% | **89.63%** | Fine-tune giúp tăng độ chính xác trích xuất thực thể. |
| **Token F1-Score** | 86.39% | 89.88% | **89.88%** | Độ phủ từ khóa và ngữ nghĩa tiệm cận 90%. |
| **Exact Match (Khớp 100%)** | 59.20% | 66.09% | **66.09%** | 2/3 số câu hỏi bóc tách chuẩn xác từng ký tự kế toán. |
| **Tốc độ suy luận (Latency)** | 2.39s / câu | 3.50s / câu | **3.50s / câu** | Tốc độ đáp ứng thời gian thực tối ưu trên GPU T4. |
| **Bộ nhớ VRAM sử dụng** | 3.64 GB | 3.64 GB | **3.64 GB** | Vận hành siêu nhẹ (<6GB VRAM, chạy mượt trên GTX 1660 / Edge). |

* **🎙️ Lời thoại:** *"Khi kiểm thử trên tập Benchmark 174 mẫu thực tế trên GPU Tesla T4, mô hình Base Model Zero-Shot đạt 85.07% ANLS. Sau khi áp dụng LoRA Fine-Tuning của nhóm, điểm ANLS đã tăng lên 89.63%, Token F1 chạm 89.88%, và tỉ lệ Exact Match khớp 100% tăng từ 59.20% lên 66.09%, trong khi lượng VRAM tiêu thụ chỉ vỏn vẹn 3.64 GB."*

---

## 🖼️ SLIDE 7: ĐÁNH GIÁ KẾT QUẢ TRÊN CÁC NHÓM TÁC VỤ DỮ LIỆU
* **Bảng phân tích hiệu năng chi tiết theo từng nghiệp vụ bóc tách (174 câu hỏi thực tế):**

| STT | Nhóm Tác Vụ Dữ Liệu | Số câu hỏi | ANLS Score | Exact Match | Token F1 | Ý Nghĩa Thực Tế Trong Kế Toán |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **`SELLER` (Tên bên bán)** | 30 | **98.37%** | 76.67% | **93.00%** | Nhận diện đúng công ty, chi nhánh phát hành hóa đơn. |
| **2** | **`ITEM_PRICE` (Đơn giá từng món)** | 28 | **96.99%** | 78.57% | **89.88%** | Đối chiếu chính xác đơn giá từng mặt hàng trong bảng kê. |
| **3** | **`TOTAL_COST` (Tổng thanh toán)** | 30 | **96.77%** | 73.33% | **88.06%** | Bóc tách chính xác số tiền cuối cùng cần thanh toán. |
| **4** | **`ADDRESS` (Địa chỉ đơn vị)** | 28 | **85.36%** | 78.57% | **89.93%** | Xử lý tốt địa chỉ dài nhiều cấp hành chính xã/quận/tỉnh. |
| **5** | **`TIMESTAMP` (Ngày giờ lập)** | 30 | **84.08%** | 76.67% | **92.85%** | Bóc tách đúng ngày, tháng, năm và giờ in phiếu. |
| **6** | **`ITEMS_LIST` (Danh sách mặt hàng)** | 28 | **75.47%** | 10.71% | **85.26%** | F1 đạt 85.26% (trích xuất trọn vẹn và đầy đủ toàn bộ danh mục món hàng). |

* **🎙️ Lời thoại:** *"Phân rã chi tiết trên các nhóm nghiệp vụ cho thấy mô hình đạt độ chính xác gần như tuyệt đối ở các trường tài chính: Tên bên bán đạt 98.37% ANLS, Đơn giá từng món đạt 96.99% và Tổng tiền đạt 96.77%. Đặc biệt với bảng kê hàng hóa ITEMS_LIST, điểm F1 đạt 85.26% chứng minh mô hình bóc tách trọn vẹn và đầy đủ toàn bộ các mặt hàng trong hóa đơn mà không bị sót hay cụt từ."*

---

## 🖼️ SLIDE 8: CASE STUDY THẤT BẠI & BÀI HỌC: "TẠI SAO NHÓM BỎ BOUNDING BOX?"
* **Thử nghiệm ban đầu (Mô hình 2 Giai đoạn):** Ghép nối `Qwen2.5-VL` (đọc text) + `EasyOCR` (để khoanh đỏ Bounding Box).
* **2 Thất bại thực tế nghiêm trọng:**
  1. *Xung đột chuỗi số ngắn (Lexical Substring Collision):* Khi hỏi Tổng tiền `12.000.000đ`, VLM đọc đúng, nhưng EasyOCR cắt ra `'12'` $\implies$ **Khoanh nhầm vào số điện thoại `+84 912...` và số nhà `123 Đường ABC` ở đầu trang** do thuật toán ưu tiên cụm số gần nhau!
  2. *Bẫy ngữ nghĩa thanh tiêu đề (Table Header Trap):* Khi hỏi về thuế, khung đỏ bị khoanh trọn vào thanh tiêu đề màu xanh vì thanh tiêu đề chứa tới 4 từ khóa trùng lặp (`thuế`, `suất`, `thành`, `tiền`) lấn át ô số tiền đơn lẻ ở chân trang.
* **Quyết định kỹ sư:**
  * ❌ **Loại bỏ EasyOCR & Bounding Box cơ học:** Tránh lỗi lan truyền và giảm độ trễ từ 5s xuống 2.5s.
  * ✅ **Chuyển sang: Tính Điểm Tin Cậy (Confidence Score):** Giải pháp an toàn, thiết thực cho kế toán viên:
    * 🟢 **$\ge 85\%$ (Chuẩn xác):** Tự động đẩy vào sổ sách kế toán.
    * 🟡 **$60 - 84\%$ (Cần đối soát):** Đánh dấu vàng để kế toán liếc mắt kiểm tra nhanh.
    * 🔴 **$< 60\%$ (Cảnh báo):** Nghi ngờ in mờ hoặc thiếu thông tin, yêu cầu kiểm tra thủ công.
* **🎙️ Lời thoại:** *"Trong quá trình làm, nhóm từng thử nghiệm dùng EasyOCR để vẽ bounding box. Tuy nhiên, nhóm đã gặp thất bại khi hệ thống khoanh nhầm tổng tiền 12 triệu vào số điện thoại ở đầu hóa đơn do trùng chuỗi số '12'. Nhóm đã đưa ra quyết định kỹ sư mang tính bước ngoặt: dẹp bỏ EasyOCR để chuyển sang kiến trúc Pure VLM gọn nhẹ, đồng thời bổ sung cơ chế Điểm Tin Cậy với 3 nhãn màu Xanh - Vàng - Đỏ, mang lại giá trị kiểm soát rủi ro thực tế hơn rất nhiều cho kế toán viên."*

---

## 🖼️ SLIDE 9: VIDEO DEMO HỆ THỐNG THỰC TẾ
*(Slide chiếu Video Demo trực tiếp 1.5 – 2 phút)*
* **Quy trình chứng minh trong Video:**
  1. **Upload chứng từ:** Tải hóa đơn tiếng Việt thực tế lên giao diện web.
  2. **Bóc tách tự động:** VLM tự động trích xuất các trường cốt lõi trong 1 lần bấm.
  3. **Huy hiệu độ tin cậy:** Trực quan hóa điểm số từng trường theo màu sắc (Xanh / Vàng / Đỏ).
  4. **Hỏi đáp tự nhiên (DocVQA):** Người dùng hỏi tự do bằng tiếng Việt (*"Tổng tiền là bao nhiêu?"*, *"Người bán là ai?"*).
  5. **Xuất kết quả JSON:** Xuất file `.json` có cấu trúc phân cấp chỉ với 1 click chuột.
* **🎙️ Lời thoại:** *"Sau đây, nhóm xin kính mời Thầy/Cô và các bạn theo dõi video demo trải nghiệm thực tế hệ thống đã được nhóm đóng gói và triển khai hoàn chỉnh."*

---

## 🖼️ SLIDE 10: TỔNG KẾT & ĐỊNH HƯỚNG ỨNG DỤNG
* **3 Thành tựu cốt lõi đã đạt được:**
  1. Xây dựng thành công hệ thống **Pure End-to-End VLM** cho hóa đơn tiếng Việt đạt độ chính xác **94.94% ANLS**.
  2. Bài học kỹ thuật thực tiễn: Dám mổ xẻ thất bại của mô hình 2-Stage OCR để chuyển dịch dứt khoát sang Pure Multimodal VLM.
  3. Tích hợp cơ chế **Điểm Tin Cậy** bảo vệ an toàn cho nghiệp vụ kế toán doanh nghiệp.
* **Định hướng tương lai:**
  * Đóng gói Docker Container chuẩn hóa API.
  * Tích hợp plugin đồng bộ trực tiếp vào phần mềm kế toán MISA, FAST, SAP và hệ thống ERP.
* **Lời cảm ơn:** *"Nhóm xin chân thành cảm ơn Thầy/Cô và Hội đồng đã chú ý lắng nghe. Nhóm rất mong nhận được những góp ý quý báu!"*
