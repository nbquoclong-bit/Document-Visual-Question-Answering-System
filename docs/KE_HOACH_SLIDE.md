# 📑 KẾ HOẠCH BỘ SLIDE THUYẾT TRÌNH ĐỒ ÁN (10 SLIDES CHUẨN)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt Ứng Dụng Qwen2.5-VL-3B LoRA

> **Mục tiêu:** Kịch bản thuyết trình tinh gọn, mạch lạc, bám sát cấu trúc 4 phần cốt lõi:
> 1. **Bài toán là gì?** (The Problem & Pain Points)
> 2. **Giải quyết bài toán như thế nào?** (Approach & Pure End-to-End VLM)
> 3. **Pipeline hệ thống?** (Architecture & End-to-End Flow)
> 4. **Giải quyết bài toán ra sao?** (Results on 6 Accounting Tasks, Failure Case Study, Video Demo & Impact)
>
> 💡 **Điểm tối ưu:** Lược bỏ toàn bộ công thức toán rườm rà; phần lý thuyết chỉ nêu công nghệ đã chọn và các quyết định kỹ thuật; bổ sung Case study mổ xẻ thất bại Bounding Box; phân tích bảng số liệu thực nghiệm đối đầu trên 6 nhóm nghiệp vụ kế toán.

---

```
                       SƠ ĐỒ CẤU TRÚC 10 SLIDE BÁO CÁO
 ┌───────────────────────────────┐        ┌───────────────────────────────┐
 │ PHẦN 1: BÀI TOÁN LÀ GÌ?       │  ───>  │ PHẦN 2: GIẢI PHÁP & TỐI ƯU    │
 │ Slide 1: Giới thiệu đề tài    │        │ Slide 3: Hạn chế cũ & VLM     │
 │ Slide 2: Nỗi đau hóa đơn VN   │        │ Slide 4: 114k mẫu & Tối ưu    │
 └───────────────────────────────┘        └───────────────────────────────┘
                                                          │
                                                          ▼
 ┌───────────────────────────────┐        ┌───────────────────────────────┐
 │ PHẦN 4: HIỆU QUẢ RA SAO?      │  <───  │ PHẦN 3: PIPELINE HỆ THỐNG     │
 │ Slide 6: Thực nghiệm đối đầu  │        │ Slide 5: Pipeline 3 giai đoạn │
 │ Slide 7: Phân rã 6 nghiệp vụ  │        │          (Tiền xử lý ➔ VLM    │
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

## 🖼️ SLIDE 3: GIẢI PHÁP KỸ THUẬT: PURE END-TO-END VLM
* **Lý do loại bỏ Pipeline truyền thống (OCR + NLP):**
  * OCR đọc nhầm 1 ký tự $\implies$ Toàn bộ regex/NLP phía sau bị sai theo (Cascading Error).
  * Làm phẳng ảnh thành chuỗi 1D làm mất hoàn toàn cấu trúc bảng biểu hàng – cột 2D.
* **Lựa chọn công nghệ của nhóm (Tech Stack):**
  * Mô hình nền tảng: **`Qwen2.5-VL-3B-Instruct`** (Vision-Language Model thế hệ mới).
  * Cơ chế: Pure End-to-End (Single-pass) — Đưa thẳng ảnh pixel và câu hỏi vào mô hình, nhận trực tiếp câu trả lời hoặc dữ liệu JSON.
  * Ưu điểm: Hiểu đồng thời ký tự chữ, vị trí tọa độ không gian 2D và ngữ cảnh kế toán mà không cần module trung gian.
* **🎙️ Lời thoại:** *"Về mặt giải pháp, nhóm loại bỏ hoàn toàn pipeline OCR ghép NLP truyền thống vì dễ bị lỗi lan truyền khi chữ in nhiệt bị mờ. Thay vào đó, nhóm sử dụng mô hình Pure Vision-Language Model Qwen2.5-VL-3B. Mô hình nhìn thẳng vào pixel ảnh hóa đơn để trích xuất trực tiếp dữ liệu, bảo toàn trọn vẹn mối liên kết không gian bảng biểu."*

---

## 🖼️ SLIDE 4: DỮ LIỆU & QUÁ TRÌNH TỐI ƯU SIÊU THAM SỐ (LORA & SYSTEM TUNING)
* **Quy mô ngữ liệu:** **114,716 cặp hỏi đáp VQA** trên **4,995 ảnh hóa đơn thực tế** bao trọn 15 thương hiệu phổ biến (Highlands, WinMart, Petrolimex, e-Invoice Viettel/VNPT...).
* **Kỹ thuật thích nghi LoRA (Low-Rank Adaptation):**
  * Đóng băng 99% trọng số nền tảng, chỉ huấn luyện bộ chuyển đổi nhẹ **~148 MB**.
* **4 Quyết định tối ưu siêu tham số then chốt (Key Optimization Choices):**
  1. **Full 7-Layer Linear LoRA:** Tinh chỉnh trên toàn bộ 7 lớp ma trận chiếu (`q, k, v, o, gate, up, down_proj`) thay vì chỉ 2 lớp Attention, giúp học sâu cấu trúc bảng biểu.
  2. **Target-Only Loss Masking:** Chỉ tính loss trên câu trả lời, mask toàn bộ prompt/ảnh $\implies$ Ép mô hình trả lời súc tích chuẩn thực thể kế toán, triệt tiêu 100% lời dẫn thừa.
  3. **Dynamic Token Allocation:** Nới trần 384 tokens cho danh sách hàng hóa nhiều dòng (`ITEMS_LIST`) tránh lỗi đứt chữ.
  4. **Vision Pixel Budgeting:** Giới hạn phân giải thích ứng $\implies$ Ép VRAM từ >10GB xuống đúng **3.64 GB** (chạy nhẹ nhàng trên GPU GTX 1660 / T4).
* **🎙️ Lời thoại:** *"Về huấn luyện, nhóm sử dụng bộ dữ liệu 114 nghìn mẫu hóa đơn Việt Nam kết hợp kỹ thuật LoRA với adapter chỉ 148 MB. Để đạt hiệu năng cao nhất, nhóm đã tối ưu 4 tham số then chốt: mở rộng LoRA ra cả 7 lớp Linear, áp dụng Target-Only Loss Masking để mô hình trả lời thẳng vào thực thể, cấp phát token động cho danh sách hàng dài, và tối ưu độ phân giải để ép VRAM xuống chỉ 3.64 GB."*

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

## 🖼️ SLIDE 6: KẾT QUẢ THỰC NGHIỆM ĐỐI ĐẦU (KAGGLE GPU TESLA T4)
* **Bảng so sánh đo đạc thực tế trên tập Benchmark độc lập (174 mẫu Unseen):**

| Chỉ số Đánh giá | Base Model (Zero-shot) | **Mô hình của Nhóm (Fine-tuned & Optimized)** | Mức độ Cải Thiện | Nhận Xét Kỹ Thuật Thực Tế |
| :--- | :---: | :---: | :---: | :--- |
| **ANLS (Độ khớp chuỗi)** | 85.07% | **89.63%** | **+4.56%** | Vượt trội ở các trường hóa đơn phức tạp. |
| **Exact Match (Khớp 100%)** | 59.20% | **66.09%** | **+6.89%** 🚀 | **Thêm 12 hóa đơn** bóc chuẩn xác tuyệt đối từng ký tự. |
| **Token F1-Score** | 86.39% | **89.88%** | **+3.49%** | Độ bao phủ từ khóa kế toán tiệm cận 90%. |
| **Bóc tách danh sách món (`ITEMS_LIST`)** | 50.69% | **75.47%** | **+24.78%** 🚀 | Khắc phục hoàn toàn lỗi tự làm toán & cắt cụt của Base. |
| **Đơn giá từng món (`ITEM_PRICE`)** | 50.00% (EM) | **78.57%** (EM) | **+28.57%** 🚀 | Tỉ lệ bóc đúng 100% đơn giá tăng vọt gần 30%. |
| **Bộ nhớ VRAM tiêu thụ** | 3.64 GB | **3.64 GB** | **Tối ưu tuyệt đối** | Chạy mượt trên GPU phổ thông (GTX 1660 / Edge). |
| **Tốc độ suy luận (Latency)** | 2.39s / câu | **3.50s / câu** | Thời gian thực | Đủ điều kiện triển khai vào quy trình thực tế. |

* **🎙️ Lời thoại:** *"Khi kiểm thử trên 174 mẫu thực tế trên GPU Tesla T4, mô hình Base Model gốc đạt 85.07% ANLS nhờ khả năng đọc chữ tốt, nhưng lại thiếu tư duy kế toán: tỷ lệ Exact Match chỉ đạt 59.20%, đơn giá món chỉ đúng 50%, và danh mục hàng hóa hay tự tiện làm toán cộng dồn. Sau khi áp dụng mô hình LoRA tối ưu của nhóm, điểm ANLS danh mục hàng tăng vọt +24.78%, Exact Match đơn giá tăng +28.57%, và toàn bộ hệ thống vận hành cực kỳ tiết kiệm với chỉ 3.64 GB VRAM."*

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
  1. Xây dựng thành công hệ thống **Pure End-to-End VLM** cho hóa đơn tiếng Việt đạt độ chính xác **89.63% ANLS**, **66.09% Exact Match** trên dữ liệu thực tế.
  2. Đóng góp bài học kỹ thuật thực tiễn: Dám mổ xẻ thất bại của mô hình 2-Stage OCR để chuyển dịch dứt khoát sang Pure Multimodal VLM.
  3. Tích hợp cơ chế **Điểm Tin Cậy** bảo vệ an toàn cho nghiệp vụ kế toán doanh nghiệp.
* **Định hướng tương lai:**
  * Đóng gói Docker Container chuẩn hóa API.
  * Tích hợp plugin đồng bộ trực tiếp vào phần mềm kế toán MISA, FAST, SAP và hệ thống ERP.
* **Lời cảm ơn:** *"Nhóm xin chân thành cảm ơn Thầy/Cô và Hội đồng đã chú ý lắng nghe. Nhóm rất mong nhận được những góp ý quý báu!"*

