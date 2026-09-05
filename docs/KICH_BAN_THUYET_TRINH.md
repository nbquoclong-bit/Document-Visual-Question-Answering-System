# 🎙️ KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (FULL SCRIPT 10 SLIDES)
## Đề Tài: Hệ Thống Document Visual Question Answering & Bóc Tách Hóa Đơn Tiếng Việt Ứng Dụng Qwen2.5-VL-3B LoRA
* **Đơn vị:** Machine Learning & IoT Lab — Đại học Bách Khoa ĐHQG-HCM
* **Tổng thời lượng khuyến nghị:** 8 – 10 phút (Bao gồm 1.5 – 2 phút Video Demo)
* **Phong cách trình bày:** Tự tin, gãy gọn, phong thái kỹ sư thực chiến; **tuyệt đối không giảng giải lý thuyết dông dài**, tập trung vào **nỗi đau thực tế, quyết định kỹ thuật, tối ưu hóa tham số và kết quả đo đạc thực nghiệm**.

---

## ⏱️ PHÂN BỔ THỜI LƯỢNG TIÊU CHUẨN

| Thứ Tự | Tên Slide | Nội Dung Cốt Lõi | Thời Lượng |
| :---: | :--- | :--- | :---: |
| **Slide 1** | Trang Tiêu Đề | Giới thiệu đề tài, giảng viên và nhóm thực hiện | **30s** |
| **Slide 2** | Nỗi Đau Thực Tế | 3 khó khăn lớn của hóa đơn kế toán Việt Nam | **60s** |
| **Slide 3** | Giải Pháp Kỹ Thuật | Tại sao bỏ OCR cũ để chuyển sang Pure End-to-End VLM | **60s** |
| **Slide 4** | Dữ Liệu & Tối Ưu LoRA | 114k mẫu & 4 quyết định tối ưu siêu tham số then chốt | **75s** |
| **Slide 5** | Pipeline Hệ Thống | 3 giai đoạn: Tiền xử lý ➔ VLM ➔ Điểm Tin Cậy Xanh/Vàng/Đỏ | **60s** |
| **Slide 6** | Thực Nghiệm Đối Đầu | Base Model vs Mô hình của nhóm trên GPU Tesla T4 | **90s** |
| **Slide 7** | Phân Rã 6 Nghiệp Vụ | Đơn giá (+28%), Danh mục hàng (+24%), Tên bên bán (98%) | **60s** |
| **Slide 8** | Case Study Thất Bại | Tại sao nhóm bỏ Bounding Box / EasyOCR để sang Điểm Tin Cậy | **75s** |
| **Slide 9** | Video Demo Thực Tế | Trình diễn video trích xuất tự động & hỏi đáp DocVQA | **90s – 120s** |
| **Slide 10**| Tổng Kết & Định Hướng | Đóng gói API, tích hợp ERP MISA/SAP, kết thúc bài báo cáo | **45s** |

---

## 📑 CHI TIẾT LỜI THOẠI TỪNG SLIDE (WORD-FOR-WORD SCRIPT)

### 🖼️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Thời lượng:** `00:00 - 00:30` (30 giây)
* **Trọng tâm:** Chào hỏi trang trọng, nêu bật tính ứng dụng thực tế của đề tài.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Kính chào Thầy/Cô Chủ tịch Hội đồng và quý Thầy/Cô phản biện.*  
> 
> *Hôm nay, nhóm chúng em xin phép được báo cáo đề tài: **'Hệ thống Document Visual Question Answering và Bóc Tách Hóa Đơn Tiếng Việt'**, ứng dụng mô hình Vision-Language thế hệ mới **Qwen2.5-VL-3B** kết hợp tinh chỉnh **LoRA** và cơ chế **Đo lường Độ Tin Cậy cho nghiệp vụ kế toán**.*  
> 
> *Đề tài được thực hiện dưới sự hướng dẫn của Thầy/Cô [Tên GVHD], cùng các thành viên trong nhóm chúng em. Sau đây em xin đại diện nhóm bắt đầu phần trình bày."*

* 💡 *Mẹo trình bày:* Đứng thẳng, mắt nhìn bao quát toàn thể hội đồng, giọng mở đầu rõ ràng, dứt khoát.

---

### 🖼️ SLIDE 2: BÀI TOÁN LÀ GÌ? NỖI ĐAU THỰC TẾ TRONG KẾ TOÁN
* **Thời lượng:** `00:30 - 01:30` (60 giây)
* **Trọng tâm:** 3 nỗi đau kinh điển khi xử lý chứng từ kế toán Việt Nam.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Thưa Thầy/Cô, mỗi ngày tại các doanh nghiệp Việt Nam có hàng trăm nghìn hóa đơn bán lẻ và chứng từ điện tử phát sinh. Tuy nhiên, quy trình nhập liệu hiện nay vẫn đang đối mặt với **3 nút thắt rất lớn**:*
> 
> * **Thứ nhất là chất lượng in ấn:** Hóa đơn in nhiệt tại các chuỗi siêu thị, nhà hàng rất dễ bay màu, mờ nét, hoặc ảnh chụp từ điện thoại thường xuyên bị nghiêng góc và dính bóng đổ.
> * **Thứ hai là bố cục tự do:** Hóa đơn không tuân theo một mẫu cố định nào. Hóa đơn Highlands khác WinMart, khác cây xăng Petrolimex, hay các định dạng e-Invoice của Viettel, VNPT.
> * **Thứ ba là rủi ro nghiệp vụ:** Nhập liệu thủ công mất từ 2 đến 3 phút mỗi tờ, và tỷ lệ gõ nhầm số tiền hay mã số thuế lên tới 5 đến 8%, gây rủi ro phạt thuế rất nghiêm trọng cho doanh nghiệp.
> 
> *Bài toán nhóm đặt ra là: **Làm sao để một hệ thống AI có thể tự động bóc tách chuẩn xác toàn bộ dữ liệu này ra định dạng JSON trong vài giây, đồng thời cho phép kế toán viên đặt câu hỏi tự nhiên bằng tiếng Việt để đối soát ngay lập tức?**"*

* 💡 *Mẹo trình bày:* Nhấn giọng ở các con số thực tế: *"2 đến 3 phút"*, *"5 đến 8% sai sót"*.

---

### 🖼️ SLIDE 3: GIẢI PHÁP KỸ THUẬT: PURE END-TO-END VLM
* **Thời lượng:** `01:30 - 02:30` (60 giây)
* **Trọng tâm:** Lý do loại bỏ OCR truyền thống; chọn Pure VLM. **Không giảng giải lý thuyết cơ bản**.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Để giải bài toán này, cách tiếp cận truyền thống là dùng một module OCR đọc chữ ra rồi dùng NLP hoặc Regex để bóc tách. Nhưng nhóm **quyết định loại bỏ hoàn toàn đường ống cũ này vì 2 lý do chí mạng**:*
> 
> * **Một là Lỗi lan truyền (Cascading Error):** Nếu OCR đọc nhầm 1 ký tự số mờ nét, thì toàn bộ chuỗi Regex phía sau sẽ sai dây chuyền.
> * **Hai là Mất liên kết không gian 2D:** Biến ảnh phẳng thành một chuỗi chữ 1D sẽ phá vỡ hoàn toàn mối quan hệ hàng và cột trong các bảng kê hàng hóa.
> 
> *Thay vào đó, giải pháp của nhóm là **Pure End-to-End Vision-Language Model** với kiến trúc nền tảng **Qwen2.5-VL-3B**.  
> Cơ chế này đưa thẳng pixel ảnh hóa đơn vào mô hình trong một lượt suy luận duy nhất (Single-pass). Mô hình đóng vai trò như mắt người kế toán: **nhìn trực tiếp mặt chữ, hiểu vị trí tọa độ 2D và hiểu ngữ cảnh tài chính cùng một lúc** mà không cần bất kỳ module trung gian nào."*

* 💡 *Mẹo trình bày:* Chỉ tay vào sơ đồ chuyển đổi trên slide từ `OCR + NLP (Lỗi lan truyền)` sang `Pure End-to-End (Single-pass)`.

---

### 🖼️ SLIDE 4: DỮ LIỆU & QUÁ TRÌNH TỐI ƯU SIÊU THAM SỐ (LORA & TUNING)
* **Thời lượng:** `02:30 - 03:45` (75 giây)
* **Trọng tâm:** Ngữ liệu 114k & 4 quyết định tối ưu tham số mang lại giá trị kỹ thuật cốt lõi.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Để mô hình am hiểu chứng từ Việt Nam, nhóm đã xây dựng một bộ ngữ liệu đồ sộ gồm **114,716 cặp hỏi đáp VQA** trên gần **5,000 ảnh hóa đơn thực tế**, bao phủ 15 chuỗi bán lẻ và nhà cung cấp phổ biến nhất.
> 
> *Nhóm sử dụng kỹ thuật **LoRA (Low-Rank Adaptation)**: đóng băng 99% trọng số nền tảng và chỉ huấn luyện một bộ Adapter gọn nhẹ **148 MB**.  
> Đặc biệt, để đạt hiệu năng tối đa, nhóm đã đưa ra **4 quyết định tối ưu tham số then chốt**:*
> 
> 1. **Mở rộng LoRA ra toàn bộ 7 lớp ma trận Linear:** Thay vì chỉ fine-tune 2 lớp Attention thông thường, nhóm nhắm vào cả 7 lớp chiếu bao gồm cả khối MLP (`gate, up, down_proj`), giúp mô hình ghi nhớ sâu cấu trúc bảng biểu.
> 2. **Kỹ thuật Target-Only Loss Masking:** Nhóm đóng băng gradient của câu hỏi và ảnh, **chỉ tính 100% loss trên câu trả lời kế toán**. Kỹ thuật này ép mô hình trả lời thẳng vào thực thể, loại bỏ hoàn toàn tật nói dài rườm rà.
> 3. **Cấp phát Token Động (Dynamic Token Budgeting):** Cấp trần 384 tokens cho danh mục hàng hóa nhiều dòng thay vì để cố định 96 tokens, giải quyết triệt để lỗi đứt chữ.
> 4. **Tối ưu dải phân giải (Vision Pixel Budgeting):** Ép mức tiêu thụ VRAM từ trên 10GB xuống đúng **3.64 GB**, giúp hệ thống chạy mượt trên cả GPU tầm trung như GTX 1660."*

* 💡 *Mẹo trình bày:* Nói dứt khoát 4 điểm nhấn, thể hiện rõ tư duy thực nghiệm kỹ sư (Ablation Study).

---

### 🖼️ SLIDE 5: KIẾN TRÚC PIPELINE HỆ THỐNG TOÀN DIỆN
* **Thời lượng:** `03:45 - 04:45` (60 giây)
* **Trọng tâm:** 3 giai đoạn khép kín và cơ chế đo độ tin cậy.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Toàn bộ hệ thống được nhóm đóng gói thành một Pipeline khép kín gồm **3 giai đoạn tinh gọn**:*
> 
> * **Giai đoạn 0 - Tiền xử lý ảnh thông minh:** Khi kế toán viên upload ảnh chụp bị nghiêng hoặc file PDF, hệ thống tự động xoay thẳng góc (Deskew), tăng cường độ tương phản cho hóa đơn in nhiệt bị mờ và chuẩn hóa kích thước pixel.
> * **Giai đoạn 1 - VLM Inference Engine:** Mô hình Qwen2.5-VL kết hợp LoRA Adapter tiến hành trích xuất toàn bộ trường dữ liệu hoặc trả lời câu hỏi tự nhiên bằng tiếng Việt chỉ trong một lần suy luận.
> * **Giai đoạn 2 - Đo lường Độ Tin Cậy (Confidence Scoring):** Đây là chốt chặn an toàn của nhóm:
>   * Hệ thống đo xác suất Logits của từng token được sinh ra.
>   * Kết hợp đối soát logic nghiệp vụ (*Format Sanity Check*) như kiểm tra mã số thuế 10-13 số hay định dạng tiền tệ.
> 
> *Kết quả xuất ra bảng trực quan kèm 3 nhãn màu an toàn: **🟢 Xanh (chính xác tuyệt đối), 🟡 Vàng (cần liếc mắt kiểm tra), và 🔴 Đỏ (cảnh báo nghi ngờ mờ nét)**, sẵn sàng xuất file JSON cho phần mềm kế toán."*

* 💡 *Mẹo trình bày:* Nhấn mạnh giá trị của 3 màu huy hiệu Xanh - Vàng - Đỏ trong việc bảo vệ an toàn cho kế toán viên.

---

### 🖼️ SLIDE 6: KẾT QUẢ THỰC NGHIỆM ĐỐI ĐẦU (KAGGLE GPU TESLA T4)
* **Thời lượng:** `04:45 - 06:15` (90 giây)
* **Trọng tâm:** Slide đinh của bài báo cáo! Đối đầu Base vs Nhóm, nhấn mạnh con số thực tế.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Đây là bảng kết quả thực nghiệm khách quan được nhóm đo đạc độc lập trên **174 mẫu hóa đơn kiểm định Unseen** chạy trực tiếp trên GPU Tesla T4:*
> 
> * Nhìn vào bảng, ta thấy mô hình **Base Model gốc** đạt **85.07% ANLS** vì khả năng đọc chữ tổng quát của Qwen2.5 rất tốt. **Tuy nhiên, Base Model hoàn toàn thiếu tư duy nghiệp vụ kế toán**:
>   * Tỷ lệ khớp tuyệt đối Exact Match chỉ đạt **59.20%**.
>   * Ở tác vụ bóc tách danh mục món hàng `ITEMS_LIST`, Base Model tự tiện tính toán cộng dồn giá và bị đứt chữ, khiến điểm ANLS chỉ đạt **50.69%** và Exact Match bằng **0%**.
>   * Ở đơn giá từng món `ITEM_PRICE`, Base Model chỉ đoán đúng **50%**.
> 
> *Sau khi áp dụng mô hình LoRA tối ưu của nhóm:
>   * Điểm **ANLS tổng thể tăng lên 89.63%**, Token F1 chạm **89.88%**.
>   * Tỷ lệ khớp tuyệt đối **Exact Match tăng vọt lên 66.09%** — tức là có thêm hơn 12 tờ hóa đơn bóc trúng tuyệt đối 100% từng ký tự.
>   * Đặc biệt, điểm ANLS danh mục hàng hóa **tăng vọt +24.78% (đạt 75.47%)**, và Exact Match đơn giá từng món **tăng vọt +28.57% (đạt 78.57%)**.
>   * Toàn bộ quá trình suy luận chỉ tiêu tốn **3.64 GB VRAM** với độ trễ chỉ **3.5 giây/câu hỏi**, hoàn toàn đủ điều kiện triển khai vào môi trường thực tế."*

* 💡 *Mẹo trình bày:* Dùng laser/tay chỉ rõ 3 mức tăng trưởng đột phá: `+6.89% Exact Match`, `+24.78% ITEMS_LIST`, `+28.57% ITEM_PRICE`.

---

### 🖼️ SLIDE 7: ĐÁNH GIÁ KẾT QUẢ TRÊN CÁC NHÓM TÁC VỤ DỮ LIỆU
* **Thời lượng:** `06:15 - 07:15` (60 giây)
* **Trọng tâm:** Phân rã độ chính xác trên 6 trường kế toán thực tế.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Đi sâu vào phân rã chi tiết trên 6 nhóm nghiệp vụ bóc tách tài chính, kết quả cho thấy mô hình của nhóm đạt độ chính xác gần như tuyệt đối ở các trường thông tin tiền tệ quan trọng:
> 
> * **Tên bên bán (`SELLER`):** Đạt **98.37% ANLS**, nhận diện đúng công ty và chuỗi cửa hàng.
> * **Đơn giá từng món (`ITEM_PRICE`):** Đạt **96.99% ANLS** và **78.57% Exact Match**, giúp đối soát đơn giá từng dòng hàng.
> * **Tổng tiền thanh toán (`TOTAL_COST`):** Đạt **96.77% ANLS** và Token F1 **88.06%**, bắt trọn số tiền thanh toán cuối cùng.
> * **Địa chỉ đơn vị (`ADDRESS`):** Đạt **85.36% ANLS**, xử lý chuẩn xác các địa chỉ dài gồm 4 cấp hành chính xã, huyện, tỉnh.
> * **Ngày giờ lập (`TIMESTAMP`):** Đạt **84.08% ANLS**, bóc đúng cả ngày tháng và giờ in phiếu.
> * Và cuối cùng, với bảng kê nhiều dòng **`ITEMS_LIST`**, điểm F1 đạt **85.26%**, minh chứng mô hình đã bắt trọn danh sách món hàng mà không bị bỏ sót."*

* 💡 *Mẹo trình bày:* Đọc lướt nhanh các trường điểm cao (>96%) và nhấn mạnh vào giá trị nghiệp vụ kế toán của từng trường.

---

### 🖼️ SLIDE 8: CASE STUDY THẤT BẠI: "TẠI SAO NHÓM BỎ BOUNDING BOX?"
* **Thời lượng:** `07:15 - 08:30` (75 giây)
* **Trọng tâm:** Câu chuyện kỹ thuật đắt giá nhất bài báo cáo. Giảng viên luôn đánh giá cao sinh viên dám phân tích lỗi sai thực tế.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Thưa Thầy/Cô, một trong những bài học kỹ thuật giá trị nhất của nhóm trong đồ án này là: **Tại sao nhóm dẹp bỏ EasyOCR và Bounding Box cơ học?**
> 
> *Ban đầu, nhóm cũng ghép nối mô hình theo cách cổ điển: dùng Qwen2.5-VL đọc câu trả lời rồi dùng EasyOCR quét lại ảnh để khoanh khung đỏ Bounding Box. Nhưng khi thử nghiệm trên hóa đơn thực tế, hệ thống gặp **2 thất bại rất nặng**:*
> 
> 1. **Thất bại 1 - Xung đột chuỗi số ngắn:** Khi hỏi tổng tiền hóa đơn là `12.000.000đ`, VLM đọc đúng, nhưng EasyOCR chỉ cắt được cụm số `'12'`. Kết quả là thuật toán khoanh nhầm vào **số điện thoại `+84 912...` và số nhà `123 Đường ABC` ở đầu trang** vì chuỗi số giống nhau!
> 2. **Thất bại 2 - Bẫy thanh tiêu đề:** Khi hỏi về thuế suất, khung đỏ bị hút chặt vào thanh tiêu đề màu xanh vì thanh tiêu đề chứa tới 4 từ khóa trùng lặp (`thuế, suất, thành, tiền`), lấn át con số thực sự ở chân trang.
> 
> *Trước thất bại này, nhóm đã đưa ra **quyết định kỹ sư dứt khoát**: Loại bỏ EasyOCR để tránh lỗi lan truyền và giảm độ trễ từ 5 giây xuống 3.5 giây. Đồng thời, nhóm chuyển hướng sang **tính Điểm Tin Cậy (Confidence Score)**.  
> Đối với kế toán viên, việc biết một trường dữ liệu đạt 95% độ tin cậy (nhãn Xanh) để yên tâm hạch toán, hay cảnh báo nhãn Vàng để kiểm tra lại, mang lại giá trị thực tế cao hơn rất nhiều so với một khung chữ nhật vẽ méo mó trên màn hình."*

* 💡 *Mẹo trình bày:* Nói với thái độ say mê, hào hứng, tự hào về bài học kinh nghiệm kỹ thuật của nhóm.

---

### 🖼️ SLIDE 9: VIDEO DEMO HỆ THỐNG THỰC TẾ
* **Thời lượng:** `08:30 - 10:00` (90 – 120 giây)
* **Trọng tâm:** Trình chiếu video quay màn hình hệ thống hoạt động thật.

> 🗣️ **Lời thoại thuyết minh (Nói đệm theo nhịp chạy của Video):**
> 
> *"Sau đây, nhóm xin kính mời Thầy/Cô và Hội đồng cùng theo dõi video trải nghiệm thực tế hệ thống đã được nhóm đóng gói hoàn chỉnh:*
> 
> * *(Giây 10 - 30):* Đầu tiên, người dùng tải lên một ảnh hóa đơn in nhiệt từ chuỗi cửa hàng, chữ bị mờ và hơi nghiêng góc. Module Stage 0 tự động cân chỉnh góc xoay.
> * *(Giây 30 - 60):* Chỉ với 1 click 'Trích xuất tự động', mô hình bóc tách toàn bộ thông tin: Tên bên bán, Địa chỉ, Tổng tiền, Mã số thuế. Bên cạnh mỗi trường đều có **huy hiệu màu Xanh/Vàng** thể hiện độ tin cậy.
> * *(Giây 60 - 90):* Tiếp theo là tính năng **DocVQA hỏi đáp tự nhiên**: Kế toán gõ câu hỏi: *'Tổng tiền trước thuế là bao nhiêu?'* hay *'Hóa đơn này mua những món gì?'*, mô hình trả lời ngay lập tức bằng tiếng Việt chính xác.
> * *(Giây 90 - 110):* Cuối cùng, kế toán bấm 'Xuất JSON', toàn bộ chứng từ được cấu trúc hóa phân cấp sạch sẽ, sẵn sàng nạp thẳng vào cơ sở dữ liệu."*

* 💡 *Mẹo trình bày:* Khớp lời nói với diễn biến trên màn hình video, không để video chạy trong im lặng.

---

### 🖼️ SLIDE 10: TỔNG KẾT & ĐỊNH HƯỚNG ỨNG DỤNG
* **Thời lượng:** `10:00 - 10:45` (45 giây)
* **Trọng tâm:** Đóng gói kết quả, định hướng sản phẩm và lời cảm ơn.

> 🗣️ **Lời thoại thuyết minh:**
> 
> *"Tóm lại, đồ án của nhóm đã đạt được **3 kết quả then chốt**:*
> 
> 1. Xây dựng thành công hệ thống **Pure End-to-End VLM** chuyên biệt cho hóa đơn tiếng Việt đạt độ chính xác **89.63% ANLS**, **66.09% Exact Match** trên dữ liệu thực nghiệm thật, vận hành siêu nhẹ chỉ với **3.64 GB VRAM**.
> 2. Đúc kết bài học kỹ thuật thực tế về việc loại bỏ bẫy OCR truyền thống để chuyển đổi hoàn toàn sang mô hình đa phương thức.
> 3. Tích hợp cơ chế kiểm soát rủi ro với **Điểm Tin Cậy**, sẵn sàng giải quyết bài toán kế toán thực tiễn.
> 
> *Trong tương lai, nhóm sẽ đóng gói hệ thống dưới dạng Docker Container và phát triển các plugin tích hợp trực tiếp vào các phần mềm kế toán phổ biến như **MISA, FAST, SAP hay hệ thống ERP doanh nghiệp**.*
> 
> *Nhóm chúng em xin chân thành cảm ơn quý Thầy/Cô trong Hội đồng đã lắng nghe. Nhóm rất mong nhận được những nhận xét và câu hỏi góp ý quý báu từ quý Thầy/Cô!"*

---

## 🛡️ BỘ CÂU HỎI PHẢN BIỆN DỰ KIẾN TỪ HỘI ĐỒNG (CHEAT SHEET Q&A)

### ❓ Câu hỏi 1: *"Tại sao mô hình Base đã đạt 85% ANLS rồi mà nhóm vẫn cần phải Fine-tune LoRA?"*
> **Trả lời mẫu ăn điểm:**  
> *"Dạ thưa Thầy/Cô, 85% của Base Model phản ánh khả năng nhận diện ký tự OCR của Qwen2.5 rất tốt. Tuy nhiên, **Base Model hoàn toàn thiếu tư duy kế toán**: nó không biết cách trích xuất chuẩn thực thể. Ví dụ ở bảng kê danh mục hàng `ITEMS_LIST`, Base Model tự tiện làm toán nhân chia và cộng dồn giá khiến ANLS chỉ đạt 50% và Exact Match bằng 0%. Đơn giá từng món Base Model cũng chỉ đoán đúng 50%. Sau khi nhóm Fine-tune LoRA, tỷ lệ Exact Match đơn giá tăng vọt lên **78.57% (+28.57%)** và ANLS danh mục hàng tăng lên **75.47% (+24.78%)**. Đó là sự khác biệt giữa một mô hình biết đọc chữ và một mô hình hiểu nghiệp vụ kế toán ạ."*

---

### ❓ Câu hỏi 2: *"Mô hình của nhóm có cần tầng Regex hậu xử lý để gọt bỏ từ ngữ rườm rà không?"*
> **Trả lời mẫu ăn điểm:**  
> *"Dạ thưa Thầy/Cô, ban đầu nhóm có chuẩn bị một bộ lọc Regex. Nhưng nhờ quyết định tối ưu áp dụng kỹ thuật **Target-Only Loss Masking** trong quá trình huấn luyện LoRA, mô hình đã tự động triệt tiêu thói quen nói dài dòng của LLM gốc. Kết quả đo đạc thực tế cho thấy câu trả lời thô của mô hình đã sạch bóng 100% rác ngôn ngữ, khớp thẳng vào thực thể kế toán mà không còn phụ thuộc vào regex hậu xử lý ạ."*

---

### ❓ Câu hỏi 3: *"Tại sao nhóm lại bỏ Bounding Box trong khi người dùng thích nhìn thấy khung đỏ trên ảnh?"*
> **Trả lời mẫu ăn điểm:**  
> *"Dạ thưa Thầy/Cô, việc vẽ khung đỏ chỉ đẹp mắt khi demo, nhưng trong nghiệp vụ kế toán thực tế, nó gặp lỗi **Xung đột chuỗi số ngắn (Lexical Substring Collision)**: khi tìm số tiền 12 triệu, EasyOCR quét ra số '12' và khoanh nhầm vào số điện thoại hoặc số nhà ở đầu trang. Sai sót này gây hiểu nhầm rất nguy hiểm cho kế toán viên. Vì vậy, nhóm quyết định thay Bounding Box bằng **Điểm Tin Cậy (Confidence Score) phân cấp 3 màu Xanh - Vàng - Đỏ**. Kế toán viên nhìn vào màu sắc là biết trường nào an toàn để duyệt tự động, trường nào cần kiểm tra lại, mang lại giá trị kiểm soát rủi ro thực chất hơn rất nhiều ạ."*

---

### ❓ Câu hỏi 4: *"Hệ thống này có thể chạy trên máy tính văn phòng thông thường không?"*
> **Trả lời mẫu ăn điểm:**  
> *"Dạ hoàn toàn được ạ. Nhờ áp dụng kỹ thuật **Vision Pixel Budgeting**, nhóm đã ép mức tiêu thụ bộ nhớ của mô hình xuống đúng **3.64 GB VRAM**. Hệ thống có thể vận hành mượt mà trên các dòng card đồ họa phổ thông giá rẻ như GTX 1660 (6GB VRAM) hoặc triển khai qua Docker container chỉ tốn 1 GPU Tesla T4 giá rẻ trên đám mây ạ."*
