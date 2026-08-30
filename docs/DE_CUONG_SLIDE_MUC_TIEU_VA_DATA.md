# 📑 ĐỀ CƯƠNG SLIDE THUYẾT TRÌNH: MỤC TIÊU SẢN PHẨM & BỘ DỮ LIỆU (DATASET)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt

> **Dành riêng cho các bạn làm slide (PowerPoint / Canva / Google Slides).**  
> Bản đề cương này tập trung **100% vào Mục Tiêu Sản Phẩm & Toàn Bộ Công Tác Dữ Liệu Đã Hoàn Thành**. Mỗi slide đã có sẵn **Tiêu đề**, **Gợi ý bố cục**, **Nội dung copy-paste ngắn gọn** và **Lời thoại thuyết trình (Speaker Notes)**.

---

```
                   CẤU TRÚC BỘ SLIDE: MỤC TIÊU SẢN PHẨM & DỮ LIỆU
 ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
 │       SLIDES 1 - 4      │ ──> │       SLIDES 5 - 8      │ ──> │      SLIDES 9 - 12      │
 │ BỐI CẢNH, MỤC TIÊU      │     │  QUY TRÌNH & ĐẶC TẢ     │     │ CHẤT LƯỢNG DATA, DEMO   │
 │ & GIÁ TRỊ SẢN PHẨM      │     │  DỮ LIỆU (114K VQA)     │     │ & KẾ HOẠCH BƯỚC TỚI     │
 └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

### 🖥️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề lớn:** HỆ THỐNG TRÍCH XUẤT THÔNG TIN HÓA ĐƠN THÔNG MINH (DOCUMENT VQA)
* **Tiêu đề phụ:** Báo Cáo Giai Đoạn 1: Mục Tiêu Sản Phẩm & Xây Dựng Tập Dữ Liệu 114,000 Mẫu
* **Thông tin nhóm:**
  * Giảng viên hướng dẫn: [Tên Thầy/Cô]
  * Thành viên thực hiện: [Tên các bạn trong nhóm]
  * Lớp / Ngành / Năm học: 2025 - 2026
* **Gợi ý thiết kế:** Tone màu xanh dương công nghệ (Tech Blue), hình ảnh hóa đơn số hóa kết hợp biểu tượng AI.
* **Speaker Notes:** *"Kính chào Thầy/Cô và các bạn, hôm nay nhóm xin phép báo cáo về Mục tiêu sản phẩm và Kết quả xây dựng bộ dữ liệu đa tác vụ cho hệ thống trích xuất hóa đơn tiếng Việt."*

---

### 🖥️ SLIDE 2: BỐI CẢNH & THỰC TRẠNG BÀI TOÁN
* **Tiêu đề:** Thực Trạng Xử Lý Hóa Đơn Tại Việt Nam
* **Bố cục:** 2 cột (Khối lượng thực tế vs Hạn chế xử lý thủ công)
* **Nội dung hiển thị:**
  * **Khối lượng hóa đơn khổng lồ:** Hàng triệu giao dịch bán lẻ, ăn uống, siêu thị và hóa đơn điện tử phát sinh mỗi ngày.
  * **Hạn chế của phương pháp thủ công:**
    * ⏳ Tốn nhiều nhân sự và thời gian nhập liệu vào phần mềm kế toán.
    * ⚠️ Dễ xảy ra sai sót khi gõ lại số tiền, mã số thuế, ngày tháng.
    * 📄 Hóa đơn in nhiệt dễ bay màu, rách mép, nhàu nát theo thời gian.
* **Speaker Notes:** *"Hóa đơn bán lẻ tại Việt Nam có số lượng rất lớn và định dạng phức tạp, việc nhập liệu thủ công gây tốn kém thời gian và dễ sai sót."*

---

### 🖥️ SLIDE 3: MỤC TIÊU SẢN PHẨM & GIÁ TRỊ THỰC TIỄN
* **Tiêu đề:** Mục Tiêu Phát Triển Sản Phẩm
* **Bố cục:** 3 Cards giải pháp (Tự động hóa, Linh hoạt, Chuẩn hóa CSDL)
* **Nội dung hiển thị:**
  * 🎯 **Mục tiêu cốt lõi:** Xây dựng hệ thống **Document VQA** cho phép người dùng hỏi đáp và trích xuất mọi thông tin trên hóa đơn bằng ngôn ngữ tự nhiên tiếng Việt.
  * 🚀 **Giá trị thực tiễn mang lại:**
    1. **Tự động hóa 100%:** Rút ngắn thời gian xử lý hóa đơn từ 2 phút xuống **dưới 2 giây**.
    2. **Không phụ thuộc Template:** Hiểu linh hoạt mọi mẫu hóa đơn mà không cần vẽ khung cố định.
    3. **Xuất dữ liệu chuẩn:** Xuất trực tiếp định dạng **JSON có cấu trúc** để tích hợp vào các hệ thống ERP, CRM, phần mềm kế toán (MISA, FAST).
* **Speaker Notes:** *"Mục tiêu của nhóm là tạo ra một sản phẩm có thể đọc và hiểu bất kỳ hóa đơn nào, trả về kết quả chính xác dạng JSON trong vòng 2 giây."*

---

### 🖥️ SLIDE 4: ĐỊNH HƯỚNG GIẢI PHÁP - END-TO-END VISION LANGUAGE
* **Tiêu đề:** Định Hướng Kiến Trúc Sản Phẩm
* **Bố cục:** Sơ đồ so sánh (Pipeline Cũ vs Pipeline Đề Xuất)
* **Nội dung hiển thị:**
  * ❌ **Pipeline cũ (OCR + NLP):** OCR đọc chữ $\rightarrow$ NLP phân tích $\implies$ Dễ gãy khúc và lỗi lan truyền (Cascading Error).
  * 💡 **Giải pháp End-to-End VLM:**
    * Nhận trực tiếp **Ảnh hóa đơn + Câu hỏi tiếng Việt**.
    * Mô hình nhìn nhận đồng thời cả **chữ viết + vị trí không gian 2D** (dòng tiền, cột bảng biểu).
    * Xuất thẳng câu trả lời chính xác hoặc chuỗi JSON phân cấp.

---

### 🖥️ SLIDE 5: QUY TRÌNH XÂY DỰNG DỮ LIỆU (DATA PIPELINE)
* **Tiêu đề:** Quy Trình Thu Thập & Chuẩn Hóa Dữ Liệu
* **Bố cục:** Sơ đồ 4 bước tuyến tính (Pipeline Workflow)
* **Nội dung hiển thị:**
  * **Bước 1: Thu thập & Số hóa (Collection):** Tổng hợp 4,995 ảnh hóa đơn thực tế từ 15 thương hiệu lớn tại VN.
  * **Bước 2: Tiền xử lý (Preprocessing):** Chuẩn hóa tỷ lệ khung hình, xử lý xoay ảnh, tối ưu hóa độ phân giải động.
  * **Bước 3: Gán nhãn Đa tầng (Multi-task Labeling):** Xây dựng câu hỏi VQA đơn trường, đa trường, và trích xuất cấu trúc JSON.
  * **Bước 4: Phân chia & Kiểm định (Splitting):** Chia tập Train (85%), Validation (15%) và Benchmark Test độc lập.

---

### 🖥️ SLIDE 6: TỔNG QUAN BỘ DỮ LIỆU (114,716 CẶP VQA)
* **Tiêu đề:** Quy Mô & Phân Bổ Tập Dữ Liệu
* **Bố cục:** 3 Khối số liệu lớn (Thống kê chính)
* **Nội dung hiển thị:**
  * 📸 **Tổng số ảnh hóa đơn:** **4,995 ảnh** độ phân giải cao.
  * 💬 **Tổng số cặp câu hỏi VQA:** **114,716 mẫu** (Bao phủ toàn diện mọi tình huống trích xuất).
  * 📊 **Phân chia tập dữ liệu:**
    * 📦 **Tập Train Master:** `97,508` mẫu (~85%) – Phục vụ huấn luyện mô hình.
    * 🔍 **Tập Validation Master:** `17,208` mẫu (~15%) – Phục vụ kiểm soát hàm mất mát.
    * 🎯 **Tập Benchmark Test:** `174` câu hỏi độc lập – Phục vụ đo lường định lượng.

---

### 🖥️ SLIDE 7: ĐỘ PHỦ 15 LOẠI MẪU HÓA ĐƠN THỰC TẾ
* **Tiêu đề:** Phân Bổ 15 Nhóm Mẫu Hóa Đơn Chuẩn Hóa
* **Bố cục:** 4 nhóm lĩnh vực (Logo thương hiệu / Icons)
* **Nội dung hiển thị:**
  * ☕ **Chuỗi Cafe & F&B (33%):** Highlands Coffee, Phúc Long, Starbucks, KFC, Jollibee.
  * 🏪 **Cửa Hàng Tiện Lợi & Mini Mart (27%):** 7-Eleven, Circle K, GS25, Minimart An An.
  * 🛒 **Đại Siêu Thị (20%):** WinMart / WinMart+, Lotte Mart, Bách Hóa Xanh.
  * 🧾 **Hóa Đơn Điện Tử & Biên Lai (20%):** Viettel e-Invoice, VNPT e-Invoice, Mẫu chuẩn C45-BB.
* **Speaker Notes:** *"Tập dữ liệu được phân bố đồng đều với 333 ảnh cho mỗi loại trong số 15 mẫu hóa đơn phổ biến nhất tại Việt Nam."*

---

### 🖥️ SLIDE 8: ĐẶC TẢ 7 TÁC VỤ TRÍCH XUẤT (TASK TAXONOMY)
* **Tiêu đề:** Đa Dạng Hóa Các Dạng Câu Hỏi VQA
* **Bố cục:** Bảng thống kê 7 trường thông tin
* **Nội dung hiển thị:**

| Tác vụ VQA (Task) | Số lượng câu hỏi | Ý nghĩa trích xuất |
| :--- | :---: | :--- |
| **`SELLER`** | 9,990 câu | Tên bên bán, cửa hàng, chi nhánh, công ty phát hành |
| **`TOTAL_COST`** | 9,990 câu | Tổng tiền thanh toán cuối cùng (đầy đủ dấu phân cách) |
| **`TIMESTAMP`** | 9,990 câu | Ngày giờ lập phiếu (đa dạng định dạng dd/mm/yyyy) |
| **`ADDRESS`** | 9,324 câu | Địa chỉ cửa hàng, trụ sở doanh nghiệp |
| **`ITEM_PRICE` & `QTY`** | 46,118 câu | Chi tiết đơn giá và số lượng từng món hàng |
| **`FULL_JSON`** | 9,990 câu | Toàn bộ cấu trúc phân cấp hóa đơn dạng JSON |
| **`BOUNDING_BOX`** | 9,990 câu | Tọa độ vùng văn bản phục vụ định vị thị giác |

---

### 🖥️ SLIDE 9: CHUẨN HÓA CẤU TRÚC JSON PHÂN CẤP
* **Tiêu đề:** Chuẩn Hóa Dữ Liệu Đầu Ra Dạng JSON
* **Bố cục:** Hiển thị khối mã JSON mẫu (Code Block)
* **Nội dung hiển thị:**
  * Cấu trúc JSON được chuẩn hóa đồng nhất cho tất cả 15 loại hóa đơn:
```json
{
  "seller": "HIGHLANDS COFFEE - VẠN HẠNH MALL",
  "address": "Số 11 Sư Vạn Hạnh, Phường 12, Quận 10, TP.HCM",
  "timestamp": "28/06/2026 09:15",
  "total_cost": "109,000đ",
  "items": [
    {"name": "Trà Sen Vàng (L)", "qty": "1", "amount": "55,000"},
    {"name": "Phin Sữa Đá (M)", "qty": "1", "amount": "54,000"}
  ]
}
```
* **Ý nghĩa:** Giúp các phần mềm kế toán có thể parse tự động 100% mà không cần viết regex riêng.

---

### 🖥️ SLIDE 10: KIỂM SOÁT CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY)
* **Tiêu đề:** Tiêu Chuẩn Kiểm Định Chất Lượng Dữ Liệu
* **Bố cục:** 3 tiêu chí đảm bảo chất lượng (Data Integrity)
* **Nội dung hiển thị:**
  * ✅ **Tính nhất quán ký tự:** 100% dấu tiếng Việt và đơn vị tiền tệ (VNĐ, đ, dấu phẩy) được kiểm tra chéo tự động.
  * ✅ **Khử nhiễu nhãn (Label Sanitization):** Loại bỏ các mẫu bị thiếu trường hoặc câu hỏi mơ hồ.
  * ✅ **Độc lập tập Test:** Tập Test Benchmark (174 câu hỏi) hoàn toàn tách biệt, không nằm trong tập huấn luyện để đảm bảo tính khách quan khi đánh giá.

---

### 🖥️ SLIDE 11: MÔ HÌNH HÓA SẢN PHẨM & TRẢI NGHIỆM NGƯỜI DÙNG
* **Tiêu đề:** Thiết Kế Giao Diện Sản Phẩm (Gradio Web UI)
* **Bố cục:** Hình ảnh mockup giao diện người dùng
* **Nội dung hiển thị:**
  * **Trải nghiệm kéo thả:** Người dùng tải ảnh chụp hóa đơn từ điện thoại hoặc máy scan.
  * **2 Chế độ trích xuất linh hoạt:**
    * 🔹 **Chế độ VQA:** Hỏi nhanh 1 trường thông tin bất kỳ (*"Tổng tiền?", "Bên bán?"*).
    * 🔹 **Chế độ Auto-JSON:** Tự động trích xuất toàn bộ hóa đơn thành JSON và tải về file `.json`.
  * **Mã nguồn giao diện:** Đã xây dựng hoàn chỉnh tại [`model/demo_gradio.py`](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py).

---

### 🖥️ SLIDE 12: TỔNG KẾT GIAI ĐOẠN 1 & KẾ HOẠCH TIẾP THEO
* **Tiêu đề:** Tổng Kết Công Việc & Lộ Trình Kế Tiếp
* **Bố cục:** 2 cột (Đã Hoàn Thành vs Kế Hoạch Tiếp Theo)
* **Nội dung hiển thị:**
  * 🏁 **Kết quả Giai đoạn 1 (Đã hoàn thành 100%):**
    * Xác định rõ ràng mục tiêu sản phẩm và giải pháp End-to-End VLM.
    * Xây dựng và chuẩn hóa thành công **114,716 cặp VQA** trên **15 loại hóa đơn**.
    * Hoàn thiện thiết kế giao diện Web Demo và các kịch bản kiểm thử.
  * 🔄 **Lộ trình Giai đoạn 2:**
    * Tiến hành huấn luyện mô hình Vision-Language với kỹ thuật LoRA.
    * Đánh giá định lượng trên bộ chỉ số học thuật (ANLS, Exact Match, Token F1).
* **Lời cảm ơn:** *"Xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý theo dõi!"*
