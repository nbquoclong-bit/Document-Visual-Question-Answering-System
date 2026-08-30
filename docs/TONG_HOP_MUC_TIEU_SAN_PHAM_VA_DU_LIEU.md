# 📘 TỔNG HỢP TOÀN DIỆN: MỤC TIÊU SẢN PHẨM & CÔNG TÁC DỮ LIỆU (DATASET)
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt

> **Mục đích tài liệu:** Đây là kho dữ liệu và thông tin đầy đủ, chi tiết nhất về **Bối cảnh, Mục tiêu sản phẩm, Giá trị thực tiễn, Luồng hệ thống, Tác vụ Định vị Bounding Box và Toàn bộ công tác kỹ thuật dữ liệu (114,716 mẫu VQA)**.  
> Các thành viên trong nhóm có thể tự do trích xuất, lựa chọn bất kỳ bảng số liệu, sơ đồ hay thông tin nào từ tài liệu này để đưa vào bài thuyết trình và slide theo ý muốn.

---

## 📌 MỤC LỤC TỔNG QUAN

1. [Phần 1: Bối Cảnh Bài Toán & Mục Tiêu Sản Phẩm](#phần-1-bối-cảnh-bài-toán--mục-tiêu-sản-phẩm)
2. [Phần 2: Người Dùng Mục Tiêu & Trải Nghiệm Sử Dụng](#phần-2-người-dùng-mục-tiêu--trải-nghiệm-sử-dụng)
3. [Phần 3: Đặc Tả Đầu Vào, Đầu Ra & Luồng Hệ Thống](#phần-3-đặc-tả-đầu-vào-đầu-ra--luồng-hệ-thống)
4. [Phần 4: Đặc Tả Tác Vụ Định Vị Tọa Độ Bounding Box (Visual Grounding)](#phần-4-đặc-tả-tác-vụ-định-vị-tọa-độ-bounding-box-visual-grounding)
5. [Phần 5: Quy Trình Xây Dựng & Chuẩn Hóa Dữ Liệu](#phần-5-quy-trình-xây-dựng--chuẩn-hóa-dữ-liệu)
6. [Phần 6: Thống Kê Chi Tiết Bộ Dữ Liệu 114,716 Mẫu VQA](#phần-6-thống-kê-chi-tiết-bộ-dữ-liệu-114716-mẫu-vqa)
7. [Phần 7: Phân Loại 15 Mẫu Hóa Đơn & 8 Nhóm Tác Vụ](#phần-7-phân-loại-15-mẫu-hóa-đơn--8-nhóm-tác-vụ)
8. [Phần 8: Tiêu Chuẩn Kiểm Soát Chất Lượng Dữ Liệu](#phần-8-tiêu-chuẩn-kiểm-soát-chất-lượng-dữ-liệu)
9. [Phần 9: Bản Đồ File Dữ Liệu Trong Repository](#phần-9-bản-đồ-file-dữ-liệu-trong-repository)

---

# PHẦN 1: BỐI CẢNH BÀI TOÁN & MỤC TIÊU SẢN PHẨM

### 1.1. Bối cảnh thực tế tại thị trường Việt Nam
* **Làn sóng chuyển đổi số trong kế toán - tài chính:** Hàng triệu hóa đơn bán lẻ (in nhiệt) và hóa đơn điện tử (e-Invoice) được phát hành mỗi ngày tại các chuỗi siêu thị, nhà hàng, cửa hàng tiện lợi và doanh nghiệp.
* **Đặc thù hóa đơn Việt Nam:**
  * Chất lượng in ấn không đồng đều (chữ in nhiệt dễ mờ, mất nét sau vài ngày).
  * Đa dạng font chữ tiếng Việt có dấu, chữ nghiêng, chữ viết hoa, logo thương hiệu đan xen.
  * Bố cục bảng biểu (table layout) phức tạp, nhiều cột đơn giá, số lượng, tiền trước thuế, VAT, chiết khấu, tổng thanh toán.

### 1.2. Hạn chế của các phương pháp truyền thống
1. **Nhập liệu thủ công (Manual Data Entry):**
   * Tốn trung bình 2–3 phút cho 1 hóa đơn dài.
   * Tỷ lệ gõ sai số tiền hoặc mã số thuế do lỗi con người lên tới 5–8%.
   * Chi phí nhân sự kế toán cao, dễ quá tải vào các kỳ quyết toán thuế cuối tháng/quý.
2. **Hệ thống OCR truyền thống kết hợp Regex/Rule-based (Tesseract, PaddleOCR + NLP):**
   * **Lỗi lan truyền (Cascading Error):** Nếu OCR đọc sai `8` thành `B` hoặc `0` thành `O`, module NLP phía sau sẽ trích xuất sai toàn bộ số tiền.
   * **Mất thông tin không gian 2D:** OCR làm phẳng ảnh thành 1 chuỗi văn bản 1D, làm mất mối quan hệ giữa nhãn bên trái (*"Tổng tiền thanh toán"*) và con số nằm ở cột đối diện bên phải.
   * **Rập khuôn theo mẫu (Template-dependent):** Khi cửa hàng đổi mẫu in hóa đơn, toàn bộ rule/regex cũ bị sụp đổ, phải lập trình lại từ đầu.

### 1.3. Mục tiêu & Tầm nhìn của sản phẩm
* **Tên sản phẩm:** Hệ thống Trích Xuất & Định Vị Hóa Đơn Thông Minh (**Document VQA & Visual Grounding for Vietnamese Invoices**).
* **Định hướng công nghệ:** Ứng dụng **Vision-Language Model (VLM) End-to-End**, nhận trực tiếp ảnh pixel và câu hỏi tự nhiên để trích xuất thẳng kết quả văn bản, JSON và **khung bao tọa độ Bounding Box** mà không qua bước OCR trung gian.
* **Mục tiêu định lượng (KPIs):**
  * **Tốc độ xử lý:** Dưới **2 giây / hóa đơn** trên phần cứng GPU phổ thông.
  * **Tự do mẫu mã:** Hoạt động chính xác trên mọi loại hóa đơn không cần định nghĩa khung mẫu trước.
  * **Khả năng tích hợp:** Xuất trực tiếp cấu trúc **JSON phân cấp** để kết nối vào các hệ thống phần mềm kế toán (MISA, FAST, SAP) và ERP doanh nghiệp.
  * **Tự động hóa:** Giảm **95%** thời gian nhập liệu thủ công của kế toán viên.

---

# PHẦN 2: NGƯỜI DÙNG MỤC TIÊU & TRẢI NGHIỆM SỬ DỤNG

### 2.1. Đối tượng người dùng mục tiêu (User Personas)

```
┌──────────────────────────┬─────────────────────────────────────────────────┐
│ Nhóm Người Dùng          │ Nhu Cầu & Bài Toán Cần Giải Quyết               │
├──────────────────────────┼─────────────────────────────────────────────────┤
│ 1. Kế toán doanh nghiệp  │ Trích xuất tự động mã số thuế, tổng tiền, ngày  │
│                          │ hóa đơn, chi tiết VAT để lập báo cáo tài chính. │
├──────────────────────────┼─────────────────────────────────────────────────┤
│ 2. Chủ cửa hàng / Thủ kho│ Quản lý danh sách mặt hàng nhập kho, số lượng,  │
│                          │ đơn giá từ các phiếu thu, hóa đơn nhà cung cấp. │
├──────────────────────────┼─────────────────────────────────────────────────┤
│ 3. Người dùng cá nhân    │ Chụp ảnh hóa đơn mua sắm để tự động theo dõi và │
│                          │ phân loại chi tiêu hàng tháng.                  │
└──────────────────────────┴─────────────────────────────────────────────────┘
```

### 2.2. Kịch bản sử dụng thực tế (Use Cases)
1. **Truy vấn đơn lẻ (Single-field VQA):** Người dùng hỏi nhanh một trường thông tin bất kỳ:
   * *"Hóa đơn này của công ty nào?"* $\rightarrow$ Trả về: `HIGHLANDS COFFEE`.
   * *"Tổng tiền thanh toán cuối cùng là bao nhiêu?"* $\rightarrow$ Trả về: `109,000`.
2. **Trích xuất toàn bộ (Full Document Structuring):** Người dùng tải ảnh lên và chọn chế độ *"Trích xuất JSON"*, hệ thống lập tức trả về file JSON chứa toàn bộ thông tin bên bán, thời gian, danh sách mặt hàng và tổng tiền.
3. **Định vị vùng văn bản (Visual Grounding / Bounding Box):** Hệ thống vẽ khung hình chữ nhật nổi bật (Bounding Box) trực tiếp lên ảnh tại vị trí dòng tiền hoặc tên cửa hàng để người dùng kiểm chứng nhanh bằng mắt thường.

---

# PHẦN 3: ĐẶC TẢ ĐẦU VÀO, ĐẦU RA & LUỒNG HỆ THỐNG

### 3.1. Sơ đồ luồng sản phẩm (System Flow)

```
   ┌───────────────────────┐
   │ Ảnh Hóa Đơn Đầu Vào   │ (Ảnh chụp điện thoại / File scan / Hóa đơn điện tử)
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐     ┌────────────────────────────┐
   │ Bộ Tiền Xử Lý Ảnh     │ <── │ Câu Hỏi Tự Nhiên Tiếng Việt│
   │ (Dynamic Resolution)  │     │ (Hỏi giá trị / Bounding Box│
   └──────────┬────────────┘     └─────────────┬──────────────┘
              │                                │
              ▼                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │             MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)            │
   │           Nhìn ảnh 2D + Hiểu ngữ cảnh câu hỏi            │
   └──────────────────────────┬───────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐      ┌───────────────┐      ┌───────────────┐
│ 1. Text Trực │      │ 2. Dữ Liệu    │      │ 3. Khung Tọa  │
│    Diện      │      │    JSON Chuẩn │      │    Độ Bounding│
│    "109,000" │      │    Hóa Phân   │      │    Box [x,y,  │
│              │      │    Cấp CSDL   │      │    w,h]       │
└──────────────┘      └───────────────┘      └───────────────┘
```

---

# PHẦN 4: ĐẶC TẢ TÁC VỤ ĐỊNH VỊ TỌA ĐỘ BOUNDING BOX (VISUAL GROUNDING)

### 4.1. Tác vụ Bounding Box là gì?
Trong bài toán Document VQA nâng cao, **Bounding Box (Visual Grounding)** là khả năng mô hình không chỉ đọc hiểu nội dung chữ mà còn **xác định chính xác tọa độ vị trí không gian của vùng chứa nội dung đó trên ảnh hóa đơn**.

```
  ┌───────────────────────────────────────────────────────────────┐
  │ [Ảnh Hóa Đơn]                                                 │
  │                                                               │
  │   HIGHLANDS COFFEE  <── 🟥 [Bounding Box: (97, 16, 282, 37)]  │
  │   Số 11 Sư Vạn Hạnh, Q.10                                     │
  │   --------------------------------------------------------    │
  │   1. Trà Sen Vàng (L)     55,000                              │
  │   2. Phin Sữa Đá  (M)     54,000                              │
  │   --------------------------------------------------------    │
  │   TỔNG TIỀN: 109,000đ  <── 🟩 [Bounding Box: (307, 456, 359)] │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 4.2. Quy chuẩn Tọa độ Bounding Box trong Hệ thống
Tọa độ Bounding Box được biểu diễn theo chuẩn 4 chiều:
$$\text{Bounding Box} = [y_{\min}, x_{\min}, y_{\max}, x_{\max}]$$
Trong đó:
* $(x_{\min}, y_{\min})$: Tọa độ góc trên bên trái của vùng chữ.
* $(x_{\max}, y_{\max})$: Tọa độ góc dưới bên phải của vùng chữ.
* Tọa độ được chuẩn hóa theo tỷ lệ kích thước pixel của ảnh thực tế.

### 4.3. Thống kê & Ví dụ Mẫu Bounding Box trong Dataset

Trong bộ dữ liệu **114,716 mẫu VQA**, có tổng cộng **9,990 mẫu gán nhãn Bounding Box** chuyên biệt:
* **`GROUNDING_SELLER` (4,995 câu):** Định vị tọa độ của tên thương hiệu / đơn vị bán hàng.
* **`GROUNDING_TOTAL` (4,995 câu):** Định vị tọa độ của dòng tổng tiền thanh toán cuối cùng.

#### 📌 Ví dụ thực tế từ bộ dữ liệu (`vlm_train_master.json`):
* **Ví dụ 1 (Highlands Coffee):**
  * *Câu hỏi:* `"Tìm và định vị vùng chứa tên đơn vị bán hàng trên hóa đơn?"`
  * *Đáp án đầu ra:* `{"text": "HIGHLANDS COFFEE LANDMARK", "box": [97, 16, 282, 37]}`
* **Ví dụ 2 (Circle K):**
  * *Câu hỏi:* `"Tìm và định vị vùng chứa tên đơn vị bán hàng trên hóa đơn?"`
  * *Đáp án đầu ra:* `{"text": "Circle K Lê Lợi", "box": [154, 21, 250, 44]}`
* **Ví dụ 3 (Tổng tiền thanh toán):**
  * *Câu hỏi:* `"Tìm và định vị vùng chứa tổng tiền thanh toán trên hóa đơn?"`
  * *Đáp án đầu ra:* `{"text": "561,000", "box": [307, 456, 359, 477]}`

### 4.4. Giá trị thực tiễn của Bounding Box đối với Sản phẩm
1. **Kiểm chứng mắt người (Human-in-the-loop):** Kế toán viên có thể nhìn trực tiếp khung màu nổi bật được vẽ trên ảnh để kiểm tra tính xác thực của số tiền mà không cần mất thời gian rà soát toàn bộ tờ hóa đơn dài.
2. **Tính giải thích được (Explainability / Transparency):** Chứng minh mô hình thực sự "nhìn" đúng vùng văn bản chứa thông tin thay vì học vẹt hay suy đoán mò.
3. **Cắt ảnh trích xuất con dấu / chữ ký:** Cho phép tự động crop đúng vùng Bounding Box để lưu trữ làm bằng chứng kiểm toán điện tử.

---

# PHẦN 5: QUY TRÌNH XÂY DỰNG & CHUẨN HÓA DỮ LIỆU

Nhóm đã hoàn thành quy trình xây dựng dữ liệu qua 5 giai đoạn nghiêm ngặt:

```
  1. THU THẬP & SỐ HÓA       2. TIỀN XỬ LÝ THỊ GIÁC       3. THIẾT KẾ SCHEMA & NHÃN
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────────┐
│ Thu thập 4,995 ảnh    │──>│ Xoay ảnh chuẩn góc,   │──>│ Định nghĩa 8 nhóm trường  │
│ 15 mẫu hóa đơn VN     │   │ cắt viền, lọc mờ nhòe │   │ VQA đơn trường, JSON, BBox│
└───────────────────────┘   └───────────────────────┘   └─────────────┬─────────────┘
                                                                      │
                                                                      ▼
  5. PHÂN CHIA TẬP DATA      4. SINH TỰ ĐỘNG & KIỂM ĐỊNH              │
┌───────────────────────┐   ┌───────────────────────────┐             │
│ Train: 85% (97k mẫu)  │<──│ Sinh 114,716 cặp VQA,     │<────────────┘
│ Val  : 15% (17k mẫu)  │   │ kiểm tra chéo 100% nhãn   │
│ Test : 174 mẫu độc lập│   │ dấu tiếng Việt, box, tiền │
└───────────────────────┘   └───────────────────────────┘
```

1. **Giai đoạn 1 - Thu thập ảnh hóa đơn:** Tổng hợp 4,995 hóa đơn thực tế trải đều 15 thương hiệu và mẫu phiếu kế toán tại Việt Nam.
2. **Giai đoạn 2 - Tiền xử lý thị giác:** Chuẩn hóa hệ màu RGB, khử nhiễu, căn chỉnh góc xoay thẳng đứng, tối ưu độ phân giải động.
3. **Giai đoạn 3 - Xây dựng Schema đặc tả nhãn:** Phân loại rõ ràng 8 nhóm tác vụ từ mức độ đơn giản (tên cửa hàng, tổng tiền) đến phức tạp (danh sách món, đơn giá, số lượng, JSON và Bounding Box).
4. **Giai đoạn 4 - Tạo lập & Làm sạch câu hỏi VQA:** Tạo câu hỏi tự nhiên phong phú với nhiều biến thể cách hỏi trong tiếng Việt (ví dụ: *"Tổng tiền?"*, *"Tổng thanh toán?"*, *"Khách cần trả bao nhiêu?"* đều trỏ về cùng một thực thể `TOTAL_COST`).
5. **Giai đoạn 5 - Phân tách tập dữ liệu khoa học:** Đảm bảo không xảy ra hiện tượng rò rỉ dữ liệu (*Data Leakage*) giữa tập huấn luyện và kiểm định.

---

# PHẦN 6: THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU 114,716 MẪU VQA

### 6.1. Bảng tổng quan quy mô dữ liệu

| Thông Số Định Lượng | Giá Trị Cụ Thể | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :--- |
| **Tổng số ảnh hóa đơn** | **4,995 ảnh** | Phủ kín 15 danh mục template hóa đơn thực tế |
| **Tổng số cặp câu hỏi VQA** | **114,716 mẫu** | Cung cấp ngữ cảnh phong phú cho mô hình học |
| **Tập Train Master (`vlm_train_master.json`)** | **97,508 mẫu** (~85%) | Dùng cho quá trình huấn luyện và tối ưu trọng số |
| **Tập Validation Master (`vlm_val_master.json`)** | **17,208 mẫu** (~15%) | Dùng kiểm soát hàm mất mát và chống quá khớp |
| **Tập Benchmark Test Độc Lập** | **174 mẫu** | Bộ kiểm định độc lập đo lường các chỉ số học thuật |
| **Dung lượng file Train Master** | **34.0 MB** | Đã tối ưu hóa lưu trữ JSON chuẩn UTF-8 |
| **Dung lượng file Validation Master** | **5.98 MB** | Đã đồng bộ lên GitHub main |

---

# PHẦN 7: PHÂN LOẠI 15 MẪU HÓA ĐƠN & 8 NHÓM TÁC VỤ

### 7.1. Phân bổ cân bằng 15 loại mẫu hóa đơn thực tế (333 ảnh / mẫu)

```
┌─────────────────────────┬──────────────────────┬──────────────┬───────────────────┐
│ Lĩnh Vực / Nhóm Ngành   │ Tên Mẫu Template     │ Số Lượng Ảnh │ Đặc Điểm Nhận Dạng│
├─────────────────────────┼──────────────────────┼──────────────┼───────────────────┤
│                         │ cafe_highlands       │ 333 ảnh      │ In nhiệt, logo đỏ │
│                         │ cafe_phuclong        │ 333 ảnh      │ Khổ dài, nhiều món│
│ 1. Cafe & Ăn uống (F&B) │ cafe_starbucks       │ 333 ảnh      │ Tiếng Anh/Việt    │
│                         │ restaurant_jollibee  │ 333 ảnh      │ Bảng combo món    │
│                         │ restaurant_kfc       │ 333 ảnh      │ Font chữ lớn      │
├─────────────────────────┼──────────────────────┼──────────────┼───────────────────┤
│                         │ convenience_7eleven  │ 333 ảnh      │ Mã vạch, điểm tích│
│ 2. Cửa hàng tiện lợi &  │ convenience_circlek  │ 333 ảnh      │ In nhiệt hẹp      │
│    Siêu thị mini        │ convenience_gs25     │ 333 ảnh      │ Chiết khấu TV     │
│                         │ minimart_anan        │ 333 ảnh      │ Mẫu phiếu bán lẻ  │
├─────────────────────────┼──────────────────────┼──────────────┼───────────────────┤
│                         │ supermarket_winmart  │ 333 ảnh      │ Bảng kê dài, VAT  │
│ 3. Chuỗi Siêu thị       │ supermarket_lotte    │ 333 ảnh      │ 2 cột giá & lượng │
│                         │ supermarket_bachhoaxanh 333 ảnh    │ Cân ký, thực phẩm │
├─────────────────────────┼──────────────────────┼──────────────┼───────────────────┤
│ 4. Hóa đơn điện tử &    │ einvoice_viettel     │ 333 ảnh      │ Khổ A4/A5, bảng kê│
│    Biên lai chuẩn bộ    │ einvoice_vnpt        │ 333 ảnh      │ Chữ ký số, MST    │
│                         │ receipt_c45_bb       │ 333 ảnh      │ Mẫu chuẩn Bộ TC   │
├─────────────────────────┼──────────────────────┼──────────────┼───────────────────┤
│ TỔNG CỘNG               │ 15 Loại Mẫu Hóa Đơn  │ 4,995 Ảnh    │ Cân bằng 100%     │
└─────────────────────────┴──────────────────────┴──────────────┴───────────────────┘
```

### 7.2. Phân bổ 8 nhóm tác vụ VQA (Task Taxonomy)

| Mã Tác Vụ (Field) | Số Lượng Câu Hỏi | Mục Tiêu Trích Xuất & Ví Dụ |
| :--- | :---: | :--- |
| **`SELLER`** | **9,990 câu** | Nhận diện tên đơn vị bán hàng, chi nhánh, công ty phát hành.<br>*Ví dụ: "Hóa đơn này của đơn vị nào phát hành?" $\rightarrow$ `WINMART+`* |
| **`TOTAL_COST`** | **9,990 câu** | Trích xuất chính xác tổng số tiền thanh toán cuối cùng.<br>*Ví dụ: "Tổng tiền cần trả là bao nhiêu?" $\rightarrow$ `184,800`* |
| **`TIMESTAMP`** | **9,990 câu** | Trích xuất ngày, tháng, năm và giờ lập phiếu.<br>*Ví dụ: "Thời gian in hóa đơn là lúc nào?" $\rightarrow$ `29/06/2026 18:40`* |
| **`ADDRESS`** | **9,324 câu** | Trích xuất địa chỉ cửa hàng, trụ sở doanh nghiệp.<br>*Ví dụ: "Địa chỉ nơi mua hàng ở đâu?" $\rightarrow$ `Số 356 Hai Bà Trưng, Q.1`* |
| **`ITEM_PRICE`** | **31,756 câu** | Trích xuất đơn giá của từng mặt hàng cụ thể.<br>*Ví dụ: "Giá của Coca Cola Lon 320ml là bao nhiêu?" $\rightarrow$ `10,000`* |
| **`ITEM_QTY`** | **14,362 câu** | Trích xuất số lượng mua của từng mặt hàng.<br>*Ví dụ: "Khách mua mấy lon Coca Cola?" $\rightarrow$ `4`* |
| **`FULL_JSON`** | **9,990 câu** | Trích xuất toàn bộ cấu trúc phân cấp hóa đơn dạng JSON.<br>*Gồm đầy đủ: seller, timestamp, address, total_cost, items.* |
| **`BOUNDING_BOX`** *(Visual Grounding)* | **9,990 câu** | **Định vị tọa độ khung bao `[ymin, xmin, ymax, xmax]` trên ảnh phục vụ kiểm chứng trực quan.** |

---

# PHẦN 8: TIÊU CHUẨN KIỂM SOÁT CHẤT LƯỢNG DỮ LIỆU

Để bộ dữ liệu đạt tiêu chuẩn học thuật cao nhất, nhóm đã áp dụng 4 nguyên tắc kiểm định:

1. **Chuẩn hóa Bộ Ký Tự Tiếng Việt (Unicode Normalization Form C - NFC):**
   * Đảm bảo 100% ký tự tiếng Việt có dấu (như `ơ`, `ư`, `ê`, `đ`, dấu hỏi, ngã, nặng) đồng nhất, không bị lỗi gãy font hay nhầm lẫn tổ hợp.
2. **Chuẩn hóa Định Dạng Số & Tiền Tệ:**
   * Các con số được giữ nguyên dấu phân cách hàng nghìn (dấu phẩy `,` hoặc dấu chấm `.`) đúng theo quy ước ghi trên từng hóa đơn thực tế.
3. **Đa dạng hóa Câu hỏi Ngôn ngữ tự nhiên:**
   * Mỗi trường thông tin có ít nhất 5–10 cách diễn đạt câu hỏi khác nhau (ví dụ: *"Tổng tiền"*, *"Tổng cộng"*, *"Số tiền thanh toán"*, *"Thành tiền cuối cùng"*), giúp mô hình có khả năng khái quát hóa vượt trội.
4. **Kiểm tra cú pháp JSON & Tọa độ Bounding Box:**
   * 100% mẫu câu hỏi `FULL_JSON` và `BOUNDING_BOX` được kiểm tra cú pháp hợp lệ bằng `json.loads()`, đảm bảo tọa độ không vượt quá khung ảnh $0 \le \text{coord} \le \text{dim}$.

---

# PHẦN 9: BẢN ĐỒ FILE DỮ LIỆU TRONG REPOSITORY

Toàn bộ tập dữ liệu đã được lưu trữ và đồng bộ trên GitHub repository:

* 📦 [**`model/data/vlm_train_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_train_master.json) – 97,508 mẫu Train Master (~34.0 MB)
* 📦 [**`model/data/vlm_val_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_val_master.json) – 17,208 mẫu Validation Master (~5.98 MB)
* 📊 [**`model/data/dataset_summary.json`**](file:///d:/STUDY/MLIoT/project/model/data/dataset_summary.json) – File tổng hợp phân bổ 15 mẫu hóa đơn
* 🎯 [**`datasets/val_benchmark_upload/multitemplate_validation_questions.json`**](file:///d:/STUDY/MLIoT/project/datasets/val_benchmark_upload/multitemplate_validation_questions.json) – 174 câu hỏi kiểm định độc lập
* 🐍 [**`model/demo_gradio.py`**](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py) – Ứng dụng Web Demo tương tác người dùng
