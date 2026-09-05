# 🎬 KỊCH BẢN QUAY VIDEO DEMO SẢN PHẨM HOÀN CHỈNH (SHOOTING SCRIPT)
## Đề tài: Hệ Thống DocVQA & Bóc Tách Hóa Đơn Tiếng Việt (Qwen2.5-VL LoRA)
* **Thời lượng mục tiêu:** **2 phút (120 giây)** — Chuẩn thời lượng trình chiếu tại Slide 9 và Hội đồng phản biện.
* **Mục tiêu video:** Thể hiện trọn vẹn **toàn bộ 5 trụ cột của sản phẩm** (Tiền xử lý ảnh ➔ Bóc tách VLM ➔ Điểm Tin Cậy 3 màu ➔ Hỏi đáp DocVQA ➔ Xuất JSON ERP).
* **Định dạng:** Full HD 1080p (1920x1080), 60fps, tỷ lệ 16:9, toàn màn hình không có tab rác.

---

## 📋 CHECKLIST CHUẨN BỊ TRƯỚC KHI BẤM NÚT QUAY (3 PHÚT SETUP)

1. **Chuẩn bị 2 file hóa đơn thực tế để ngoài Desktop:**
   * `hoa_don_highlands.jpg` hoặc `hoa_don_sieu_thi.jpg`: Hóa đơn in nhiệt có danh sách món dài (3–5 món), ảnh chụp hơi nghiêng nhẹ góc để phô diễn tính năng Deskew.
   * `hoa_don_vat_dien_tu.pdf`: Hóa đơn GTGT điện tử (PDF) để chứng minh hệ thống nhận cả ảnh chụp lẫn PDF.
2. **Khởi chạy ứng dụng:**
   * Bật Backend & Frontend (hoặc Gradio Demo).
   * Mở trình duyệt Chrome/Edge, truy cập giao diện hệ thống.
   * Bấm **`F11`** để vào chế độ Toàn màn hình (ẩn thanh bookmark, ẩn taskbar Windows).
   * Phóng to trình duyệt lên **110% hoặc 125%** (`Ctrl` + con lăn chuột) để chữ to rõ khi chiếu lên màn hình lớn.
3. **Phần mềm quay màn hình:**
   * Bấm **`Win + Alt + R`** (Xbox Game Bar có sẵn trên Windows 10/11) để bắt đầu và kết thúc quay. File video `.mp4` sẽ tự lưu trong thư mục `C:\Users\<User>\Videos\Captures`.

---

## ⏱️ BẢNG PHÂN CẢNH CHI TIẾT TỪNG GIÂY (STORYBOARD 120S)

```
00:00 ─── 00:20 ─── 00:50 ─────── 01:20 ────────── 01:45 ───── 02:00
  Mở đầu &     Tiền xử lý &    Hỏi đáp DocVQA      Minh chứng &    Kết thúc &
  Tổng quan    Bóc tách VLM    Tiếng Việt          Xuất JSON       Tích hợp ERP
```

---

### 📍 PHÂN CẢNH 1: TỔNG QUAN GIAO DIỆN & TIẾP NHẬN CHỨNG TỪ (`00:00 - 00:20`)
* **Thời lượng:** 20 giây
* **Trụ cột phô diễn:** Giao diện Web hiện đại + Khả năng nhận đa định dạng (Ảnh chụp/PDF).

| Giây | Thao tác chuột & Màn hình hiển thị | Tính năng làm nổi bật | Lời thoại thuyết minh (hoặc Chữ phụ đề) |
| :---: | :--- | :--- | :--- |
| **00:00 - 00:08** | Chuột di nhẹ qua tiêu đề hệ thống **DocVQA System** và bảng điều khiển 3 vùng: Vùng tải tài liệu (Trái), Vùng xem chứng từ (Giữa), Vùng kết quả kế toán (Phải). | Giao diện Dashboard chuyên nghiệp, thân thiện với kế toán. | *"Chào mừng Thầy/Cô đến với giao diện hoàn chỉnh của Hệ thống DocVQA bóc tách hóa đơn tiếng Việt."* |
| **00:08 - 00:20** | Kéo thả file `hoa_don_sieu_thi.jpg` (ảnh in nhiệt hơi nghiêng) vào khung Upload. Ảnh tải lên thành công, hiển thị xem trước rõ nét. | Nhận diện kéo thả tức thì, hỗ trợ cả PNG, JPG và PDF. | *"Kế toán viên chỉ cần kéo thả một tờ hóa đơn in nhiệt thực tế từ điện thoại vào hệ thống."* |

---

### 📍 PHÂN CẢNH 2: TIỀN XỬ LÝ STAGE 0 & BÓC TÁCH VLM SINGLE-PASS (`00:20 - 00:50`)
* **Thời lượng:** 30 giây
* **Trụ cột phô diễn:** Stage 0 (Tự xoay thẳng Deskew/Tăng tương phản) + VLM Engine (Bóc tách trọn vẹn trường thông tin).

| Giây | Thao tác chuột & Màn hình hiển thị | Tính năng làm nổi bật | Lời thoại thuyết minh (hoặc Chữ phụ đề) |
| :---: | :--- | :--- | :--- |
| **00:20 - 00:25** | Bấm nút **🚀 "Trích xuất thông tin"** (nút màu xanh nổi bật). Icon xoay tải dữ liệu hiện lên mượt mà. | Kích hoạt chuỗi xử lý tự động khép kín. | *"Chỉ với một cú click, hệ thống tự động kích hoạt chuỗi xử lý khép kín."* |
| **00:25 - 00:35** | Màn hình hoàn tất sau ~3 giây. Ảnh hóa đơn ở giữa được **tự động xoay thẳng góc (Deskew)** và tăng tương phản chữ in mờ. | **Stage 0 Preprocessing** tự động khắc phục ảnh mờ, nghiêng. | *"Stage 0 tự động cân chỉnh góc xoay thẳng thớm và khử nhiễu nền cho chữ in nhiệt bị mờ."* |
| **00:35 - 00:50** | Bảng kê **Fields Ledger** hiện đầy đủ: <br>• **Tên bên bán:** CÔNG TY CP TM BÁCH HÓA XANH<br>• **Ngày lập:** 14/06/2026 18:30<br>• **Tổng thanh toán:** 348.000đ<br>• **Mã số thuế:** 0310438543 | **VLM Single-pass:** Trích xuất đồng thời thực thể và ngữ cảnh không gian 2D. | *"Mô hình Qwen2.5-VL LoRA lập tức trích xuất toàn bộ các thực thể cốt lõi với độ chính xác cao nhất."* |

---

### 📍 PHÂN CẢNH 3: CƠ CHẾ ĐIỂM TIN CẬY 3 MÀU BẢO VỆ KẾ TOÁN (`00:50 - 01:15`)
* **Thời lượng:** 25 giây
* **Trụ cột phô diễn:** Stage 2 Confidence Scoring (🟢 Xanh / 🟡 Vàng / 🔴 Đỏ) + Format Sanity Check.

| Giây | Thao tác chuột & Màn hình hiển thị | Tính năng làm nổi bật | Lời thoại thuyết minh (hoặc Chữ phụ đề) |
| :---: | :--- | :--- | :--- |
| **00:50 - 01:02** | Rê chuột vào huy hiệu **🟢 Xanh (98%)** cạnh trường *Tên bên bán* và **🟢 Xanh (96%)** cạnh *Tổng tiền*. Tooltip giải thích: "Độ tin cậy cao - Tự động ghi sổ". | **Confidence Scoring:** Phân cấp an toàn dựa trên Logits và Format Sanity Check. | *"Điểm sáng cốt lõi của hệ thống là cơ chế Điểm Tin Cậy: Các trường đạt trên 85% nhãn Xanh được tự động duyệt vào sổ sách."* |
| **01:02 - 01:15** | Rê chuột vào một trường nhãn **🟡 Vàng (72%)** (ví dụ: Địa chỉ hoặc ghi chú mờ). Hệ thống nhắc: "Cần kế toán đối soát nhanh". | Kiểm soát rủi ro kế toán thực tế (Risk Governance). | *"Với các trường bị mờ đạt nhãn Vàng, hệ thống cảnh báo kế toán viên liếc mắt kiểm tra, triệt tiêu rủi ro sai lệch thuế."* |

---

### 📍 PHÂN CẢNH 4: HỎI ĐÁP TỰ NHIÊN DOCVQA & BẢNG KÊ HÀNG HÓA (`01:15 - 01:45`)
* **Thời lượng:** 30 giây
* **Trụ cột phô diễn:** Interactive DocVQA tiếng Việt + Khắc phục lỗi cắt cụt danh mục món (`ITEMS_LIST`).

| Giây | Thao tác chuột & Màn hình hiển thị | Tính năng làm nổi bật | Lời thoại thuyết minh (hoặc Chữ phụ đề) |
| :---: | :--- | :--- | :--- |
| **01:15 - 01:25** | Click vào khung chat **QA Console**. Nhập câu hỏi: <br>`Danh sách các mặt hàng trên hóa đơn gồm những gì?` <br>Bấm **Gửi**. | Khả năng đọc hiểu câu hỏi ngôn ngữ tự nhiên tiếng Việt. | *"Bên cạnh trích xuất form mẫu, kế toán có thể trò chuyện trực tiếp với tài liệu bằng câu hỏi tự nhiên."* |
| **01:25 - 01:35** | Mô hình trả về kết quả mượt mà: <br>`1. Mì tôm Hảo Hảo chua cay: 3 x 3.500`<br>`2. Nước suối Lavie 500ml: 2 x 6.000`<br>`3. Cafe sữa Wake-up 240ml: 4 x 12.000` | **Dynamic Token Allocation 384 tokens:** Bóc tách trọn vẹn 100% món, không bị cụt dòng. | *"Nhờ cơ chế cấp phát token động, mô hình liệt kê trọn vẹn toàn bộ danh mục mặt hàng mà không bị đứt chữ giữa chừng."* |
| **01:35 - 01:45** | Nhập nhanh câu hỏi nghiệp vụ thứ hai: <br>`Hóa đơn này thanh toán bằng hình thức nào?` <br>Mô hình trả lời ngay: `Tiền mặt (Cash)`. | Khả năng suy luận ngữ cảnh sâu của VLM. | *"Mô hình hiểu sâu cả các trường thứ yếu như phương thức thanh toán, thuế suất hay phụ phí."* |

---

### 📍 PHÂN CẢNH 5: XUẤT CẤU TRÚC JSON & TÍCH HỢP ERP DOANH NGHIỆP (`01:45 - 02:00`)
* **Thời lượng:** 15 giây
* **Trụ cột phô diễn:** Data Export, tính sẵn sàng tích hợp phần mềm kế toán MISA/SAP/ERP.

| Giây | Thao tác chuột & Màn hình hiển thị | Tính năng làm nổi bật | Lời thoại thuyết minh (hoặc Chữ phụ đề) |
| :---: | :--- | :--- | :--- |
| **01:45 - 01:53** | Bấm nút **📥 "Xuất JSON"** (Download). <br>Mở nhanh cửa sổ file `.json` vừa tải về (hoặc popup JSON Viewer). | Cấu trúc dữ liệu chuẩn hóa, phân cấp rõ ràng theo schema kế toán. | *"Cuối cùng, toàn bộ chứng từ được đóng gói thành file JSON phân cấp chuẩn chỉ trong một lần bấm."* |
| **01:53 - 02:00** | Quay lại màn hình chính, nút "Tải tài liệu mới" sẵn sàng. <br>Hiện dòng kết thúc: **"Document VQA Engine - Sẵn sàng tích hợp ERP MISA / FAST / SAP"**. | Sản phẩm hoàn chỉnh, tính thực chiến cao. | *"Dữ liệu sẵn sàng nạp trực tiếp vào hệ thống ERP doanh nghiệp. Xin cảm ơn Thầy/Cô đã theo dõi!"* |

---

## 🎯 BÍ QUYẾT QUAY VIDEO "ĂN ĐIỂM TUYỆT ĐỐI"

1. **Thao tác chuột như một chuyên gia:**
   * Không rê chuột vòng vòng hoặc lắc chuột lung tung.
   * Di chuột thẳng đến nút bấm, click dứt khoát, sau đó **giữ yên con trỏ chuột trong 1.5 giây** để người xem kịp nhìn thấy vị trí click.
2. **Chuẩn bị văn bản mẫu để Paste nhanh:**
   * Tạo 1 file Notepad nhỏ bên ngoài chứa sẵn câu hỏi:
     `Danh sách các mặt hàng trên hóa đơn gồm những gì?`
   * Khi quay đến đoạn hỏi đáp, chỉ cần bấm `Ctrl + V` dán vào khung chat thay vì ngồi gõ từng phím (vừa nhanh, vừa tránh bị gõ nhầm chữ).
3. **Âm thanh và Lời bình:**
   * Bạn có thể thu âm giọng nói đệm theo đúng các mốc thời gian trên (giọng đọc chậm rãi, rõ ràng).
   * Hoặc nếu không muốn lồng tiếng, bạn có thể dùng công cụ dựng video (như CapCut / Clipchamp) **chèn chữ phụ đề chạy phía dưới màn hình (Subtitles)** kèm theo một đoạn nhạc nền công nghệ nhẹ nhàng (Tech Background Music - âm lượng nhỏ 15%).
4. **Chèn Video vào Slide:**
   * Sau khi quay xong file `.mp4`, chèn trực tiếp vào giữa **Slide 9** trong PowerPoint/Canva.
   * Đặt chế độ video: *"Play Automatically"* (Tự động phát khi bấm chuyển sang Slide 9) và *"Full Screen"* để buổi báo cáo diễn ra trơn tru nhất!
