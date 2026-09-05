# 📋 DANH SÁCH HÓA ĐƠN MẪU & CÂU HỎI QUAY VIDEO DEMO
> Tài liệu chuẩn bị sẵn dữ liệu và câu hỏi để người thuyết trình copy-paste nhanh khi quay clip demo (Slide 9).

---

## 📂 THƯ MỤC CHỨA TOÀN BỘ ẢNH HÓA ĐƠN MẪU
Toàn bộ ảnh hóa đơn kiểm định thực tế có sẵn tại:
```text
d:\STUDY\MLIoT\project cuối kì\Document-Visual-Question-Answering-System\datasets\val_benchmark_upload\images\
```

---

## ☕ 1. MẪU HÓA ĐƠN 1: CHUỖI F&B HIGHLANDS COFFEE (KHUYÊN DÙNG CHÍNH ĐỂ QUAY)
* **Tên file ảnh:** `cafe_highlands_val_001.png`
* **Đường dẫn file:**
  `d:\STUDY\MLIoT\project cuối kì\Document-Visual-Question-Answering-System\datasets\val_benchmark_upload\images\cafe_highlands_val_001.png`
* **Đặc điểm nổi bật:** Hóa đơn in nhiệt thực tế, danh sách đồ uống dài (6 món) để phô diễn tính năng sinh đủ không bị đứt chữ.

### 💬 Danh sách câu hỏi để Copy - Paste nhanh vào khung Demo:

1. **Câu hỏi 1 (Trích xuất toàn bộ JSON):**
   ```text
   Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON đầy đủ tất cả các trường.
   ```
   * *Đáp án mô hình trả về:* Full JSON chứa SELLER, TIMESTAMP, TOTAL_COST, ADDRESS, ITEMS_LIST.

2. **Câu hỏi 2 (Bóc tách danh sách món - ITEMS_LIST):**
   ```text
   Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?
   ```
   * *Đáp án chuẩn (Ground Truth):*
     `Trà Sen Vàng Size M, Cà Phê Đen Đá Size M, Trà Thạch Đào Size L, Bánh Tiramisu, Freeze Trà Xanh Size M, Phin Sữa Đá Size L`

3. **Câu hỏi 3 (Bóc tách tổng tiền - TOTAL_COST):**
   ```text
   Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?
   ```
   * *Đáp án chuẩn:* `796,068`

4. **Câu hỏi 4 (Bóc tách tên quán - SELLER):**
   ```text
   Tên đơn vị / người bán hàng trên hóa đơn là gì?
   ```
   * *Đáp án chuẩn:* `HIGHLANDS COFFEE TRẦN HƯNG ĐẠO`

5. **Câu hỏi 5 (Bóc tách đơn giá từng món - ITEM_PRICE):**
   ```text
   Thành tiền của Trà Sen Vàng Size M là bao nhiêu?
   ```
   * *Đáp án chuẩn:* `135,000`

---

## 🛒 2. MẪU HÓA ĐƠN 2: BÁN LẺ SIÊU THỊ WINMART
* **Tên file ảnh:** `supermarket_winmart_val_001.png`
* **Đường dẫn file:**
  `d:\STUDY\MLIoT\project cuối kì\Document-Visual-Question-Answering-System\datasets\val_benchmark_upload\images\supermarket_winmart_val_001.png`
* **Đặc điểm nổi bật:** Hóa đơn siêu thị nhiều dòng hàng tạp hóa, có mã vạch và ngày giờ mua sắm.

### 💬 Danh sách câu hỏi để Copy - Paste:

1. **Câu hỏi 1 (Danh sách hàng hóa):**
   ```text
   Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?
   ```
   * *Đáp án chuẩn:*
     `Coca Cola Lon 320ml, Nước rửa chén Sunlight 750ml, Khăn giấy Pulppy 100 tờ, Mì tôm Hảo Hảo chua cay`

2. **Câu hỏi 2 (Tổng tiền):**
   ```text
   Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?
   ```
   * *Đáp án chuẩn:* `94.050`

3. **Câu hỏi 3 (Thời gian lập):**
   ```text
   Ngày giờ lập hóa đơn là khi nào?
   ```
   * *Đáp án chuẩn:* `12/06/2026 13:46`

4. **Câu hỏi 4 (Đơn giá):**
   ```text
   Thành tiền của Coca Cola Lon 320ml là bao nhiêu?
   ```
   * *Đáp án chuẩn:* `10.000`

---

## 🏢 3. MẪU HÓA ĐƠN 3: HÓA ĐƠN ĐIỆN TỬ DOANH NGHIỆP (VIETTEL E-INVOICE)
* **Tên file ảnh:** `einvoice_viettel_val_001.png`
* **Đường dẫn file:**
  `d:\STUDY\MLIoT\project cuối kì\Document-Visual-Question-Answering-System\datasets\val_benchmark_upload\images\einvoice_viettel_val_001.png`
* **Đặc điểm nổi bật:** Chứng từ kế toán chính thức, tổng giá trị lớn (>24 triệu), có tên công ty, địa chỉ hành chính 4 cấp, thiết bị văn phòng.

### 💬 Danh sách câu hỏi để Copy - Paste:

1. **Câu hỏi 1 (Tên công ty phát hành):**
   ```text
   Tên đơn vị / người bán hàng trên hóa đơn là gì?
   ```
   * *Đáp án chuẩn:* `CÔNG TY CỔ PHẦN ĐẦU TƯ & PHÁT TRIỂN HƯNG PHÁT`

2. **Câu hỏi 2 (Tổng tiền thanh toán):**
   ```text
   Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?
   ```
   * *Đáp án chuẩn:* `24,389,200đ`

3. **Câu hỏi 3 (Địa chỉ doanh nghiệp):**
   ```text
   Địa chỉ của đơn vị bán hàng là ở đâu?
   ```
   * *Đáp án chuẩn:* `Số 322 Lý Thường Kiệt, Quận Ba Đình, Hà Nội`

4. **Câu hỏi 4 (Danh sách thiết bị & dịch vụ):**
   ```text
   Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?
   ```
   * *Đáp án chuẩn:*
     `Bút bi Thiên Long FO-03, Giấy Double A A4 70gsm, Dịch vụ Bảo trì Hệ thống mạng, Máy in HP LaserJet Pro, Mực in Canon Cartridge`

---

## 🎯 KỊCH BẢN ĐỀ XUẤT ĐỂ QUAY CLIP MƯỢT NHẤT TRONG 90 GIÂY
* **Bước 1:** Kéo thả ảnh `cafe_highlands_val_001.png` vào hệ thống.
* **Bước 2:** Bấm nút *"Trích xuất thông tin"* $\implies$ Hiện bảng kê kế toán và huy hiệu **🟢 Xanh** Tên bên bán, Tổng tiền.
* **Bước 3:** Copy câu hỏi: `Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?` dán vào khung chat $\implies$ Mô hình trả lời đủ 6 món hàng.
* **Bước 4:** Copy câu hỏi: `Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?` dán vào khung chat $\implies$ Mô hình trả lời `796,068`.
* **Bước 5:** Bấm *"Xuất JSON"* $\implies$ Kết thúc video hoàn hảo!
