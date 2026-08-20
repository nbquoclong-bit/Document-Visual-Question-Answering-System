# 📑 Data Preprocessing & Labels (FUNSD Format)

Kho lưu trữ này chứa toàn bộ dữ liệu nhãn (Bounding box, Text, Label) đã được làm sạch và chuẩn hóa về **định dạng FUNSD** phục vụ cho bài toán Information Extraction (KIE).

## 🗂️ Cấu trúc dữ liệu hiện tại trên Git
Trên Git hiện chỉ lưu các file JSON (rất nhẹ) để phục vụ cho việc tracking code. Dữ liệu được chia làm 3 tập chính:

* `datasets/MCOCR/`: Tập dữ liệu hóa đơn tiếng Việt từ cuộc thi MC_OCR.
* `datasets/SROIE/`: Tập dữ liệu hóa đơn tiếng Anh SROIE 2019 (đã được map nhãn tự động).
* `datasets/vietnamese-receipts-v3/`: Tập dữ liệu hóa đơn thu thập thêm.

## 📥 Hướng dẫn lấy Dữ liệu Ảnh (Dành cho đội KIE)
Để train model, các bạn bắt buộc phải có file ảnh. Vì lý do tối ưu dung lượng, toàn bộ ảnh (bao gồm ảnh gốc và ảnh đã qua Augmentation) được lưu trữ riêng trên Google Drive. 

**Các bạn thực hiện theo các bước sau để setup môi trường train:**

1. **Clone repo này về máy:**
   ```bash
   git clone <https://github.com/nbquoclong-bit/Document-Visual-Question-Answering-System>

2. **Tải file Ảnh từ Google Drive:**
   * Truy cập link sau để tải toàn bộ ảnh Train/Val: **[https://drive.google.com/file/d/1-4VbizQvv_goFrmC3DZF5j2wJUenco5X/view?usp=sharing]**
   * Giải nén file vừa tải về.

3. **Đồng bộ Dữ liệu:**
   * Trong file tải về từ Drive sẽ có các thư mục ảnh tương ứng. 
   * Hãy copy các thư mục ảnh đó và dán vào bên cạnh các thư mục chứa JSON trong folder `datasets/`.
   * Cấu trúc hoàn thiện trên máy của bạn để sẵn sàng train sẽ trông như thế này:
     ```text
     datasets/
     ├── MCOCR/
     │   ├── images/       <-- (Lấy từ Drive)
     │   └── funsd_json/   <-- (Kéo từ Git về)
     ├── SROIE/
     │   ├── images/       <-- (Lấy từ Drive)
     │   └── funsd_json/   <-- (Kéo từ Git về)
     └── vietnamese-receipts-v3/
         ├── images/       <-- (Lấy từ Drive)
         └── funsd_json/   <-- (Kéo từ Git về)
     ```

---
**Phụ trách module Data & Preprocessing:** [Nguyễn Văn Nhật Nam]