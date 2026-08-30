# 📑 ĐỀ CƯƠNG CHI TIẾT LÀM SLIDE THUYẾT TRÌNH ĐỒ ÁN
## Đề Tài: Xây Dựng Hệ Thống Document Visual Question Answering (DocVQA) Cho Hóa Đơn Tiếng Việt Sử Dụng Qwen2.5-VL và LoRA

> **Dành riêng cho các thành viên trong nhóm thiết kế Slide (PowerPoint / Canva / Google Slides).**  
> Mỗi slide bên dưới được chia sẵn: **Tiêu đề**, **Bố cục hình ảnh/Icon gợi ý**, **Nội dung gạch đầu dòng ngắn gọn (Copy-paste trực tiếp)** và **Lời thoại thuyết trình (Speaker Notes)**.

---

```
                       CẤU TRÚC BỘ SLIDE (12 SLIDES CHÍNH + 3 BACKUP)
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │ Slide 1-3     │ ──> │ Slide 4-7     │ ──> │ Slide 8-10    │ ──> │ Slide 11-12   │
 │ ĐẶT VẤN ĐỀ    │     │ PHƯƠNG PHÁP   │     │ KẾT QUẢ       │     │ DEMO WEB      │
 │ & MỤC TIÊU    │     │ & CÔNG NGHỆ   │     │ & THỰC NGHIỆM │     │ & KẾT LUẬN    │
 └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

---

### 🖥️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề lớn:** HỆ THỐNG TRÍCH XUẤT THÔNG TIN HÓA ĐƠN THÔNG MINH (DOCUMENT VQA)
* **Tiêu đề phụ:** Ứng Dụng Mô Hình Vision-Language Thế Hệ Mới Qwen2.5-VL & Kỹ Thuật Fine-Tuning LoRA
* **Thông tin nhóm:**
  * Giảng viên hướng dẫn: [Tên Thầy/Cô]
  * Thành viên thực hiện: [Tên các bạn trong nhóm]
  * Lớp / Ngành / Năm học: 2025 - 2026
* **Gợi ý thiết kế:** Nền màu xanh đậm công nghệ (Deep Blue / Dark Slate), icon trí tuệ nhân tạo (AI), robot đọc hóa đơn.

---

### 🖥️ SLIDE 2: ĐẶT VẤN ĐỀ & GIỚI HẠN CỦA PIPELINE TRUYỀN THỐNG
* **Tiêu đề:** Thách Thức Khi Trích Xuất Hóa Đơn Tiếng Việt
* **Bố cục:** So sánh 2 cột (Pipeline Truyền Thống vs Thực Tế Phức Tạp)
* **Nội dung hiển thị:**
  * **Hóa đơn thực tế tại VN rất đa dạng:** Chữ in nhiệt mờ, giấy gấp nếp, chụp nghiêng, bảng biểu nhiều cột, font nghệ thuật.
  * **Hạn chế của Pipeline OCR + NLP cũ:**
    * ❌ **Lỗi lan truyền thác (Cascading Error):** OCR đọc sai 1 ký tự (`8` thành `B`, `0` thành `O`) $\implies$ NLP trích xuất sai toàn bộ số tiền/ngày tháng.
    * ❌ **Mất thông tin không gian 2D:** OCR làm phẳng ảnh thành văn bản 1 chiều, mất liên kết giữa nhãn bên trái và số tiền bên phải.
* **Speaker Notes:** *"Các hệ thống trước đây phải qua 2 giai đoạn riêng biệt nên lỗi nối tiếp nhau. Nhóm đề xuất tiếp cận theo hướng End-to-End Multimodal VLM."*

---

### 🖥️ SLIDE 3: GIẢI PHÁP ĐỀ XUẤT - END-TO-END MULTIMODAL VLM
* **Tiêu đề:** Kiến Trúc Giải Pháp Đột Phá
* **Bố cục:** Sơ đồ dòng dữ liệu từ Ảnh $\rightarrow$ Mô hình VLM $\rightarrow$ Kết quả
* **Nội dung hiển thị:**
  * **Đầu vào:** Nhận trực tiếp **Pixel ảnh hóa đơn + Câu hỏi tự nhiên** (Tiếng Việt).
  * **Mô hình cốt lõi:** **Qwen2.5-VL-3B-Instruct** (Mô hình thị giác - ngôn ngữ mới nhất 2025).
  * **Đầu ra:** Trả về trực tiếp trường thông tin hoặc **Cấu trúc JSON chuẩn hóa**.
  * **Ưu điểm vượt trội:** Nhìn trực tiếp ngữ cảnh 2 chiều, loại bỏ hoàn toàn OCR trung gian.

---

### 🖥️ SLIDE 4: MÔ HÌNH NỀN TẢNG QWEN2.5-VL-3B
* **Tiêu đề:** Sức Mạnh Công Nghệ Của Qwen2.5-VL-3B
* **Bố cục:** 3 khối tính năng chính (Cards/Boxes)
* **Nội dung hiển thị:**
  * 👁️ **2D Spatial M-RoPE:** Mã hóa vị trí không gian 2 chiều, giúp mô hình dóng hàng cột số tiền chuẩn xác.
  * 📐 **Native Dynamic Resolution:** Xử lý ảnh ở mọi tỉ lệ khung hình mà không làm biến dạng chữ in nhỏ.
  * 🧠 **Dung lượng 3B tối ưu:** Hiệu năng tương đương model 7B cũ nhưng vừa vặn trong **1 GPU Tesla T4 (16GB VRAM)**.

---

### 🖥️ SLIDE 5: KỸ THUẬT PEFT / LORA (LOW-RANK ADAPTATION)
* **Tiêu đề:** Tối Ưu Hóa Tham Số Bằng LoRA
* **Bố cục:** Hình minh họa ma trận $W = W_0 + \frac{\alpha}{r}(B \times A)$
* **Nội dung hiển thị:**
  * **Đóng băng 99.67% mô hình gốc:** Chống hiện tượng quên tri thức (*Catastrophic Forgetting*).
  * **LoRA Configuration:**
    * $\text{Rank } r = 16, \text{Alpha } \alpha = 32$ ($\text{Scaling Ratio} = 2.0$).
    * $\text{Dropout} = 0.05$ (Chống học vẹt).
  * **Can thiệp cả 7 ma trận chiếu tuyến tính:** `q, k, v, o` (Attention) và `gate, up, down` (MLP).
  * **Kết quả:** Chỉ huấn luyện **0.33% tham số** (~10.2M params), Adapter cực nhẹ **~75 MB**.

---

### 🖥️ SLIDE 6: TỰ ĐỘNG TỐI ƯU SIÊU THAM SỐ (AutoML / BAYESIAN OPTIMIZATION)
* **Tiêu đề:** Phương Pháp Luận Chọn Siêu Tham Số
* **Bố cục:** Biểu đồ đường cong Learning Rate và quy trình Optuna TPE
* **Nội dung hiển thị:**
  * **Không chọn tham số cảm tính:** Ứng dụng **Bayesian Optimization (Optuna TPE)** kết hợp **Gradient-based LR Finder**.
  * **LR Finder:** Dùng đạo hàm $\frac{d\mathcal{L}}{d\text{LR}}$ tìm điểm dốc nhất $\implies \text{LR} = 2 \times 10^{-4}$.
  * **ASHA Pruning:** Tự động cắt tỉa các trial kém hiệu quả để tiết kiệm tài nguyên GPU.
  * **Bộ tham số tối ưu:** $\text{LR} = 2 \times 10^{-4}, r=16, \alpha=32, \text{Effective Batch Size} = 16, \text{Native FP16}$.

---

### 🖥️ SLIDE 7: KỸ THUẬT TARGET-ONLY LOSS MASKING
* **Tiêu đề:** Bí Quyết Đạt Exact Match Tuyệt Đối
* **Bố cục:** So sánh Trước và Sau khi áp dụng Masking nhãn `-100`
* **Nội dung hiển thị:**
  * ❌ **Vấn đề của Base Model:** Sinh lời dẫn lan man (*"Dựa vào ảnh hóa đơn, số tiền là..."*) $\implies$ Bị phạt 0 điểm ANLS.
  * 💡 **Giải pháp:** Gán nhãn `ignore_index = -100` cho toàn bộ Vision tokens và Question tokens.
  * 🎯 **Công thức:** $\mathcal{L} = -\frac{1}{|T_{\text{target}}|} \sum_{t \in T_{\text{target}}} \log P(y_t \mid \dots)$
  * ✅ **Kết quả:** Triệt tiêu 100% lời dẫn thừa, ép mô hình trả về chính xác chuỗi kết quả đích.

---

### 🖥️ SLIDE 8: BỘ DỮ LIỆU HUẤN LUYỆN & BENCHMARK
* **Tiêu đề:** Dữ Liệu Thực Nghiệm (15 Loại Hóa Đơn)
* **Bố cục:** Lưới logo 15 thương hiệu và bảng thống kê số lượng
* **Nội dung hiển thị:**
  * **Tổng quy mô:** **114,716 cặp VQA** trên **4,995 ảnh hóa đơn**.
  * **Phân chia:** Train Master (97k), Val Master (17k), Benchmark Test (174 câu hỏi độc lập).
  * **15 Mẫu hóa đơn thực tế:**
    * *Cafe & F&B:* Highlands Coffee, Phúc Long, Starbucks, KFC, Jollibee.
    * *Cửa hàng tiện lợi:* 7-Eleven, Circle K, GS25, Minimart An An.
    * *Siêu thị:* WinMart, Lotte Mart, Bách Hóa Xanh.
    * *e-Invoice & Biên lai:* Viettel e-Invoice, VNPT e-Invoice, Mẫu C45-BB.

---

### 🖥️ SLIDE 9: BẢNG SO SÁNH ĐỐI CHỨNG HIỆU NĂNG (KEY SLIDE ⭐)
* **Tiêu đề:** Kết Quả Định Lượng Trên 15 Loại Hóa Đơn
* **Bố cục:** Bảng số liệu to rõ, nổi bật các con số tăng trưởng màu xanh lá
* **Nội dung bảng:**

| Chỉ số (Metric) | Base Model (Zero-Shot) | Fine-Tuned (LoRA) | Mức Cải Thiện |
| :--- | :---: | :---: | :---: |
| **ANLS (DocVQA Metric)** | `2.22%` | **`100.00%`** | **+97.78%** 🚀 |
| **Exact Match (EM %)** | `2.22%` | **`100.00%`** | **Gấp 45 lần** 💥 |
| **Token F1-Score** | `40.09%` | **`100.00%`** | **+59.91%** 🎯 |
| **Độ trễ trung bình** | `2.85s` / câu | **`2.15s`** / câu | **Nhanh hơn 25%** ⚡ |
| **VRAM GPU** | `7.5 GB` | `8.2 GB` | **Vừa vặn Tesla T4** |

---

### 🖥️ SLIDE 10: PHÂN TÍCH THEO DANH MỤC & CẤU TRÚC JSON
* **Tiêu đề:** Khả Năng Trích Xuất Đa Trường & Xuất JSON
* **Bố cục:** Chia làm 2 phần (Biểu đồ cột theo nhóm trường + Ví dụ JSON đầu ra)
* **Nội dung hiển thị:**
  * **Đạt 100% trên cả 7 nhóm trường:** `SELLER`, `TOTAL_COST`, `TIMESTAMP`, `ADDRESS`, `ITEM_PRICE`, `ITEM_QTY`.
  * **Trích xuất JSON phân cấp:**
    ```json
    {
      "seller": "HIGHLANDS COFFEE",
      "timestamp": "28/06/2026 09:15",
      "total_cost": "109,000",
      "items": [{"name": "Trà Sen Vàng", "qty": 2, "price": "109,000"}]
    }
    ```

---

### 🖥️ SLIDE 11: DEMO ỨNG DỤNG WEB TRÊN GPU THỰC TẾ
* **Tiêu đề:** Triển Khai Thực Tế (Gradio Web Application)
* **Bố cục:** Ảnh chụp màn hình giao diện Web Demo kéo thả hóa đơn
* **Nội dung hiển thị:**
  * Giao diện trực quan: Upload ảnh chụp $\rightarrow$ Chọn câu hỏi / Nhập câu hỏi tự do.
  * Tốc độ phản hồi: ~2 giây trên GPU Tesla T4.
  * Trả về kết quả song song: Câu trả lời ngắn + JSON phân tích + Bounding Box định vị.

---

### 🖥️ SLIDE 12: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
* **Tiêu đề:** Tổng Kết & Hướng Mở Rộng
* **Bố cục:** 2 khối (Thành tựu đạt được vs Hướng mở rộng)
* **Nội dung hiển thị:**
  * **Thành tựu:** Xây dựng thành công hệ thống DocVQA tiếng Việt đạt độ chính xác tuyệt đối, triển khai gọn nhẹ trên GPU đơn.
  * **Hướng phát triển:** Mở rộng sang chứng từ kế toán đa trang (PDF), hợp đồng pháp lý và hóa đơn viết tay.
* **Lời cảm ơn:** *"Xin chân thành cảm ơn Quý Thầy Cô trong Hội đồng đã lắng nghe!"*

---

### 🛡️ BACKUP SLIDES: SẴN SÀNG TRẢ LỜI PHẢN BIỆN (Q&A)

1. **Tại sao không dùng OCR + LLM?**  
   $\rightarrow$ Tránh lỗi lan truyền (Cascading Error) và giữ nguyên thông tin liên kết không gian 2 chiều.
2. **Tại sao chọn LoRA Rank 16 mà không phải 64?**  
   $\rightarrow$ Đã kiểm chứng qua Optuna TPE: $r=16$ cho điểm hội tụ tương đương $r=64$ nhưng tiết kiệm 75% bộ nhớ.
3. **Mô hình có chạy được trên GPU phổ thông không?**  
   $\rightarrow$ Có, mô hình Native FP16 chỉ chiếm ~8.2GB VRAM, chạy mượt trên RTX 3060/3070/4060 hoặc Tesla T4 miễn phí trên Kaggle.
