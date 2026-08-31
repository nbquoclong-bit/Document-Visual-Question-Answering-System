# 🎓 ĐỀ CƯƠNG THUYẾT TRÌNH BÁO CÁO: CASE STUDY PHÂN TÍCH CÁC THỬ NGHIỆM THẤT BẠI TRONG MODULE VISUAL GROUNDING (BOUNDING BOX)

> **Dự án:** Hệ thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt  
> **Chủ đề thuyết trình:** Phân Tích Thực Nghiệm Thất Bại & Bài Học Kinh Nghiệm Kỹ Thuật trong Module Bounding Box  
> **Đối tượng báo cáo:** Giảng viên hướng dẫn & Hội đồng bảo vệ đồ án môn học  

---

## 💡 1. TẠI SAO BÁO CÁO VỀ THẤT BẠI LẠI ĐẠT ĐIỂM HÀN LÂM RẤT CAO?
Trong nghiên cứu Khoa học Dữ liệu và Kỹ thuật Máy học (Machine Learning Engineering), một bài báo cáo chỉ trình bày kết quả màu hồng thường thiếu tính thuyết phục. Việc **mổ xẻ chi tiết nguyên nhân thất bại ở mức độ toán học và kiến trúc hệ thống** chứng minh:
1. Bạn thực sự hiểu sâu cơ chế bên trong của mô hình Vision-Language (Attention, 2D M-RoPE, Float16/BFloat16, Gradient Accumulation), không dùng mô hình như một "hộp đen".
2. Bạn có tư duy kỹ thuật thực nghiệm (Empirical Debugging) có phương pháp, biết cách khoanh vùng và giải thích nguyên nhân gốc rễ (Root Cause Analysis).

---

## 📑 2. DÀN Ý SLIDE THUYẾT TRÌNH CHI TIẾT (7 SLIDE CHUẨN)

```
SLIDE 1: Đặt Vấn Đề: Tại Sao Bounding Box Cho Hóa Đơn Lại Cực Kỳ Khó?
SLIDE 2: Thất Bại #1: Tiếp Cận 2-Stage Pipeline (VLM + Heuristic OCR)
SLIDE 3: Mổ Xẻ 2 Lỗi Kinh Điển: Lexical Collision & Header Semantic Trap
SLIDE 4: Thất Bại #2: Tiếp Cận End-to-End V2 (Native Multi-Task Grounding)
SLIDE 5: Mổ Xẻ Hiện Tượng "Loss: NaN" & Tràn Số Float16 Trên 2D M-RoPE
SLIDE 6: Điểm Sáng Đối Chứng: Mô Hình V1 Đạt 94.94% ANLS & Giải Pháp Hybrid
SLIDE 7: 4 Bài Học Kinh Nghiệm Quý Giá Cho Kỹ Sư Machine Learning
```

---

### 🖼️ SLIDE 1: ĐẶT VẤN ĐỀ - TẠI SAO BOUNDING BOX HÓA ĐƠN LẠI KHÓ?
* **Mục tiêu:** Không chỉ trích xuất chữ/số (ví dụ: `12.000.000đ`), mà phải **khoanh đúng vùng chữ nhật** trên hóa đơn để kế toán đối soát trực quan.
* **Đặc thù phức tạp của hóa đơn tiếng Việt:**
  * Mật độ thông tin cực kỳ dày đặc (hàng trăm số điện thoại, số nhà, mã số thuế, số tiền nằm san sát nhau).
  * Bố cục bảng biểu (Table Grid) phức tạp với các dải tiêu đề màu sắc nổi bật.

---

### 🖼️ SLIDE 2 & 3: THẤT BẠI #1 - TIẾP CẬN 2-STAGE PIPELINE (VLM + HEURISTIC OCR)

* **Ý tưởng ban đầu:** 
  $$\text{Ảnh Hóa Đơn} \xrightarrow{\text{Qwen2.5-VL}} \text{Văn bản Trả lời (Text: 12.000.000đ)} \xrightarrow{\text{EasyOCR + Matcher}} \text{Vẽ Bounding Box}$$
* **Thực tế phát sinh 2 lỗi nghiêm trọng:**

#### 🔴 Lỗi 1: Xung đột xâu con chữ số (Lexical Substring Collision)
* *Hiện tượng:* Hỏi tổng tiền `12.000.000đ`, mô hình trả lời đúng `12.000.000đ`, nhưng khung đỏ lại khoanh lên đầu trang vào `Số ĐT: +84 912...` và `Số nhà: 123...`.
* *Bản chất lỗi:*
  * Thuật toán tách từ chia `12.000.000` thành token con `'12'`.
  * Chuỗi con `'12'` xuất hiện trong số điện thoại (`912`) và số nhà (`123`).
  * Do 2 dòng này nằm gần nhau ($Y < 25\text{px}$), giải thuật phân cụm (Clustering) đã gom chúng lại và tính điểm cao hơn số tiền đơn lẻ ở chân trang $\rightarrow$ **Khoanh sai hoàn toàn vị trí!**

#### 🔴 Lỗi 2: Bẫy ngữ nghĩa thanh tiêu đề (Table Header Semantic Trap)
* *Hiện tượng:* Khi hỏi về tiền thuế hoặc thành tiền, khung đỏ khoanh trọn vào thanh tiêu đề màu xanh: `Thành tiền | Thuế suất GTGT | Thuế GTGT | Tổng cộng`.
* *Bản chất lỗi:*
  * Câu trả lời có giải thích chứa nhiều từ vựng kế toán.
  * Thanh tiêu đề chứa tới 4 từ khóa trùng khớp liên tiếp, trong khi ô giá trị số ở đáy bảng chỉ chứa chữ số.
  * Thuật toán Overlap Score ưu tiên từ ngữ đã chọn nhầm thanh tiêu đề thay vì ô số liệu.

---

### 🖼️ SLIDE 4 & 5: THẤT BẠI #2 - TIẾP CẬN END-TO-END V2 (NATIVE GROUNDING & LOSS: NAN)

* **Ý tưởng ban đầu:** Huấn luyện mô hình sinh trực tiếp `{"answer": "...", "box": [ymin, xmin, ymax, xmax]}` qua cơ chế 2D M-RoPE không cần OCR ngoài.
* **Thực tế phát sinh lỗi nghiêm trọng (Ghi nhận trong `evaluation_report_v2.json`):**
  ```json
  "loss_history": [
    { "epoch": 1, "loss": NaN },
    { "epoch": 2, "loss": NaN },
    { "epoch": 3, "loss": NaN }
  ],
  "prediction": "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  ```

#### 💥 Mổ xẻ bản chất Toán học & Phần cứng của Lỗi `Loss: NaN`:
1. **Giới hạn dải động của kiểu dữ liệu Float16:**
   - Trên GPU Tesla T4, ta cấu hình `torch_dtype = torch.float16`.
   - Kiểu `float16` chỉ biểu diễn được giá trị tối đa là $65,504$ (so với $3.4 \times 10^{38}$ của float32).
2. **Cơ chế 2D Multimodal Rotary Embedding (2D M-RoPE):**
   - Kiến trúc Qwen2.5-VL tính toán vị trí pixel theo ma trận sin/cos 2 chiều $X, Y$. Khi xử lý ảnh độ phân giải lớn ($512 \times 512$) cùng với chuỗi câu hỏi và tọa độ dài, ma trận tích vô hướng Attention Logits ($Q \cdot K^T / \sqrt{d}$) vượt quá ngưỡng $65,504$.
   - Giá trị bị biến thành Vô cực (`+Inf`), sau đó đi qua hàm Softmax và Cross-Entropy Loss lập tức trở thành **`NaN` (Not a Number)**.
3. **Hiện tượng Sụp Đổ Mô Hình (Model Collapse):**
   - Khi gradient bị `NaN`, toàn bộ ma trận trọng số LoRA bị xóa sạch giá trị.
   - Khi suy luận, xác suất của các từ vựng bị đều nhau, mô hình rơi vào vòng lặp vô hạn sinh ký tự chấm than `!`.

---

### 🖼️ SLIDE 6: ĐIỂM SÁNG ĐỐI CHỨNG - BẢN V1 VẪN ĐẠT 94.94% ANLS & GIẢI PHÁP HYBRID

* **Thành tựu cốt lõi (V1):**
  * Mô hình LoRA V1 (148.7 MB) bóc tách văn bản tiếng Việt cực kỳ xuất sắc (**94.94% ANLS**, **73.45% Token F1**).
* **Giải pháp "Chữa Cháy Kỹ Thuật" (Hybrid Hardening):**
  1. **Strict Digits Equality:** Với chuỗi số $\ge 4$ chữ số, chỉ chấp nhận so khớp khi 100% chuỗi số bằng nhau (`cand_digits == token_digits`), cấm so khớp xâu con.
  2. **Header Blacklisting:** Khai báo danh sách đen `LABEL_BLACKLIST` tự động loại trừ thanh tiêu đề bảng biểu.
  3. **Word-Boundary Regex (`\b`):** Khớp trọn vẹn từng món hàng trong bảng kê.

---

### 🖼️ SLIDE 7: 4 BÀI HỌC KINH NGHIỆM QUÝ GIÁ (LESSONS LEARNED)

| STT | Bài học đúc kết | Ý nghĩa thực tiễn cho Kỹ sư ML |
| :---: | :--- | :--- |
| **1** | **Precision Matters trong VLM** | Với các mô hình Vision Transformer dùng RoPE phức tạp, **không bao giờ dùng `float16` trần** mà phải dùng `bfloat16` (trên A100/V100) hoặc dùng `GradScaler` với `float32` cho attention logits. |
| **2** | **Đừng tin vào so khớp chuỗi ngây thơ** | Trong tài liệu tài chính, dữ liệu số có cấu trúc hoàn toàn khác văn bản thông thường. Thuật toán so khớp chuỗi ngây thơ (naive substring match) sẽ luôn thất bại trước số điện thoại và địa chỉ. |
| **3** | **Lỗi lan truyền trong hệ thống 2-Stage** | Pipeline ghép nối 2 mô hình (VLM + OCR) luôn tạo ra sai số kép. Tương lai của xử lý tài liệu bắt buộc phải là **Single-Pass End-to-End Multimodal Modeling**. |
| **4** | **Giá trị của việc đọc Log và Metric đối chứng** | Việc theo dõi biểu đồ Loss và log từng epoch giúp phát hiện sớm sự sụp đổ mô hình thay vì chỉ nhìn vào kết quả đầu ra cuối cùng. |

---

## 🎯 CÂU KẾT ẤN TƯỢNG ĐỂ BẠN KẾT THÚC BÀI THUYẾT TRÌNH:

> *"Thưa Thầy và Hội đồng, dự án của chúng em không chỉ dừng lại ở một con số chính xác 94.94% của bài toán hỏi đáp văn bản, mà giá trị học thuật lớn nhất nhóm thu nhận được chính là việc **thử nghiệm đến tận cùng hai kiến trúc Bounding Box, trực tiếp đối mặt với hiện tượng tràn số FP16 trên 2D M-RoPE và tìm ra bản chất toán học của các lỗi so khớp không gian**. Đây là hành trang thực tế quý giá nhất cho nhóm trên con đường nghiên cứu AI chuyên sâu."*
