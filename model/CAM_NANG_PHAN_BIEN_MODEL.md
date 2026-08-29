# 🎓 CẨM NANG HỎI - ĐÁP PHẢN BIỆN VỀ PHẦN MODEL (DOCVQA & VLM)
> **Tài liệu bí kíp dành cho người thuyết trình phần Model:** Giúp bạn hiểu sâu 100% bản chất kỹ thuật, tự tin trả lời bất kỳ câu hỏi "xoáy" nào từ Giảng viên, Mentor hoặc Hội đồng chấm đồ án.

---

## ❓ CÂU 1: Tại sao nhóm lại dùng End-to-End VLM (Qwen2-VL) mà không dùng Pipeline truyền thống (OCR + LayoutLM / Regex)?

### 💡 Trả lời:
- **Nhược điểm của Pipeline truyền thống:** Bị hiện tượng **Lỗi rò rỉ dây chuyền (Cascading Error)**. Pipeline truyền thống phải qua 3 bước riêng biệt: `Image -> OCR (Paddle/Tesseract) -> KIE/LayoutLM -> Regex/LLM`. Nếu bước OCR đọc sai 1 ký tự (do hóa đơn mờ, nhăn, góc chụp nghiêng), toàn bộ các module phía sau sẽ sai theo 100%.
- **Ưu điểm của End-to-End VLM:** Nhận trực tiếp ảnh và câu hỏi cùng lúc trong không gian nhúng đa phương thức (Multimodal Embedding). Mô hình vừa "nhìn" bức ảnh, vừa "đọc" câu hỏi để suy luận ngữ nghĩa trực tiếp, không phụ thuộc vào text OCR trung gian, từ đó triệt tiêu hoàn toàn lỗi cascading error.

---

## ❓ CÂU 2: Cơ chế M-RoPE (Multimodal Rotary Position Embedding) trong Qwen2-VL là gì và tại sao nó đặc biệt quan trọng với hóa đơn?

### 💡 Trả lời:
- Các mô hình ngôn ngữ thông thường (LLM) dùng RoPE 1D – chỉ đánh số thứ tự từ trái sang phải trên một dòng văn bản.
- Tuy nhiên, hóa đơn là tài liệu mang **cấu trúc không gian 2D** (dòng, cột, bảng, khoảng cách lề). 
- **M-RoPE phân rã vị trí thành 3 thành phần tọa độ:**
  1. $pos_t$: Vị trí thời gian (Time/Sequence)
  2. $pos_h$: Vị trí dòng dọc (Height - Dòng trên/dưới)
  3. $pos_w$: Vị trí cột ngang (Width - Cột trái/phải)
- **Ý nghĩa thực tế:** Giúp mô hình hiểu rằng từ `"Tổng tiền"` ở bên trái và con số `"796,068"` ở tận cùng bên phải nằm trên **cùng một dòng ngang**, từ đó liên kết chính xác nhãn và giá trị thực thể.

---

## ❓ CÂU 3: QLoRA là gì? Tại sao chỉ train 0.2% tham số mà Adapter lại chỉ nặng 73.9 MB?

### 💡 Trả lời:
- **Bản chất LoRA (Low-Rank Adaptation):** Đóng băng toàn bộ trọng số gốc $W_0 \in \mathbb{R}^{d \times k}$ (không chỉnh sửa 2 tỷ tham số gốc). Thay vào đó, chèn vào 2 ma trận phân rã hạng thấp $A \in \mathbb{R}^{d \times r}$ và $B \in \mathbb{R}^{r \times k}$ với hạng $r = 16 \ll d$.
- Khi cập nhật trọng số: $\Delta W = B \times A$.
- Nhóm can thiệp vào **7 khối module quan trọng nhất** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Tổng số tham số cần học chỉ chiếm **~0.2%** tổng mạng. Khi xuất file lưu trữ dưới dạng FP16, kích thước file `adapter_model.safetensors` **chỉ vỏn vẹn 73.9 MB** (so với 4.2 GB của base model), cực kỳ thuận tiện để cập nhật qua mạng (OTA) cho các hệ thống nhúng / IoT.

---

## ❓ CÂU 4: Giải thích công thức tính ANLS và tại sao ngưỡng phạt lại là $\tau = 0.5$?

### 💡 Trả lời:
- **ANLS (Average Normalized Levenshtein Similarity)** là thước đo chuẩn trong cuộc thi quốc tế **DocVQA Challenge**.
- Khoảng cách Levenshtein $d_L(p, gt)$ đếm số bước chèn, xóa, sửa ký tự.
- Khoảng cách chuẩn hóa: $NL = \frac{d_L(p, gt)}{\max(|p|, |gt|)}$.
- Điểm cho từng mẫu:
  $$\text{ANLS} = \begin{cases} 1 - NL & \text{nếu } NL < 0.5 \\ 0.0 & \text{nếu } NL \ge 0.5 \end{cases}$$
- **Ý nghĩa ngưỡng $\tau = 0.5$:**
  - Nếu sai lệch $< 50\%$ (ví dụ lỗi OCR mất 1 dấu thanh tiếng Việt): Mô hình vẫn nhận điểm tương ứng (ví dụ: `0.85` hay `0.90`).
  - Nếu sai lệch $\ge 50\%$ (mô hình bịa đặt thông tin hoặc trả lời sai thực thể): Bị phạt thẳng về **0 điểm** để đảm bảo tính nghiêm ngặt của nghiệp vụ kế toán.

---

## ❓ CÂU 5: Tại sao Exact Match (EM) của Base Model chỉ đạt 30% mà sau khi gắn LoRA lại tăng vọt lên 75% - 80%?

### 💡 Trả lời:
- **Base Model gốc:**
  1. Hay trả lời lan man theo phong cách đàm thoại (VD: hỏi ngày thì trả lời *"Ngày giờ lập hóa đơn là 16:41 ngày 31/05/2026."* thay vì chỉ xuất `31/05/2026 16:41`).
  2. Bộ Tokenizer gốc của Qwen được huấn luyện chủ yếu bằng tiếng Anh và tiếng Trung nên rất hay đọc sai dấu tiếng Việt (VD: `TRÀN HUNG ĐẢO`).
- **Sau khi Fine-tune LoRA:**
  1. Mô hình được học phong cách trả lời ngắn gọn, trực diện, đúng chuẩn kế toán (chỉ lấy đúng chuỗi giá trị).
  2. Mô hình được học lại toàn bộ bộ từ vựng và dấu thanh tiếng Việt trên các hóa đơn thực tế, giúp khớp chính xác 100% từng ký tự.

---

## ❓ CÂU 6: Thời gian xử lý (Latency) của mô hình là bao lâu? Làm sao để tối ưu khi demo?

### 💡 Trả lời:
- **Trên CPU cục bộ:** Mất 2–3 phút do nạp ma trận 2 tỷ tham số trên RAM và tính toán bằng CPU.
- **Trên GPU NVIDIA Tesla T4 (16GB VRAM):** Thời gian suy luận chỉ mất **~1.2 – 1.5 giây / câu hỏi** (với VRAM tiêu thụ ~4.2 GB đến 6.8 GB).
- **Kỹ thuật tối ưu tốc độ của nhóm:**
  1. Sử dụng nửa độ chính xác **FP16**.
  2. Khống chế độ phân giải ảnh thích ứng: `min_pixels=256*28*28`, `max_pixels=1024*28*28` để giảm số lượng visual tokens truyền vào Transformer.
  3. Cố định `max_new_tokens=128` và `do_sample=False` để mô hình sinh câu trả lời nhanh nhất mà không tốn tài nguyên tìm kiếm.
