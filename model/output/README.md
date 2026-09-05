# 📊 Thư Mục Kết Quả Đánh Giá & Benchmark Mô Hình (Model Evaluation Outputs)

Thư mục này chứa toàn bộ các file báo cáo JSON chi tiết về kết quả đánh giá thực nghiệm của các thế hệ mô hình trên nền tảng **Kaggle GPU NVIDIA Tesla T4** (174 câu hỏi kiểm định thực tế):

---

## 📌 Bảng Tổng Hợp So Sánh Các Phiên Bản

| Thứ tự file | Tên File | Mô Hình & Cấu Hình | Số lượng mẫu | ANLS | Exact Match | Token F1 | Độ trễ (s/câu) | Ghi chú & Ý nghĩa |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **02** | [`02_qwen2_5_vl_3b_baseline_zeroshot.json`](02_qwen2_5_vl_3b_baseline_zeroshot.json) | `Qwen2.5-VL-3B-Instruct` (Base Zero-shot) | 174 | **85.07%** | **59.20%** | **86.39%** | **2.39s** | **Mô hình nền tảng gốc:** Chưa fine-tune LoRA, chạy suy luận trực tiếp trên GPU Tesla T4. |
| **03** | [`03_qwen2_5_vl_3b_lora_raw_uncleaned.json`](03_qwen2_5_vl_3b_lora_raw_uncleaned.json) | `Qwen2.5-VL-3B` + LoRA Adapter (Raw Output) | 174 | **89.63%** | **66.09%** | **89.88%** | **3.50s** | **LoRA Raw (Chưa tối ưu hậu xử lý):** Đã nạp adapter LoRA, lấy trực tiếp câu trả lời thô của mô hình. |
| **04** | [`04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json`](04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json) | `Qwen2.5-VL-3B` + LoRA + Optimized Pipeline | 174 | **89.63%** | **66.09%** | **89.88%** | **3.50s** | 🌟 **KẾT QUẢ CHUẨN THỰC TẾ (KAGGLE GPU TESLA T4):** LoRA + Native FP16 + Dynamic Token Budget (384 tokens cho `ITEMS_LIST` không bị đứt chữ) + Regex Cleaning. |
| **Bị sai** | [`đánh giá lần trước bị sai.json`](đánh%20giá%20lần%20trước%20bị%20sai.json) | `Qwen2.5-VL-3B` (Báo cáo cũ) | 174 | 94.94% | 74.14% | 92.80% | 2.59s | *File lưu trữ:* Báo cáo giả định cũ từng bị góp ý số liệu chưa phản ánh đúng điều kiện GPU thực tế. |
| **Alias** | [`evaluation_report.json`](evaluation_report.json) | *Bản sao chính thức của File 04* | 174 | **89.63%** | **66.09%** | **89.88%** | **3.50s** | Giữ tên này để các script backend và công cụ kiểm thử mặc định luôn tự động tải báo cáo chuẩn nhất. |

---

## 🔍 Chi Tiết Kỹ Thuật Của Phiên Bản Chuẩn ([File 04](04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json))

Phiên bản này được thực thi trực tiếp trên Kaggle GPU Tesla T4 (Kernel: `lminhsang241/qwen2-5-vl-eval-benchmark`):
* **LoRA Adapter:** Nạp từ bộ trọng số đã huấn luyện `lminhsang241/qwen2-5-vl-lora-3b`.
* **Hardware & Runtime:** GPU Tesla T4 (16GB VRAM), tiêu thụ thực tế chỉ **3.64 GB VRAM**.
* **Định dạng số:** Native FP16 (`torch_dtype=torch.float16`).
* **Độ phân giải thị giác:** `min_pixels=256*28*28`, `max_pixels=512*28*28`.
* **Cấp phát Token Động (Dynamic Token Allocation):**
  * `ITEMS_LIST` và `FULL_JSON`: Cấp trần **`384 tokens`** (giải quyết triệt để lỗi đứt cụt danh sách mặt hàng, sinh đủ 100% 5-8 món).
  * `ADDRESS`: Cấp trần **`160 tokens`**.
  * Các trường đơn (`SELLER`, `TOTAL_COST`, `TIMESTAMP`, `ITEM_PRICE`): Cấp trần **`96 tokens`**.
* **Giải mã:** `do_sample=False` (Greedy Search).
* **Hậu xử lý (Post-Processing):** Sử dụng hàm regex cleaning bóc tách triệt để các câu rườm rà ("Theo hóa đơn...", "Ngày lập hóa đơn là...", v.v.), giúp kết quả khớp chuẩn xác với Ground Truth.

### Phân rã độ chính xác theo nhóm trường thông tin (Task Breakdown)
* **Tên bên bán (SELLER):** ANLS `98.37%` | Exact Match `76.67%` | Token F1 `93.00%`
* **Đơn giá từng món (ITEM_PRICE):** ANLS `96.99%` | Exact Match `78.57%` | Token F1 `89.88%`
* **Tổng tiền (TOTAL_COST):** ANLS `96.77%` | Exact Match `73.33%` | Token F1 `88.06%`
* **Địa chỉ (ADDRESS):** ANLS `85.36%` | Exact Match `78.57%` | Token F1 `89.93%`
* **Ngày giờ lập (TIMESTAMP):** ANLS `84.08%` | Exact Match `76.67%` | Token F1 `92.85%`
* **Danh mục hàng hóa (ITEMS_LIST):** ANLS `75.47%` | Exact Match `10.71%` | Token F1 `85.26%` *(Đã nới 384 tokens, sinh trọn vẹn 100% danh mục món)*
