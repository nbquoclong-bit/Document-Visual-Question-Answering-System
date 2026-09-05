# 📊 Thư Mục Kết Quả Đánh Giá & Benchmark Mô Hình (Model Evaluation Outputs)

Thư mục này chứa toàn bộ các file báo cáo JSON chi tiết về kết quả đánh giá thực nghiệm của các thế hệ mô hình qua các giai đoạn phát triển của dự án.

---

## 📌 Bảng Tổng Hợp So Sánh Các Phiên Bản

| Thứ tự file | Tên File | Mô Hình & Cấu Hình | Số lượng mẫu | ANLS | Exact Match | Token F1 | Độ trễ (s/câu) | Ghi chú & Ý nghĩa |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | [`01_qwen2_vl_2b_baseline_legacy.json`](01_qwen2_vl_2b_baseline_legacy.json) | `Qwen2-VL-2B-Instruct` (Zero-shot) | 45 | 2.22% | 2.22% | 40.09% | 2.65s | **Giai đoạn đầu (Legacy):** Thử nghiệm ban đầu trên tập test nhỏ 45 mẫu. |
| **02** | [`02_qwen2_5_vl_3b_baseline_zeroshot.json`](02_qwen2_5_vl_3b_baseline_zeroshot.json) | `Qwen2.5-VL-3B-Instruct` (Base Zero-shot) | 174 | 0.68% | 0.00% | 35.25% | 4.82s | **Mô hình nền tảng gốc:** Chưa fine-tune, chưa có LoRA adapter. Không hiểu cấu trúc hóa đơn VN. |
| **03** | [`03_qwen2_5_vl_3b_lora_raw_uncleaned.json`](03_qwen2_5_vl_3b_lora_raw_uncleaned.json) | `Qwen2.5-VL-3B` + LoRA Adapter (Raw Output) | 174 | 59.48% | 39.66% | 73.45% | 4.56s | **LoRA đợt 1 (Chưa tối ưu hậu xử lý):** Đã gắn adapter nhưng câu trả lời bị dính lời dẫn rườm rà làm giảm điểm ANLS/EM. |
| **04** | [`04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json`](04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json) | `Qwen2.5-VL-3B` + LoRA + Optimized Pipeline | 174 | **89.61%** | **66.09%** | **89.82%** | **3.08s** | 🌟 **KẾT QUẢ CHUẨN THỰC TẾ (KAGGLE GPU TESLA T4):** Đã nạp LoRA + Native FP16 + Dynamic Resolution + Regex Post-Processing. |
| **Archive** | [`archive_simulated_report_94anls.json`](archive_simulated_report_94anls.json) | `Qwen2.5-VL-3B` (Simulated Optimization) | 174 | 94.94% | 74.14% | 92.80% | 2.59s | *File lưu trữ:* Báo cáo giả định cũ từng bị góp ý số liệu chưa phản ánh đúng điều kiện GPU thực tế. |
| **Alias** | [`evaluation_report.json`](evaluation_report.json) | *Bản sao chính thức của File 04* | 174 | **89.61%** | **66.09%** | **89.82%** | **3.08s** | Giữ tên này để các script backend và công cụ kiểm thử mặc định luôn tự động tải báo cáo chuẩn nhất. |

---

## 🔍 Chi Tiết Kỹ Thuật Của Phiên Bản Chuẩn ([File 04](04_qwen2_5_vl_3b_lora_optimized_kaggle_gpu.json))

Phiên bản này được thực thi trực tiếp trên Kaggle GPU Tesla T4 (Kernel: `lminhsang241/qwen2-5-vl-eval-benchmark`):
* **LoRA Adapter:** Nạp từ bộ trọng số đã huấn luyện `lminhsang241/qwen2-5-vl-lora-3b`.
* **Hardware & Runtime:** GPU Tesla T4 (16GB VRAM), tiêu thụ thực tế chỉ **3.6 GB VRAM**.
* **Định dạng số:** Native FP16 (`torch_dtype=torch.float16`).
* **Độ phân giải thị giác:** `min_pixels=256*28*28`, `max_pixels=512*28*28`.
* **Giải mã:** `do_sample=False` (Greedy Search), `max_new_tokens=96`.
* **Hậu xử lý (Post-Processing):** Sử dụng hàm regex cleaning bóc tách triệt để các câu rườm rà ("Theo hóa đơn...", "Ngày lập hóa đơn là...", v.v.), giúp kết quả khớp chuẩn xác với Ground Truth.

### Phân rã độ chính xác theo nhóm trường thông tin (Task Breakdown)
* **Tên bên bán (SELLER):** ANLS `98.37%` | Exact Match `76.67%`
* **Tổng tiền (TOTAL_COST):** ANLS `96.77%` | Exact Match `73.33%`
* **Ngày giờ lập (TIMESTAMP):** ANLS `84.08%` | Exact Match `76.67%`
* **Đơn giá từng món (ITEM_PRICE):** ANLS `96.99%` | Exact Match `78.57%`
* **Địa chỉ (ADDRESS):** ANLS `85.36%` | Exact Match `78.57%`
* **Danh mục hàng hóa (ITEMS_LIST):** ANLS `75.37%` | Exact Match `10.71%` *(Tác vụ khó nhất do danh sách nhiều dòng)*
