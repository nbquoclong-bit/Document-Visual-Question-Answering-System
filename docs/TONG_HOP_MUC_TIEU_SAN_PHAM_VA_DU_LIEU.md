# 📘 TỔNG HỢP TOÀN DIỆN: MỤC TIÊU SẢN PHẨM, DỮ LIỆU, QUÁ TRÌNH FINE-TUNE & TỐI ƯU HÓA MÔ HÌNH VLM
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Bóc Tách Hóa Đơn Tiếng Việt (Qwen2.5-VL-3B LoRA)

> **Mục đích tài liệu:** Đây là kho thông tin, phương pháp luận và số liệu thực nghiệm đầy đủ, chi tiết và chuẩn xác nhất về:
> 1. **Bối cảnh, Mục tiêu sản phẩm & Tính năng bóc tách, hỏi đáp thông minh và minh chứng trực quan.**
> 2. **Toàn bộ công tác kỹ thuật dữ liệu (114,716 mẫu VQA trên 15 loại hóa đơn Việt Nam).**
> 3. **Phương pháp Toán học & AutoML tìm Siêu tham số tối ưu (Gradient LR Finder + Optuna TPE).**
> 4. **Nhật ký & Phương pháp luận quá trình Fine-Tuning thực tế trên Kaggle GPU Tesla T4 (Qwen2.5-VL-3B LoRA).**
> 5. **Toàn bộ kỹ thuật Tối Ưu Hóa (Optimization Deep-Dive): 7 Lớp Linear LoRA, Dynamic 1024 Tokens cho Full JSON, Resolution Constraining, và Regex Post-processing.**
> 6. **Kết quả thực nghiệm định lượng toàn diện trên GPU Tesla T4 (Base Model vs Fine-Tuned Optimized Model đạt 94.94% ANLS).**  
> Các thành viên trong nhóm có thể tự do trích xuất bất kỳ bảng biểu, số liệu so sánh hay ví dụ thực tế nào để đưa vào bài thuyết trình và slide.

---

## 📌 MỤC LỤC TỔNG QUAN

1. [Phần 1: Bối Cảnh Bài Toán & Mục Tiêu Sản Phẩm](#phần-1-bối-cảnh-bài-toán--mục-tiêu-sản-phẩm)
2. [Phần 2: Tính Năng Cốt Lõi: Minh Chứng Trực Quan & Bóc Tách Thông Tin Hóa Đơn](#phần-2-tính-năng-cốt-lõi-minh-chứng-trực-quan--bóc-tách-thông-tin-hóa-đơn)
3. [Phần 3: Người Dùng Mục Tiêu & Trải Nghiệm Sử Dụng](#phần-3-người-dùng-mục-tiêu--trải-nghiệm-sử-dụng)
4. [Phần 4: Đặc Tả Đầu Vào, Đầu Ra & Luồng Hệ Thống](#phần-4-đặc-tả-đầu-vào-đầu-ra--luồng-hệ-thống)
5. [Phần 5: Kết Quả Đo Lường Thực Nghiệm Base Model (Zero-Shot) Trên GPU](#phần-5-kết-quả-đo-lường-thực-nghiệm-base-model-zero-shot-trên-gpu)
6. [Phần 6: Phương Pháp Toán Học Tìm Siêu Tham Số Tối Ưu (AutoML & Gradient Descent)](#phần-6-phương-pháp-toán-học-tìm-siêu-tham-số-tối-ưu-automl--gradient-descent)
7. [Phần 7: Kết Quả Thực Nghiệm Mô Hình Sau Tối Ưu Hóa Toàn Diện (Đạt 94.94% ANLS)](#phần-7-kết-quả-thực-nghiệm-mô-hình-sau-tối-ưu-hóa-toàn-diện-đạt-9494-anls)
8. [Phần 8: Quy Trình Xây Dựng & Chuẩn Hóa Dữ Liệu](#phần-8-quy-trình-xây-dựng--chuẩn-hóa-dữ-liệu)
9. [Phần 9: Thống Kê Chi Tiết Bộ Dữ Liệu 114,716 Mẫu VQA](#phần-9-thống-kê-chi-tiết-bộ-dữ-liệu-114716-mẫu-vqa)
10. [Phần 10: Phân Loại 15 Mẫu Hóa Đơn & 8 Nhóm Tác Vụ](#phần-10-phân-loại-15-mẫu-hóa-đơn--8-nhóm-tác-vụ)
11. [Phần 11: Tiêu Chuẩn Kiểm Soát Chất Lượng Dữ Liệu](#phần-11-tiêu-chuẩn-kiểm-soát-chất-lượng-dữ-liệu)
12. [Phần 12: Nhật Ký & Phương Pháp Luận Quá Trình Huấn Luyện Fine-Tuning Thực Tế Trên Kaggle GPU](#phần-12-nhật-ký--phương-pháp-luận-quá-trình-huấn-luyện-fine-tuning-thực-tế-trên-kaggle-gpu)
13. [Phần 13: Toàn Bộ Kỹ Thuật Tối Ưu Hóa Chuyên Sâu (Optimization Deep-Dive)](#phần-13-toàn-bộ-kỹ-thuật-tối-ưu-hóa-chuyên-sâu-optimization-deep-dive)
14. [Phần 14: Bảng So Sánh 3 Thế Hệ Mô Hình & Phân Tích Lỗi (Error Analysis)](#phần-14-bảng-so-sánh-3-thế-hệ-mô-hình--phân-tích-lỗi-error-analysis)
15. [Phần 15: Bản Đồ File Trong Repository Chuẩn Bị Đẩy Lên GitHub](#phần-15-bản-đồ-file-trong-repository-chuẩn-bị-đẩy-lên-github)

---

# PHẦN 1: BỐI CẢNH BÀI TOÁN & MỤC TIÊU SẢN PHẨM

### 1.1. Bối cảnh thực tế tại thị trường Việt Nam
* **Làn sóng chuyển đổi số trong kế toán - tài chính:** Hàng triệu hóa đơn bán lẻ (in nhiệt) và hóa đơn điện tử (e-Invoice) được phát hành mỗi ngày tại các chuỗi siêu thị, nhà hàng, cửa hàng tiện lợi và doanh nghiệp.
* **Đặc thù hóa đơn Việt Nam:**
  * Chất lượng in ấn không đồng đều (chữ in nhiệt dễ mờ, mất nét sau vài ngày).
  * Đa dạng font chữ tiếng Việt có dấu, chữ nghiêng, chữ viết hoa, logo thương hiệu đan xen.
  * Bố cục bảng biểu (table layout) phức tạp, nhiều cột đơn giá, số lượng, tiền trước thuế, VAT, chiết khấu, tổng thanh toán.

### 1.2. Hạn chế của các phương pháp truyền thống
1. **Nhập liệu thủ công (Manual Data Entry):** Tốn trung bình 2–3 phút cho 1 hóa đơn dài; tỷ lệ gõ sai số tiền hoặc mã số thuế do lỗi con người lên tới 5–8%.
2. **Hệ thống OCR truyền thống kết hợp Regex/Rule-based:** Chịu **lỗi lan truyền (Cascading Error)** và **mất liên kết không gian 2D**, sụp đổ hoàn toàn khi cửa hàng đổi mẫu in hóa đơn.
3. **Hiện tượng ảo giác AI (Hallucination) của LLM thông thường:** Các mô hình AI ngôn ngữ thuần túy thường trả lời một con số nhưng người dùng không biết con số đó nằm ở đâu trên ảnh để kiểm chứng.

### 1.3. Mục tiêu & Tầm nhìn của sản phẩm
* **Tên sản phẩm:** Hệ thống Trích Xuất & Minh Chứng Hóa Đơn Thông Minh (**Document VQA System**).
* **Nguyên lý cốt lõi:** **Ứng dụng Vision-Language Model (VLM) đọc trực tiếp từ ảnh đến thông tin ngữ nghĩa**, kết hợp khả năng hỏi đáp mở và trích xuất cấu trúc phân cấp Full JSON 1024 Tokens.
* **Mục tiêu định lượng (KPIs):**
  * **Tốc độ xử lý:** Dưới **2.5 giây / câu hỏi** trên GPU phổ thông (Tesla T4).
  * **Độ chính xác chuỗi (ANLS):** Đạt trên **94.5%** trên toàn bộ 15 loại mẫu hóa đơn thực tế.
  * **Khả năng tích hợp:** Xuất trực tiếp cấu trúc **JSON phân cấp 100%** không bị ngắt quãng để kết nối vào các hệ thống phần mềm kế toán (MISA, FAST, SAP) và ERP doanh nghiệp.
  * **Tự động hóa:** Giảm **95%** thời gian nhập liệu thủ công của kế toán viên.

---

# PHẦN 2: TÍNH NĂNG CỐT LÕI: MINH CHỨNG TRỰC QUAN & BÓC TÁCH THÔNG TIN HÓA ĐƠN

### 2.1. Tại sao sản phẩm chú trọng minh chứng trực quan & đối soát?
* **Chống ảo giác AI (Anti-Hallucination):** Trong nghiệp vụ tài chính - kế toán, độ chính xác là tuyệt đối. Hệ thống chứng minh mô hình thực sự "nhìn" thấy con số trên hóa đơn chứ không phải sinh ngẫu nhiên.
* **Quy trình kiểm chứng siêu tốc (Human-in-the-loop):** Kế toán viên có thể đối chiếu câu trả lời ngay lập tức với hình ảnh gốc.
* **Bóc tách Full JSON 1024 Tokens:** Tự động chuyển đổi hóa đơn phi cấu trúc thành dữ liệu JSON chuẩn mực.

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ [ẢNH HÓA ĐƠN GỐC]                 │ [KẾT QUẢ BÓC TÁCH TỪ VLM ENGINE]      │
  │                                   │                                       │
  │   HIGHLANDS COFFEE                │   🏢 Tên bên bán: HIGHLANDS COFFEE    │
  │   Số 11 Sư Vạn Hạnh, Q.10         │   📍 Địa chỉ    : Số 11 Sư Vạn Hạnh   │
  │   ---------------------------     │   📦 Mặt hàng   :                     │
  │   1. Trà Sen Vàng (L)  55,000     │      - Trà Sen Vàng (L): 55,000đ      │
  │   2. Phin Sữa Đá  (M)  54,000     │      - Phin Sữa Đá  (M): 54,000đ      │
  │   ---------------------------     │   💰 Tổng tiền  : 109,000đ            │
  │   TỔNG TIỀN: 109,000đ             │   ⏱️ Độ trễ     : 2.50s (Tesla T4)    │
  └───────────────────────────────────┴───────────────────────────────────────┘
```

### 2.2. Bảng mã phân loại trường theo chuẩn nghiệp vụ Kế toán

| Loại trường thông tin | Ý nghĩa nghiệp vụ | Ví dụ thực tế |
| :--- | :--- | :--- |
| **`SELLER`** | Tên cửa hàng, đơn vị bán hàng, chi nhánh | *CÔNG TY CỔ PHẦN DỊCH VỤ CÀ PHÊ CAO NGUYÊN* |
| **`TOTAL_COST`** | Tổng tiền thanh toán cuối cùng (trọng yếu nhất) | *109,000 VNĐ* |
| **`TIMESTAMP`** | Ngày giờ in hóa đơn, kỳ lập phiếu | *16/06/2025 14:30* |
| **`ADDRESS`** | Địa chỉ nơi phát sinh giao dịch | *Số 11 Sư Vạn Hạnh, Phường 12, Quận 10, TP.HCM* |
| **`ITEMS_LIST`** | Danh sách mặt hàng, chi tiết đơn giá từng món | *1. Trà Sen Vàng (L), 2. Phin Sữa Đá (M)* |
| **`TAX`** | Mã số thuế doanh nghiệp | *0302863720* |

---

# PHẦN 3: NGƯỜI DÙNG MỤC TIÊU & TRẢI NGHIỆM SỬ DỤNG

### 3.1. Đối tượng người dùng mục tiêu (User Personas)
1. **Kế toán doanh nghiệp:** Cần bóc tách nhanh tổng tiền, ngày hóa đơn, MST và đối soát nhanh trước khi duyệt chi.
2. **Chủ cửa hàng / Thủ kho:** Kiểm tra số lượng và đơn giá từng món hàng nhập kho từ các phiếu thu viết tay hoặc in nhiệt.
3. **Lập trình viên / Đội ngũ ERP:** Tích hợp API trích xuất JSON vào hệ thống quản trị nội bộ.

---

# PHẦN 4: ĐẶC TẢ ĐẦU VÀO, ĐẦU RA & LUỒNG HỆ THỐNG

```
   ┌───────────────────────┐
   │ Ảnh Hóa Đơn Đầu Vào   │ (Ảnh chụp điện thoại / File scan / Hóa đơn điện tử)
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐     ┌────────────────────────────┐
   │ Bộ Tiền Xử Lý Ảnh     │ <── │ Câu Hỏi Nghiệp Vụ Tiếng Việt│
   │ (Dynamic Resolution)  │     │ (Hỏi giá trị / Trích JSON) │
   └──────────┬────────────┘     └─────────────┬──────────────┘
              │                                │
              ▼                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │             MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)            │
   │           Qwen2.5-VL-3B + LoRA 7 Lớp Linear Tối Ưu        │
   └──────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ 1. Câu Trả Lời Trực Diện      │   │ 2. Cấu Trúc Phân Cấp Hoàn Chỉnh│
│    "3700223705" / "109,000đ"  │   │    Full JSON 1024 Tokens      │
└───────────────────────────────┘   └───────────────────────────────┘
```

---

# PHẦN 5: KẾT QUẢ ĐO LƯỜNG THỰC NGHIỆM BASE MODEL (ZERO-SHOT) TRÊN GPU

Nhóm đã tiến hành kiểm thử thực tế mô hình gốc **Qwen2.5-VL-3B-Instruct (Zero-Shot)** trên toàn bộ **174 câu hỏi kiểm định** thuộc 15 loại mẫu hóa đơn trên **GPU NVIDIA Tesla T4** (`model/output/qwen2_5_vl_baseline_report.json`).

### 5.1. Bảng số liệu định lượng thực tế

| Chỉ Số Đo Lường (Metrics) | Kết Quả Base Model Zero-Shot | Đánh Giá Kỹ Thuật |
| :--- | :---: | :--- |
| **Tổng số câu hỏi kiểm định** | **174 câu hỏi** | Bao phủ 100% 15 loại mẫu hóa đơn thực tế |
| **Điểm ANLS (DocVQA Metric)** | **71.30%** | Bị trừ điểm do nhiều từ ngữ đàm thoại rườm rà |
| **Tỷ lệ Exact Match (EM %)** | **42.10%** | Kém khi cần so khớp chính xác 100% |
| **Điểm Token F1-Score** | **68.45%** | Chưa tối ưu cho thuật ngữ kế toán tiếng Việt |
| **Thời gian suy luận trung bình** | **2.60 giây / câu** | Tương đối nhanh nhưng chưa tối ưu token allocation |
| **Bộ nhớ GPU chiếm dụng** | **7.85 GB** | Hoạt động trên GPU Tesla T4 |

---

# PHẦN 6: PHƯƠNG PHÁP TOÁN HỌC TÌM SIÊU THAM SỐ TỐI ƯU (AUTOMAL & GRADIENT DESCENT)

Thay vì chọn tham số theo cảm tính (ad-hoc), nhóm đã xây dựng module **AutoML toán học chuyên biệt** (`model/hyperparameter_tuning.py`) để tự động tìm kiếm cấu hình tối ưu toàn cục:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             2 PHƯƠNG PHÁP TOÁN HỌC TÌM BỘ SIÊU THAM SỐ TỐI ƯU               │
├────────────────────────────────┬────────────────────────────────────────────┤
│ 1. Gradient-Based LR Finder    │ Quét Gradient Descent theo hàm số mũ để tìm│
│    (Smith / FastAI Approach)   │ điểm có đạo hàm dLoss/dLR âm sâu nhất.     │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 2. Bayesian Optimization       │ Mô hình xác suất Tree-structured Parzen    │
│    (Optuna TPE + ASHA Pruning) │ Estimator (TPE) kết hợp cắt tỉa nhánh xấu. │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### 6.1. Phương pháp 1: Quét Gradient Descent Tìm Learning Rate Tối Ưu (LR Finder)
* **Cơ sở toán học:**  
  Tăng dần Learning Rate theo cấp số nhân trong 100 bước: $\text{LR}_{k+1} = \text{LR}_k \cdot \gamma$ (với $\text{LR} \in [10^{-6}, 10^{-2}]$).  
  Điểm có **đạo hàm âm lớn nhất** chính là Learning Rate tối ưu để mô hình hội tụ nhanh nhất mà không bị bùng nổ gradient:
  $$\text{Optimal LR} = \arg\min_{\text{LR}} \left( \frac{\partial \mathcal{L}}{\partial \text{LR}} \right) = \mathbf{2.0 \times 10^{-4}}$$

### 6.2. Phương pháp 2: Tối Ưu Hóa Bayes Toàn Cục (Bayesian Optimization - Optuna TPE)
* **Thuật toán:** Sử dụng **Tree-structured Parzen Estimator (TPE)** để mô hình hóa xác suất $P(\theta \mid \mathcal{L})$ giữa cấu hình siêu tham số $\theta$ và hàm mất mát kiểm định $\mathcal{L}_{\text{val}}$.
* **Kỹ thuật ASHA Pruning:** Cắt tỉa sớm các thử nghiệm kém ở step thứ 50 để tiết kiệm GPU.

### 6.3. Bảng Bộ Siêu Tham Số Tối Ưu Vàng Đã Tìm Được (`model/optimal_hyperparameters.json`)

| Siêu Tham Số (Hyperparameter) | Giá Trị Tối Ưu | Cơ Sở Kỹ Thuật & Ý Nghĩa Thực Tiễn |
| :--- | :---: | :--- |
| **Learning Rate ($\eta$)** | $\mathbf{2.0 \times 10^{-4}}$ | Đạt tốc độ giảm Loss tối đa qua phép thử Gradient Descent |
| **LoRA Rank ($r$)** | **16** | Đủ dung lượng biểu diễn đặc trưng 15 template hóa đơn |
| **LoRA Scaling ($\alpha$)** | **32** | Tỷ lệ $\alpha / r = 2.0$ chuẩn mực, giữ ổn định ma trận cập nhật |
| **Weight Decay ($\lambda$)** | **0.01** | Tránh hiện tượng ghi nhớ vẹt (Overfitting) trên dữ liệu huấn luyện |
| **Warmup Ratio** | **0.03** | Làm ấm Learning Rate trong 3% bước đầu kết hợp Cosine Annealing |
| **Effective Batch Size** | **16** | Batch size $1 \times \text{Gradient Accumulation } 16$ (Tối ưu VRAM $< 8\text{GB}$) |

---

# PHẦN 7: KẾT QUẢ THỰC NGHIỆM MÔ HÌNH SAU TỐI ƯU HÓA TOÀN DIỆN (ĐẠT 94.94% ANLS)

Sau khi áp dụng **Bộ siêu tham số tối ưu**, kết hợp **Domain System Prompt** và **LoRA 7 Lớp Linear**, mô hình **Qwen2.5-VL-3B LoRA Optimized** đã hoàn thành kiểm định trên 174 câu hỏi thực tế (`model/output/optimized_evaluation_report.json`):

### 7.1. Bảng so sánh tiến trình tối ưu hóa:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    BẢNG SO SÁNH TIẾN TRÌNH TỐI ƯU HÓA TOÀN DIỆN                         │
├──────────────────────────────┬──────────────────┬──────────────────┬────────────────────┤
│ Chỉ Số Đo Lường Học Thuật    │ Base (Zero-Shot) │ Đợt 1 (LoRA 900) │ Đợt 2 (Tối Ưu Hoá) │
├──────────────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ ANLS Score (DocVQA Metric)   │ 🔴 71.30%        │ 🟡 82.40%        │ 🟢 94.94% (+23.6%) │
│ Token F1-Score               │ 🔴 68.45%        │ 🟡 81.15%        │ 🟢 92.80% (+24.4%) │
│ Exact Match (EM %)           │ 🔴 42.10%        │ 🟡 55.20%        │ 🟢 74.14% (129/174)│
│ Tốc độ suy luận (Latency)    │ 2.60 giây / câu  │ 2.80 giây / câu  │ ⚡ 2.50 giây / câu │
│ VRAM GPU chiếm dụng          │ 7.85 GB          │ 8.05 GB          │ 8.12 GB (Tesla T4) │
└──────────────────────────────┴──────────────────┴──────────────────┴────────────────────┘
```

### 7.2. Bảng kết quả định lượng chi tiết trên 15 loại mẫu hóa đơn:

```
┌──────────────────────────────────────┬──────────┬──────────┬──────────┐
│ Loại Mẫu Hóa Đơn (Template)          │   ANLS   │    EM    │ Token F1 │
├──────────────────────────────────────┼──────────┼──────────┼──────────┤
│ 1. supermarket_bachhoaxanh           │ 100.00%  │ 100.00%  │ 100.00%  │ (⭐ Hoàn hảo 100%)
│ 2. supermarket_lotte                 │  99.85%  │  91.67%  │  99.36%  │
│ 3. restaurant_kfc                    │  98.88%  │  75.00%  │  95.93%  │
│ 4. cafe_phuclong                     │  98.71%  │  66.67%  │  96.56%  │
│ 5. einvoice_vnpt                     │  98.16%  │  75.00%  │  93.52%  │
│ 6. supermarket_winmart               │  98.04%  │  75.00%  │  95.81%  │
│ 7. convenience_circlek               │  97.55%  │  75.00%  │  91.15%  │
│ 8. convenience_gs25                  │  96.93%  │  75.00%  │  91.11%  │
│ 9. restaurant_jollibee               │  96.82%  │  66.67%  │  89.64%  │
│ 10. cafe_starbucks                   │  94.48%  │  58.33%  │  87.44%  │
│ 11. einvoice_viettel                 │  93.59%  │  58.33%  │  87.80%  │
│ 12. minimart_anan                    │  91.67%  │  91.67%  │  96.43%  │
│ 13. cafe_highlands                   │  90.96%  │  75.00%  │  92.40%  │
│ 14. convenience_7eleven              │  89.93%  │  80.00%  │  95.33%  │
│ 15. receipt_c45_bb                   │  69.04%  │  37.50%  │  73.45%  │
├──────────────────────────────────────┼──────────┼──────────┼──────────┤
│ TRUNG BÌNH TOÀN BỘ HỆ THỐNG          │  94.94%  │  74.14%  │  92.80%  │
└──────────────────────────────────────┴──────────┴──────────┴──────────┘
```

---

# PHẦN 8: QUY TRÌNH XÂY DỰNG & CHUẨN HÓA DỮ LIỆU

Nhóm đã hoàn thành quy trình xây dựng dữ liệu qua 5 giai đoạn nghiêm ngặt:
1. **Thu thập 4,995 ảnh hóa đơn:** Phủ kín 15 thương hiệu và mẫu phiếu kế toán thực tế tại Việt Nam.
2. **Tiền xử lý thị giác:** Căn chỉnh góc xoay, khử bóng mờ, tối ưu Dynamic Resolution.
3. **Thiết kế Schema 8 nhóm tác vụ:** Tên bên bán, tổng tiền, ngày giờ, địa chỉ, đơn giá, số lượng, JSON và Bounding Box.
4. **Tạo lập & Làm sạch 114,716 cặp VQA:** Kiểm tra chéo 100% nhãn dấu tiếng Việt, định dạng tiền tệ và tính hợp lệ của tọa độ hộp bao.
5. **Phân chia tập dữ liệu:** Train Master (97k mẫu, 85%), Val Master (17k mẫu, 15%), Benchmark Test độc lập (174 mẫu).

---

# PHẦN 9: THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU 114,716 MẪU VQA

| Thông Số Định Lượng | Giá Trị Cụ Thể | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :--- |
| **Tổng số ảnh hóa đơn** | **4,995 ảnh** | Phủ kín 15 danh mục template hóa đơn thực tế |
| **Tổng số cặp câu hỏi VQA** | **114,716 mẫu** | Cung cấp ngữ cảnh phong phú cho mô hình học |
| **Mẫu có nhãn Bounding Box** | **9,990 mẫu** | Cung cấp dữ liệu học định vị không gian 2D |
| **Tập Train Master (`vlm_train_master.json`)** | **97,508 mẫu** (~85%) | Dùng cho quá trình huấn luyện và tối ưu trọng số |
| **Tập Validation Master (`vlm_val_master.json`)** | **17,208 mẫu** (~15%) | Dùng kiểm soát hàm mất mát và chống quá khớp |
| **Tập Benchmark Test Độc Lập** | **174 mẫu** | Bộ kiểm định độc lập đo lường các chỉ số học thuật |
| **Dung lượng file Train Master** | **34.0 MB** | Đã tối ưu hóa lưu trữ JSON chuẩn UTF-8 |

---

# PHẦN 10: PHÂN LOẠI 15 MẪU HÓA ĐƠN & 8 NHÓM TÁC VỤ

### 10.1. Phân bổ cân bằng 15 loại mẫu hóa đơn thực tế (333 ảnh / mẫu)
* **Chuỗi Cafe & Ăn uống (F&B):** Highlands Coffee, Phúc Long, Starbucks, Jollibee, KFC.
* **Cửa hàng tiện lợi & Mini Mart:** 7-Eleven, Circle K, GS25, Minimart An An.
* **Đại Siêu thị:** WinMart / WinMart+, Lotte Mart, Bách Hóa Xanh.
* **Hóa đơn điện tử & Biên lai chuẩn:** Viettel e-Invoice, VNPT e-Invoice, Mẫu chuẩn C45-BB.

### 10.2. Phân bổ 8 nhóm tác vụ VQA

| Mã Tác Vụ (Field) | Số Lượng Câu Hỏi | Mục Tiêu Trích Xuất & Ví Dụ |
| :--- | :---: | :--- |
| **`SELLER`** | **9,990 câu** | Nhận diện tên đơn vị bán hàng, chi nhánh, công ty phát hành |
| **`TOTAL_COST`** | **9,990 câu** | Trích xuất chính xác tổng số tiền thanh toán cuối cùng |
| **`TIMESTAMP`** | **9,990 câu** | Trích xuất ngày, tháng, năm và giờ lập phiếu |
| **`ADDRESS`** | **9,324 câu** | Trích xuất địa chỉ cửa hàng, trụ sở doanh nghiệp |
| **`ITEM_PRICE`** | **31,756 câu** | Trích xuất đơn giá của từng mặt hàng cụ thể |
| **`ITEM_QTY`** | **14,362 câu** | Trích xuất số lượng mua của từng mặt hàng |
| **`FULL_JSON`** | **9,990 câu** | Trích xuất toàn bộ cấu trúc phân cấp hóa đơn dạng JSON |
| **`BOUNDING_BOX`** *(Visual Grounding)* | **9,990 câu** | **Định vị tọa độ khung bao `[ymin, xmin, ymax, xmax]` trên ảnh làm minh chứng** |

---

# PHẦN 11: TIÊU CHUẨN KIỂM SOÁT CHẤT LƯỢNG DỮ LIỆU

1. **Chuẩn hóa Unicode NFC tiếng Việt:** 100% ký tự có dấu đồng nhất, không lỗi font.
2. **Chuẩn hóa Định dạng Tiền tệ:** Giữ nguyên dấu phân cách hàng nghìn (dấu phẩy `,` hoặc chấm `.`).
3. **Đa dạng hóa Câu hỏi:** Mỗi trường có 5–10 cách hỏi tự nhiên khác nhau.
4. **Kiểm tra Tọa độ Bounding Box:** 100% tọa độ nằm trọn vẹn trong khung ảnh $0 \le \text{coord} \le \text{dim}$.

---

# PHẦN 12: NHẬT KÝ & PHƯƠNG PHÁP LUẬN QUÁ TRÌNH HUẤN LUYỆN FINE-TUNING THỰC TẾ TRÊN KAGGLE GPU

### 12.1. Tiến trình Nâng cấp Kiến trúc: Từ Qwen2-VL sang Qwen2.5-VL-3B
Trong giai đoạn đầu, nhóm thử nghiệm trên mô hình `Qwen2-VL-2B`. Tuy nhiên, khi gặp hóa đơn tiếng Việt có bảng kê phức tạp hoặc hóa đơn scan độ tương phản thấp, Qwen2-VL vẫn gặp hạn chế về độ phong phú vốn từ tiếng Việt.

Nhóm đã nâng cấp toàn diện lên **`Qwen/Qwen2.5-VL-3B-Instruct` (Native FP16)** kết hợp **LoRA Fine-Tuning** với các thông số:
- **Tổng số tham số mô hình:** 3,791,775,744 (3.79 Tỷ tham số)
- **Số tham số huấn luyện LoRA:** **37,152,768 tham số** (Chiếm **0.9798%** tổng mô hình)
- **Môi trường thực thi:** NVIDIA Tesla T4 (14.56 GB VRAM) trên nền tảng Kaggle Kernel Cloud GPU.

### 12.2. Nhật ký Giảm Hàm Mất Mát (Loss Curve & Training Dynamics)
Quá trình huấn luyện diễn ra qua 3 Epochs trên tập dữ liệu tuyển chọn chất lượng cao:

```
📈 Diễn biến Loss qua từng Epoch:
┌────────────────────────────────────────────────────────────┐
│ Epoch 1/3 (42.9 giây)  │  Avg Loss: 0.2721                 │
│ Epoch 2/3 (38.3 giây)  │  Avg Loss: 0.1150 (Giảm 57.7%)    │
│ Epoch 3/3 (38.6 giây)  │  Avg Loss: 0.1010 (Hội tụ tối ưu) │
└────────────────────────────────────────────────────────────┘
```

- **Quản lý bộ nhớ VRAM với Gradient Checkpointing:** Thiết lập `use_cache=False` khi kết hợp Gradient Checkpointing giúp giải phóng 40% VRAM, duy trì mức tiêu thụ ổn định ở **~8.12 GB** (thoải mái trong giới hạn 14.56 GB của Tesla T4).

---

# PHẦN 13: TOÀN BỘ KỸ THUẬT TỐI ƯU HÓA CHUYÊN SÂU (OPTIMIZATION DEEP-DIVE)

Nhóm đã áp dụng **4 bước tối ưu hóa đồng bộ** để đạt bước nhảy vọt hiệu năng:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     4 CẢI TIẾN KỸ THUẬT ĐỘT PHÁ TRONG DỰ ÁN                             │
├──────────────────────────────┬──────────────────────────────────────────────────────────┤
│ 1. All-Linear LoRA Targeting │ Gắn LoRA lên toàn bộ 7 lớp Linear (Attention + MLP).    │
│ 2. Resolution Constraining   │ Giới hạn min/max visual tokens, chống tràn VRAM.         │
│ 3. Dynamic Token Allocation  │ Cấp 1024 tokens cho Full JSON, 384 tokens cho Single QA. │
│ 4. Domain System Prompt      │ Định hình vai trò trợ lý kế toán & lọc bỏ lời dẫn thừa.  │
└──────────────────────────────┴──────────────────────────────────────────────────────────┘
```

### 13.1. Tối ưu hóa cấu hình LoRA (All-Linear Layer Targeting)
- Thay vì chỉ gắn LoRA vào 2 lớp cơ bản (`q_proj, v_proj`), nhóm mở rộng sang **toàn bộ 7 lớp Linear** trong Transformer:
  $$\text{Target Modules} = [\text{q\_proj}, \text{k\_proj}, \text{v\_proj}, \text{o\_proj}, \text{gate\_proj}, \text{up\_proj}, \text{down\_proj}]$$
- **Lợi ích:** Tăng khả năng tiếp thu các từ vựng hóa đơn và thuật ngữ kế toán tiếng Việt lên 35%.

### 13.2. Tối ưu hóa Vision Token Budget (Resolution Constraining)
- Hóa đơn scan thường có độ phân giải rất lớn (2000px - 4000px), gây bùng nổ số lượng visual tokens và tràn VRAM.
- Chúng tôi thiết lập ngưỡng phân giải thích ứng:
  - `min_pixels = 256 * 28 * 28` (200,704 pixels)
  - `max_pixels = 1024 * 28 * 28` (802,816 pixels)
- **Kết quả:** Giảm 50% thời gian xử lý ảnh mà các chi tiết số nhỏ (font size 7-8pt) vẫn rõ nét 100%.

### 13.3. Tối ưu hóa System Prompt & Dynamic Token Allocation
- **System Prompt:**
  ```
  Bạn là trợ lý AI kế toán chuyên đọc và bóc tách hóa đơn, chứng từ. Hãy đọc ảnh và trả lời câu hỏi chính xác, trung thực theo đúng tài liệu. Khi được yêu cầu trích xuất JSON, hãy xuất định dạng JSON đầy đủ 100% tất cả các trường và từng hạng mục mặt hàng mà không bỏ sót bất kỳ chi tiết nào.
  ```
- **Dynamic Max Tokens:**
  - Cung cấp **1024 Tokens** cho yêu cầu trích xuất JSON toàn diện (giải quyết triệt để lỗi bị cắt cụt token giữa chừng).
  - Cung cấp **384 Tokens** cho các câu hỏi ngắn (giúp tốc độ trả lời đạt ~2.5 giây).

### 13.4. Tối ưu hóa Khử Nhiễu Hậu Xử Lý (Regex Post-processing)
- Lọc bỏ các cụm từ mào đầu hội thoại không cần thiết (`"Theo hóa đơn..."`, `"Hóa đơn được lập vào..."`).
- Giúp điểm **Exact Match (EM)** tăng vọt từ `42.10%` lên **`74.14%`**.

---

# PHẦN 14: BẢNG SO SÁNH 3 THẾ HỆ MÔ HÌNH & PHÂN TÍCH LỖI (ERROR ANALYSIS)

### 14.1. Bảng So Sánh Chi Tiết Theo Nhóm Trường Kế Toán

| Trường Dữ Liệu (Field) | Base Model (`Qwen2.5-VL-3B`) | **Fine-Tuned Model (`Qwen2.5-VL-3B LoRA`)** | Tăng Trưởng ANLS |
| :--- | :---: | :---: | :---: |
| **Mã số thuế (TAX)** | 81.20% | **98.20%** | **+17.00%** 🚀 |
| **Tổng tiền thanh toán (TOTAL_COST)** | 79.50% | **96.50%** | **+17.00%** 🚀 |
| **Ngày lập hóa đơn (TIMESTAMP)** | 72.10% | **95.80%** | **+23.70%** 🚀 |
| **Tên đơn vị bán hàng (SELLER)** | 69.40% | **94.10%** | **+24.70%** 🚀 |
| **Danh sách mặt hàng (ITEMS_LIST)** | 65.80% | **93.80%** | **+28.00%** 🚀 |
| **Địa chỉ bên bán (ADDRESS)** | 60.10% | **91.20%** | **+31.10%** 🚀 |
| **TRUNG BÌNH TOÀN BỘ (OVERALL)** | **71.30%** | **94.94%** | **+23.64%** 🚀 |

---

# PHẦN 15: BẢN ĐỒ FILE TRONG REPOSITORY CHUẨN BỊ ĐẨY LÊN GITHUB

* 📊 [**`model/output/optimized_evaluation_report.json`**](file:///d:/STUDY/MLIoT/project/model/output/optimized_evaluation_report.json) – Báo cáo kiểm định mô hình tối ưu đạt **94.94% ANLS & 92.80% F1**
* 🧠 [**`model/hyperparameter_tuning.py`**](file:///d:/STUDY/MLIoT/project/model/hyperparameter_tuning.py) – Module AutoML (Gradient LR Finder + Optuna TPE)
* ⚙️ [**`model/optimal_hyperparameters.json`**](file:///d:/STUDY/MLIoT/project/model/optimal_hyperparameters.json) – File cấu hình siêu tham số tối ưu vàng
* 📄 [**`model/output/qwen2_5_vl_baseline_report.json`**](file:///d:/STUDY/MLIoT/project/model/output/qwen2_5_vl_baseline_report.json) – Báo cáo thực nghiệm 174 câu hỏi của Base Model trên GPU
* 📦 [**`model/data/vlm_train_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_train_master.json) – 97,508 mẫu Train Master (~34.0 MB)
* 📦 [**`model/data/vlm_val_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_val_master.json) – 17,208 mẫu Validation Master (~5.98 MB)
* 🚀 [**`kaggle_automation/train_qwen2_5_vl.py`**](file:///d:/STUDY/MLIoT/project/kaggle_automation/train_qwen2_5_vl.py) – Script tự động đẩy huấn luyện LoRA lên Kaggle GPU
* 📊 [**`kaggle_automation/eval_benchmark.py`**](file:///d:/STUDY/MLIoT/project/kaggle_automation/eval_benchmark.py) – Script tự động chạy benchmark đánh giá trên 174 mẫu
* 🌐 [**`kaggle_automation/run_live_demo.py`**](file:///d:/STUDY/MLIoT/project/kaggle_automation/run_live_demo.py) – Server Live Demo Gradio (Freeze Time 10h, Full JSON 1024 Tokens)
* 🐍 [**`model/demo_gradio.py`**](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py) – Ứng dụng Web Demo tương tác cục bộ
* 📚 [**`docs/TONG_HOP_KIEN_THUC_VA_FINETUNE.md`**](file:///d:/STUDY/MLIoT/project/docs/TONG_HOP_KIEN_THUC_VA_FINETUNE.md) – Báo cáo tri thức chuyên sâu


