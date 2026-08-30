# 📘 TỔNG HỢP TOÀN DIỆN: MỤC TIÊU SẢN PHẨM, DỮ LIỆU & KẾT QUẢ MÔ HÌNH TỐI ƯU
## Đề Tài: Hệ Thống Document Visual Question Answering (DocVQA) & Visual Grounding Cho Hóa Đơn Tiếng Việt

> **Mục đích tài liệu:** Đây là kho thông tin và số liệu thực nghiệm đầy đủ, chi tiết và chuẩn xác nhất về:
> 1. **Bối cảnh, Mục tiêu sản phẩm & Tính năng Bounding Box minh chứng trực quan.**
> 2. **Toàn bộ công tác kỹ thuật dữ liệu (114,716 mẫu VQA trên 15 loại hóa đơn).**
> 3. **Phương pháp Toán học & AutoML tìm Siêu tham số tối ưu (Gradient LR Finder + Optuna TPE).**
> 4. **Kết quả thực nghiệm định lượng toàn diện trên GPU Tesla T4 (Base Model vs Fine-Tuned vs Optimized Model đạt 94.94% ANLS).**  
> Các thành viên trong nhóm có thể tự do trích xuất bất kỳ bảng biểu, số liệu so sánh hay ví dụ thực tế nào để đưa vào bài thuyết trình và slide.

---

## 📌 MỤC LỤC TỔNG QUAN

1. [Phần 1: Bối Cảnh Bài Toán & Mục Tiêu Sản Phẩm](#phần-1-bối-cảnh-bài-toán--mục-tiêu-sản-phẩm)
2. [Phần 2: Tính Năng Cốt Lõi: Minh Chứng Trực Quan Bằng Bounding Box (Visual Grounding)](#phần-2-tính-năng-cốt-lõi-minh-chứng-trực-quan-bằng-bounding-box-visual-grounding)
3. [Phần 3: Người Dùng Mục Tiêu & Trải Nghiệm Sử Dụng](#phần-3-người-dùng-mục-tiêu--trải-nghiệm-sử-dụng)
4. [Phần 4: Đặc Tả Đầu Vào, Đầu Ra & Luồng Hệ Thống](#phần-4-đặc-tả-đầu-vào-đầu-ra--luồng-hệ-thống)
5. [Phần 5: Kết Quả Đo Lường Thực Nghiệm Base Model (Zero-Shot) Trên GPU](#phần-5-kết-quả-đo-lường-thực-nghiệm-base-model-zero-shot-trên-gpu)
6. [Phần 6: Phương Pháp Toán Học Tìm Siêu Tham Số Tối Ưu (AutoML & Gradient Descent)](#phần-6-phương-pháp-toán-học-tìm-siêu-tham-số-tối-ưu-automl--gradient-descent)
7. [Phần 7: Kết Quả Thực Nghiệm Mô Hình Sau Tối Ưu Hóa Toàn Diện (Đạt 94.94% ANLS)](#phần-7-kết-quả-thực-nghiệm-mô-hình-sau-tối-ưu-hóa-toàn-diện-đạt-9494-anls)
8. [Phần 8: Quy Trình Xây Dựng & Chuẩn Hóa Dữ Liệu](#phần-8-quy-trình-xây-dựng--chuẩn-hóa-dữ-liệu)
9. [Phần 9: Thống Kê Chi Tiết Bộ Dữ Liệu 114,716 Mẫu VQA](#phần-9-thống-kê-chi-tiết-bộ-dữ-liệu-114716-mẫu-vqa)
10. [Phần 10: Phân Loại 15 Mẫu Hóa Đơn & 8 Nhóm Tác Vụ](#phần-10-phân-loại-15-mẫu-hóa-đơn--8-nhóm-tác-vụ)
11. [Phần 11: Tiêu Chuẩn Kiểm Soát Chất Lượng Dữ Liệu](#phần-11-tiêu-chuẩn-kiểm-soát-chất-lượng-dữ-liệu)
12. [Phần 12: Bản Đồ File Trong Repository](#phần-12-bản-đồ-file-trong-repository)

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
* **Tên sản phẩm:** Hệ thống Trích Xuất & Minh Chứng Hóa Đơn Thông Minh (**Document VQA with Visual Grounding Proof**).
* **Nguyên lý cốt lõi:** **100% CÂU TRẢ LỜI CỦA MÔ HÌNH PHẢI ĐI KÈM BOUNDING BOX MINH CHỨNG TRỰC QUAN TRÊN ẢNH.**
* **Mục tiêu định lượng (KPIs):**
  * **Tốc độ xử lý:** Dưới **2 giây / câu hỏi** trên GPU phổ thông.
  * **Khả năng minh chứng:** Tự động vẽ Bounding Box màu sắc nổi bật khoanh vùng đúng vị trí thông tin để người dùng kiểm chứng mắt thường trong 0.5 giây.
  * **Khả năng tích hợp:** Xuất trực tiếp cấu trúc **JSON phân cấp** để kết nối vào các hệ thống phần mềm kế toán (MISA, FAST, SAP) và ERP doanh nghiệp.
  * **Tự động hóa:** Giảm **95%** thời gian nhập liệu thủ công của kế toán viên.

---

# PHẦN 2: TÍNH NĂNG CỐT LÕI: MINH CHỨNG TRỰC QUAN BẰNG BOUNDING BOX (VISUAL GROUNDING)

### 2.1. Tại sao sản phẩm BẮT BUỘC phải có Bounding Box minh chứng?
* **Chống ảo giác AI (Anti-Hallucination):** Trong nghiệp vụ tài chính - kế toán, độ chính xác là tuyệt đối. Bounding Box là bằng chứng không thể chối cãi chứng minh mô hình thực sự "nhìn" thấy con số trên hóa đơn chứ không phải sinh ngẫu nhiên.
* **Quy trình kiểm chứng siêu tốc (Human-in-the-loop):** Kế toán viên chỉ cần nhìn vào khung màu đỏ/xanh được vẽ sẵn trên màn hình để xác nhận trong 1 giây mà không phải dò tìm toàn bộ tờ hóa đơn dài.
* **Trích xuất ảnh bằng chứng (Audit Trail Cropping):** Tự động crop đúng vùng Bounding Box để lưu trữ làm bằng chứng kiểm toán điện tử.

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ [ẢNH HÓA ĐƠN GỐC]                 │ [ẢNH MINH CHỨNG BOUNDING BOX ĐẦU RA]  │
  │                                   │                                       │
  │   HIGHLANDS COFFEE                │   ┌───────────────────────────┐       │
  │   Số 11 Sư Vạn Hạnh, Q.10         │   │📍 SELLER: HIGHLANDS COFFEE│       │
  │   ---------------------------     │   └───────────────────────────┘       │
  │   1. Trà Sen Vàng (L)  55,000     │   Số 11 Sư Vạn Hạnh, Q.10             │
  │   2. Phin Sữa Đá  (M)  54,000     │   ---------------------------         │
  │   ---------------------------     │   1. Trà Sen Vàng (L)  55,000         │
  │   TỔNG TIỀN: 109,000đ             │   2. Phin Sữa Đá  (M)  54,000         │
  │                                   │   ---------------------------         │
  │                                   │   ┌───────────────────────────┐       │
  │                                   │   │📍 TOTAL_COST: 109,000đ    │       │
  │                                   │   └───────────────────────────┘       │
  └───────────────────────────────────┴───────────────────────────────────────┘
```

### 2.2. Bảng mã màu phân loại Bounding Box theo chuẩn nghiệp vụ

| Loại trường thông tin | Mã màu hiển thị | Ý nghĩa nghiệp vụ |
| :--- | :---: | :--- |
| **`SELLER`** | 🟢 **Xanh lá cây (Emerald)** | Tên cửa hàng, đơn vị bán hàng, chi nhánh |
| **`TOTAL_COST`** | 🔴 **Đỏ nổi bật (Alizarin)** | Tổng tiền thanh toán cuối cùng (trọng yếu nhất) |
| **`TIMESTAMP`** | 🔵 **Xanh dương (Peter River)** | Ngày giờ in hóa đơn, kỳ lập phiếu |
| **`ADDRESS`** | 🟠 **Màu Cam (Carrot)** | Địa chỉ nơi phát sinh giao dịch |
| **`ITEMS_LIST` / `ITEM_PRICE`** | 🟣 **Màu Tím (Amethyst)** | Danh sách mặt hàng, chi tiết đơn giá từng món |
| **`TAX` (Mã số thuế)** | 🟢 **Xanh ngọc (Turquoise)** | Mã số thuế doanh nghiệp |

---

# PHẦN 3: NGƯỜI DÙNG MỤC TIÊU & TRẢI NGHIỆM SỬ DỤNG

### 3.1. Đối tượng người dùng mục tiêu (User Personas)
1. **Kế toán doanh nghiệp:** Cần bóc tách nhanh tổng tiền, ngày hóa đơn, MST và muốn nhìn thấy ngay Bounding Box để đối soát trước khi duyệt chi.
2. **Chủ cửa hàng / Thủ kho:** Kiểm tra số lượng và đơn giá từng món hàng nhập kho từ các phiếu thu viết tay hoặc in nhiệt.
3. **Người dùng cá nhân:** Chụp ảnh hóa đơn mua sắm để theo dõi chi tiêu và lưu trữ bằng chứng mua hàng.

---

# PHẦN 4: ĐẶC TẢ ĐẦU VÀO, ĐẦU RA & LUỒNG HỆ THỐNG

### 4.1. Sơ đồ luồng sản phẩm (System Flowchart)

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
│ 1. Text Trực │      │ 2. Dữ Liệu    │      │ 3. Ảnh Minh   │
│    Diện      │      │    JSON Chuẩn │      │    Chứng Kèm  │
│    "109,000" │      │    Hóa Phân   │      │    Bounding   │
│              │      │    Cấp CSDL   │      │    Box Màu 📍 │
└──────────────┘      └───────────────┘      └───────────────┘
```

---

# PHẦN 5: KẾT QUẢ ĐO LƯỜNG THỰC NGHIỆM BASE MODEL (ZERO-SHOT) TRÊN GPU

Nhóm đã tiến hành kiểm thử thực tế mô hình gốc **Qwen2.5-VL-3B-Instruct (Zero-Shot)** trên toàn bộ **174 câu hỏi kiểm định** thuộc 15 loại mẫu hóa đơn trên **GPU NVIDIA Tesla T4** (`model/output/qwen2_5_vl_baseline_report.json`).

### 5.1. Bảng số liệu định lượng thực tế

| Chỉ Số Đo Lường (Metrics) | Kết Quả Base Model Zero-Shot | Đánh Giá Kỹ Thuật |
| :--- | :---: | :--- |
| **Tổng số câu hỏi kiểm định** | **174 câu hỏi** | Bao phủ 100% 15 loại mẫu hóa đơn thực tế |
| **Điểm ANLS (DocVQA Metric)** | **0.68%** | Gần như bằng 0 theo tiêu chuẩn quốc tế |
| **Tỷ lệ Exact Match (EM %)** | **0.00%** | **0 / 174 câu trả lời đúng 100%** ⚠️ |
| **Điểm Token F1-Score** | **35.25%** | Chỉ trích xuất được rải rác một vài từ khóa |
| **Thời gian suy luận trung bình** | **3.76 giây / câu** | Chậm do phải sinh thêm nhiều từ dẫn nhập thừa |
| **Bộ nhớ GPU chiếm dụng** | **3.64 GB** | Hoạt động nhẹ nhàng trên GPU Tesla T4 |

### 5.2. Minh chứng các trường hợp lỗi thực tế (Empirical Proof):
* **Lỗi 1 (Preamble Chatter):** Hóa đơn VNPT (`einvoice_vnpt_val_001.png`), câu hỏi địa chỉ $\rightarrow$ Base Model sinh: `"The address of the selling company is at Số 99 Nguyễn Huệ, Quận 7, TP. Hồ Chí Minh."` $\implies$ **ANLS = 0.0, Exact Match = 0%**.
* **Lỗi 2 (Thừa văn phong đàm thoại):** Hóa đơn Highlands, hỏi tổng tiền $\rightarrow$ Sinh: `"Dựa trên hình ảnh hóa đơn bạn cung cấp, tổng tiền thanh toán là 109,000 VNĐ."` $\implies$ **ANLS = 0.0, Exact Match = 0%**.

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
| **Effective Batch Size** | **16** | Batch size $1 \times \text{Gradient Accumulation } 16$ (Tối ưu VRAM $< 5\text{GB}$) |

---

# PHẦN 7: KẾT QUẢ THỰC NGHIỆM MÔ HÌNH SAU TỐI ƯU HÓA TOÀN DIỆN (ĐẠT 94.94% ANLS)

Sau khi áp dụng **Bộ siêu tham số tối ưu**, kết hợp **Strict System Prompt** và **2,400 mẫu dữ liệu đa tác vụ**, mô hình **Qwen2.5-VL-3B LoRA Optimized** đã hoàn thành kiểm định trên 174 câu hỏi thực tế (`model/output/optimized_evaluation_report.json`):

### 7.1. Bảng so sánh 3 giai đoạn tiến hóa hiệu năng:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    BẢNG SO SÁNH TIẾN TRÌNH TỐI ƯU HÓA TOÀN DIỆN                         │
├──────────────────────────────┬──────────────────┬──────────────────┬────────────────────┤
│ Chỉ Số Đo Lường Học Thuật    │ Base (Zero-Shot) │ Đợt 1 (LoRA 900) │ Đợt 2 (Tối Ưu Hoá) │
├──────────────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ ANLS Score (DocVQA Metric)   │ 🔴 0.68%         │ 🟡 59.48%        │ 🟢 94.94% (Gấp 139)│
│ Token F1-Score               │ 🔴 35.25%        │ 🟡 73.45%        │ 🟢 92.80% (Gấp 2.6)│
│ Exact Match (EM %)           │ 🔴 0.00% (0/174) │ 🟡 39.66%        │ 🟢 74.14% (129/174)│
│ Tốc độ suy luận (Latency)    │ 3.76 giây / câu  │ 4.56 giây / câu  │ ⚡ 2.59 giây / câu │
│ VRAM GPU chiếm dụng          │ 3.64 GB          │ 4.96 GB          │ 5.28 GB (Tesla T4) │
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

# PHẦN 12: BẢN ĐỒ FILE TRONG REPOSITORY

* 📊 [**`model/output/optimized_evaluation_report.json`**](file:///d:/STUDY/MLIoT/project/model/output/optimized_evaluation_report.json) – Báo cáo kiểm định mô hình tối ưu đạt **94.94% ANLS & 92.80% F1**
* 🧠 [**`model/hyperparameter_tuning.py`**](file:///d:/STUDY/MLIoT/project/model/hyperparameter_tuning.py) – Module AutoML (Gradient LR Finder + Optuna TPE)
* ⚙️ [**`model/optimal_hyperparameters.json`**](file:///d:/STUDY/MLIoT/project/model/optimal_hyperparameters.json) – File cấu hình siêu tham số tối ưu vàng
* 📄 [**`model/output/qwen2_5_vl_baseline_report.json`**](file:///d:/STUDY/MLIoT/project/model/output/qwen2_5_vl_baseline_report.json) – Báo cáo thực nghiệm 174 câu hỏi của Base Model trên GPU
* 📦 [**`model/data/vlm_train_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_train_master.json) – 97,508 mẫu Train Master (~34.0 MB)
* 📦 [**`model/data/vlm_val_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_val_master.json) – 17,208 mẫu Validation Master (~5.98 MB)
* 🐍 [**`model/stage1_vlm/src/visual_grounding.py`**](file:///d:/STUDY/MLIoT/project/model/stage1_vlm/src/visual_grounding.py) – Module vẽ Bounding Box màu sắc & badge phân loại
* 🐍 [**`model/demo_gradio.py`**](file:///d:/STUDY/MLIoT/project/model/demo_gradio.py) – Ứng dụng Web Demo tương tác người dùng
