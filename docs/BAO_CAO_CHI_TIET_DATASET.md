# 📊 BÁO CÁO CHI TIẾT BỘ DỮ LIỆU HUẤN LUYỆN & KIỂM ĐỊNH (DATASET REPORT)
## Dự Án Document Visual Question Answering & Visual Grounding Cho 15 Loại Hóa Đơn Tiếng Việt

---

## 1. Thống Kê Tổng Quan Bộ Dữ Liệu

* **Tổng số ảnh hóa đơn:** `4,995 ảnh`
* **Tổng số cặp câu hỏi - câu trả lời (VQA pairs):** `114,716 mẫu`
* **Định dạng lưu trữ:** JSON cấu trúc chuẩn (`image_path`, `template`, `field`, `question`, `answer`)
* **Tập Train Master (`model/data/vlm_train_master.json`):** `97,508 mẫu` (~85%)
* **Tập Validation Master (`model/data/vlm_val_master.json`):** `17,208 mẫu` (~15%)
* **Tập Benchmark Test Độc Lập:** `174 mẫu` (đại diện độc lập trên 30 ảnh kiểm thử)

---

## 2. Phân Bổ 15 Loại Mẫu Hóa Đơn Thực Tế (Mỗi mẫu 333 ảnh)

```
┌──────────────────────────────────────┬────────────────┬────────────────────┐
│ Loại Mẫu Hóa Đơn (Template)          │ Số Lượng Ảnh   │ Ngành Hàng / Lĩnh Vực│
├──────────────────────────────────────┼────────────────┼────────────────────┤
│ 1. cafe_highlands                    │ 333            │ F&B / Chuỗi Cafe   │
│ 2. cafe_phuclong                     │ 333            │ F&B / Chuỗi Trà    │
│ 3. cafe_starbucks                    │ 333            │ F&B / Chuỗi Cafe   │
│ 4. restaurant_jollibee               │ 333            │ F&B / Fast Food    │
│ 5. restaurant_kfc                    │ 333            │ F&B / Fast Food    │
│ 6. convenience_7eleven               │ 333            │ Cửa Hàng Tiện Lợi  │
│ 7. convenience_circlek               │ 333            │ Cửa Hàng Tiện Lợi  │
│ 8. convenience_gs25                  │ 333            │ Cửa Hàng Tiện Lợi  │
│ 9. minimart_anan                     │ 333            │ Siêu Thị Mini      │
│ 10. supermarket_bachhoaxanh          │ 333            │ Chuỗi Siêu Thị     │
│ 11. supermarket_lotte                │ 333            │ Đại Siêu Thị       │
│ 12. supermarket_winmart              │ 333            │ Chuỗi Siêu Thị     │
│ 13. einvoice_viettel                 │ 333            │ Hóa Đơn Điện Tử    │
│ 14. einvoice_vnpt                    │ 333            │ Hóa Đơn Điện Tử    │
│ 15. receipt_c45_bb                   │ 333            │ Phiếu Thu Chuẩn Bộ │
└──────────────────────────────────────┴────────────────┴────────────────────┘
```

---

## 3. Phân Bổ Theo 8 Nhóm Tác Vụ Trích Xuất & Định Vị (Task Breakdown)

| Tác vụ trích xuất (Field) | Số câu hỏi | Cấu trúc dữ liệu & Ví dụ câu hỏi thực tế |
| :--- | :---: | :--- |
| **`SELLER`** | 9,990 | Text: *"Hóa đơn này do cửa hàng / công ty nào phát hành?"* $\rightarrow$ `HIGHLANDS COFFEE` |
| **`TOTAL_COST`** | 9,990 | Text: *"Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"* $\rightarrow$ `109,000` |
| **`TIMESTAMP`** | 9,990 | Text: *"Hóa đơn được lập vào ngày giờ nào?"* $\rightarrow$ `28/06/2026 09:15` |
| **`ADDRESS`** | 9,324 | Text: *"Địa chỉ cửa hàng xuất hóa đơn ở đâu?"* $\rightarrow$ `Số 11 Sư Vạn Hạnh, Q.10` |
| **`LINE_ITEM_LISTS`** | 9,324 | Text: *"Tên, Số lượng, Đơn giá?"* $\rightarrow$ `Sữa tươi - Số lượng: 2 - Đơn giá: 20.000đ` |
| **`ITEM_PRICE`** | 31,756 | Text: *"Đơn giá của Trà Sen Vàng là bao nhiêu?"* $\rightarrow$ `55,000` |
| **`ITEM_QTY`** | 14,362 | Text: *"Số lượng của Phin Sữa Đá là bao nhiêu?"* $\rightarrow$ `1` |
| **`FULL_JSON`** | 9,990 | JSON: `{"seller": "...", "timestamp": "...", "total_cost": "...", "items": [...]}` |
| **`BOUNDING_BOX`** *(Visual Grounding)* | **9,990** | **Tọa độ Bounding Box:** `{"text": "HIGHLANDS COFFEE", "box": [ymin, xmin, ymax, xmax]}` |

---

## 4. Chi Tiết Tác Vụ Bounding Box (Visual Grounding)

Trong tập dữ liệu có **9,990 cặp hỏi đáp định vị tọa độ**:
* **`GROUNDING_SELLER` (4,995 câu):** Định vị tọa độ của tên bên bán/thương hiệu.  
  *Ví dụ:* `"Tìm và định vị vùng chứa tên đơn vị bán hàng trên hóa đơn?"` $\rightarrow$ `{"text": "HIGHLANDS COFFEE LANDMARK", "box": [97, 16, 282, 37]}`
* **`GROUNDING_TOTAL` (4,995 câu):** Định vị tọa độ của số tổng tiền.  
  *Ví dụ:* `"Tìm và định vị vùng chứa tổng tiền thanh toán trên hóa đơn?"` $\rightarrow$ `{"text": "561,000", "box": [307, 456, 359, 477]}`

---

## 5. Vị Trí Lưu Trữ Dữ Liệu Trên Hệ Thống

* 📦 [**`model/data/vlm_train_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_train_master.json) (~34 MB, 97,508 mẫu)
* 📦 [**`model/data/vlm_val_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_val_master.json) (~5.9 MB, 17,208 mẫu)
* 📊 [**`model/data/dataset_summary.json`**](file:///d:/STUDY/MLIoT/project/model/data/dataset_summary.json)
* 🔗 Đã được đồng bộ 100% trên GitHub `main`.
