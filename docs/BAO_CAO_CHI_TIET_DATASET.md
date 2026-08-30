# 📊 BÁO CÁO CHI TIẾT BỘ DỮ LIỆU HUẤN LUYỆN & KIỂM ĐỊNH (DATASET REPORT)
## Dự Án Document Visual Question Answering Cho 15 Loại Hóa Đơn Tiếng Việt

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

## 3. Phân Bổ Theo 7 Nhóm Tác Vụ Trích Xuất (Task Breakdown)

| Tác vụ trích xuất (Field) | Số câu hỏi | Ví dụ câu hỏi thực tế |
| :--- | :---: | :--- |
| **`SELLER`** | 9,990 | *"Hóa đơn này do cửa hàng / công ty nào phát hành?"* |
| **`TOTAL_COST`** | 9,990 | *"Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"* |
| **`TIMESTAMP`** | 9,990 | *"Hóa đơn được lập vào ngày giờ nào?"* |
| **`ADDRESS`** | 9,324 | *"Địa chỉ cửa hàng xuất hóa đơn ở đâu?"* |
| **`ITEM_PRICE`** | 31,756 | *"Đơn giá của [Tên mặt hàng] là bao nhiêu?"* |
| **`ITEM_QTY`** | 14,362 | *"Số lượng của [Tên mặt hàng] là bao nhiêu?"* |
| **`FULL_JSON`** | 9,990 | *"Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON?"* |
| **`BOUNDING_BOX`** | 9,990 | *"Tọa độ vùng chữ của tổng tiền trên ảnh là bao nhiêu?"* |

---

## 4. Vị Trí Lưu Trữ Dữ Liệu Trên Hệ Thống

* 📦 [**`model/data/vlm_train_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_train_master.json) (~34 MB)
* 📦 [**`model/data/vlm_val_master.json`**](file:///d:/STUDY/MLIoT/project/model/data/vlm_val_master.json) (~5.9 MB)
* 📊 [**`model/data/dataset_summary.json`**](file:///d:/STUDY/MLIoT/project/model/data/dataset_summary.json)
* 🔗 Đã được đồng bộ 100% trên GitHub `main`.
