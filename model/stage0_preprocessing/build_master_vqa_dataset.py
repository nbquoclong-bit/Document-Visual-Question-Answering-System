import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_text(t):
    return " ".join(str(t).strip().split()) if t else ""

# 1. BỘ CÂU HỎI ĐA DẠNG NGÔN NGỮ TỰ NHIÊN (DATA AUGMENTATION)
QUESTION_TEMPLATES = {
    "SELLER": [
        "Tên đơn vị / người bán hàng trên hóa đơn là gì?",
        "Hóa đơn này được phát hành bởi công ty / cửa hàng nào?",
        "Nhà cung cấp / bên bán trên hóa đơn là ai?",
        "Tên quán / thương hiệu trên hóa đơn là gì?",
        "Tên đơn vị bán hàng là gì?"
    ],
    "TOTAL_COST": [
        "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?",
        "Khách hàng phải thanh toán tổng cộng bao nhiêu tiền?",
        "Tổng cộng số tiền trên hóa đơn là bao nhiêu?",
        "Số tiền cần thanh toán là bao nhiêu?",
        "Tổng tiền trên hóa đơn là bao nhiêu?"
    ],
    "TIMESTAMP": [
        "Ngày giờ lập hóa đơn là khi nào?",
        "Hóa đơn này được xuất vào ngày tháng năm nào?",
        "Thời gian in hóa đơn / thanh toán là lúc nào?",
        "Ngày lập chứng từ là ngày nào?"
    ],
    "ADDRESS": [
        "Địa chỉ của đơn vị bán hàng là ở đâu?",
        "Cửa hàng / công ty phát hành hóa đơn nằm ở địa chỉ nào?",
        "Địa chỉ nơi mua hàng là gì?",
        "Địa chỉ trụ sở bên bán là ở đâu?"
    ],
    "ITEMS_LIST": [
        "Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?",
        "Hóa đơn này bao gồm những sản phẩm / dịch vụ nào?",
        "Các món ăn / thức uống / dịch vụ đã mua là gì?",
        "Liệt kê tất cả các mặt hàng có trên hóa đơn?"
    ],
    "FULL_JSON": [
        "Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON?",
        "Xuất cấu trúc JSON chi tiết của hóa đơn gồm bên bán, tổng tiền, ngày lập và danh sách mặt hàng?",
        "Bóc tách cấu trúc dữ liệu hóa đơn dạng JSON chuẩn kế toán?"
    ]
}

def build_master_vqa_dataset():
    print("=" * 85)
    print("🚀 BẮT ĐẦU XÂY DỰNG BỘ DỮ LIỆU ĐA NHIỆM CHUẨN MỤC TIÊU DỰ ÁN (MULTI-TASK DATASET)")
    print("=" * 85)

    base_dir = Path("datasets")
    out_dir = Path("model/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    v3_dir = base_dir / "vietnamese-receipts-v3"
    
    all_vqa_records = []
    stats = {
        "total_images": 0,
        "total_vqa_pairs": 0,
        "seller_questions": 0,
        "total_cost_questions": 0,
        "timestamp_questions": 0,
        "address_questions": 0,
        "line_items_list_questions": 0,
        "item_price_questions": 0,
        "item_qty_questions": 0,
        "full_json_extraction_questions": 0,
        "bounding_box_grounding_questions": 0,
        "template_distribution": defaultdict(int)
    }

    # Quét dữ liệu từ Vietnamese Receipts V3
    print("🔍 Đang quét và xử lý 15 loại hóa đơn thương mại...")
    for split in ["train", "val"]:
        lbl_dir = v3_dir / split / "labels"
        img_dir = v3_dir / split / "images"
        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.json"):
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            img_fname = data.get("file_name", f"{lbl_file.stem}.png")
            img_path = img_dir / img_fname
            if not img_path.exists():
                img_path = Path(f"datasets/vietnamese-receipts-v3/{split}/images/{img_fname}")

            annotations = data.get("annotations", [])
            if not annotations:
                continue

            stats["total_images"] += 1
            template_name = lbl_file.stem.split("_val_")[0].split("_train_")[0]
            stats["template_distribution"][template_name] += 1

            seller, total, timestamp, address = "", "", "", ""
            seller_box, total_box, timestamp_box, address_box = None, None, None, None
            items = []
            curr_item = {}

            for a in annotations:
                lbl = a.get("label", "").upper()
                txt = clean_text(a.get("text", ""))
                box = a.get("box")
                if lbl == "SELLER" and not seller:
                    seller = txt
                    seller_box = box
                elif lbl == "TOTAL_COST" and not total:
                    total = txt
                    total_box = box
                elif lbl == "TIMESTAMP" and not timestamp:
                    timestamp = txt
                    timestamp_box = box
                elif lbl == "ADDRESS" and not address:
                    address = txt
                    address_box = box
                elif lbl == "ITEM_NAME":
                    if curr_item and "name" in curr_item:
                        items.append(curr_item)
                    curr_item = {"name": txt, "box": box}
                elif lbl == "ITEM_QTY":
                    curr_item["qty"] = txt
                elif lbl == "ITEM_PRICE":
                    curr_item["price"] = txt
                elif lbl == "ITEM_AMOUNT":
                    curr_item["amount"] = txt

            if curr_item and "name" in curr_item:
                items.append(curr_item)

            img_path_str = str(img_path).replace("\\", "/")

            # -------------------------------------------------------------
            # NHIỆM VỤ 1: HỎI ĐÁP HEADER & FOOTER ĐƠN LẺ (VQA KEY-VALUES)
            # -------------------------------------------------------------
            if seller:
                for q in QUESTION_TEMPLATES["SELLER"][:2]:
                    all_vqa_records.append({
                        "image_path": img_path_str,
                        "template": template_name,
                        "field": "SELLER",
                        "question": q,
                        "answer": seller
                    })
                    stats["seller_questions"] += 1

            if total:
                for q in QUESTION_TEMPLATES["TOTAL_COST"][:2]:
                    all_vqa_records.append({
                        "image_path": img_path_str,
                        "template": template_name,
                        "field": "TOTAL_COST",
                        "question": q,
                        "answer": total
                    })
                    stats["total_cost_questions"] += 1

            if timestamp:
                for q in QUESTION_TEMPLATES["TIMESTAMP"][:2]:
                    all_vqa_records.append({
                        "image_path": img_path_str,
                        "template": template_name,
                        "field": "TIMESTAMP",
                        "question": q,
                        "answer": timestamp
                    })
                    stats["timestamp_questions"] += 1

            if address:
                for q in QUESTION_TEMPLATES["ADDRESS"][:2]:
                    all_vqa_records.append({
                        "image_path": img_path_str,
                        "template": template_name,
                        "field": "ADDRESS",
                        "question": q,
                        "answer": address
                    })
                    stats["address_questions"] += 1

            # -------------------------------------------------------------
            # NHIỆM VỤ 2: BÓC TÁCH CHI TIẾT BẢNG MẶT HÀNG & DỊCH VỤ (LINE-ITEMS)
            # -------------------------------------------------------------
            if items:
                item_names = [it["name"] for it in items if it.get("name")]
                if item_names:
                    for q in QUESTION_TEMPLATES["ITEMS_LIST"][:2]:
                        all_vqa_records.append({
                            "image_path": img_path_str,
                            "template": template_name,
                            "field": "ITEMS_LIST",
                            "question": q,
                            "answer": ", ".join(item_names[:6])
                        })
                        stats["line_items_list_questions"] += 1

                for it in items[:4]:
                    name = it.get("name")
                    amt = it.get("amount") or it.get("price")
                    qty = it.get("qty")
                    if name and amt:
                        all_vqa_records.append({
                            "image_path": img_path_str,
                            "template": template_name,
                            "field": "ITEM_PRICE",
                            "question": f"Thành tiền của {name} là bao nhiêu?",
                            "answer": amt
                        })
                        all_vqa_records.append({
                            "image_path": img_path_str,
                            "template": template_name,
                            "field": "ITEM_PRICE",
                            "question": f"Giá / Phí của {name} trên hóa đơn là bao nhiêu?",
                            "answer": amt
                        })
                        stats["item_price_questions"] += 2
                    if name and qty:
                        all_vqa_records.append({
                            "image_path": img_path_str,
                            "template": template_name,
                            "field": "ITEM_QTY",
                            "question": f"Số lượng của {name} là bao nhiêu?",
                            "answer": qty
                        })
                        stats["item_qty_questions"] += 1

            # -------------------------------------------------------------
            # NHIỆM VỤ 3: BÓC TÁCH TOÀN BỘ HÓA ĐƠN DẠNG JSON (STRUCTURAL JSON)
            # -------------------------------------------------------------
            if seller or total:
                structured_dict = {}
                if seller: structured_dict["seller"] = seller
                if address: structured_dict["address"] = address
                if timestamp: structured_dict["timestamp"] = timestamp
                if total: structured_dict["total_cost"] = total
                if items:
                    structured_dict["items"] = [
                        {"name": it.get("name"), "qty": it.get("qty", "1"), "amount": it.get("amount") or it.get("price", "")}
                        for it in items[:5] if it.get("name")
                    ]
                
                json_answer = json.dumps(structured_dict, ensure_ascii=False)
                for q_json in QUESTION_TEMPLATES["FULL_JSON"][:2]:
                    all_vqa_records.append({
                        "image_path": img_path_str,
                        "template": template_name,
                        "field": "FULL_JSON",
                        "question": q_json,
                        "answer": json_answer
                    })
                    stats["full_json_extraction_questions"] += 1

            # -------------------------------------------------------------
            # NHIỆM VỤ 4: ĐỊNH VỊ BOUNDING BOX (VISUAL GROUNDING)
            # -------------------------------------------------------------
            if total and total_box:
                all_vqa_records.append({
                    "image_path": img_path_str,
                    "template": template_name,
                    "field": "GROUNDING_TOTAL",
                    "question": "Tìm và định vị vùng chứa tổng tiền thanh toán trên hóa đơn?",
                    "answer": json.dumps({"text": total, "box": total_box}, ensure_ascii=False)
                })
                stats["bounding_box_grounding_questions"] += 1

            if seller and seller_box:
                all_vqa_records.append({
                    "image_path": img_path_str,
                    "template": template_name,
                    "field": "GROUNDING_SELLER",
                    "question": "Tìm và định vị vùng chứa tên đơn vị bán hàng trên hóa đơn?",
                    "answer": json.dumps({"text": seller, "box": seller_box}, ensure_ascii=False)
                })
                stats["bounding_box_grounding_questions"] += 1

    stats["total_vqa_pairs"] = len(all_vqa_records)
    random.seed(42)
    random.shuffle(all_vqa_records)

    split_idx = int(len(all_vqa_records) * 0.85)
    train_records = all_vqa_records[:split_idx]
    val_records = all_vqa_records[split_idx:]

    train_path = out_dir / "vlm_train_master.json"
    val_path = out_dir / "vlm_val_master.json"
    summary_path = out_dir / "dataset_summary.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 85)
    print(f"🎉 HOÀN TẤT ĐÓNG GÓI BỘ DỮ LIỆU ĐA NHIỆM TOÀN DIỆN: {out_dir}")
    print("=" * 85)
    print(f"- 📁 Tệp Train độc lập : {train_path} ({len(train_records)} mẫu VQA)")
    print(f"- 📁 Tệp Val độc lập   : {val_path} ({len(val_records)} mẫu VQA)")
    print(f"- 📊 Báo cáo Thống kê : {summary_path}")
    print(f"  • 1. Tên bên bán (SELLER)              : {stats['seller_questions']} câu")
    print(f"  • 2. Tổng tiền (TOTAL_COST)            : {stats['total_cost_questions']} câu")
    print(f"  • 3. Ngày giờ lập (TIMESTAMP)          : {stats['timestamp_questions']} câu")
    print(f"  • 4. Địa chỉ (ADDRESS)                 : {stats['address_questions']} câu")
    print(f"  • 5. Danh sách mặt hàng (ITEMS_LIST)   : {stats['line_items_list_questions']} câu")
    print(f"  • 6. Giá/Phí từng món (ITEM_PRICE)     : {stats['item_price_questions']} câu")
    print(f"  • 7. Số lượng món (ITEM_QTY)           : {stats['item_qty_questions']} câu")
    print(f"  • 8. Xuất toàn bộ JSON (FULL_JSON)     : {stats['full_json_extraction_questions']} câu")
    print(f"  • 9. Định vị Bounding Box (GROUNDING)  : {stats['bounding_box_grounding_questions']} câu")
    print("=" * 85)

if __name__ == "__main__":
    build_master_vqa_dataset()
