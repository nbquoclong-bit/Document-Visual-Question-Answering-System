import os
import sys
import json
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def build_multitemplate_validation_benchmark():
    print("=" * 80)
    print("🚀 THIẾT LẬP TẬP BENCHMARK ĐA DẠNG 15 LOẠI HÓA ĐƠN (GỒM CẢ LINE-ITEMS & DỊCH VỤ)")
    print("=" * 80)

    val_img_dir = Path("datasets/vietnamese-receipts-v3/val/images")
    val_label_dir = Path("datasets/vietnamese-receipts-v3/val/labels")
    
    out_dir = Path("datasets/val_benchmark_upload")
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    for p in out_images_dir.glob("*"):
        if p.is_file():
            p.unlink()

    templates = [
        "einvoice_viettel",
        "einvoice_vnpt",
        "receipt_c45_bb",
        "supermarket_winmart",
        "supermarket_lotte",
        "supermarket_bachhoaxanh",
        "convenience_circlek",
        "convenience_gs25",
        "convenience_7eleven",
        "cafe_highlands",
        "cafe_phuclong",
        "cafe_starbucks",
        "restaurant_kfc",
        "restaurant_jollibee",
        "minimart_anan"
    ]

    all_benchmark_samples = []
    sample_id = 1
    copied_images = set()

    for t in templates:
        for idx_str in ["001", "002"]:
            img_name = f"{t}_val_{idx_str}.png"
            label_name = f"{t}_val_{idx_str}.json"
            
            img_src = val_img_dir / img_name
            label_src = val_label_dir / label_name
            
            if img_src.exists() and label_src.exists():
                shutil.copy2(img_src, out_images_dir / img_name)
                copied_images.add(img_name)
                
                with open(label_src, "r", encoding="utf-8") as f:
                    label_data = json.load(f)
                
                annotations = label_data.get("annotations", [])
                
                # 1. Thu thập thông tin header/footer
                seller, total, timestamp, address = None, None, None, None
                for a in annotations:
                    lbl = a.get("label", "").upper()
                    txt = a.get("text", "").strip()
                    if lbl == "SELLER" and not seller:
                        seller = txt
                    elif lbl == "TOTAL_COST" and not total:
                        total = txt
                    elif lbl == "TIMESTAMP" and not timestamp:
                        timestamp = txt
                    elif lbl == "ADDRESS" and not address:
                        address = txt

                if seller:
                    all_benchmark_samples.append({
                        "id": sample_id,
                        "template": t,
                        "image_name": img_name,
                        "field": "SELLER",
                        "question": "Tên đơn vị / người bán hàng trên hóa đơn là gì?",
                        "ground_truth": seller
                    })
                    sample_id += 1
                    
                if total:
                    all_benchmark_samples.append({
                        "id": sample_id,
                        "template": t,
                        "image_name": img_name,
                        "field": "TOTAL_COST",
                        "question": "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?",
                        "ground_truth": total
                    })
                    sample_id += 1
                    
                if timestamp:
                    all_benchmark_samples.append({
                        "id": sample_id,
                        "template": t,
                        "image_name": img_name,
                        "field": "TIMESTAMP",
                        "question": "Ngày giờ lập hóa đơn là khi nào?",
                        "ground_truth": timestamp
                    })
                    sample_id += 1
                    
                if address:
                    all_benchmark_samples.append({
                        "id": sample_id,
                        "template": t,
                        "image_name": img_name,
                        "field": "ADDRESS",
                        "question": "Địa chỉ của đơn vị bán hàng là ở đâu?",
                        "ground_truth": address
                    })
                    sample_id += 1

                # 2. Thu thập danh sách mặt hàng / dịch vụ
                items = []
                current_item = {}
                for a in annotations:
                    lbl = a.get("label", "").upper()
                    txt = a.get("text", "").strip()
                    if lbl == "ITEM_NAME":
                        if current_item and "name" in current_item:
                            items.append(current_item)
                        current_item = {"name": txt}
                    elif lbl == "ITEM_QTY":
                        current_item["qty"] = txt
                    elif lbl == "ITEM_PRICE":
                        current_item["price"] = txt
                    elif lbl == "ITEM_AMOUNT":
                        current_item["amount"] = txt
                if current_item and "name" in current_item:
                    items.append(current_item)

                # Thêm câu hỏi danh sách mặt hàng/dịch vụ
                if items:
                    item_names_str = ", ".join([it["name"] for it in items[:6]])
                    all_benchmark_samples.append({
                        "id": sample_id,
                        "template": t,
                        "image_name": img_name,
                        "field": "ITEMS_LIST",
                        "question": "Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?",
                        "ground_truth": item_names_str
                    })
                    sample_id += 1
                    
                    # Thêm câu hỏi tra cứu giá cho món đầu tiên
                    first_item = items[0]
                    first_name = first_item["name"]
                    first_val = first_item.get("amount") or first_item.get("price")
                    if first_val:
                        all_benchmark_samples.append({
                            "id": sample_id,
                            "template": t,
                            "image_name": img_name,
                            "field": "ITEM_PRICE",
                            "question": f"Thành tiền của {first_name} là bao nhiêu?",
                            "ground_truth": first_val
                        })
                        sample_id += 1

    print(f"✅ Đã copy {len(copied_images)} ảnh thuộc 15 loại hóa đơn vào {out_images_dir}!")
    print(f"📋 Tổng số câu hỏi kiểm định toàn diện tạo ra: {len(all_benchmark_samples)} câu hỏi.")

    questions_file = out_dir / "multitemplate_validation_questions.json"
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_samples, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu file câu hỏi kiểm định: {questions_file}")

    dataset_metadata = {
        "title": "docvqa-benchmark-dataset",
        "id": "lminhsang241/docvqa-benchmark-dataset",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(out_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    return all_benchmark_samples

if __name__ == "__main__":
    build_multitemplate_validation_benchmark()
