import os
import json
import glob
from collections import defaultdict

def parse_funsd_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entities = defaultdict(list)
    for item in data.get("form", []):
        text = item.get("text", "").strip()
        label = item.get("label", "OTHER")
        if not text or label.upper() == "OTHER":
            continue
        entities[label.upper()].append(text)
        
    return entities

def build_unseen_test_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    val_funsd_dir = os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_val_funsd")
    val_images_dir = os.path.join(project_root, "datasets/vietnamese-receipts-v3/val/images")
    
    if not os.path.exists(val_funsd_dir):
        print(f"[Error] Khong tim thay thu muc JSON: {val_funsd_dir}")
        return
        
    json_files = glob.glob(os.path.join(val_funsd_dir, "*.json"))
    print(f"[Dataset] Da tim thay {len(json_files)} file nhan JSON va tap anh thuc te tu 'vietnamese-receipts-v3'!")
    
    test_records = []
    found_images = 0
    
    for json_path in json_files:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        img_name = basename + ".png"
        img_path = os.path.join(val_images_dir, img_name)
        
        if not os.path.exists(img_path):
            # Thu tim .jpg
            img_path = os.path.join(val_images_dir, basename + ".jpg")
            if not os.path.exists(img_path):
                img_path = None
            else:
                found_images += 1
        else:
            found_images += 1
            
        entities = parse_funsd_json(json_path)
        if not entities:
            continue
            
        # 1. Tên bên bán
        if "SELLER" in entities:
            seller = " ".join(entities["SELLER"])
            test_records.append({
                "image_path": img_path,
                "source_file": os.path.basename(json_path),
                "question": "Tên cửa hàng / bên bán là gì?",
                "ground_truth": seller
            })
            
        # 2. Tổng tiền
        if "TOTAL_COST" in entities:
            total = " ".join(entities["TOTAL_COST"])
            test_records.append({
                "image_path": img_path,
                "source_file": os.path.basename(json_path),
                "question": "Tổng tiền thanh toán trên hóa đơn?",
                "ground_truth": total
            })
            
        # 3. Sản phẩm
        if "ITEM_NAME" in entities:
            items = ", ".join(entities["ITEM_NAME"])
            test_records.append({
                "image_path": img_path,
                "source_file": os.path.basename(json_path),
                "question": "Đã mua những sản phẩm / món ăn nào?",
                "ground_truth": items
            })
            
        # 4. Thời gian
        if "TIMESTAMP" in entities:
            ts = " ".join(entities["TIMESTAMP"])
            test_records.append({
                "image_path": img_path,
                "source_file": os.path.basename(json_path),
                "question": "Ngày giờ lập hóa đơn là khi nào?",
                "ground_truth": ts
            })

    output_path = os.path.join(base_dir, "test_unseen_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
        
    print(f"- Da khop duoc {found_images}/{len(json_files)} file anh thuc te!")
    print(f"- Da dong goi thanh cong {len(test_records)} cau hoi kiem thu tu vietnamese-receipts-v3 vao file: model/test_unseen_dataset.json")

if __name__ == "__main__":
    build_unseen_test_dataset()
