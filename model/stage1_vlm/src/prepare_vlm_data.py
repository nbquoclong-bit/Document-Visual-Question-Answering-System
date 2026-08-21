import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import os
import json
import glob
from collections import defaultdict

def find_all_images(image_dirs):
    """Quét đệ quy tìm tất cả ảnh (mọi định dạng hoa/thường) và lập chỉ mục theo basename."""
    image_index = {}
    valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}
    
    for img_dir in image_dirs:
        if not os.path.exists(img_dir):
            continue
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    basename = os.path.splitext(file)[0]
                    image_index[basename] = os.path.join(root, file)
                    # Lưu thêm biến thể không có prefix/suffix
                    clean_name = basename.replace("mcocr_public_", "").replace("mcocr_val_", "")
                    image_index[clean_name] = os.path.join(root, file)
                    
    return image_index

def convert_funsd_to_vqa(data_dirs, image_index, output_path):
    vqa_records = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"[Warning] Directory {data_dir} does not exist!")
            continue
            
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        for json_path in json_files:
            basename = os.path.splitext(os.path.basename(json_path))[0]
            
            # Kiểm tra ảnh tồn tại trong chỉ mục
            target_image = None
            if basename in image_index:
                target_image = image_index[basename]
            else:
                alt_name = basename.replace("mcocr_public_", "").replace("_ver2", "")
                if alt_name in image_index:
                    target_image = image_index[alt_name]
            
            # Nếu không tìm thấy ảnh theo tên exact, lấy ảnh đầu tiên cùng folder để demo test
            if not target_image:
                continue
                
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            entities = defaultdict(list)
            for item in data.get("form", []):
                text = item.get("text", "").strip()
                label = item.get("label", "OTHER")
                
                if not text or label.upper() == "OTHER":
                    continue
                    
                entities[label.upper()].append(text)
            
            if entities:
                # 1. Full JSON extraction record
                output_dict = {lbl: " ".join(texts) for lbl, texts in entities.items()}
                vqa_records.append({
                    "image_path": os.path.abspath(target_image),
                    "instruction": "Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON.",
                    "output": json.dumps(output_dict, ensure_ascii=False)
                })
                
                # 2. Single-field VQA QA pairs for targeted instruction tuning
                if "SELLER" in entities:
                    vqa_records.append({
                        "image_path": os.path.abspath(target_image),
                        "instruction": "Tên cửa hàng / bên bán là gì?",
                        "output": " ".join(entities["SELLER"])
                    })
                if "TOTAL_COST" in entities:
                    vqa_records.append({
                        "image_path": os.path.abspath(target_image),
                        "instruction": "Tổng tiền thanh toán trên hóa đơn?",
                        "output": " ".join(entities["TOTAL_COST"])
                    })
                if "TIMESTAMP" in entities:
                    vqa_records.append({
                        "image_path": os.path.abspath(target_image),
                        "instruction": "Ngày giờ lập hóa đơn là khi nào?",
                        "output": " ".join(entities["TIMESTAMP"])
                    })
                if "ITEM_NAME" in entities:
                    vqa_records.append({
                        "image_path": os.path.abspath(target_image),
                        "instruction": "Đã mua những sản phẩm / món ăn nào?",
                        "output": ", ".join(entities["ITEM_NAME"])
                    })
                if "TAX" in entities or "VAT" in entities:
                    tax_val = " ".join(entities.get("TAX", entities.get("VAT", [])))
                    vqa_records.append({
                        "image_path": os.path.abspath(target_image),
                        "instruction": "Thuế VAT / Tiền thuế là bao nhiêu?",
                        "output": tax_val
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vqa_records, f, ensure_ascii=False, indent=2)
        
    print(f"[INFO] Successfully created {output_path} with {len(vqa_records)} VQA records!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
    
    train_data_dirs = [
        os.path.join(project_root, "datasets/SROIE/sroie_train_funsd_ver_2"),
        os.path.join(project_root, "datasets/MCOCR/mcocr_train_funsd"),
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_train_funsd")
    ]
    
    test_data_dirs = [
        os.path.join(project_root, "datasets/SROIE/sroie_train_val_ver_2"),
        os.path.join(project_root, "datasets/MCOCR/mcocr_val_funsd"),
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_val_funsd")
    ]
    
    image_dirs = [
        os.path.join(project_root, "datasets")
    ]
    
    print("[INFO] Scanning for images in datasets...")
    image_index = find_all_images(image_dirs)
    print(f"[INFO] Found {len(image_index)} image files.")
    
    out_dir = os.path.abspath(os.path.join(base_dir, "../../data"))
    os.makedirs(out_dir, exist_ok=True)
    
    train_out = os.path.join(out_dir, "vlm_train.json")
    test_out = os.path.join(out_dir, "vlm_test.json")
    
    print("[INFO] Processing Train data...")
    convert_funsd_to_vqa(train_data_dirs, image_index, train_out)
    
    print("[INFO] Processing Test data...")
    convert_funsd_to_vqa(test_data_dirs, image_index, test_out)
