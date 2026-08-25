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
                    full_path = os.path.abspath(os.path.join(root, file))
                    image_index[basename] = full_path
                    # Lưu thêm biến thể không có prefix/suffix
                    clean_name = basename.replace("mcocr_public_", "").replace("mcocr_val_", "")
                    image_index[clean_name] = full_path
                    
    return image_index

def clean_text(text: str) -> str:
    """Làm sạch khoảng trắng thừa và chuẩn hóa chuỗi."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())

def convert_funsd_to_vqa(data_dirs, image_index, output_path):
    """Chuyển đổi các nhãn FUNSD tiếng Việt (MCOCR & vietnamese-receipts-v3) sang VQA samples."""
    vqa_records = []
    processed_files = 0
    missing_images = 0
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"[Warning] Directory {data_dir} does not exist (skip).")
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
            
            if not target_image:
                missing_images += 1
                continue
                
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Cannot read {json_path}: {e}")
                continue
                
            entities = defaultdict(list)
            items_list = []
            
            for item in data.get("form", []):
                raw_text = clean_text(item.get("text", ""))
                label = item.get("label", "OTHER")
                
                if not raw_text or not label or label.upper() == "OTHER":
                    continue
                    
                lbl = label.upper()
                entities[lbl].append(raw_text)
                if lbl == "ITEM_NAME":
                    items_list.append(raw_text)
            
            if not entities:
                continue
                
            processed_files += 1
            img_abs = os.path.abspath(target_image)
            
            # 1. Full JSON extraction sample (KIE)
            output_dict = {}
            for k in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "TAX", "VAT"]:
                if k in entities:
                    output_dict[k] = " ".join(entities[k])
            if items_list:
                output_dict["ITEMS"] = items_list
            if not output_dict:
                output_dict = {lbl: " ".join(texts) for lbl, texts in entities.items()}
                
            vqa_records.append({
                "image_path": img_abs,
                "instruction": "Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON.",
                "output": json.dumps(output_dict, ensure_ascii=False)
            })
            
            # 2. Targeted VQA QA pairs cho từng trường thực thể tiếng Việt
            if "SELLER" in entities:
                vqa_records.append({
                    "image_path": img_abs,
                    "instruction": "Tên cửa hàng / bên bán trên hóa đơn là gì?",
                    "output": " ".join(entities["SELLER"])
                })
            
            if "ADDRESS" in entities:
                vqa_records.append({
                    "image_path": img_abs,
                    "instruction": "Địa chỉ cửa hàng / bên bán là ở đâu?",
                    "output": " ".join(entities["ADDRESS"])
                })
                
            if "TOTAL_COST" in entities:
                vqa_records.append({
                    "image_path": img_abs,
                    "instruction": "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?",
                    "output": " ".join(entities["TOTAL_COST"])
                })
                
            if "TIMESTAMP" in entities:
                vqa_records.append({
                    "image_path": img_abs,
                    "instruction": "Ngày giờ lập hóa đơn là khi nào?",
                    "output": " ".join(entities["TIMESTAMP"])
                })
                
            if items_list:
                vqa_records.append({
                    "image_path": img_abs,
                    "instruction": "Các sản phẩm / món hàng được mua trên hóa đơn là gì?",
                    "output": ", ".join(items_list)
                })
                
            if "TAX" in entities or "VAT" in entities:
                tax_val = " ".join(entities.get("TAX", entities.get("VAT", [])))
                if tax_val:
                    vqa_records.append({
                        "image_path": img_abs,
                        "instruction": "Thuế VAT / Tiền thuế trên hóa đơn là bao nhiêu?",
                        "output": tax_val
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vqa_records, f, ensure_ascii=False, indent=2)
        
    print(f"[INFO] Processed {processed_files} json files (missing images: {missing_images}).")
    print(f"[INFO] Successfully created {output_path} with {len(vqa_records)} VQA records!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
    
    # CHỈ SỬ DỤNG 100% DỮ LIỆU TIẾNG VIỆT (vietnamese-receipts-v3 & MCOCR)
    train_data_dirs = [
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_train_funsd"),
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/train/funsd_json"),
        os.path.join(project_root, "datasets/VN_receipts_train_funsd"),
        os.path.join(project_root, "datasets/MCOCR/mcocr_train_funsd"),
        os.path.join(project_root, "datasets/MCOCR/train/funsd_json"),
        os.path.join(project_root, "datasets/mcocr_train_funsd"),
    ]
    
    test_data_dirs = [
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_val_funsd"),
        os.path.join(project_root, "datasets/vietnamese-receipts-v3/val/funsd_json"),
        os.path.join(project_root, "datasets/VN_receipts_val_funsd"),
        os.path.join(project_root, "datasets/MCOCR/mcocr_val_funsd"),
        os.path.join(project_root, "datasets/MCOCR/val/funsd_json"),
        os.path.join(project_root, "datasets/mcocr_val_funsd"),
    ]
    
    image_dirs = [
        os.path.join(project_root, "datasets/vietnamese-receipts-v3"),
        os.path.join(project_root, "datasets/MCOCR"),
        os.path.join(project_root, "datasets"),
    ]
    
    print("[INFO] Scanning for Vietnamese images in datasets (vietnamese-receipts-v3 & MCOCR)...")
    image_index = find_all_images(image_dirs)
    print(f"[INFO] Found {len(image_index)} Vietnamese image files.")
    
    out_dir = os.path.abspath(os.path.join(base_dir, "../../data"))
    os.makedirs(out_dir, exist_ok=True)
    
    train_out = os.path.join(out_dir, "vlm_train.json")
    test_out = os.path.join(out_dir, "vlm_test.json")
    
    print("\n[INFO] Processing Vietnamese Train data...")
    convert_funsd_to_vqa(train_data_dirs, image_index, train_out)
    
    print("\n[INFO] Processing Vietnamese Validation / Test data...")
    convert_funsd_to_vqa(test_data_dirs, image_index, test_out)
