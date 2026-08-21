import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import os
import json
import glob
from collections import defaultdict

def find_all_images(image_dirs):
    """Quét đệ quy tìm tất cả ảnh và lập chỉ mục theo basename."""
    image_index = {}
    for img_dir in image_dirs:
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            # Dùng ** để tìm đệ quy
            for filepath in glob.glob(os.path.join(img_dir, '**', ext), recursive=True):
                basename = os.path.splitext(os.path.basename(filepath))[0]
                image_index[basename] = filepath
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
            
            # Kiểm tra ảnh tồn tại
            if basename not in image_index:
                continue
                
            image_path = image_index[basename]
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Gom nhóm các từ theo nhãn
            entities = defaultdict(list)
            for item in data.get("form", []):
                text = item.get("text", "").strip()
                label = item.get("label", "OTHER")
                
                if not text or label.upper() == "OTHER":
                    continue
                    
                entities[label.upper()].append(text)
            
            # Nếu có dữ liệu trích xuất được
            if entities:
                # Ghép các từ có cùng nhãn bằng khoảng trắng
                output_dict = {lbl: " ".join(texts) for lbl, texts in entities.items()}
                
                record = {
                    "image_path": os.path.abspath(image_path),
                    "instruction": "Trích xuất thông tin hóa đơn dưới dạng JSON.",
                    "output": json.dumps(output_dict, ensure_ascii=False)
                }
                vqa_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vqa_records, f, ensure_ascii=False, indent=2)
        
    print(f"[INFO] Successfully created {output_path} with {len(vqa_records)} records!")

if __name__ == "__main__":
    train_data_dirs = [
        "../../../datasets/sroie_train_funsd",
        "../../../datasets/mcocr_train_funsd"
    ]
    test_data_dirs = [
        "../../../datasets/sroie_test_funsd",
        "../../../datasets/mcocr_test_funsd"
    ]
    image_dirs = [
        "../../../datasets/sroie_images",
        "../../../datasets/mcocr_images"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dirs = [os.path.abspath(os.path.join(base_dir, d)) for d in train_data_dirs]
    test_data_dirs = [os.path.abspath(os.path.join(base_dir, d)) for d in test_data_dirs]
    image_dirs = [os.path.abspath(os.path.join(base_dir, d)) for d in image_dirs]
    
    print("[INFO] Scanning for images...")
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
