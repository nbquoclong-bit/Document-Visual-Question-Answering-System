import os
import sys
import json
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def build_multitemplate_validation_benchmark():
    print("=" * 80)
    print("🚀 ĐANG THIẾT LẬP TẬP BENCHMARK ĐA DẠNG 15 LOẠI HÓA ĐƠN TIẾNG VIỆT")
    print("=" * 80)

    val_img_dir = Path("datasets/vietnamese-receipts-v3/val/images")
    val_label_dir = Path("datasets/vietnamese-receipts-v3/val/labels")
    
    out_dir = Path("datasets/val_benchmark_upload")
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Xóa ảnh cũ
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
        # Lấy 2 ảnh đại diện cho mỗi template: _val_001.png và _val_002.png
        for idx_str in ["001", "002"]:
            img_name = f"{t}_val_{idx_str}.png"
            label_name = f"{t}_val_{idx_str}.json"
            
            img_src = val_img_dir / img_name
            label_src = val_label_dir / label_name
            
            if img_src.exists() and label_src.exists():
                # Copy ảnh vào thư mục upload
                shutil.copy2(img_src, out_images_dir / img_name)
                copied_images.add(img_name)
                
                with open(label_src, "r", encoding="utf-8") as f:
                    label_data = json.load(f)
                    
                annotations = {a.get("label"): a.get("text") for a in label_data.get("annotations", [])}
                
                seller = annotations.get("SELLER")
                total = annotations.get("TOTAL_COST")
                timestamp = annotations.get("TIMESTAMP")
                address = annotations.get("ADDRESS")
                
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

    print(f"✅ Đã copy {len(copied_images)} ảnh thuộc 15 loại hóa đơn vào {out_images_dir}!")
    print(f"📋 Tổng số câu hỏi kiểm định tạo ra: {len(all_benchmark_samples)} câu hỏi.")

    # Lưu file metadata câu hỏi
    questions_file = out_dir / "multitemplate_validation_questions.json"
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_samples, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu file câu hỏi kiểm định: {questions_file}")

    # Tạo file dataset-metadata.json cho Kaggle Dataset
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
