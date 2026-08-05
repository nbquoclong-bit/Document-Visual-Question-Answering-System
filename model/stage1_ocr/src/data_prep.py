import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def download_mcocr_instructions():
    print("=== HUONG DAN DOWNLOAD MC_OCR 2021 ===")
    print("1. Truy cap: https://aiexp.ai/dataset/94")
    print("2. Dang ky tai khoan")
    print("3. Download mcocr_public.zip (~1.5GB)")
    print("4. Giai nen vao: stage1_ocr/data/mcocr_public/")
    print("5. Cau truc thu muc mong doi:")
    print("   mcocr_public/")
    print("   ├── mcocr_train_images/")
    print("   ├── mcocr_train_labels.json")
    print("   ├── mcocr_val_images/")
    print("   └── mcocr_val_labels.json")


def parse_mcocr_annotation(annotation_path: str) -> List[Dict]:
    import json
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for img_id, info in data.items():
        img_path = info.get('img_path', '')
        label = info.get('label', {})
        annos = info.get('annos', [])
        records.append({
            'img_id': img_id,
            'img_path': img_path,
            'label': label,
            'annos': annos,
        })
    return records


def convert_to_paddle_format(records: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    for rec in records:
        img_path = rec['img_path']
        if not os.path.exists(img_path):
            continue
        annotations = []
        for anno in rec.get('annos', []):
            text = anno.get('text', '')
            bbox = anno.get('bbox', [])  # [x1, y1, x2, y2]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                annotations.append(f"{x1},{y1},{x2},{y2},{text}")
        if annotations:
            line = f"{img_path}\t{' '.join(annotations)}"
            lines.append(line)
    out_path = os.path.join(output_dir, 'paddle_train.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def prepare_dataset(mcocr_dir: str, output_dir: str):
    download_mcocr_instructions()
    train_anno = os.path.join(mcocr_dir, 'mcocr_train_labels.json')
    if not os.path.exists(train_anno):
        raise FileNotFoundError(f"Khong tim thay: {train_anno}. Vui long download MC_OCR 2021.")
    records = parse_mcocr_annotation(train_anno)
    print(f"Tong cong {len(records)} anh trong tap train")
    out_path = convert_to_paddle_format(records, output_dir)
    print(f"Da chuyen doi xong: {out_path}")
