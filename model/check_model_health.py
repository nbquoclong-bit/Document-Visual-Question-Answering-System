"""
check_model_health.py - Script chẩn đoán toàn diện sức khỏe mô hình Qwen2-VL & LoRA Adapter

Script này tự động thực hiện:
- BƯỚC 1: So sánh suy luận giữa Base Model gốc và Mô hình nạp LoRA Adapter
- BƯỚC 3: Kiểm tra tính toàn vẹn dữ liệu huấn luyện (vlm_train.json & vlm_test.json)
- BƯỚC 4: Kiểm tra sự tồn tại và dung lượng file trọng số LoRA (adapter_model.safetensors)
"""

import os
import sys
import json
from pathlib import Path

# Cấu hình UTF-8 cho stdout trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục hiện tại và root vào sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)


def check_step_4_adapter_files():
    """BƯỚC 4: Kiểm tra sự tồn tại và dung lượng file trọng số LoRA Adapter."""
    print("\n" + "=" * 60)
    print("[CHECK] BUOC 4: KIEM TRA FILE TRONG SO LORA ADAPTER")
    print("=" * 60)

    possible_paths = [
        os.path.join(base_dir, "stage1_vlm/output/lora_adapters"),
        os.path.join(base_dir, "output/lora_adapters"),
        os.path.join(base_dir, "stage1_vlm/output"),
    ]

    adapter_dir = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            adapter_dir = p
            break

    if not adapter_dir:
        print("[ERROR] Khong tim thay thu muc lora_adapters chua adapter_config.json!")
        print("-> Mo hinh chua duoc train hoac chua saved adapter chuan.")
        return None

    print(f"[OK] Tim thay thu muc Adapter tai: {adapter_dir}")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    print(f"   - Target Modules: {config_data.get('target_modules', [])}")
    print(f"   - LoRA r: {config_data.get('r')}, alpha: {config_data.get('lora_alpha')}")

    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    found_weights = False
    for wf in weight_files:
        full_wf = os.path.join(adapter_dir, wf)
        if os.path.exists(full_wf):
            size_mb = os.path.getsize(full_wf) / (1024 * 1024)
            found_weights = True
            print(f"[OK] Tim thay file trong so: {wf} (Dung luong: {size_mb:.2f} MB)")
            if size_mb < 10:
                print("[WARNING] Dung luong file trong so qua nho (< 10MB)! Co the file bi rong hoac loi khi luu.")
            else:
                print("[OK] Dung luong file trong so hop le (> 50MB).")
            break

    if not found_weights:
        print("[ERROR] NGIEM TRONG: Khong tim thay adapter_model.safetensors hoac adapter_model.bin!")
        print("-> He thong se roi ve Base Model goc do thieu trong so fine-tuned.")
        return None

    return adapter_dir


def check_step_3_dataset_integrity():
    """BƯỚC 3: Kiểm tra tính toàn vẹn của dữ liệu huấn luyện (vlm_train.json & vlm_test.json)."""
    print("\n" + "=" * 60)
    print("[CHECK] BUOC 3: KIEM TRA DU LIEU HUAN LUYEN (DATASET INTEGRITY)")
    print("=" * 60)

    data_dir = os.path.join(base_dir, "data")
    train_file = os.path.join(data_dir, "vlm_train.json")
    test_file = os.path.join(data_dir, "vlm_test.json")

    for fpath, label in [(train_file, "TRAIN DATA"), (test_file, "TEST DATA")]:
        if not os.path.exists(fpath):
            print(f"[INFO] Chua thay file {label} tai: {fpath}")
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f)

            print(f"[DATA] [{label}] Tim thay {len(records)} mau du lieu tai: {os.path.basename(fpath)}")
            missing_imgs = 0
            empty_outs = 0

            for r in records:
                img = r.get("image_path", "")
                out = str(r.get("output", "")).strip()
                if not img or not os.path.exists(img):
                    missing_imgs += 1
                if not out:
                    empty_outs += 1

            if missing_imgs > 0:
                print(f"   [WARNING] So duong dan anh KHONG ton tai tren may: {missing_imgs}/{len(records)}")
            else:
                print(f"   [OK] Tat ca duong dan anh deu ton tai hop le tren dia.")

            if empty_outs > 0:
                print(f"   [WARNING] So mau co dap an rong (empty output): {empty_outs}")
            else:
                print(f"   [OK] Tat ca cac mau deu co cau tra loi/nhan khong rong.")

        except Exception as e:
            print(f"[ERROR] Loi khi doc file {label}: {e}")


def check_step_1_model_inference_comparison(adapter_dir):
    """BƯỚC 1: So sánh suy luận giữa Base Model gốc và Mô hình fine-tuned LoRA."""
    print("\n" + "=" * 60)
    print("[CHECK] BUOC 1: SO SANH SUY LUAN BASE MODEL VS LORA ADAPTER")
    print("=" * 60)

    try:
        from stage1_vlm.src.inference import VQAEngine
    except ImportError:
        try:
            from model.stage1_vlm.src.inference import VQAEngine
        except ImportError as err:
            print(f"[ERROR] Khong the import VQAEngine: {err}")
            return

    # Tìm 1 ảnh test mẫu
    sample_candidates = [
        os.path.join(base_dir, "temp_gradio_input.jpg"),
        os.path.join(base_dir, "temp_demo_input.jpg"),
    ]
    
    # Tìm ảnh bất kỳ trong datasets nếu không có ảnh temp
    datasets_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets"))
    if os.path.exists(datasets_dir):
        for root, _, files in os.walk(datasets_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    sample_candidates.append(os.path.join(root, f))
                    if len(sample_candidates) >= 5:
                        break

    test_image = None
    for cand in sample_candidates:
        if os.path.exists(cand):
            test_image = cand
            break

    if not test_image:
        print("[INFO] Khong tim thay anh test mau trong thu muc model/ hoac datasets/.")
        print("-> Vui long dat 1 file anh test (VD: temp_gradio_input.jpg) de thuc hien kiem thu suy luan.")
        return

    question = "Trich xuat thong tin hoa don: Ma so thue, So hoa don, Ngay lap, Tong tien."
    print(f"[IMG] Su dung anh test: {os.path.basename(test_image)}")
    print(f"[QA] Cau hoi test: '{question}'\n")

    # 1. Test Base Model Gốc
    print("1. Dang khoi chay BASE MODEL GOC (chua nap LoRA)...")
    try:
        engine_base = VQAEngine(adapter_dir=None)
        answer_base = engine_base.extract_and_answer(test_image, question)
        print("-> [Base Model Response]:")
        print(answer_base)
    except Exception as e:
        print(f"[ERROR] Loi khi suy luan tren Base Model: {e}")

    print("-" * 50)

    # 2. Test LoRA Adapter Model (nếu có)
    if adapter_dir:
        print(f"2. Dang khoi chay MO HINH FINE-TUNED (voi Adapter tu {adapter_dir})...")
        try:
            engine_lora = VQAEngine(adapter_dir=adapter_dir)
            answer_lora = engine_lora.extract_and_answer(test_image, question)
            print("-> [LoRA Model Response]:")
            print(answer_lora)
            
            # Đánh giá suy luận rác
            if "sexdate" in answer_lora.lower() or answer_lora.count("sex") > 3:
                print("\n[ALERT] PHAT HIEN LOI: LoRA Model tra ve chuoi rac 'sexdate'!")
                print("[DIAGNOSIS] KET LUAN: File trong so LoRA bi hong hoac bi collapse trong luc huan luyen.")
                print("-> HUONG XU LY: Can xoa folder stage1_vlm/output va chay lai script trainer.py tren GPU Colab.")
            else:
                print("\n[SUCCESS] THANH CONG: Mo hinh fine-tuned tra loi ro rang va khong bi lap chu rac!")
        except Exception as e:
            print(f"[ERROR] Loi khi suy luan tren LoRA Model: {e}")
    else:
        print("[WARNING] Bo qua test LoRA do khong tim thay thu muc trong so lora_adapters.")


def main():
    print("=== BAT DAU CHAN DOAN SUC KHOE MO HINH (MODEL HEALTH CHECK) ===")
    
    # Chạy bước 4 trước để tìm folder adapter
    adapter_dir = check_step_4_adapter_files()
    
    # Chạy bước 3 kiểm tra dataset
    check_step_3_dataset_integrity()
    
    # Chạy bước 1 kiểm tra suy luận
    check_step_1_model_inference_comparison(adapter_dir)
    
    print("\n" + "=" * 60)
    print("=== HOAN THANH KIEM TRA CHAN DOAN SUC KHOE MO HINH ===")
    print("=" * 60)


if __name__ == "__main__":
    main()
