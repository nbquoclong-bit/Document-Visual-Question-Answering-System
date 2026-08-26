import os
import sys
import time
import argparse
from PIL import Image

# Thêm đường dẫn model vào sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from stage1_vlm.src.inference import VQAEngine

def find_adapter_dir():
    import zipfile
    possible_zips = [
        "/kaggle/working/qwen2_vl_lora_adapters.zip",
        os.path.join(base_dir, "output", "qwen2_vl_lora_adapters.zip"),
        os.path.join(base_dir, "qwen2_vl_lora_adapters.zip"),
        os.path.join(project_root, "qwen2_vl_lora_adapters.zip")
    ]
    for z in possible_zips:
        if os.path.exists(z):
            extract_target = os.path.join(base_dir, "output", "lora_adapters")
            if not os.path.exists(os.path.join(extract_target, "adapter_config.json")):
                os.makedirs(extract_target, exist_ok=True)
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(extract_target)
                    print(f"📦 Đã tự động giải nén trọng số LoRA từ {z} vào {extract_target}!")
                except Exception as err:
                    print(f"⚠️ Lỗi giải nén {z}: {err}")

    search_roots = [
        os.path.join(base_dir, "output"),
        os.path.join(base_dir, "stage1_vlm", "output"),
        "/kaggle/working",
        base_dir
    ]
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "adapter_config.json" in filenames:
                    return dirpath
    return None

def main():
    parser = argparse.ArgumentParser(description="Query trực tiếp mô hình Qwen2-VL không qua Web")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh hóa đơn")
    parser.add_argument("--question", type=str, default="Tổng tiền thanh toán trên hóa đơn là bao nhiêu?", help="Câu hỏi")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 KHỞI CHẠY QUERY THỦ CÔNG MÔ HÌNH QWEN2-VL")
    print("=" * 60)

    adapter_dir = find_adapter_dir()
    if adapter_dir:
        print(f"✅ Đã tìm thấy LoRA Adapter: {adapter_dir}")
    else:
        print("ℹ️ Chạy trực tiếp Base Model Qwen2-VL-2B-Instruct.")

    engine = VQAEngine(adapter_dir=adapter_dir)
    print("✅ Nạp mô hình thành công!\n")

    # Tìm file ảnh nếu không truyền tham số
    img_path = args.image
    if not img_path or not os.path.exists(img_path):
        candidates = [
            "temp_gradio_input.jpg",
            "invoice.png",
            "invoice.jpg",
            "test.png",
            "test.jpg"
        ]
        # Tìm trong /kaggle/working hoặc datasets
        for root, _, files in os.walk("/kaggle/working" if os.path.exists("/kaggle/working") else base_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    candidates.insert(0, os.path.join(root, f))
                    break
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break

    if not img_path or not os.path.exists(img_path):
        print("❌ Không tìm thấy file ảnh để query. Vui lòng truyền: python model/query_manual.py --image <đường_dẫn_ảnh>")
        return

    print(f"📄 Đang đọc ảnh: {img_path}")
    print(f"❓ Câu hỏi: {args.question}")

    t0 = time.time()
    answer = engine.extract_and_answer(img_path, args.question)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"🎯 KẾT QUẢ TỪ AI (Thời gian phản hồi: {elapsed:.2f}s):")
    print("=" * 60)
    print(answer)
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
