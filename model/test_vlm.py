import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage1_vlm.src.inference import VQAEngine

def find_adapter_dir():
    possible_paths = [
        "stage1_vlm/output/lora_adapters",
        "output/lora_adapters",
        "stage1_vlm/output"
    ]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
    return None

def main():
    print("🚀 Đang khởi tạo mô hình Qwen2-VL VLM Engine...")
    adapter_dir = find_adapter_dir()
    if adapter_dir:
        print(f"✅ Tìm thấy LoRA Adapter tại: {adapter_dir}")
    else:
        print("💡 Không tìm thấy folder lora_adapters. Sẽ dùng Base Model Qwen2-VL-2B-Instruct.")

    try:
        engine = VQAEngine(adapter_dir=adapter_dir)
        print("✅ Tải mô hình VQAEngine thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    # Đường dẫn ảnh test mẫu
    sample_images = ["temp_gradio_input.jpg", "temp_demo_input.jpg"]
    image_path = None
    for img in sample_images:
        if os.path.exists(img):
            image_path = img
            break

    if image_path:
        question = "Trích xuất thông tin hóa đơn: Mã số thuế, Số hóa đơn, Ngày lập, Tổng tiền."
        print(f"\nĐang đọc ảnh {image_path} và trích xuất thông tin...")
        try:
            answer = engine.extract_and_answer(image_path, question)
            print("\n--- KẾT QUẢ VQA ---")
            print(answer)
        except Exception as e:
            print(f"\n❌ Lỗi khi dự đoán: {e}")
    else:
        print("\n💡 VQAEngine sẵn sàng! Đặt file ảnh vào thư mục để test thử nghiệm CLI.")

if __name__ == "__main__":
    main()
