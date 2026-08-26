import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image

def test_lora_scales(image_path: str, question: str = "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?"):
    print("=" * 60)
    print("🔬 KIỂM THỬ ĐIỀU CHỈNH SCALE TRỌNG SỐ LORA (SCALE FACTOR)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = "/kaggle/working/Document-Visual-Question-Answering-System/model/stage1_vlm/output/lora_adapters"
    if not os.path.exists(adapter_path):
        adapter_path = "model/stage1_vlm/output/lora_adapters"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        print("✅ Đã nạp LoRA adapter.")
    else:
        print("❌ Không tìm thấy LoRA adapter.")
        return

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path,
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 768 * 28 * 28
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    # Thử nghiệm các mức scale khác nhau từ 0.0 (Base), 0.05, 0.1, 0.2, 0.5, 1.0 (Full)
    scales_to_test = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for scale in scales_to_test:
        # Cập nhật scale factor cho toàn bộ các layer LoRA
        for name, module in model.named_modules():
            if hasattr(module, "scaling"):
                if isinstance(module.scaling, dict):
                    for k in module.scaling:
                        orig_r = getattr(module, "r", {}).get(k, 16) if isinstance(getattr(module, "r", None), dict) else getattr(module, "r", 16)
                        orig_alpha = getattr(module, "lora_alpha", {}).get(k, 32) if isinstance(getattr(module, "lora_alpha", None), dict) else getattr(module, "lora_alpha", 32)
                        base_scale = orig_alpha / orig_r if orig_r else 1.0
                        module.scaling[k] = base_scale * scale
                elif isinstance(module.scaling, (int, float)):
                    orig_r = getattr(module, "r", 16)
                    orig_alpha = getattr(module, "lora_alpha", 32)
                    base_scale = orig_alpha / orig_r if orig_r else 1.0
                    module.scaling = base_scale * scale

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [out[0][len(inputs.input_ids[0]):]]
        ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        print(f"\n🔹 Scale = {scale:.2f} ({int(scale*100)}% LoRA effect):")
        print(f"   👉 Trả lời: {ans if ans else '[Chuỗi rỗng / EOS ngắt ngay]'}")

if __name__ == "__main__":
    test_img = "temp_gradio_input.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    test_lora_scales(test_img)
