import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image

def run_test(image_path: str, question: str = "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?"):
    print("=" * 60)
    print("🧪 CHƯƠNG TRÌNH KIỂM THỬ ĐỐI SOÁT QWEN2-VL (BASE MODEL VS LORA)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị: {device}")
    
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if torch.cuda.is_available() else None

    # 1. Nạp Base Model ở chế độ 4-bit (chỉ tốn ~1.5 GB VRAM)
    print("\n[1/3] Đang nạp Base Model: Qwen/Qwen2-VL-2B-Instruct...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    # Chuẩn bị Prompt với System Prompt chuẩn hóa đơn tiếng Việt
    messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia trích xuất dữ liệu hóa đơn kế toán. Hãy đọc kỹ ảnh và trả lời ngắn gọn, chính xác thông tin hoặc số tiền thực tế ghi trên hóa đơn, không giải thích dài dòng."
        },
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
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    eos_ids = [processor.tokenizer.eos_token_id]
    for special_tok in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = processor.tokenizer.convert_tokens_to_ids(special_tok)
        if isinstance(tok_id, int) and tok_id not in eos_ids:
            eos_ids.append(tok_id)

    # --- TEST 1: CHẠY TRỰC TIẾP TRÊN BASE MODEL ---
    print("\n--- 🔍 TEST 1: KẾT QUẢ TỪ BASE MODEL GỐC (Chưa gắn LoRA) ---")
    with torch.no_grad():
        out_ids = base_model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=eos_ids
        )
    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids)]
    ans_base = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
    print(f"👉 Trả lời (Base Model): {ans_base}")

    # --- TEST 2: GẮN THÊM LORA ADAPTER VỪA HUẤN LUYỆN ---
    adapter_path = "/kaggle/working/Document-Visual-Question-Answering-System/model/stage1_vlm/output/lora_adapters"
    if not os.path.exists(adapter_path):
        adapter_path = "model/stage1_vlm/output/lora_adapters"
        
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\n--- 🔍 TEST 2: KẾT QUẢ SAU KHI GẮN LORA ADAPTER ({adapter_path}) ---")
        try:
            lora_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
            lora_model.eval()
            with torch.no_grad():
                out_ids_lora = lora_model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    eos_token_id=eos_ids
                )
            trimmed_lora = [out_ids_lora[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out_ids_lora)]
            ans_lora = processor.batch_decode(trimmed_lora, skip_special_tokens=True)[0].strip()
            print(f"👉 Trả lời (LoRA Fine-tuned): {ans_lora}")
        except Exception as e:
            print(f"❌ Lỗi khi gắn LoRA: {e}")
    else:
        print(f"\n⚠️ Không tìm thấy LoRA adapter tại {adapter_path}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_img = "temp_gradio_input.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    run_test(test_img)
