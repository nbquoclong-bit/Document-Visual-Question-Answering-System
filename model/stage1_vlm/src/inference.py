import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from typing import Dict, Any
from PIL import Image

try:
    from peft import PeftModel
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft>=0.12.0", "bitsandbytes>=0.43.0", "-q"])
    from peft import PeftModel

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    import subprocess
    print("📦 [inference] Đang tự động cài đặt thư viện 'qwen-vl-utils'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils>=0.0.8", "-q"])
    from qwen_vl_utils import process_vision_info

from .model import load_model_and_processor

class VQAEngine:
    def __init__(self, adapter_dir: str = None, base_model: str = "Qwen/Qwen2-VL-2B-Instruct"):
        use_cuda = torch.cuda.is_available()
        self.model, self.processor = load_model_and_processor(
            base_model_name=base_model,
            is_training=False,
            use_4bit=use_cuda
        )
        
        if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            try:
                print(f"[VQAEngine] Loading LoRA adapters from {adapter_dir}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=False)
                print("🎯 [VQAEngine] Nạp LoRA adapter chuẩn thành công 100%!")
            except Exception as e:
                print(f"[Warning] Failed to load adapter from {adapter_dir}: {e}. Running base model.")
        elif adapter_dir:
            print(f"[Warning] Adapter dir {adapter_dir} does not contain adapter_config.json. Running base model.")
        
        self.model.eval()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if not hasattr(self.model, "hf_device_map") and not hasattr(self.model, "is_quantized"):
            self.model.to(self.device)

    def extract_and_answer(self, image_input, question: str = "Trích xuất thông tin hóa đơn và kiểm tra tính toán.") -> str:
        # Hỗ trợ cả đường dẫn ảnh (str) và đối tượng PIL.Image
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        # System prompt định hướng chuyên gia trích xuất trực tiếp giá trị
        messages = [
            {
                "role": "system",
                "content": "Bạn là hệ thống AI trích xuất thông tin hóa đơn tài chính tiếng Việt độ chính xác cao. Hãy quan sát thật kỹ các ký tự, dấu tiếng Việt và con số trên hóa đơn để trả lời chính xác, đầy đủ ngày giờ hoặc số tiền. Tuyệt đối chỉ trả về giá trị cần trích xuất, không thêm tiền tố hay giải thích."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image,
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 1280 * 28 * 28
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)

        # Lấy danh sách token kết thúc của Qwen2-VL (<|im_end|>, <|endoftext|>)
        eos_ids = [self.processor.tokenizer.eos_token_id]
        for special_tok in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = self.processor.tokenizer.convert_tokens_to_ids(special_tok)
            if isinstance(tok_id, int) and tok_id not in eos_ids:
                eos_ids.append(tok_id)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=eos_ids,
            )


        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        res_text = response[0].strip()
        
        # Nếu gắn LoRA mà bị rỗng, tự động tắt LoRA để lấy câu trả lời chuẩn xác từ Base Model
        if not res_text and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                with torch.no_grad():
                    gen_ids_base = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=3,
                        eos_token_id=eos_ids
                    )
                trim_base = [gen_ids_base[0][len(inputs["input_ids"][0]):]]
                res_text = self.processor.batch_decode(trim_base, skip_special_tokens=True)[0].strip()
                
        return res_text
