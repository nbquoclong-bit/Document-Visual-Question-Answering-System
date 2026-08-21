import torch
from typing import Dict, Any
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from .model import load_model_and_processor

class VQAEngine:
    def __init__(self, adapter_dir: str = None, base_model: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model, self.processor = load_model_and_processor(
            base_model_name=base_model,
            is_training=False
        )
        
        if adapter_dir:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            except Exception as e:
                print(f"[Warning] Failed to load adapter from {adapter_dir}: {e}. Running base model.")
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # `device_map="auto"` has already dispatched a GPU model. A CPU model
        # still needs an explicit placement for consistent inference behaviour.
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)

    def extract_and_answer(self, image_path: str, question: str = "Trích xuất thông tin hóa đơn và kiểm tra tính toán.") -> str:
        messages = [
            {
                "role": "system",
                "content": "Bạn là một chuyên gia kế toán kiểm toán Việt Nam. Hãy đọc hóa đơn và trả lời câu hỏi."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
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
        # The processor returns a BatchEncoding; moving it as a whole preserves
        # all image/video tensors. With an Accelerate device map, use the first
        # model parameter's device as the input device.
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)

        eos_ids = [self.processor.tokenizer.eos_token_id, 151643, 151645]
        eos_ids = [t for t in eos_ids if t is not None]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                repetition_penalty=1.2,
                eos_token_id=eos_ids,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return response[0].strip()

if __name__ == "__main__":
    pass
