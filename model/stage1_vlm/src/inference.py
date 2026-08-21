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
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure model is on device if device_map="auto" didn't already put it there
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
        if not hasattr(self.model, "hf_device_map"):
            inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return response[0].strip()

if __name__ == "__main__":
    # Test inference
    # engine = VQAEngine()
    # print(engine.extract_and_answer("path/to/invoice.jpg", "Tổng tiền là bao nhiêu?"))
    pass
