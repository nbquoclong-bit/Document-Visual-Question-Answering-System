import os
import sys
import torch
from contextlib import nullcontext

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image
from peft import PeftModel
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
                raise RuntimeError(f"Không thể nạp LoRA adapter từ {adapter_dir}: {e}") from e
        elif adapter_dir:
            raise FileNotFoundError(
                f"Adapter dir {adapter_dir} không chứa adapter_config.json."
            )
        
        self.model.eval()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if not hasattr(self.model, "hf_device_map") and not hasattr(self.model, "is_quantized"):
            self.model.to(self.device)

    def extract_and_answer(
        self,
        image_input=None,
        question: str = "Trích xuất thông tin hóa đơn và kiểm tra tính toán.",
        *,
        image_path=None,
        max_new_tokens: int = 256,
        use_adapter: bool = True,
    ) -> str:
        """Answer from a file path or PIL image; `image_path` keeps API compatibility."""
        if image_input is None:
            image_input = image_path
        elif image_path is not None:
            raise ValueError("Chỉ truyền một trong hai tham số image_input hoặc image_path.")
        if image_input is None:
            raise ValueError("Thiếu ảnh đầu vào cho VQAEngine.")

        # Hỗ trợ cả đường dẫn ảnh và đối tượng PIL.Image.
        if isinstance(image_input, (str, os.PathLike)):
            with Image.open(image_input) as source_image:
                image = source_image.convert("RGB")
        else:
            image = image_input

        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        # System Prompt tổng quát chuyên sâu cho Document VQA trên mọi loại chứng từ/hóa đơn
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia AI thị giác - ngôn ngữ chuyên xử lý tài liệu, hóa đơn và chứng từ tài chính tiếng Việt. "
                    "Hãy trả lời 100% bằng tiếng Việt tự nhiên, trực tiếp và chính xác theo nội dung hiển thị trên tài liệu.\n\n"
                    "CÁC NGUYÊN TẮC BÓC TÁCH VÀ SUY LUẬN TỔNG QUÁT:\n"
                    "1. Cấu trúc bảng biểu & Không gian: Khi tra cứu thông tin trong bảng hoặc danh mục, luôn đối chiếu chính xác giao điểm giữa Hàng (tên mục hàng hóa, dịch vụ, loại thuế) và Cột (tiêu đề cột). Phân biệt rõ các cột: 'Số tiền/Thành tiền trước thuế', 'Tiền thuế/VAT', và 'Tổng tiền sau thuế'.\n"
                    "2. Ngữ nghĩa thực thể tài chính:\n"
                    "   - Số tiền chịu thuế / tiền trước thuế: là giá trị tiền hàng chưa gồm thuế.\n"
                    "   - Tiền thuế: là số tiền thuế phát sinh.\n"
                    "   - Tổng thanh toán / tổng tiền: là số tiền cuối cùng người mua cần trả.\n"
                    "   - Mã số thuế: là chuỗi số định danh thuế của đúng chủ thể (bên bán hoặc bên mua theo câu hỏi).\n"
                    "   - Số hóa đơn: là số thứ tự chứng từ (không lấy ký hiệu/mẫu số).\n"
                    "3. Chống ảo giác (Anti-Hallucination): Chỉ trả lời thông tin có xuất hiện thực tế trên tài liệu. Nếu câu hỏi về một mục không tồn tại, phải trả lời rõ ràng: 'Không có thông tin về [mục được hỏi] trên tài liệu này', tuyệt đối không tự ý lấy số liệu khác bù vào.\n"
                    "4. Định dạng câu trả lời: Trả lời ngắn gọn, trực diện con số hoặc nội dung được hỏi, không giải thích lan man."
                )
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

        adapter_context = nullcontext()
        if not use_adapter and hasattr(self.model, "disable_adapter"):
            adapter_context = self.model.disable_adapter()
        with adapter_context:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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
        if use_adapter and not res_text and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                with torch.no_grad():
                    gen_ids_base = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=3,
                        eos_token_id=eos_ids
                    )
                trim_base = [gen_ids_base[0][len(inputs["input_ids"][0]):]]
                res_text = self.processor.batch_decode(trim_base, skip_special_tokens=True)[0].strip()
                
        return res_text
