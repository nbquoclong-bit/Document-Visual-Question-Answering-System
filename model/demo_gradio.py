import os
import sys
import time
import torch
from PIL import Image

try:
    import gradio as gr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    import subprocess
    print("📦 [demo_gradio] Đang tự động cài đặt các thư viện cần thiết...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "gradio>=4.0.0", 
        "qwen-vl-utils>=0.0.8", 
        "peft>=0.12.0", 
        "transformers>=4.49.0", 
        "accelerate>=0.34.2", 
        "-q"
    ])
    import gradio as gr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def find_adapter_dir():
    import zipfile
    possible_paths = [
        os.path.join(base_dir, "output", "qwen2_5_vl_lora_adapters"),
        os.path.join(base_dir, "output", "lora_adapters"),
        os.path.join(base_dir, "stage1_vlm", "output", "lora_adapters")
    ]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
    return None

print("🚀 Đang khởi tạo VQA Engine (Qwen2.5-VL-3B)...")
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1024*28*28)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)

adapter_path = find_adapter_dir()
if adapter_path:
    print(f"✅ Đã tìm thấy và nạp trọng số LoRA tại: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
else:
    print("ℹ️ Chạy Base Model Qwen2.5-VL-3B-Instruct.")
    model = base_model.eval()

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI kế toán chuyên đọc và bóc tách hóa đơn, chứng từ. "
    "Hãy đọc ảnh và trả lời câu hỏi chính xác, trung thực theo đúng tài liệu. "
    "Khi được yêu cầu trích xuất JSON, hãy xuất định dạng JSON đầy đủ 100% tất cả các trường "
    "và từng hạng mục mặt hàng mà không bỏ sót bất kỳ chi tiết nào."
)

def predict_docvqa(image, question):
    if image is None:
        return "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc chứng từ.", "0.00s", "0.00 GB"
    
    if not question or not question.strip():
        question = "Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON."
        
    t0 = time.time()
    q_lower = question.lower()
    is_json = any(k in q_lower for k in ["json", "toàn bộ", "cấu trúc", "tất cả", "hạng mục"])
    max_tokens = 1024 if is_json else 384
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{SYSTEM_PROMPT}\n\nCâu hỏi: {question.strip()}"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
    lat = time.time() - t0
    vram = (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
    return raw_response, f"{lat:.2f}s", f"{vram:.2f} GB"

with gr.Blocks(title="Document VQA Pro - Qwen2.5-VL", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Hệ Thống Document Visual Question Answering (DocVQA Pro)")
    gr.Markdown("💡 Mô hình **Qwen2.5-VL-3B (LoRA Fine-Tuned 94.94% ANLS)**. Hỗ trợ hỏi đáp hóa đơn và trích xuất **Full JSON 1024 Tokens** siêu nhanh.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📄 1. Tải lên ảnh Hóa đơn / Chứng từ")
            q_input = gr.Textbox(
                lines=2, 
                placeholder="Nhập câu hỏi hoặc chọn các nút nghiệp vụ bên dưới...",
                value="Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON đầy đủ 100% tất cả các trường và từng hạng mục mặt hàng.",
                label="💬 2. Câu hỏi cần bóc tách"
            )
            with gr.Row():
                btn_json = gr.Button("🧾 Trích xuất JSON Đầy Đủ", variant="primary", size="sm")
                btn_items = gr.Button("📦 Danh sách món hàng", size="sm")
                btn_tax = gr.Button("🔢 Mã số thuế", size="sm")
            with gr.Row():
                btn_total = gr.Button("💰 Tổng tiền", size="sm")
                btn_vendor = gr.Button("🏢 Tên bên bán", size="sm")
                btn_date = gr.Button("📅 Ngày lập", size="sm")
                btn_addr = gr.Button("📍 Địa chỉ", size="sm")
            btn_submit = gr.Button("🚀 Phân tích & Trích xuất", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            txt_output = gr.Textbox(lines=16, label="💬 3. Kết quả Trích xuất từ AI (Full JSON / Text)")
            with gr.Row():
                lat_output = gr.Textbox(label="⏱️ Tốc độ suy luận", interactive=False)
                vram_output = gr.Textbox(label="🧠 VRAM sử dụng", interactive=False)
                
    btn_json.click(fn=lambda: "Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON đầy đủ 100% tất cả các trường và từng hạng mục mặt hàng.", outputs=q_input)
    btn_items.click(fn=lambda: "Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?", outputs=q_input)
    btn_tax.click(fn=lambda: "Mã số thuế của đơn vị bán hàng trên hóa đơn là gì?", outputs=q_input)
    btn_total.click(fn=lambda: "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?", outputs=q_input)
    btn_vendor.click(fn=lambda: "Tên đơn vị / người bán hàng trên hóa đơn là gì?", outputs=q_input)
    btn_date.click(fn=lambda: "Ngày giờ lập hóa đơn là khi nào?", outputs=q_input)
    btn_addr.click(fn=lambda: "Địa chỉ của đơn vị bán hàng là ở đâu?", outputs=q_input)
    
    btn_submit.click(
        fn=predict_docvqa, 
        inputs=[img_input, q_input], 
        outputs=[txt_output, lat_output, vram_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
