import os
import sys
import time
import re
import numpy as np
import torch
from PIL import Image

try:
    import gradio as gr
    import easyocr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    import subprocess
    print("📦 [demo_gradio] Đang tự động cài đặt các thư viện cần thiết...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "gradio>=4.0.0", 
        "easyocr",
        "qwen-vl-utils>=0.0.8", 
        "peft>=0.12.0", 
        "transformers>=4.49.0", 
        "accelerate>=0.34.2", 
        "-q"
    ])
    import gradio as gr
    import easyocr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from stage1_vlm.src.visual_grounding import perform_smart_grounding

def find_adapter_dir():
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

print("🔍 Đang nạp EasyOCR Engine...")
reader = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
print("🎉 Đã sẵn sàng phục vụ!")

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI kế toán chuyên đọc và bóc tách hóa đơn, chứng từ. "
    "Hãy đọc ảnh và trả lời câu hỏi chính xác, trung thực theo đúng tài liệu. "
    "Khi được yêu cầu trích xuất JSON, hãy xuất định dạng JSON đầy đủ 100% tất cả các trường "
    "và từng hạng mục mặt hàng mà không bỏ sót bất kỳ chi tiết nào."
)

def predict_docvqa(image, question, enable_bbox):
    if image is None:
        return None, "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc chứng từ.", "0.00s", "0.00 GB"
    
from schema_adapter import SchemaAdapter
import json

def predict_docvqa(image, question, enable_bbox=True, schema_format="Canonical (Quốc tế)"):
    if image is None:
        return None, "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc chứng từ.", "0.00s", "0.00 GB"
    if not question or not question.strip():
        question = "Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON."
        
    t0 = time.time()
    q_lower = question.lower()
    is_json = any(k in q_lower for k in ["json", "toàn bộ", "cấu trúc", "tất cả", "hạng mục", "misa", "sap", "fast"])
    max_tokens = 1024 if is_json else 384
    
    # 1. OCR tokens nếu người dùng bật BBox và không phải trích JSON
    ocr_results = []
    if enable_bbox and not is_json:
        try:
            img_np = np.array(image.convert("RGB"))
            ocr_results = reader.readtext(img_np)
        except Exception:
            pass
            
    # 2. VLM Inference
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{SYSTEM_PROMPT}\n\nCâu hỏi: {question.strip()}"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
    clean_ans = str(raw_response).strip()
    if not is_json:
        for p in [r'^Hóa đơn được lập vào ngày\s*', r'^Theo thông tin trong phiếu thanh toán, ngày lập hóa đơn là\s*', r'^Theo hóa đơn bán lẻ, các mặt hàng/dịch vụ được mua bao gồm:\s*', r'^Theo hóa đơn, các mặt hàng/dịch vụ được mua bao gồm:\s*', r'^The address of the selling company is at\s*']:
            clean_ans = re.sub(p, '', clean_ans, flags=re.IGNORECASE).strip()
    else:
        # Áp dụng Enterprise Schema Adapter nếu là tác vụ trích xuất JSON
        adapted_dict = SchemaAdapter.adapt(clean_ans, target_format=schema_format)
        clean_ans = json.dumps(adapted_dict, ensure_ascii=False, indent=2)
            
    # 3. Vẽ Bounding Box tối giản (1 màu, không nhãn chữ)
    annotated_img = perform_smart_grounding(image, clean_ans, question, ocr_results, enable_bbox=enable_bbox)
    
    lat = time.time() - t0
    vram = (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
    return annotated_img, clean_ans, f"{lat:.2f}s", f"{vram:.2f} GB"

with gr.Blocks(title="Document Visual QA Pro - Qwen2.5-VL", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Hệ Thống Document Visual Question Answering (DocVQA Pro)")
    gr.Markdown("💡 Mô hình **Qwen2.5-VL-3B (LoRA Fine-Tuned 89.63% ANLS)**. Hỗ trợ hỏi đáp hóa đơn, trích xuất **Full JSON** với **Enterprise Schema Adapter (MISA / SAP / FAST)**.")
    
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
                schema_choice = gr.Radio(
                    choices=["Canonical (Quốc tế)", "MISA meInvoice", "SAP S/4HANA ERP", "FAST Accounting"],
                    value="Canonical (Quốc tế)",
                    label="🏢 Chuẩn Hóa Schema Kế Toán (Enterprise Adapter)"
                )
            
            chk_bbox = gr.Checkbox(value=True, label="🎯 Hiển thị Bounding Box minh chứng trực quan (1 màu, không nhãn chữ)")
            
            with gr.Row():
                btn_json = gr.Button("🧾 Trích xuất JSON (Canonical)", variant="primary", size="sm")
                btn_misa = gr.Button("🏢 Xuất MISA JSON", size="sm")
                btn_sap = gr.Button("🌐 Xuất SAP ERP", size="sm")
                btn_fast = gr.Button("⚡ Xuất FAST JSON", size="sm")
            with gr.Row():
                btn_total = gr.Button("💰 Tổng tiền", size="sm")
                btn_items = gr.Button("📦 Danh sách món hàng", size="sm")
                btn_tax = gr.Button("🔢 Mã số thuế", size="sm")
                btn_vendor = gr.Button("🏢 Tên bên bán", size="sm")
            with gr.Row():
                btn_date = gr.Button("📅 Ngày lập", size="sm")
                btn_addr = gr.Button("📍 Địa chỉ", size="sm")
                
            btn_submit = gr.Button("🚀 Phân tích & Trích xuất", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="🎯 3. Ảnh Đối Soát Minh Chứng (Bounding Box)")
            txt_output = gr.Textbox(lines=16, label="💬 4. Kết quả Trích xuất từ AI (Full JSON / Text)")
            with gr.Row():
                lat_output = gr.Textbox(label="⏱️ Tốc độ suy luận", interactive=False)
                vram_output = gr.Textbox(label="🧠 VRAM sử dụng", interactive=False)
                
    json_prompt = "Trích xuất toàn bộ thông tin quan trọng của hóa đơn dưới dạng JSON đầy đủ 100% tất cả các trường và từng hạng mục mặt hàng."
    btn_json.click(fn=lambda: (json_prompt, "Canonical (Quốc tế)"), outputs=[q_input, schema_choice])
    btn_misa.click(fn=lambda: (json_prompt, "MISA meInvoice"), outputs=[q_input, schema_choice])
    btn_sap.click(fn=lambda: (json_prompt, "SAP S/4HANA ERP"), outputs=[q_input, schema_choice])
    btn_fast.click(fn=lambda: (json_prompt, "FAST Accounting"), outputs=[q_input, schema_choice])
    
    btn_items.click(fn=lambda: "Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?", outputs=q_input)
    btn_tax.click(fn=lambda: "Mã số thuế của đơn vị bán hàng trên hóa đơn là gì?", outputs=q_input)
    btn_total.click(fn=lambda: "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?", outputs=q_input)
    btn_vendor.click(fn=lambda: "Tên đơn vị / người bán hàng trên hóa đơn là gì?", outputs=q_input)
    btn_date.click(fn=lambda: "Ngày giờ lập hóa đơn là khi nào?", outputs=q_input)
    btn_addr.click(fn=lambda: "Địa chỉ của đơn vị bán hàng là ở đâu?", outputs=q_input)
    
    btn_submit.click(
        fn=predict_docvqa, 
        inputs=[img_input, q_input, chk_bbox, schema_choice], 
        outputs=[img_output, txt_output, lat_output, vram_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
