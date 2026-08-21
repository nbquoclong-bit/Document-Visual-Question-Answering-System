import os
import sys
import torch
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage1_vlm.src.inference import VQAEngine

print("🚀 Đang khởi tạo VQA Engine (Qwen2-VL)...")
adapter_path = "output/lora_adapters" if os.path.exists("output/lora_adapters") else None
engine = VQAEngine(adapter_dir=adapter_path)

def predict_vqa(image, question):
    if image is None:
        return "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc tài liệu."
    
    if not question or question.strip() == "":
        question = "Trích xuất các thông tin quan trọng trên hóa đơn: Tên đơn vị bán, Mã số thuế, Số hóa đơn, Ngày lập và Tổng tiền thanh toán."
    
    temp_img_path = "temp_gradio_input.jpg"
    image.save(temp_img_path)
    
    try:
        res = engine.extract_and_answer(temp_img_path, question)
        return res
    except Exception as e:
        return f"❌ Lỗi khi xử lý: {str(e)}"

with gr.Blocks(title="Document Visual Question Answering System") as demo:
    gr.Markdown("# 📄 Document Visual Question Answering System (Qwen2-VL VLM)")
    gr.Markdown("Hệ thống Hỏi - Đáp và Trích xuất dữ liệu hóa đơn tự động ứng dụng Qwen2-VL-2B & QLoRA Fine-tuning.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Tải lên ảnh Hóa đơn / Chứng từ / Tài liệu")
            q_input = gr.Textbox(
                lines=2, 
                placeholder="Nhập câu hỏi (VD: Mã số thuế người bán là gì? Tổng tiền thanh toán là bao nhiêu?)",
                label="Câu hỏi / Yêu cầu trích xuất"
            )
            btn_submit = gr.Button("🔍 Trích xuất & Giải đáp", variant="primary")
        
        with gr.Column():
            txt_output = gr.Textbox(lines=10, label="Kết quả phản hồi từ Qwen2-VL")
            
    btn_submit.click(fn=predict_vqa, inputs=[img_input, q_input], outputs=txt_output)

if __name__ == "__main__":
    demo.launch(share=True)
