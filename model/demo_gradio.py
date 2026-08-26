import os
import sys
import torch
import gradio as gr

# Tự động nạp đường dẫn module
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from stage1_vlm.src.inference import VQAEngine
except ImportError:
    from model.stage1_vlm.src.inference import VQAEngine

def find_adapter_dir():
    candidates = [
        os.path.join(base_dir, "stage1_vlm", "output", "lora_adapters"),
        os.path.join(base_dir, "output", "lora_adapters"),
        os.path.join(base_dir, "stage1_vlm", "output"),
        "/kaggle/working/Document-Visual-Question-Answering-System/model/stage1_vlm/output/lora_adapters",
        "/kaggle/working/lora_adapters",
        "output/lora_adapters"
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
    return None

print("🚀 Đang khởi tạo VQA Engine (Qwen2-VL)...")
adapter_path = find_adapter_dir()
if adapter_path:
    print(f"✅ Đã tìm thấy trọng số LoRA tại: {adapter_path}")
else:
    print("ℹ️ Chạy trực tiếp Base Model Qwen2-VL-2B (Không tìm thấy LoRA adapter cục bộ).")

engine = VQAEngine(adapter_dir=adapter_path)

def predict_vqa(image, question):
    if image is None:
        return "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc tài liệu."
    
    if not question or question.strip() == "":
        question = "Trích xuất các thông tin quan trọng: Tên đơn vị bán, Mã số thuế, Ngày lập, Tổng tiền thanh toán."
    
    temp_img_path = os.path.join(base_dir, "temp_gradio_input.jpg")
    image.save(temp_img_path)
    
    try:
        res = engine.extract_and_answer(temp_img_path, question)
        return res
    except Exception as e:
        return f"❌ Lỗi khi xử lý: {str(e)}"

with gr.Blocks(title="Document VQA System - Qwen2-VL") as demo:
    gr.Markdown("# 📄 Hệ Thống Hỏi - Đáp & Bóc Tách Hóa Đơn (Qwen2-VL)")
    gr.Markdown("Trợ lý AI hỗ trợ kế toán: Trích xuất thông tin, hỏi đáp số tiền, ngày lập, tên quán từ hóa đơn tiếng Việt.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Tải lên ảnh Hóa đơn / Chứng từ")
            q_input = gr.Textbox(
                lines=2, 
                placeholder="Nhập câu hỏi (VD: Tổng tiền thanh toán là bao nhiêu? Tên đơn vị bán là gì? Ngày lập hóa đơn?)",
                label="Câu hỏi / Yêu cầu trích xuất"
            )
            btn_submit = gr.Button("🔍 Trích xuất & Trả lời", variant="primary")
            
            gr.Examples(
                examples=[
                    ["Tổng tiền thanh toán là bao nhiêu?"],
                    ["Tên đơn vị bán / Cửa hàng là gì?"],
                    ["Mã số thuế của người bán là gì?"],
                    ["Ngày lập hóa đơn là ngày nào?"],
                    ["Trích xuất toàn bộ thông tin quan trọng dưới dạng JSON."]
                ],
                inputs=[q_input],
                label="Gợi ý câu hỏi nhanh"
            )
        
        with gr.Column():
            txt_output = gr.Textbox(lines=12, label="Kết quả phản hồi từ AI (Qwen2-VL)")
            
    btn_submit.click(fn=predict_vqa, inputs=[img_input, q_input], outputs=txt_output)

if __name__ == "__main__":
    demo.launch(share=True)

