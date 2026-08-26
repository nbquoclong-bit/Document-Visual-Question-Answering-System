import os
import sys
import torch

try:
    import gradio as gr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
except ImportError:
    import subprocess
    print("📦 [demo_gradio] Đang tự động cài đặt các thư viện cần thiết...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "gradio>=4.0.0", 
        "qwen-vl-utils>=0.0.8", 
        "peft>=0.12.0", 
        "bitsandbytes>=0.43.3", 
        "transformers>=4.45.0", 
        "accelerate>=0.34.0", 
        "-q"
    ])
    import gradio as gr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

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
    import zipfile
    
    # 1. Tự động giải nén nếu có file zip trong /kaggle/working hoặc thư mục model/output
    possible_zips = [
        "/kaggle/working/qwen2_vl_lora_adapters_golden.zip",
        "/kaggle/working/qwen2_vl_lora_adapters.zip",
        os.path.join(base_dir, "output", "qwen2_vl_lora_adapters_golden.zip"),
        os.path.join(base_dir, "output", "qwen2_vl_lora_adapters.zip"),
        os.path.join(base_dir, "qwen2_vl_lora_adapters_golden.zip"),
        os.path.join(base_dir, "qwen2_vl_lora_adapters.zip"),
        os.path.join(project_root, "qwen2_vl_lora_adapters_golden.zip"),
        os.path.join(project_root, "qwen2_vl_lora_adapters.zip")
    ]
    for z in possible_zips:
        if os.path.exists(z):
            extract_target = os.path.join(base_dir, "output", "lora_adapters")
            if not os.path.exists(os.path.join(extract_target, "adapter_config.json")):
                os.makedirs(extract_target, exist_ok=True)
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(extract_target)
                    print(f"📦 Đã tự động giải nén trọng số LoRA từ {z} vào {extract_target}!")
                except Exception as err:
                    print(f"⚠️ Lỗi giải nén {z}: {err}")

    # 2. Tìm kiếm đệ quy thư mục chứa 'adapter_config.json'
    search_roots = [
        os.path.join(base_dir, "output"),
        os.path.join(base_dir, "stage1_vlm", "output"),
        "/kaggle/working",
        base_dir
    ]
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "adapter_config.json" in filenames:
                    return dirpath
    return None

print("🚀 Đang khởi tạo VQA Engine (Qwen2-VL)...")
adapter_path = find_adapter_dir()
if adapter_path:
    print(f"✅ Đã tìm thấy và nạp trọng số LoRA tại: {adapter_path}")
else:
    print("ℹ️ Chạy trực tiếp Base Model Qwen2-VL-2B (Không tìm thấy LoRA adapter cục bộ).")

engine = VQAEngine(adapter_dir=adapter_path)

def predict_vqa(image, question):
    if image is None:
        return "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc tài liệu."
    
    if not question or question.strip() == "":
        question = "Trích xuất các thông tin quan trọng: Tên đơn vị bán, Mã số thuế, Ngày lập, Tổng tiền thanh toán."
    
    try:
        res = engine.extract_and_answer(image, question)
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
                    ["Tổng tiền thanh toán trên hóa đơn là bao nhiêu?"],
                    ["Tên cửa hàng / bên bán trên hóa đơn là gì?"],
                    ["Địa chỉ cửa hàng / bên bán là ở đâu?"],
                    ["Ngày giờ lập hóa đơn là khi nào?"],
                    ["Thuế VAT / Tiền thuế trên hóa đơn là bao nhiêu?"],
                    ["Các sản phẩm / món hàng được mua trên hóa đơn là gì?"],
                    ["Trích xuất toàn bộ thông tin hóa đơn dưới dạng JSON."]
                ],
                inputs=[q_input],
                label="💡 Gợi ý câu hỏi chuẩn hóa đơn"
            )
        
        with gr.Column():
            txt_output = gr.Textbox(lines=12, label="Kết quả phản hồi từ AI (Qwen2-VL)")
            
    btn_submit.click(fn=predict_vqa, inputs=[img_input, q_input], outputs=txt_output)

if __name__ == "__main__":
    demo.launch(share=True)

