import os
import sys
import time
import torch
from PIL import Image

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
        "transformers>=4.46.2", 
        "accelerate>=0.34.2", 
        "-q"
    ])
    import gradio as gr
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from stage1_vlm.src.visual_grounding import highlight_prediction_on_image

try:
    from stage1_vlm.src.inference import VQAEngine
except ImportError:
    from model.stage1_vlm.src.inference import VQAEngine

def find_adapter_dir():
    import zipfile
    possible_zips = [
        os.path.join(base_dir, "output", "qwen2_vl_lora_adapters_golden.zip"),
        os.path.join(base_dir, "stage1_vlm", "output", "lora_adapters"),
        os.path.join(base_dir, "output", "lora_adapters")
    ]
    for p in possible_zips:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
        if p.endswith(".zip") and os.path.exists(p):
            target = os.path.join(base_dir, "output", "lora_adapters")
            try:
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(target)
                return target
            except Exception:
                pass
    return None

print("🚀 Đang khởi tạo VQA Engine (Qwen2-VL)...")
adapter_path = find_adapter_dir()
if adapter_path:
    print(f"✅ Đã tìm thấy và nạp trọng số LoRA tại: {adapter_path}")
else:
    print("ℹ️ Chạy trực tiếp Base Model Qwen2-VL-2B (Không tìm thấy LoRA adapter cục bộ).")

engine = VQAEngine(adapter_dir=adapter_path)

def predict_vqa_with_bounding_box(image, question):
    if image is None:
        return None, "⚠️ Vui lòng tải lên ảnh hóa đơn hoặc tài liệu.", "0.00s"
    
    if not question or question.strip() == "":
        question = "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"
    
    t0 = time.time()
    try:
        ans = engine.extract_and_answer(image, question)
    except Exception as e:
        ans = f"❌ Lỗi khi xử lý: {str(e)}"
    lat = time.time() - t0
    
    # Xác định loại trường để vẽ Bounding Box màu tương ứng
    q_lower = question.lower()
    if "tổng tiền" in q_lower or "thanh toán" in q_lower or "tiền" in q_lower:
        f_type = "TOTAL_COST"
    elif "tên" in q_lower or "quán" in q_lower or "công ty" in q_lower or "bên bán" in q_lower:
        f_type = "SELLER"
    elif "ngày" in q_lower or "giờ" in q_lower or "thời gian" in q_lower:
        f_type = "TIMESTAMP"
    elif "danh sách" in q_lower or "món" in q_lower or "dịch vụ" in q_lower:
        f_type = "ITEMS_LIST"
    elif "địa chỉ" in q_lower:
        f_type = "ADDRESS"
    else:
        f_type = "DEFAULT"
        
    annotated_img = highlight_prediction_on_image(image, ans, field_type=f_type)
    return annotated_img, ans, f"{lat:.2f} giây"

with gr.Blocks(title="Document VQA System - Qwen2-VL with Bounding Box") as demo:
    gr.Markdown("# 🧾 Hệ Thống Document VQA Hóa Đơn & Bounding Box Minh Chứng")
    gr.Markdown("**Bóc tách thông tin hóa đơn tiếng Việt** và **tự động vẽ Bounding Box** xác thực trực quan lên ảnh.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="1. Tải lên ảnh Hóa đơn / Chứng từ")
            q_input = gr.Textbox(
                lines=2, 
                placeholder="Nhập câu hỏi (VD: Tổng tiền là bao nhiêu? Danh sách dịch vụ? Tên bên bán? Phí đậu xe?)",
                value="Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?",
                label="2. Câu hỏi cần bóc tách"
            )
            btn_submit = gr.Button("🔍 Bóc Tách & Vẽ Bounding Box", variant="primary")
            
            gr.Markdown("### 💡 Gợi ý câu hỏi chuẩn nghiệp vụ Kế toán:")
            gr.Examples(
                examples=[
                    ["Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"],
                    ["Tên đơn vị / người bán hàng trên hóa đơn là gì?"],
                    ["Danh sách các mặt hàng / dịch vụ được mua trên hóa đơn gồm những gì?"],
                    ["Địa chỉ của đơn vị bán hàng là ở đâu?"],
                    ["Ngày giờ lập hóa đơn là khi nào?"],
                    ["Mã số thuế của bên bán là gì?"]
                ],
                inputs=[q_input]
            )
        
        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="3. Ảnh đối soát kèm Bounding Box (Visual Grounding)")
            txt_output = gr.Textbox(lines=4, label="4. Kết quả bóc tách giá trị thực thể (Entity Value)")
            lat_output = gr.Textbox(label="Độ trễ phản hồi (Latency)")
            
    btn_submit.click(
        fn=predict_vqa_with_bounding_box, 
        inputs=[img_input, q_input], 
        outputs=[img_output, txt_output, lat_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
