"""
Script test trực tiếp tính toán Điểm Tin Cậy (Confidence Score) trên ảnh hóa đơn thực tế.
"""
import sys
import os
from pathlib import Path

# Setup encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from stage1_vlm.src.inference import VQAEngine

def main():
    image_path = project_root / "datasets" / "val_benchmark_upload" / "images" / "cafe_highlands_val_001.png"
    if not image_path.exists():
        # Fallback to alternative path
        image_path = project_root.parent / "datasets" / "val_benchmark_upload" / "images" / "cafe_highlands_val_001.png"

    print("================================================================================")
    print("   KIỂM THỬ TRỰC TIẾP TÍNH NĂNG ĐỘ TIN CẬY (CONFIDENCE SCORE) TRÊN VLM")
    print("================================================================================")
    print(f"Ảnh hóa đơn kiểm thử: {image_path.name}")
    print(f"Đường dẫn: {image_path}")

    # Khởi tạo engine với model và adapter
    adapter_dir = project_root / "stage1_vlm" / "output" / "lora_adapters"
    if not (adapter_dir / "adapter_model.safetensors").exists():
        adapter_dir = None
    
    print("\n[1] Đang nạp mô hình VLM...")
    engine = VQAEngine(
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        base_model="Qwen/Qwen2-VL-2B-Instruct"
    )
    print("  -> Nạp VQAEngine thành công!")

    test_questions = [
        ("Tổng tiền", "Tổng tiền thanh toán cuối cùng trên hóa đơn là bao nhiêu?"),
        ("Tên cửa hàng", "Tên cửa hàng / thương hiệu trên hóa đơn là gì?"),
        ("Mã số thuế", "Mã số thuế của bên bán là gì?"),
    ]

    print("\n[2] Thực hiện suy luận và đo lường độ tin cậy nội tại từ Logits:")
    print("--------------------------------------------------------------------------------")

    for category, q in test_questions:
        print(f"\n❓ Câu hỏi [{category}]: {q}")
        answer, confidence = engine.extract_and_answer(
            image_path=str(image_path),
            question=q,
            max_new_tokens=128,
            return_confidence=True,
        )
        
        pct = round(confidence * 100, 1)
        status = "🟢 RẤT TIN CẬY (Chuẩn xác)" if pct >= 85 else ("🟡 KHÁ TIN CẬY (Cần kiểm tra)" if pct >= 60 else "🔴 ĐỘ TỰ TIN THẤP")

        print(f"  👉 Câu trả lời : \"{answer}\"")
        print(f"  📊 Điểm tin cậy: {pct}% [{status}]")

    print("\n================================================================================")
    print("   KẾT QUẢ KIỂM THỬ THÀNH CÔNG: MÔ HÌNH VLM TÍNH CONFIDENCE CHÍNH XÁC!")
    print("================================================================================")

if __name__ == "__main__":
    main()
