"""
Kịch bản kiểm thử đối chứng: Các trường hợp Độ Tin Cậy Cao (Xanh), Trung Bình (Vàng) và Thấp (Đỏ).
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

backend_dir = Path(__file__).resolve().parent.parent / "backend" / "backend-docvqa" / "backend"
sys.path.insert(0, str(backend_dir))

from app.services import vlm_service

def test_cases():
    print("=" * 85)
    print("   BẢNG ĐỐI CHỨNG THỰC NGHIỆM: CÁC TRƯỜNG HỢP ĐỘ TIN CẬY CAO, TRUNG BÌNH & THẤP")
    print("=" * 85)

    cases = [
        # --- CA 1: ĐỘ TIN CẬY CAO (XANH LÁ) ---
        {
            "group": "🟢 ĐỘ TIN CẬY CAO (≥ 85%)",
            "scenario": "Ảnh rõ nét, chữ số sắc cạnh, đúng định dạng kế toán chuẩn",
            "field_key": "tax_code",
            "extracted_value": "0302863720",
            "vlm_conf": 0.96,  # Token probs cao, margin dứt khoát
        },
        {
            "group": "🟢 ĐỘ TIN CẬY CAO (≥ 85%)",
            "scenario": "Tổng tiền có đầy đủ số và phân cách hàng nghìn",
            "field_key": "total_amount",
            "extracted_value": "109.000",
            "vlm_conf": 0.95,
        },
        {
            "group": "🟢 ĐỘ TIN CẬY CAO (≥ 85%)",
            "scenario": "Tên thương hiệu lớn hiển thị rõ ở đầu hóa đơn",
            "field_key": "store_name",
            "extracted_value": "HIGHLANDS COFFEE",
            "vlm_conf": 0.97,
        },

        # --- CA 2: ĐỘ TIN CẬY TRUNG BÌNH (VÀNG CAM) ---
        {
            "group": "🟡 ĐỘ TIN CẬY TRUNG BÌNH (60% - 84%)",
            "scenario": "Chữ in nhiệt bị mờ 1 chữ số (VLM phân vân giữa số 3 và số 8, margin hẹp)",
            "field_key": "total_amount",
            "extracted_value": "138.000",
            "vlm_conf": 0.65,  # Min prob thấp do mờ nét
        },
        {
            "group": "🟡 ĐỘ TIN CẬY TRUNG BÌNH (60% - 84%)",
            "scenario": "Ngày lập hóa đơn bị thiếu năm (chỉ có ngày và tháng)",
            "field_key": "invoice_date",
            "extracted_value": "16/06 14:30",
            "vlm_conf": 0.75,
        },
        {
            "group": "🟡 ĐỘ TIN CẬY TRUNG BÌNH (60% - 84%)",
            "scenario": "Mã số thuế chỉ đọc được 8 số (nghi ngờ bị che khuất hoặc mất nét)",
            "field_key": "tax_code",
            "extracted_value": "03028637",
            "vlm_conf": 0.70,
        },

        # --- CA 3: ĐỘ TIN CẬY THẤP (ĐỎ HỒNG - NGHI NGỜ ẢO GIÁC / THIẾU THÔNG TIN) ---
        {
            "group": "🔴 ĐỘ TIN CẬY THẤP (< 60%)",
            "scenario": "Hỏi thông tin KHÔNG TỒN TẠI trên hóa đơn (Hạn sử dụng)",
            "field_key": "qa_answer",
            "extracted_value": "Không có thông tin về hạn sử dụng trên hóa đơn này",
            "vlm_conf": 0.35,  # Mô hình bất định, logits phân tán
        },
        {
            "group": "🔴 ĐỘ TIN CẬY THẤP (< 60%)",
            "scenario": "Mã số thuế bị ảo giác, đọc nhầm sang ký hiệu chữ cái",
            "field_key": "tax_code",
            "extracted_value": "HD-VAT-01",  # Lẫn chữ, không phải MST
            "vlm_conf": 0.48,
        },
        {
            "group": "🔴 ĐỘ TIN CẬY THẤP (< 60%)",
            "scenario": "Tổng tiền bị đọc sai thành chuỗi văn bản không thể quy đổi",
            "field_key": "total_amount",
            "extracted_value": "Không rõ số tiền",
            "vlm_conf": 0.30,
        },
        {
            "group": "🔴 ĐỘ TIN CẬY THẤP (< 60%)",
            "scenario": "Tên cửa hàng bị mờ hoàn toàn hoặc rách giấy (chỉ đọc được 1 ký tự)",
            "field_key": "store_name",
            "extracted_value": "H",
            "vlm_conf": 0.32,
        },
    ]

    current_group = ""
    for c in cases:
        if c["group"] != current_group:
            current_group = c["group"]
            print(f"\n{current_group}:")
            print("-" * 85)

        fmt_score = vlm_service._calculate_format_confidence(c["field_key"], c["extracted_value"])
        vlm_conf = c["vlm_conf"]

        # Nếu là câu trả lời không có thông tin, vlm_service sẽ tự động giới hạn trần confidence
        if any(w in c["extracted_value"].lower() for w in ["không có", "không rõ"]):
            vlm_conf = min(vlm_conf, 0.38)

        final_conf = round(0.55 * vlm_conf + 0.45 * fmt_score, 2)
        pct = int(final_conf * 100)

        badge = "🟢 RẤT TIN CẬY" if pct >= 85 else ("🟡 CẦN ĐỐI SOÁT" if pct >= 60 else "🔴 ĐỘ TỰ TIN THẤP (CẢNH BÁO)")

        print(f" • Tình huống : {c['scenario']}")
        print(f"   Trường     : {c['field_key']:<14} = \"{c['extracted_value']}\"")
        print(f"   VLM Logits : {c['vlm_conf'] * 100:.0f}%  |  Format Sanity: {fmt_score * 100:.0f}%  ==>  Tổng kết: {pct}% [{badge}]")
        print()

    print("=" * 85)

if __name__ == "__main__":
    test_cases()
