import { Receipt, ShieldCheck } from "lucide-react";

const KEY_LABELS = {
  store_name: "Tên cửa hàng",
  invoice_number: "Số hóa đơn",
  tax_code: "Mã số thuế",
  invoice_date: "Ngày hoá đơn",
  total_amount: "Tổng cộng",
  address: "Địa chỉ",
  items: "Sản phẩm",
  vlm_response: "Phản hồi VLM",
};

export default function FieldsLedger({ fields }) {
  if (!fields || fields.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-6 text-center text-sm text-ink-soft">
        <Receipt size={22} strokeWidth={1.5} />
        Chưa có trường thông tin nào được trích xuất.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5">
      {fields.map((field, idx) => {
        const conf = field.confidence != null ? Math.round(field.confidence * 100) : null;
        let badgeStyle = "bg-emerald-50 text-emerald-800 border-emerald-300";
        let statusText = "Rất tin cậy";

        if (conf != null) {
          if (conf < 60) {
            badgeStyle = "bg-rose-50 text-rose-800 border-rose-300";
            statusText = "Độ tin cậy thấp";
          } else if (conf < 85) {
            badgeStyle = "bg-amber-50 text-amber-800 border-amber-300";
            statusText = "Khá tin cậy";
          }
        }

        return (
          <div
            key={`${field.key}-${idx}`}
            className="flex flex-col gap-1 rounded-sm border border-line/60 bg-paper/40 p-2.5 transition-colors hover:bg-paper/70"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-ink-soft">
                {KEY_LABELS[field.key] || field.key}
              </span>

              {conf != null && (
                <span
                  className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-[11px] font-semibold ${badgeStyle}`}
                  title={`Độ tin cậy mô hình: ${conf}% (${statusText})`}
                >
                  <ShieldCheck size={11} className="shrink-0" />
                  {conf}%
                  <span className="text-[10px] font-normal text-ink-soft">
                    {conf >= 85 ? "• Chuẩn xác" : conf >= 60 ? "• Đối soát" : "• Xem lại"}
                  </span>
                </span>
              )}
            </div>

            <div className="break-words font-mono text-[13.5px] font-medium leading-5 text-ink">
              {field.value}
            </div>
          </div>
        );
      })}

      <div className="mt-2 flex items-center justify-between border-t border-dotted border-line pt-2 text-[11.5px] text-ink-soft">
        <span className="flex items-center gap-1">
          <ShieldCheck size={13} className="text-ledger" />
          Độ tin cậy End-to-End Qwen2.5-VL (Logits &amp; Format)
        </span>
        <span className="font-mono text-[10.5px]">≥85% Chuẩn xác</span>
      </div>
    </div>
  );
}
