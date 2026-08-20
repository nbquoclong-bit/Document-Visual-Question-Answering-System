import { Receipt } from "lucide-react";

const KEY_LABELS = {
  store_name: "Tên cửa hàng",
  invoice_date: "Ngày hoá đơn",
  total_amount: "Tổng cộng",
};

export default function FieldsLedger({ fields, onSelectField }) {
  if (!fields || fields.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-6 text-center text-sm text-ink-soft">
        <Receipt size={22} strokeWidth={1.5} />
        Chưa có trường thông tin nào được trích xuất.
      </div>
    );
  }

  return (
    <div>
      {fields.map((field, idx) => (
        <button
          key={`${field.key}-${idx}`}
          onClick={() => field.bbox && onSelectField(field.bbox)}
          disabled={!field.bbox}
          className="group flex w-full items-baseline gap-2 border-b border-dotted border-line py-2.5 text-left last:border-b-0 disabled:cursor-default"
        >
          <span className="whitespace-nowrap text-xs uppercase tracking-wide text-ink-soft">
            {KEY_LABELS[field.key] || field.key}
          </span>
          <span className="-translate-y-0.5 flex-1 border-b border-dotted border-line" />
          <span className="whitespace-nowrap font-mono text-[13.5px] font-medium text-ink group-enabled:group-hover:text-stamp">
            {field.value}
          </span>
        </button>
      ))}
    </div>
  );
}
