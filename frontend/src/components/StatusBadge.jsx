import { Circle, Loader2, CheckCircle2, XCircle, FileQuestion } from "lucide-react";

const CONFIG = {
  idle: { label: "Chưa có tài liệu", icon: FileQuestion, className: "text-ink-soft border-ink-soft/40 bg-transparent" },
  uploaded: { label: "Đã tải lên", icon: Circle, className: "text-warn border-warn bg-warn/10" },
  processing: { label: "Đang xử lý", icon: Loader2, className: "text-warn border-warn bg-warn/10", spin: true },
  processed: { label: "Đã xử lý", icon: CheckCircle2, className: "text-ledger border-ledger bg-ledger/10" },
  failed: { label: "Xử lý lỗi", icon: XCircle, className: "text-stamp border-stamp bg-stamp/10" },
};

export default function StatusBadge({ status = "idle" }) {
  const cfg = CONFIG[status] || CONFIG.idle;
  const Icon = cfg.icon;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-[11px] font-medium uppercase tracking-wide whitespace-nowrap ${cfg.className}`}
    >
      <Icon size={13} className={cfg.spin ? "animate-spin" : ""} />
      {cfg.label}
    </span>
  );
}
