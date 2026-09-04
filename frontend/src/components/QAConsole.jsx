import { useState } from "react";
import { Send, Loader2, MessageSquareText, ShieldCheck } from "lucide-react";

export default function QAConsole({ qaHistory, onAsk, disabled, asking }) {
  const [question, setQuestion] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || disabled || asking) return;
    onAsk(trimmed);
    setQuestion("");
  }

  return (
    <div>
      <div className="thin-scrollbar mb-3.5 flex max-h-80 flex-col gap-3 overflow-y-auto">
        {(!qaHistory || qaHistory.length === 0) && (
          <div className="flex items-center gap-2 py-2.5 text-sm text-ink-soft">
            <MessageSquareText size={16} strokeWidth={1.5} />
            Chưa có câu hỏi nào. Hãy hỏi về nội dung hoá đơn bên dưới.
          </div>
        )}
        {qaHistory?.map((entry, idx) => {
          const conf = entry.confidence != null ? Math.round(entry.confidence * 100) : null;
          let badgeStyle = "bg-emerald-50 text-emerald-800 border-emerald-300";
          let labelText = "Rất tin cậy";

          if (conf != null) {
            if (conf < 60) {
              badgeStyle = "bg-rose-50 text-rose-800 border-rose-300";
              labelText = "Cần đối soát";
            } else if (conf < 85) {
              badgeStyle = "bg-amber-50 text-amber-800 border-amber-300";
              labelText = "Khá tin cậy";
            }
          }

          return (
            <div key={idx} className="text-[13.5px]">
              <p className="mb-1 font-mono text-ink-soft before:mr-1 before:text-stamp before:content-['›']">
                {entry.question}
              </p>
              <div className="rounded-r-sm border-l-2 border-ledger bg-paper px-2.5 py-2 text-ink">
                <p>{entry.answer}</p>
                {conf != null && (
                  <div className="mt-2 flex items-center gap-2 border-t border-dotted border-line/60 pt-1.5 text-xs">
                    <span className="text-ink-soft">Độ tin cậy VLM:</span>
                    <span
                      className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-[11px] font-semibold ${badgeStyle}`}
                      title={`Độ tin cậy toán học: ${conf}% (${labelText})`}
                    >
                      <ShieldCheck size={11} className="shrink-0" />
                      {conf}%
                      <span className="text-[10px] font-normal text-ink-soft">
                        • {labelText}
                      </span>
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          placeholder={disabled ? "Cần xử lý tài liệu trước khi hỏi" : "Vd: Tổng tiền là bao nhiêu?"}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={disabled || asking}
          className="flex-1 rounded-sm border border-line bg-paper px-3 py-2 text-[13.5px] text-ink placeholder:text-ink-soft/70 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || asking || !question.trim()}
          className="flex items-center gap-1.5 rounded-sm border border-ink bg-ink px-4 py-2 text-[13px] font-medium text-paper-raised transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {asking ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          Hỏi
        </button>
      </form>
    </div>
  );
}
