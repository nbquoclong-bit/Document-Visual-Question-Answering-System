import { useState } from "react";
import { Send, Loader2, MessageSquareText, ScanSearch } from "lucide-react";

export default function QAConsole({ qaHistory, onAsk, onShowEvidence, disabled, asking }) {
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
        {qaHistory?.map((entry, idx) => (
          <div key={idx} className="text-[13.5px]">
            <p className="mb-1 font-mono text-ink-soft before:mr-1 before:text-stamp before:content-['›']">
              {entry.question}
            </p>
            <p className="rounded-r-sm border-l-2 border-ledger bg-paper px-2.5 py-2 text-ink">
              {entry.answer}
              {entry.evidence_bbox && (
                <button
                  type="button"
                  onClick={() => onShowEvidence(entry.evidence_bbox)}
                  className="ml-2 inline-flex items-center gap-1 text-[11px] text-stamp underline"
                >
                  <ScanSearch size={11} />
                  xem bằng chứng
                </button>
              )}
              {!entry.evidence_bbox && (
                <span className="ml-2 text-[11px] text-ink-soft">
                  VLM chưa trả về toạ độ bằng chứng.
                </span>
              )}
            </p>
          </div>
        ))}
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
