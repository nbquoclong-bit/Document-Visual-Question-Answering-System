import { useState } from "react";
import {
  uploadDocument,
  processDocument,
  askQuestion,
  getImageUrl,
  getExportUrl,
  getErrorMessage,
} from "./api";
import StatusBadge from "./components/StatusBadge";
import UploadZone from "./components/UploadZone";
import DocumentViewer from "./components/DocumentViewer";
import FieldsLedger from "./components/FieldsLedger";
import QAConsole from "./components/QAConsole";
import { RotateCcw, Sparkles, Loader2, Download, AlertTriangle } from "lucide-react";

export default function App() {
  // Trạng thái tài liệu hiện tại
  const [documentId, setDocumentId] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploaded | processing | processed | failed
  const [fields, setFields] = useState([]);
  const [ocrTokens, setOcrTokens] = useState([]);
  const [qaHistory, setQaHistory] = useState([]);
  const [activeBbox, setActiveBbox] = useState(null);
  const [isPdf, setIsPdf] = useState(false);

  // Trạng thái UI phụ
  const [processing, setProcessing] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);

  const imageUrl = documentId ? getImageUrl(documentId) : null;

  async function handleFileSelected(file) {
    setError(null);
    setActiveBbox(null);
    setFields([]);
    setOcrTokens([]);
    setQaHistory([]);
    try {
      const result = await uploadDocument(file);
      setDocumentId(result.document_id);
      setStatus(result.status);
      setIsPdf((result.original_filename || file.name).toLowerCase().endsWith(".pdf"));
    } catch (err) {
      setError(`Upload thất bại: ${getErrorMessage(err)}`);
    }
  }

  async function handleProcess() {
    if (!documentId) return;
    setProcessing(true);
    setError(null);
    setStatus("processing");
    try {
      const result = await processDocument(documentId);
      setStatus(result.status);
      setFields(result.fields || []);
      setOcrTokens(result.ocr_tokens || []);
      if (result.status === "failed") {
        setError(`Xử lý thất bại: ${result.error_message}`);
      }
    } catch (err) {
      setStatus("failed");
      setError(`Xử lý thất bại: ${getErrorMessage(err)}`);
    } finally {
      setProcessing(false);
    }
  }

  async function handleAsk(question) {
    if (!documentId) return;
    setAsking(true);
    setError(null);
    try {
      const result = await askQuestion(documentId, question);
      setQaHistory((prev) => [...prev, result]);
      if (result.evidence_bbox) setActiveBbox(result.evidence_bbox);
    } catch (err) {
      setError(`Không thể trả lời câu hỏi: ${getErrorMessage(err)}`);
    } finally {
      setAsking(false);
    }
  }

  function handleReset() {
    setDocumentId(null);
    setStatus("idle");
    setFields([]);
    setOcrTokens([]);
    setQaHistory([]);
    setActiveBbox(null);
    setIsPdf(false);
    setError(null);
  }

  const canProcess = status === "uploaded" || status === "failed";
  const canAsk = status === "processed";

  return (
    <div className="mx-auto max-w-[1180px] px-6 pb-16 pt-7">
      <header className="mb-5 flex items-baseline justify-between gap-4 border-b border-line pb-4">
        <div>
          <h1 className="font-display text-[27px] font-semibold tracking-tight text-ink">
            Sổ Hoá Đơn
          </h1>
          <p className="mt-1 text-[13px] text-ink-soft">
            Hỏi-đáp trực quan trên tài liệu — tiền xử lý &amp; trích xuất bằng Qwen2-VL
          </p>
        </div>
        <StatusBadge status={status} />
      </header>

      {error && (
        <div className="mb-3.5 flex items-start gap-2 rounded-sm border border-stamp bg-stamp/10 px-3 py-2.5 text-[13px] text-stamp">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 items-start gap-5 md:grid-cols-[1.15fr_0.85fr]">
        {/* Cột trái: ảnh tài liệu */}
        <section className="rounded-sm border border-line bg-paper-raised shadow-sm">
          <div className="flex items-center justify-between gap-2.5 border-b border-line px-4.5 py-3.5">
            <h2 className="font-display text-[15px] font-semibold uppercase tracking-wide text-ink-soft">
              Tài liệu
            </h2>
          </div>
          <div className="p-4.5">
            {!documentId ? (
              <UploadZone onFileSelected={handleFileSelected} disabled={processing} />
            ) : (
              <DocumentViewer
                imageUrl={imageUrl}
                isPdf={isPdf}
                activeBbox={activeBbox}
                ocrTokens={ocrTokens}
              />
            )}

            <div className="mt-4.5 flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={handleReset}
                disabled={!documentId}
                className="flex items-center gap-1.5 rounded-sm border border-ink px-4 py-2 text-[13px] font-medium text-ink transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <RotateCcw size={14} />
                Tải hoá đơn khác
              </button>

              {canProcess && (
                <button
                  type="button"
                  onClick={handleProcess}
                  disabled={processing}
                  className="flex items-center gap-1.5 rounded-sm border border-stamp bg-stamp px-4 py-2 text-[13px] font-medium text-paper-raised transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {processing ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                  Xử lý tài liệu
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Cột phải: field trích xuất + hỏi đáp */}
        <div className="flex flex-col gap-5">
          <section className="rounded-sm border border-line bg-paper-raised shadow-sm">
            <div className="flex items-center justify-between gap-2.5 border-b border-line px-4.5 py-3.5">
              <h2 className="font-display text-[15px] font-semibold uppercase tracking-wide text-ink-soft">
                Thông tin trích xuất
              </h2>
            </div>
            <div className="p-4.5">
              <FieldsLedger fields={fields} onSelectField={setActiveBbox} />
              {fields.some((field) => field.bbox) && (
                <p className="mt-2 text-xs text-ink-soft">Bấm vào một dòng để xem vị trí trên ảnh.</p>
              )}
            </div>
          </section>

          <section className="rounded-sm border border-line bg-paper-raised shadow-sm">
            <div className="flex items-center justify-between gap-2.5 border-b border-line px-4.5 py-3.5">
              <h2 className="font-display text-[15px] font-semibold uppercase tracking-wide text-ink-soft">
                Hỏi đáp
              </h2>
            </div>
            <div className="p-4.5">
              <QAConsole
                qaHistory={qaHistory}
                onAsk={handleAsk}
                onShowEvidence={setActiveBbox}
                disabled={!canAsk}
                asking={asking}
              />
            </div>
          </section>

          {status === "processed" && (
            <a
              href={getExportUrl(documentId)}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-center gap-1.5 rounded-sm border border-ink px-4 py-2 text-[13px] font-medium text-ink transition-opacity hover:opacity-80"
            >
              <Download size={14} />
              Xuất kết quả JSON
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
