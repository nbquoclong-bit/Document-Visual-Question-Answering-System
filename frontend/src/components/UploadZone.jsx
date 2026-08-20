import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

const ACCEPTED = ".jpg,.jpeg,.png,.pdf";

export default function UploadZone({ onFileSelected, disabled }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  function handleFiles(fileList) {
    const file = fileList?.[0];
    if (file) onFileSelected(file);
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      className={`cursor-pointer rounded-sm border-[1.5px] border-dashed px-5 py-12 text-center transition-colors
        ${dragActive ? "border-ink bg-stamp/5" : "border-line hover:border-ink hover:bg-stamp/5"}`}
    >
      <UploadCloud className="mx-auto mb-3 text-ink-soft" size={30} strokeWidth={1.5} />
      <strong className="mb-1.5 block font-display text-lg font-medium text-ink">
        Kéo thả ảnh hoá đơn vào đây
      </strong>
      <p className="text-sm text-ink-soft">hoặc bấm để chọn file — hỗ trợ JPG, PNG, PDF</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />
    </div>
  );
}
