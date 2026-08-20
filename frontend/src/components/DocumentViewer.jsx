import { useState } from "react";
import { ImageOff } from "lucide-react";

/**
 * Hiển thị ảnh gốc, vẽ đè khung đỏ (evidence bbox) lên đúng vị trí.
 *
 * bbox từ backend là toạ độ PIXEL trên ảnh gốc [x1,y1,x2,y2]. Cần biết
 * kích thước gốc (naturalWidth/naturalHeight) để quy đổi sang PHẦN TRĂM —
 * nhờ đó khung luôn đúng vị trí dù ảnh hiển thị to/nhỏ thế nào (responsive).
 */
export default function DocumentViewer({ imageUrl, activeBbox, ocrTokens = [] }) {
  const [naturalSize, setNaturalSize] = useState(null);

  if (!imageUrl) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center text-sm text-ink-soft">
        <ImageOff size={28} strokeWidth={1.5} />
        Chưa có ảnh nào được tải lên. Hãy upload một hoá đơn để bắt đầu.
      </div>
    );
  }

  function toOverlayStyle(bbox) {
    if (!bbox || !naturalSize) return null;
    const normalized = Math.max(...bbox) <= 1;
    const scaleX = normalized ? 100 : 100 / naturalSize.w;
    const scaleY = normalized ? 100 : 100 / naturalSize.h;
    return {
      left: `${bbox[0] * scaleX}%`,
      top: `${bbox[1] * scaleY}%`,
      width: `${(bbox[2] - bbox[0]) * scaleX}%`,
      height: `${(bbox[3] - bbox[1]) * scaleY}%`,
    };
  }

  const overlayStyle = toOverlayStyle(activeBbox);

  return (
    <div className="relative inline-block max-w-full leading-none">
      <img
        src={imageUrl}
        alt="Ảnh hoá đơn đã tải lên"
        className="block w-full rounded-sm"
        onLoad={(e) =>
          setNaturalSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
        }
      />
      {naturalSize &&
        ocrTokens.map((token, index) => (
          <div
            key={`${token.text}-${index}`}
            className="pointer-events-none absolute border border-ledger/25"
            style={toOverlayStyle(token.bbox)}
            title={`${token.text} (${Math.round(token.confidence * 100)}%)`}
          />
        ))}
      {overlayStyle && (
        // key theo toạ độ để React re-mount -> animation "đóng dấu" chạy lại mỗi lần đổi bằng chứng
        <div
          key={activeBbox.join(",")}
          className="animate-stamp-in absolute rounded-sm border-2 border-stamp shadow-[0_0_0_3px_rgba(168,63,50,0.15)] before:absolute before:-inset-1.5 before:rounded-sm before:border before:border-dashed before:border-stamp before:opacity-50"
          style={overlayStyle}
        />
      )}
    </div>
  );
}
