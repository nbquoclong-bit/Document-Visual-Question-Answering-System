"""PaddleOCR adapter that exposes normalized OCR tokens to the pipeline."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import List

from app.config import settings


@dataclass
class OCRToken:
    text: str
    bbox: List[float]  # [x0, y0, x1, y1], normalized to [0, 1]
    confidence: float


class OCRRuntimeError(RuntimeError):
    """Raised when the configured OCR runtime is unavailable."""


@lru_cache(maxsize=1)
def _get_ocr_engine():
    """Load PaddleOCR once per worker, avoiding a model reload for every upload."""
    # PaddlePaddle 3.x on Windows ignores PADDLE_HOME for a legacy dataset
    # import and derives its cache from USERPROFILE. Point that process-local
    # location at the writable project runtime directory before importing Paddle.
    os.environ["USERPROFILE"] = str(settings.paddle_home)
    os.environ.setdefault("PADDLE_HOME", str(settings.paddle_home))
    # PP-OCRv3 inference graphs can fail against Paddle 3.x oneDNN on Windows.
    # Disabling it trades a little CPU speed for a stable, portable demo runtime.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRRuntimeError(
            "PaddleOCR chưa được cài. Cài dependencies model trước khi xử lý tài liệu."
        ) from exc

    return PaddleOCR(
        use_angle_cls=settings.ocr_use_angle_classifier,
        lang=settings.ocr_language,
        show_log=False,
        use_gpu=settings.device.startswith("cuda"),
    )


def run_ocr(image_path: str) -> List[OCRToken]:
    """Run PaddleOCR and convert quadrilaterals into normalized, sorted boxes.

    The public contract deliberately uses normalized coordinates. Model-specific
    coordinates (PaddleOCR pixels, LayoutLMv3's 0-1000 scale) stay inside services.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise OCRRuntimeError("Pillow chưa được cài. Cài dependencies model trước khi xử lý tài liệu.") from exc

    with Image.open(Path(image_path)) as image:
        width, height = image.size

    if width <= 0 or height <= 0:
        raise OCRRuntimeError("Không thể xác định kích thước ảnh tài liệu.")

    result = _get_ocr_engine().ocr(image_path, cls=settings.ocr_use_angle_classifier)
    if not result or not result[0]:
        return []

    tokens: List[OCRToken] = []
    for quadrilateral, (text, confidence) in result[0]:
        score = float(confidence)
        if score < settings.ocr_min_confidence or not text.strip():
            continue

        xs = [point[0] for point in quadrilateral]
        ys = [point[1] for point in quadrilateral]
        tokens.append(
            OCRToken(
                text=text.strip(),
                bbox=[
                    max(0.0, min(xs) / width),
                    max(0.0, min(ys) / height),
                    min(1.0, max(xs) / width),
                    min(1.0, max(ys) / height),
                ],
                confidence=score,
            )
        )

    # PaddleOCR does not guarantee reading order for complex invoice layouts.
    return sorted(tokens, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0]))
