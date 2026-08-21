from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import cv2
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def is_pdf(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def render_pdf_pages(pdf_path: Union[str, Path], dpi: int = 300, max_pages: int = 10) -> List[Tuple[int, np.ndarray]]:
    import fitz

    doc = fitz.open(pdf_path)
    results = []
    for page_idx in range(min(len(doc), max_pages)):
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        results.append((page_idx + 1, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)))
    doc.close()
    return results


def load_image(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def load_input(path: Union[str, Path]) -> Tuple[str, Union[np.ndarray, List[Tuple[int, np.ndarray]]]]:
    if is_pdf(path):
        return "pdf", render_pdf_pages(path)
    return "image", load_image(path)
