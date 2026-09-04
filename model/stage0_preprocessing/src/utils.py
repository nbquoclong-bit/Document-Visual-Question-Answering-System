import cv2
import numpy as np
from typing import Union, Optional, List, Tuple
import yaml


from PIL import Image
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    try:
        with Image.open(str(path)) as pil_img:
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    try:
        suffix = Path(path).suffix or ".jpg"
        is_success, buffer = cv2.imencode(suffix, image)
        if is_success:
            with open(path, "wb") as f:
                f.write(buffer)
            return
    except Exception:
        pass
    cv2.imwrite(path, image)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
