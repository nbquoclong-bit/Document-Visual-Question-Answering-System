import cv2
import numpy as np
from typing import Union, Optional, List, Tuple
import yaml


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
