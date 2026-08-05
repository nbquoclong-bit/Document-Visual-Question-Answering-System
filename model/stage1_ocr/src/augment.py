import cv2
import numpy as np
import random
from typing import Tuple


def random_rotation(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def add_gaussian_noise(image: np.ndarray, std: int = 15) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def random_blur(image: np.ndarray, prob: float = 0.3) -> np.ndarray:
    if random.random() < prob:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(image, (k, k), 0)
    return image


def augment(image: np.ndarray) -> np.ndarray:
    image = random_rotation(image)
    image = add_gaussian_noise(image)
    image = random_blur(image)
    return image
