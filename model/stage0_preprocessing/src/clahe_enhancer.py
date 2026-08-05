import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, grid_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image, strength=1.5, kernel_size=3):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
    kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel
    return cv2.filter2D(image, -1, kernel)


def denoise(image, strength=10):
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
