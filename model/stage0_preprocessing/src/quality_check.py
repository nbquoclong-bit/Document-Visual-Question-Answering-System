import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityReport:
    is_blurry: bool = False
    blur_score: float = 0.0
    skew_angle: float = 0.0
    needs_deskew: bool = False
    has_background_glare: bool = False
    background_ratio: float = 0.0


class QualityAssessor:
    def __init__(self, blur_threshold: float = 100.0, skew_threshold: float = 2.0, bg_ratio_threshold: float = 0.9):
        self.blur_threshold = blur_threshold
        self.skew_threshold = skew_threshold
        self.bg_ratio_threshold = bg_ratio_threshold

    def laplacian_variance(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect_skew_angle(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None or len(lines) == 0:
            return 0.0
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta - np.pi / 2)
            angles.append(angle)
        angle = float(np.median(angles))
        return max(-45.0, min(45.0, angle))

    def detect_background_ratio(self, image: np.ndarray) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None
        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        total_area = image.shape[0] * image.shape[1]
        ratio = max_area / total_area if total_area > 0 else 0.0
        return ratio, max_contour

    def assess(self, image: np.ndarray) -> QualityReport:
        blur_score = self.laplacian_variance(image)
        skew_angle = self.detect_skew_angle(image)
        bg_ratio, _ = self.detect_background_ratio(image)
        return QualityReport(
            is_blurry=(blur_score < self.blur_threshold),
            blur_score=blur_score,
            skew_angle=skew_angle,
            needs_deskew=(abs(skew_angle) > self.skew_threshold),
            has_background_glare=(bg_ratio > self.bg_ratio_threshold),
            background_ratio=bg_ratio,
        )
