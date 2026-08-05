import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, target_size=1024):
    rect = order_points(pts)
    width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    aspect = width / height if height > 0 else 1.0
    dst = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size * aspect - 1],
        [0, target_size * aspect - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (int(target_size * aspect), target_size))
