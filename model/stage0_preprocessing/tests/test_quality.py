import unittest
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_check import QualityAssessor
from perspective import four_point_transform


class TestQualityAssessor(unittest.TestCase):
    def setUp(self):
        # Synthetic images have much lower Laplacian variance than real photos;
        # use a deterministic threshold that still separates sharp vs blurred.
        self.assessor = QualityAssessor(blur_threshold=10.0)
        self.sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(self.sharp_img, (50, 50), (150, 150), 255, -1)
        self.blur_img = cv2.GaussianBlur(self.sharp_img, (15, 15), 0)

    def test_sharp_has_high_blur_score(self):
        report = self.assessor.assess(self.sharp_img)
        self.assertFalse(report.is_blurry)

    def test_blur_has_low_blur_score(self):
        report = self.assessor.assess(self.blur_img)
        self.assertTrue(report.is_blurry)

    def test_perspective_transform_preserves_portrait_aspect_ratio(self):
        image = np.zeros((800, 400, 3), dtype=np.uint8)
        points = np.array([[0, 0], [399, 0], [399, 799], [0, 799]], dtype=np.float32)
        transformed = four_point_transform(image, points, target_size=1024)
        self.assertEqual(transformed.shape[0], 1024)
        self.assertAlmostEqual(transformed.shape[1] / transformed.shape[0], 0.5, places=2)


if __name__ == '__main__':
    unittest.main()
