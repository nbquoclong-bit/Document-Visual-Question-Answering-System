import unittest
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_check import QualityAssessor


class TestQualityAssessor(unittest.TestCase):
    def setUp(self):
        self.assessor = QualityAssessor()
        self.sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(self.sharp_img, (50, 50), (150, 150), 255, -1)
        self.blur_img = cv2.GaussianBlur(self.sharp_img, (15, 15), 0)

    def test_sharp_has_high_blur_score(self):
        report = self.assessor.assess(self.sharp_img)
        self.assertFalse(report.is_blurry)

    def test_blur_has_low_blur_score(self):
        report = self.assessor.assess(self.blur_img)
        self.assertTrue(report.is_blurry)


if __name__ == '__main__':
    unittest.main()
