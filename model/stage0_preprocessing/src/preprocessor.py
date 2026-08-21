import cv2
import numpy as np
from pathlib import Path
from .pdf_router import load_input


class PreprocessingEngine:
    def __init__(self, config_path: str | None = None):
        import yaml
        resolved_config = Path(config_path) if config_path else Path(__file__).with_name("config.py")
        with resolved_config.open('r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.quality_cfg = self.cfg.get('quality', {})
        self.clahe_cfg = self.cfg.get('clahe', {})
        self.sharp_cfg = self.cfg.get('sharpen', {})
        self.persp_cfg = self.cfg.get('perspective', {})
        self.pdf_cfg = self.cfg.get('pdf', {})

    def _rotate(self, image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _crop_perspective(self, image, contour):
        from .perspective import four_point_transform
        contour = contour.reshape(-1, 2)
        if len(contour) < 4:
            return image
        return four_point_transform(image, contour.astype(np.float32),
                                    target_size=self.persp_cfg.get('output_size', 1024))

    def process_image(self, image):
        from .quality_check import QualityAssessor
        assessor = QualityAssessor(
            blur_threshold=self.quality_cfg.get('blur_threshold', 100),
            skew_threshold=self.quality_cfg.get('skew_threshold_degrees', 2),
            bg_ratio_threshold=self.quality_cfg.get('contour_area_ratio', 0.9),
        )
        report = assessor.assess(image)
        meta = {'deskew': False, 'clahe': False, 'sharpen': False,
                'perspective_crop': False}
        out = image.copy()

        if report.needs_deskew:
            out = self._rotate(out, report.skew_angle)
            meta['deskew'] = True

        if report.is_blurry:
            from .clahe_enhancer import apply_clahe, sharpen_image
            out = apply_clahe(out, clip_limit=self.clahe_cfg.get('clip_limit', 2.0),
                              grid_size=self.clahe_cfg.get('tile_grid_size', 8))
            out = sharpen_image(out, strength=self.sharp_cfg.get('strength', 1.5),
                                kernel_size=self.sharp_cfg.get('kernel_size', 3))
            meta['clahe'] = True
            meta['sharpen'] = True

        if not report.has_background_glare:
            _, max_contour = assessor.detect_background_ratio(out)
            if max_contour is not None and len(max_contour) >= 4:
                out = self._crop_perspective(out, max_contour)
                meta['perspective_crop'] = True

        return out, meta

    def process(self, input_path: str):
        kind, data = load_input(input_path)
        if kind == 'pdf':
            results = []
            for page_idx, page_img in data:
                processed, meta = self.process_image(page_img)
                results.append((page_idx, processed, meta))
            return kind, results
        processed, meta = self.process_image(data)
        return kind, (processed, meta)
