from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from paddleocr import PaddleOCR


ocr_vi = None
ocr_en = None


def get_ocr_engine(lang: str = 'vi', use_gpu: bool = False):
    global ocr_vi, ocr_en
    if lang == 'vi':
        if ocr_vi is None:
            ocr_vi = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False, use_gpu=use_gpu)
        return ocr_vi
    else:
        if ocr_en is None:
            ocr_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=use_gpu)
        return ocr_en


def ocr_image(image: np.ndarray, lang: str = 'vi', use_gpu: bool = False) -> List[Dict]:
    engine = get_ocr_engine(lang=lang, use_gpu=use_gpu)
    result = engine.ocr(image, cls=True)
    if not result or not result[0]:
        return []
    boxes_texts = []
    for line in result[0]:
        bbox, (text, conf) = line
        boxes_texts.append({
            'text': text,
            'confidence': round(float(conf), 4),
            'bbox': [[int(x), int(y)] for x, y in bbox],
        })
    return boxes_texts


def ocr_file(input_path: str, lang: str = 'vi', use_gpu: bool = False):
    from stage0_preprocessing.src.preprocessor import PreprocessingEngine
    engine = PreprocessingEngine()
    _, data = engine.process(input_path)
    if isinstance(data, list):
        results = []
        for page_idx, page_img, meta in data:
            texts = ocr_image(page_img, lang=lang, use_gpu=use_gpu)
            results.append({'page': page_idx, 'preprocessing': meta, 'texts': texts})
        return results
    image, meta = data
    texts = ocr_image(image, lang=lang, use_gpu=use_gpu)
    return [{'preprocessing': meta, 'texts': texts}]
