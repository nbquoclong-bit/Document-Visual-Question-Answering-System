import json
from pathlib import Path
from stage0_preprocessing.src.preprocessor import PreprocessingEngine
from stage1_ocr.src.ocr_engine import ocr_image
from stage2_kie.src.kie_engine import predict, KIEConfig


def run_pipeline(input_path: str, kie_model_dir: str, lang: str = "vi") -> dict:
    engine = PreprocessingEngine()
    _, data = engine.process(input_path)
    if isinstance(data, list):
        image = data[0][1]
    else:
        image = data[0]
    import numpy as np
    import cv2
    _, buf = cv2.imencode(".jpg", image)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(buf)
        tmp_path = tmp.name
    ocr_results = ocr_image(image, lang=lang)
    words = [r["text"] for r in ocr_results]
    boxes = [r["bbox"][0] + r["bbox"][2] for r in ocr_results]
    kie_cfg = KIEConfig(model_dir=kie_model_dir)
    entities = predict(kie_cfg, tmp_path, words, boxes)
    Path(tmp_path).unlink()
    return {"status": "success", "ocr_words": len(words), "entities": entities}


if __name__ == "__main__":
    result = run_pipeline("sample_invoice.jpg", "stage2_kie/output/best_model")
    print(json.dumps(result, ensure_ascii=False, indent=2))
