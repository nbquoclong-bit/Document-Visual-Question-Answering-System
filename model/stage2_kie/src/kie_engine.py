import json
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image


LABEL_LIST = ['O', 'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'B-TAX_CODE', 'I-TAX_CODE', 'B-DATE', 'I-DATE', 'B-TOTAL_AMOUNT', 'I-TOTAL_AMOUNT']
ENTITY_TYPES = {
    'INVOICE_NUMBER': {'start': 'B-INVOICE_NUMBER', 'continue': 'I-INVOICE_NUMBER'},
    'TAX_CODE': {'start': 'B-TAX_CODE', 'continue': 'I-TAX_CODE'},
    'DATE': {'start': 'B-DATE', 'continue': 'I-DATE'},
    'TOTAL_AMOUNT': {'start': 'B-TOTAL_AMOUNT', 'continue': 'I-TOTAL_AMOUNT'},
}


class KIEConfig:
    def __init__(self, model_dir: str, max_length: int = 512):
        self.model_dir = model_dir
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


def decode_bio_tags(tokens, labels, bboxes, confidence_threshold=0.0):
    entities = {}
    current_type = None
    current_text = []
    for tok, label, bbox in zip(tokens, labels, bboxes):
        tag = LABEL_LIST[label] if label < len(LABEL_LIST) else 'O'
        if tag == 'O':
            if current_type and current_text:
                ent_key = current_type.lower()
                entities.setdefault(ent_key, {'value': '', 'confidence': 0.0, 'bbox': []})
                entities[ent_key]['value'] = ' '.join(current_text)
                entities[ent_key]['bbox'] = bbox
                current_type = None
                current_text = []
            continue
        prefix, etype = tag.split('-', 1)
        if prefix == 'B' or etype != current_type:
            if current_type and current_text:
                ent_key = current_type.lower()
                entities.setdefault(ent_key, {'value': '', 'confidence': 0.0, 'bbox': []})
                entities[ent_key]['value'] = ' '.join(current_text)
                entities[ent_key]['bbox'] = bbox
            current_type = etype
            current_text = [tok]
        else:
            current_text.append(tok)
    if current_type and current_text:
        ent_key = current_type.lower()
        entities.setdefault(ent_key, {'value': '', 'confidence': 0.0, 'bbox': []})
        entities[ent_key]['value'] = ' '.join(current_text)
        entities[ent_key]['bbox'] = bboxes[-1]
    return entities


def predict(config: KIEConfig, image_path: str, ocr_words: List[str], ocr_boxes: List[List[int]]) -> Dict[str, Any]:
    image = Image.open(image_path).convert('RGB')
    encoding = config.processor(image, ocr_words, boxes=ocr_boxes, truncation=True, padding='max_length', max_length=config.max_length, return_tensors='pt')
    inputs = {k: v.to(config.device) for k, v in encoding.items()}
    with torch.no_grad():
        logits = config.model(**inputs).logits
    preds = logits.argmax(-1).cpu().numpy()[0]
    tokens = config.processor.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    attention_mask = encoding.attention_mask[0].cpu().numpy()
    filtered_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1]
    filtered_preds = [p for p, m in zip(preds, attention_mask) if m == 1]
    filtered_boxes = [ocr_boxes[i] if i < len(ocr_boxes) else [0, 0, 0, 0] for i, m in enumerate(attention_mask) if m == 1]
    entities = decode_bio_tags(filtered_tokens, filtered_preds, filtered_boxes)
    return entities
