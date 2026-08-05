from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
from PIL import Image


class InvoiceKIEConfig:
    def __init__(self, image_dir: str = None, processor_name: str = 'microsoft/layoutlmv3-base', max_length: int = 512):
        self.image_dir = image_dir
        self.processor_name = processor_name
        self.max_length = max_length
        self.processor = LayoutLMv3Processor.from_pretrained(processor_name, apply_ocr=False)


class InvoiceKIE(Dataset):
    def __init__(self, records: list, config: InvoiceKIEConfig):
        self.records = records
        self.config = config
        self.processor = config.processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec['image_path']).convert('RGB')
        words = rec['words']
        boxes = rec['bboxes']
        word_labels = rec['labels']
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([x[k] for x in batch])
    return out
