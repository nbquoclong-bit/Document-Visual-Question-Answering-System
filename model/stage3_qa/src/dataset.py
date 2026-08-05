"""
dataset.py - Dataset class for Vietnamese accounting QA instruction tuning.

Wraps Stage-2 KIE output (JSON extracted data) alongside Vietnamese
questions and answers into a PyTorch Dataset tokenized with the
Qwen2.5 chat template, max sequence length 512.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


DEFAULT_INSTRUCTION = "Kiem tra tinh toan tren hoa don nay co dung khong?"


class Stage3QADataset(Dataset):
    """
    PyTorch Dataset for Qwen2.5-1.5B instruction tuning on Vietnamese invoices.

    Each sample dict contains:
        - instruction: Vietnamese question string
        - input: JSON string of extracted invoice fields from Stage 2 KIE
        - output: Accounting audit response string
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        default_instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.default_instruction = default_instruction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]

        instruction = record.get('instruction', self.default_instruction)
        input_data = record.get('input', '{}')
        output = record.get('output', '')

        if isinstance(input_data, dict):
            input_data = json.dumps(input_data, ensure_ascii=False, indent=2)

        messages = [
            {
                "role": "system",
                "content": "Ban la mot chuyen gia ke toan kiem toan Viet Nam.",
            },
            {
                "role": "user",
                "content": f"{instruction}\n\nDu lieu hoa don:\n{input_data}",
            },
            {"role": "assistant", "content": output},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors='pt',
        )

        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_jsonl_dataset(path: str) -> List[Dict]:
    """Load QA records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[dataset] Loaded {len(records)} records from {path}")
    return records


def load_json_dataset(path: str) -> List[Dict]:
    """Load QA records from a JSON file (array of dicts)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    print(f"[dataset] Loaded {len(data)} records from {path}")
    return data
