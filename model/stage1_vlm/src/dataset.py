"""
dataset.py - Dataset class for Multimodal Qwen2-VL VQA fine-tuning.
"""

import json
from typing import List, Dict
from torch.utils.data import Dataset

DEFAULT_INSTRUCTION = "Trích xuất thông tin hóa đơn và kiểm tra tính toán."

class VQADataset(Dataset):
    """
    PyTorch Dataset for Qwen2-VL-2B instruction tuning.
    Instead of processing tokens here, it yields the raw dict format so that
    a Custom Data Collator can dynamically pad the batch.
    """

    def __init__(
        self,
        records: List[Dict],
        default_instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        self.records = records
        self.default_instruction = default_instruction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]

        instruction = record.get('instruction', self.default_instruction)
        image_path = record.get('image_path', '')
        output = record.get('output', '')

        # Construct chat template messages
        messages = [
            {
                "role": "system",
                "content": "Bạn là một chuyên gia kế toán kiểm toán. Hãy đọc hóa đơn và trả lời."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "max_pixels": 768 * 768},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": output
            }
        ]

        return {"messages": messages}

def load_dataset_records(path: str) -> List[Dict]:
    """Load image-QA records from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    print(f"[dataset] Loaded {len(data)} records from {path}")
    return data
