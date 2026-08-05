import json
import torch
from typing import Dict, Any
from unsloth import FastLanguageModel
from stage3_qa.src.model import load_model
from peft import PeftModel


class QAEngine:
    def __init__(self, adapter_dir: str = None, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model, self.tokenizer = load_model(base_model)
        if adapter_dir:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        FastLanguageModel.for_inference(self.model)

    def answer(self, invoice_json: Dict, question: str = "Kiem tra tinh toan tren hoa don nay co dung khong?") -> str:
        context = json.dumps(invoice_json, ensure_ascii=False)
        prompt = f"[INST] {question}\n\nDu lieu hoa don: {context} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()