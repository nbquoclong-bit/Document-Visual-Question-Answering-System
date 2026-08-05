import os
import yaml
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from stage3_qa.src.model import load_model, apply_lora, save_model


def train(config_path="stage3_qa/configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model()
    model = apply_lora(model)
    data_cfg = cfg.get("data", {})
    dataset = load_dataset("json", data_files=data_cfg.get("train_data_path"), split="train")
    sft_cfg = SFTConfig(
        output_dir=cfg.get("output_dir", "./stage3_qa/output"),
        max_steps=cfg.get("max_steps", 500),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=sft_cfg, train_dataset=dataset)
    trainer.train()
    save_model(model, tokenizer, cfg.get("output_dir", "./stage3_qa/output"))


if __name__ == "__main__":
    train()