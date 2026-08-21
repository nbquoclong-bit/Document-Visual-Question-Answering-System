import os
import yaml
import torch
from transformers import Trainer, TrainingArguments
from qwen_vl_utils import process_vision_info

from stage1_vlm.src.model import load_model_and_processor, save_model
from stage1_vlm.src.dataset import VQADataset, load_dataset_records

class Qwen2VLDataCollator:
    """
    Custom Data Collator that dynamically pads Qwen2-VL inputs.
    It takes raw messages, extracts vision info, and uses the processor 
    with padding=True to avoid truncation of vision tokens.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        messages_list = [item["messages"] for item in batch]
        
        for msgs in messages_list:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # padding=True allows dynamic padding up to the longest sequence in the batch
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Labels are identical to input_ids for Causal LM. 
        # We replace padding token ids and prompt tokens with -100 so loss is calculated ONLY on completion tokens.
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        
        # Mask system + user prompt tokens
        assistant_token_ids = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for i, text_str in enumerate(texts):
            input_ids_list = inputs["input_ids"][i].tolist()
            assistant_start = -1
            for idx in range(len(input_ids_list) - len(assistant_token_ids) + 1):
                if input_ids_list[idx : idx + len(assistant_token_ids)] == assistant_token_ids:
                    assistant_start = idx + len(assistant_token_ids)
            if assistant_start != -1:
                labels[i, :assistant_start] = -100
                
        inputs["labels"] = labels
        return inputs


def train(config_path="stage1_vlm/configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model, processor = load_model_and_processor(
        base_model_name=cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        is_training=True
    )
    
    data_cfg = cfg.get("data", {})
    train_path = data_cfg.get("train_data_path", "data/vlm_train.json")
    
    records = load_dataset_records(train_path) if os.path.exists(train_path) else []
    if not records:
        print(f"[Warning] Training data file not found at {train_path}. Please prepare it before training.")
        return

    train_dataset = VQADataset(
        records=records,
    )

    # Use the custom dynamic collator instead of default
    data_collator = Qwen2VLDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./stage1_vlm/output"),
        max_steps=cfg.get("max_steps", 500),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False, # Essential for multimodal inputs
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("[trainer] Starting training...")
    trainer.train()
    
    save_model(model, processor, cfg.get("output_dir", "./stage1_vlm/output"))

if __name__ == "__main__":
    train()