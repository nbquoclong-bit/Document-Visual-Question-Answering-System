import os
import sys

# Vô hiệu hóa hoàn toàn WandB để không bị crash yêu cầu API key trên Kaggle/Colab
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
from transformers import Trainer, TrainingArguments
from qwen_vl_utils import process_vision_info

# Auto-add model directory to sys.path so imports work from any cwd
_here = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.abspath(os.path.join(_here, "../.."))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

try:
    from stage1_vlm.src.model import load_model_and_processor, save_model
    from stage1_vlm.src.dataset import VQADataset, load_dataset_records
except ImportError:
    from src.model import load_model_and_processor, save_model
    from src.dataset import VQADataset, load_dataset_records

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
        
        # Mask system + vision + user prompt tokens accurately
        # In Qwen2 tokenizer: <|im_start|> is 151644, assistant token is 77091
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")

        for i in range(inputs["input_ids"].size(0)):
            input_ids_list = inputs["input_ids"][i].tolist()
            assistant_start = -1
            # Find the LAST occurrence of <|im_start|> assistant in the sequence
            for idx in range(len(input_ids_list) - 1):
                if input_ids_list[idx] == im_start_id and input_ids_list[idx + 1] == assistant_id:
                    # Skip <|im_start|>, assistant, and newline tokens (\n = 198)
                    cur = idx + 2
                    while cur < len(input_ids_list) and input_ids_list[cur] in (198, 271):
                        cur += 1
                    assistant_start = cur
            if assistant_start != -1:
                labels[i, :assistant_start] = -100
                
        inputs["labels"] = labels
        return inputs



def train(config_path="stage1_vlm/configs/train_config.yaml"):
    # Tìm kiếm đường dẫn config linh hoạt theo thư mục thực thi
    if not os.path.exists(config_path):
        alt_config = os.path.join(_here, "../configs/train_config.yaml")
        if os.path.exists(alt_config):
            config_path = alt_config

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model, processor = load_model_and_processor(
        base_model_name=cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        is_training=True
    )
    
    data_cfg = cfg.get("data", {})
    train_path = data_cfg.get("train_data_path", "data/vlm_train.json")
    if not os.path.exists(train_path):
        alt_train = os.path.join(_model_dir, "data/vlm_train.json")
        if os.path.exists(alt_train):
            train_path = alt_train
    
    records = load_dataset_records(train_path) if os.path.exists(train_path) else []
    if not records:
        print(f"[Warning] Training data file not found at {train_path}. Please prepare it before training.")
        return

    train_dataset = VQADataset(
        records=records,
    )

    # Use the custom dynamic collator instead of default
    data_collator = Qwen2VLDataCollator(processor)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./stage1_vlm/output"),
        max_steps=cfg.get("max_steps", 400),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=4,
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        warmup_steps=20,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
        dataloader_pin_memory=use_cuda,
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