import os
import sys

# Vô hiệu hóa hoàn toàn WandB để không bị crash yêu cầu API key trên Kaggle/Colab
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
from transformers import Trainer, TrainingArguments
# Auto-add model directory to sys.path so imports work from any cwd
_here = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.abspath(os.path.join(_here, "../.."))
project_root = os.path.abspath(os.path.join(_here, "../../.."))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    import subprocess
    print("📦 [trainer] Đang tự động cài đặt thư viện 'qwen-vl-utils'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils>=0.0.8", "peft>=0.12.0", "bitsandbytes>=0.43.0", "-q"])
    from qwen_vl_utils import process_vision_info

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
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

        for i in range(inputs["input_ids"].size(0)):
            input_ids_list = inputs["input_ids"][i].tolist()
            assistant_start = -1
            
            # Tìm thẻ <|im_start|> cuối cùng trong chuỗi (chính là lượt của assistant)
            for idx in range(len(input_ids_list) - 1, -1, -1):
                if input_ids_list[idx] == im_start_id:
                    # Bỏ qua <|im_start|>, 'assistant', và ký tự xuống dòng (\n)
                    cur = idx + 1
                    while cur < len(input_ids_list) and input_ids_list[cur] not in (198, 271) and cur < idx + 4:
                        cur += 1
                    while cur < len(input_ids_list) and input_ids_list[cur] in (198, 271):
                        cur += 1
                    assistant_start = cur
                    break
                    
            if assistant_start != -1 and assistant_start < len(input_ids_list):
                labels[i, :assistant_start] = -100
            else:
                # Dự phòng: tìm vị trí <|im_start|> cuối cùng và mask toàn bộ phần trước
                last_starts = [k for k, val in enumerate(input_ids_list) if val == im_start_id]
                if last_starts:
                    labels[i, :last_starts[-1] + 3] = -100
                
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
        
    lora_cfg_dict = cfg.get("lora", {})
    model, processor = load_model_and_processor(
        base_model_name=cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        is_training=True,
        lora_r=lora_cfg_dict.get("r", 16),
        lora_alpha=lora_cfg_dict.get("lora_alpha", 32),
        lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
        target_modules=lora_cfg_dict.get("target_modules", None)
    )
    
    data_cfg = cfg.get("data", {})
    train_path = data_cfg.get("train_data_path", "data/vlm_train.json")
    
    # Tìm kiếm đệ quy file dữ liệu
    possible_paths = [
        train_path,
        os.path.join(_model_dir, "data/vlm_train.json"),
        os.path.join(project_root, "data/vlm_train.json"),
        "/kaggle/working/Document-Visual-Question-Answering-System/model/data/vlm_train.json",
        "/kaggle/working/Document-Visual-Question-Answering-System/data/vlm_train.json"
    ]
    
    actual_train_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            actual_train_path = p
            break
            
    if not actual_train_path:
        print("[trainer] ⚠️ Chưa tìm thấy vlm_train.json. Đang tự động quét và tạo tập dữ liệu ngay lập tức...")
        try:
            from stage1_vlm.src.prepare_vlm_data import find_all_images, convert_funsd_to_vqa
        except ImportError:
            from prepare_vlm_data import find_all_images, convert_funsd_to_vqa
            
        image_dirs = [
            "/kaggle/input", "/kaggle/working",
            os.path.join(project_root, "datasets/vietnamese-receipts-v3"),
            os.path.join(project_root, "datasets/MCOCR"),
            os.path.join(project_root, "datasets"),
        ]
        image_index = find_all_images(image_dirs)
        
        train_data_dirs = [
            os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_train_funsd"),
            os.path.join(project_root, "datasets/vietnamese-receipts-v3/train/funsd_json"),
            os.path.join(project_root, "datasets/VN_receipts_train_funsd"),
            os.path.join(project_root, "datasets/MCOCR/mcocr_train_funsd"),
            os.path.join(project_root, "datasets/MCOCR/train/funsd_json"),
            os.path.join(project_root, "datasets/mcocr_train_funsd"),
        ]
        out_dir = os.path.abspath(os.path.join(_model_dir, "data"))
        os.makedirs(out_dir, exist_ok=True)
        train_out = os.path.join(out_dir, "vlm_train.json")
        convert_funsd_to_vqa(train_data_dirs, image_index, train_out)
        actual_train_path = train_out

    records = load_dataset_records(actual_train_path) if actual_train_path and os.path.exists(actual_train_path) else []
    
    # Kiểm tra tính hợp lệ của đường dẫn ảnh trên môi trường hiện tại
    valid_records = [r for r in records if os.path.exists(r.get("image_path", ""))]
    if records and len(valid_records) < len(records) * 0.5:
        print("[trainer] ⚠️ Đường dẫn ảnh trong file json không khớp với máy hiện tại. Đang tự động quét và lập chỉ mục lại...")
        try:
            from stage1_vlm.src.prepare_vlm_data import find_all_images, convert_funsd_to_vqa
        except ImportError:
            from prepare_vlm_data import find_all_images, convert_funsd_to_vqa
            
        image_dirs = [
            "/kaggle/input", "/kaggle/working",
            os.path.join(project_root, "datasets/vietnamese-receipts-v3"),
            os.path.join(project_root, "datasets/MCOCR"),
            os.path.join(project_root, "datasets"),
        ]
        image_index = find_all_images(image_dirs)
        
        train_data_dirs = [
            os.path.join(project_root, "datasets/vietnamese-receipts-v3/VN_receipts_train_funsd"),
            os.path.join(project_root, "datasets/vietnamese-receipts-v3/train/funsd_json"),
            os.path.join(project_root, "datasets/VN_receipts_train_funsd"),
            os.path.join(project_root, "datasets/MCOCR/mcocr_train_funsd"),
            os.path.join(project_root, "datasets/MCOCR/train/funsd_json"),
            os.path.join(project_root, "datasets/mcocr_train_funsd"),
        ]
        out_dir = os.path.abspath(os.path.join(_model_dir, "data"))
        os.makedirs(out_dir, exist_ok=True)
        train_out = os.path.join(out_dir, "vlm_train.json")
        convert_funsd_to_vqa(train_data_dirs, image_index, train_out)
        actual_train_path = train_out
        records = load_dataset_records(actual_train_path)
    elif valid_records:
        records = valid_records

    if not records:
        print(f"[Error] Không tìm thấy ảnh hoặc dữ liệu hợp lệ tại {actual_train_path}. Vui lòng kiểm tra thư mục datasets.")
        return

    print(f"📊 [trainer] Nạp thành công {len(records)} mẫu huấn luyện hợp lệ!")

    train_dataset = VQADataset(
        records=records,
    )

    # Use the custom dynamic collator instead of default
    data_collator = Qwen2VLDataCollator(processor)

    # Đảm bảo đường dẫn output luôn chuẩn xác
    abs_output_dir = os.path.abspath(os.path.join(_here, "../output"))
    os.makedirs(abs_output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=abs_output_dir,
        max_steps=cfg.get("max_steps", 400),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        warmup_steps=30,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
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
    
    print("[trainer] 🚀 Bắt đầu huấn luyện QLoRA Golden (Dự kiến: 15-25 phút trên GPU T4)...")
    trainer.train()
    
    save_model(model, processor, cfg.get("output_dir", "./stage1_vlm/output"))

if __name__ == "__main__":
    train()