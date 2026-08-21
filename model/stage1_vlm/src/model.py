"""
model.py - Qwen2-VL-2B-Instruct with QLoRA for Vietnamese accounting VQA.

Loads the base Vision-Language Model with 4-bit quantization, applies LoRA adapters,
and provides utilities to save adapters or merge them into the base model.
"""

import gc
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

BASE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def get_quantization_config() -> BitsAndBytesConfig:
    """Return 4-bit quantization config for memory-efficient loading.
    Uses float16 compute dtype for compatibility with T4/P100 GPUs (no bfloat16).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def get_lora_config(target_modules: Optional[list] = None) -> LoraConfig:
    """Return LoRA configuration for Qwen2-VL attention + MLP layers."""
    default_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules or default_targets,
    )


def load_model_and_processor(
    base_model_name: str = BASE_MODEL_NAME,
    device_map: str = "auto",
    is_training: bool = True
) -> Tuple:
    """
    Load base VLM with 4-bit quantization and processor. Apply LoRA if training.

    Args:
        base_model_name: Hugging Face model identifier.
        device_map: Device allocation strategy.
        is_training: If True, applies QLoRA preparation.

    Returns:
        Tuple of (model, processor).
    """
    print(f"[model] Loading base model: {base_model_name}")

    processor = AutoProcessor.from_pretrained(base_model_name)
    use_cuda = torch.cuda.is_available()
    quantization_config = get_quantization_config() if is_training and use_cuda else None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map=device_map if use_cuda else None,
        low_cpu_mem_usage=True,
    )

    if is_training:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = get_lora_config()
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print("[model] Model loaded with QLoRA adapters applied.")
    else:
        print("[model] Model loaded for inference.")

    return model, processor


def save_model(
    model,
    processor,
    save_dir: str,
    merge: bool = False,
) -> Path:
    """
    Save LoRA adapters (and optionally merge with base model).

    Args:
        model: PeftModel to save.
        processor: Processor to save alongside adapters.
        save_dir: Output directory path.
        merge: If True, merge LoRA weights into base model.

    Returns:
        Path to the saved model directory.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if merge:
        print("[model] Merging LoRA adapters into base model...")
        merged_model = model.merge_and_unload()
        merged_save_path = save_path / "merged_model"
        merged_model.save_pretrained(str(merged_save_path))
        processor.save_pretrained(str(merged_save_path))
        print(f"[model] Merged model saved to: {merged_save_path}")

        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        return merged_save_path
    else:
        adapter_path = save_path / "lora_adapters"
        model.save_pretrained(str(adapter_path))
        processor.save_pretrained(str(adapter_path))
        print(f"[model] LoRA adapters saved to: {adapter_path}")
        return adapter_path


def free_memory(model=None, processor=None) -> None:
    """Explicitly free GPU memory after training or inference."""
    if model is not None:
        del model
    if processor is not None:
        del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[model] GPU memory cleared.")
