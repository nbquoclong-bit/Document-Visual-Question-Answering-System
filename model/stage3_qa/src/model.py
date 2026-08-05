"""
model.py - Qwen2.5-1.5B-Instruct with QLoRA for Vietnamese accounting QA.

Loads the base model with 4-bit quantization, applies LoRA adapters,
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
from transformers import AutoTokenizer, BitsAndBytesConfig

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError(
        "unsloth is required. Install with: pip install unsloth"
    )


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 512


def get_quantization_config() -> BitsAndBytesConfig:
    """Return 4-bit quantization config for memory-efficient loading."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_lora_config(target_modules: Optional[list] = None) -> LoraConfig:
    """Return LoRA configuration for Qwen2.5 attention + MLP layers."""
    default_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules or default_targets,
    )


def load_model_and_tokenizer(
    base_model_name: str = BASE_MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    device_map: str = "auto",
) -> Tuple:
    """
    Load base model with 4-bit quantization and apply LoRA adapters.

    Args:
        base_model_name: Hugging Face model identifier.
        max_seq_length: Maximum sequence length.
        device_map: Device allocation strategy.

    Returns:
        Tuple of (model, tokenizer) ready for training.
    """
    print(f"[model] Loading base model: {base_model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    print("[model] Model loaded with QLoRA adapters applied.")
    return model, tokenizer


def save_model(
    model,
    tokenizer,
    save_dir: str,
    merge: bool = False,
    save_tokenizer: bool = True,
) -> Path:
    """
    Save LoRA adapters (and optionally merge with base model).

    Args:
        model: PeftModel to save.
        tokenizer: Tokenizer to save alongside adapters.
        save_dir: Output directory path.
        merge: If True, merge LoRA weights into base model.
        save_tokenizer: Whether to save the tokenizer.

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
        if save_tokenizer:
            tokenizer.save_pretrained(str(merged_save_path))
        print(f"[model] Merged model saved to: {merged_save_path}")

        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        return merged_save_path
    else:
        adapter_path = save_path / "lora_adapters"
        model.save_pretrained(str(adapter_path))
        if save_tokenizer:
            tokenizer.save_pretrained(str(adapter_path))
        print(f"[model] LoRA adapters saved to: {adapter_path}")
        return adapter_path


def free_memory(model=None, tokenizer=None) -> None:
    """Explicitly free GPU memory after training or inference."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[model] GPU memory cleared.")
