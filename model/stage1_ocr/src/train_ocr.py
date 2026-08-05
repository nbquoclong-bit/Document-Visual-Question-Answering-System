"""
train_ocr.py - PaddleOCR fine-tuning for Vietnamese invoices.

Reads hyperparameters from ../configs/finetune_config.yaml, prepares
train/val Paddle-format list files via data_prep, trains with early
stopping, and saves the best checkpoint to ../output/best_model/.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple

import yaml
import paddle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config(config_path: str) -> dict:
    """Load YAML fine-tuning configuration."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_paths(cfg: dict, base: Path) -> dict:
    """Resolve relative paths from YAML against the project root."""
    data = cfg.get("data", {})
    return {
        "mc_ocr_path": base / data.get("mc_ocr_path", "./data/mcocr_public/"),
        "internal_data_path": base / data.get("internal_data_path", "./data/internal_invoices/"),
        "output_dir": base / data.get("output_dir", "./output/"),
        "train_list": base / "output" / "paddle_train.txt",
        "val_list": base / "output" / "paddle_val.txt",
        "best_model": base / "output" / "best_model",
    }


def prepare_train_val_lists(
    mc_ocr_path: Path,
    internal_path: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[str, str]:
    """Call data_prep to produce (or reuse) Paddle-format train/val list files."""
    os.makedirs(str(output_dir), exist_ok=True)
    train_list = str(output_dir / "paddle_train.txt")
    val_list = str(output_dir / "paddle_val.txt")

    if Path(train_list).exists() and Path(val_list).exists():
        print(f"Reusing existing list files: {train_list}, {val_list}")
        return train_list, val_list

    from data_prep import prepare_dataset
    records = prepare_dataset(str(mc_ocr_path), str(output_dir))

    n_total = len(records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    lines = Path(train_list).read_text(encoding="utf-8").splitlines() if Path(train_list).exists() else []
    if not lines:
        raise RuntimeError("prepare_dataset did not produce paddle_train.txt")

    with open(train_list, "w", encoding="utf-8") as f_train, \
         open(val_list, "w", encoding="utf-8") as f_val:
        f_train.writelines(lines[:n_train])
        f_val.writelines(lines[n_train:n_train + n_val])

    print(f"Wrote {n_train} train / {n_val} val samples to list files.")
    return train_list, val_list

def train_model(
    cfg: dict,
    train_list: str,
    val_list: str,
    best_model_dir: Path,
) -> None:
    """Run the PaddleOCR training loop with early stopping."""
    from paddle.vision.datasets import DatasetFolder

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augmentation", {})

    max_epochs = int(train_cfg.get("max_epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    save_epoch = int(train_cfg.get("save_epoch", 5))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    base_model = model_cfg.get("base_model", "ch_PP-OCRv3_rec")
    lang = model_cfg.get("lang", "vi")
    rot_range = float(aug_cfg.get("rotation_range", 5))
    noise_std = float(aug_cfg.get("noise_std", 15))
    blur_prob = float(aug_cfg.get("blur_prob", 0.3))

    best_model_dir.mkdir(parents=True, exist_ok=True)
    last_best_loss = float("inf")
    epochs_no_improve = 0

    # TODO: replace with real PaddleOCR / AutoML training entry once the API surface
    # is confirmed for this repo version. The block below mirrors the logical flow
    # of a recognition fine-tune (backbone -> train_head -> eval -> checkpoint).
    print("=" * 60)
    print(f"Starting OCR fine-tuning | lang={lang} | base={base_model}")
    print(f"  max_epochs={max_epochs}  batch={batch_size}  lr={lr}")
    print(f"  early_stopping_patience={patience}  save_epoch={save_epoch}")
    print(f"  augmentation: rot={rot_range} noise={noise_std} blur={blur_prob}")
    print("=" * 60)

    train_dataset = paddle.io.DatasetLoader.from_generator(
        lambda: _load_paddle_list(train_list),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataset = paddle.io.DatasetLoader.from_generator(
        lambda: _load_paddle_list(val_list),
        batch_size=batch_size,
        shuffle=False,
    )

    model = paddle.models.resnet18(pretrained=True, num_classes=len(_charset()))
    optimizer = Adam(learning_rate=lr, parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_id, batch in enumerate(train_dataset()):
            images, labels = batch
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            running_loss += loss.item()
            if batch_id % 100 == 0:
                print(f"  epoch {epoch} step {batch_id} loss={loss.item():.4f}")

        train_loss = running_loss / max(1, batch_id + 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with paddle.no_grad():
            for batch in val_dataset():
                images, labels = batch
                logits = model(images)
                loss = loss_fn(logits, labels)
                val_loss_sum += loss.item()
                val_count += 1

        val_loss = val_loss_sum / max(1, val_count)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < last_best_loss:
            last_best_loss = val_loss
            epochs_no_improve = 0
            _save_checkpoint(model, str(best_model_dir / "best_model"))
            print(f"  => New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if epoch % save_epoch == 0:
            _save_checkpoint(model, str(best_model_dir / f"epoch_{epoch:03d}"))
            print(f"  => Checkpoint epoch {epoch:03d} saved.")

    elapsed = time.time() - start
    print(f"Training complete in {elapsed/60:.1f} min.")
    print(f"Best model is at: {best_model_dir / 'best_model'}")


def _load_paddle_list(list_path: str):
    """Yield (image, label) tuples from a Paddle-format list file."""
    import cv2
    import numpy as np

    charset = _charset()

    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("\t", 1)
        if len(parts) != 2:
            continue
        img_path, annos_str = parts
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (100, 32))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        label_indices = [_char_to_index(annos_str, charset)]
        label_tensor = paddle.to_tensor(label_indices, dtype="int64")

        yield paddle.to_tensor(img).unsqueeze(0), label_tensor[0]


def _char_to_index(text: str, charset: dict) -> int:
    return charset.get(text, 0)


def _charset() -> dict:
    """Dummy character-to-index map for the stub training loop."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return {c: i + 1 for i, c in enumerate(chars)}


def _save_checkpoint(model: paddle.nn.Layer, path: str) -> None:
    paddle.save(model.state_dict(), path + ".pdparams")
    print(f"Saved checkpoint: {path}.pdparams")


def main() -> None:
    """Entry point: load config, prepare data, train, save best model."""
    config_path = str(PROJECT_ROOT / "configs" / "finetune_config.yaml")
    if not Path(config_path).exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    paths = resolve_paths(cfg, PROJECT_ROOT)

    mc_ocr_path = paths["mc_ocr_path"]
    internal_path = paths["internal_data_path"]
    output_dir = paths["output_dir"]
    train_ratio = cfg.get("data", {}).get("train_ratio", 0.8)
    val_ratio = cfg.get("data", {}).get("val_ratio", 0.1)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {config_path}")
    print(f"MC_OCR path  : {mc_ocr_path}")
    print(f"Output dir   : {output_dir}")

    train_list, val_list = prepare_train_val_lists(
        mc_ocr_path, internal_path, output_dir, train_ratio, val_ratio
    )

    train_model(cfg, train_list, val_list, paths["best_model"])


if __name__ == "__main__":
    main()
