import argparse
import yaml
import os
import numpy as np
from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoProcessor, Trainer, TrainingArguments, EarlyStoppingCallback
from stage2_kie.src.dataset import InvoiceKIE, InvoiceKIEConfig, collate_fn


seq_metric = load_metric('seqeval')


def get_label_list():
    return ['O', 'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'B-TAX_CODE', 'I-TAX_CODE', 'B-DATE', 'I-DATE', 'B-TOTAL_AMOUNT', 'I-TOTAL_AMOUNT']


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    label_list = get_label_list()
    for pred, lbl in zip(predictions, labels):
        pred_line = []
        lbl_line = []
        for p_i, l_i in zip(pred, lbl):
            if l_i == -100:
                continue
            pred_line.append(label_list[p_i])
            lbl_line.append(label_list[l_i])
        true_predictions.append(pred_line)
        true_labels.append(lbl_line)
    results = seq_metric.compute(predictions=true_predictions, references=true_labels)
    return {'precision': results['overall_precision'], 'recall': results['overall_recall'], 'f1': results['overall_f1']}


def train(config_path='stage2_kie/configs/train_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get('model_name', 'microsoft/layoutlmv3-base')
    num_labels = len(get_label_list())
    label2id = {l: i for i, l in enumerate(get_label_list())}
    id2label = {i: l for l, i in label2id.items()}
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    cfg_data = cfg.get('data', {})
    kie_config = InvoiceKIEConfig(image_dir=cfg_data.get('image_dir', ''), processor_name=model_name, max_length=cfg.get('max_length', 512))
    train_dataset = InvoiceKIE(cfg_data.get('train_records', []), kie_config)
    val_dataset = InvoiceKIE(cfg_data.get('val_records', []), kie_config)
    training_args = TrainingArguments(
        output_dir=cfg.get('output_dir', './output'),
        num_train_epochs=cfg.get('epochs', 10),
        per_device_train_batch_size=cfg.get('train_batch_size', 4),
        per_device_eval_batch_size=cfg.get('eval_batch_size', 8),
        learning_rate=cfg.get('learning_rate', 5e-5),
        weight_decay=cfg.get('weight_decay', 0.01),
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_dir=os.path.join(cfg.get('output_dir', './output'), 'logs'),
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 5))],
    )
    trainer.train()
    model.save_pretrained(os.path.join(cfg.get('output_dir', './output'), 'best_model'))
    processor.save_pretrained(os.path.join(cfg.get('output_dir', './output'), 'best_model'))


if __name__ == '__main__':
    train()
