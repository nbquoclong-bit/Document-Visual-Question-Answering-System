import json
import argparse
from pathlib import Path
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import AutoModelForTokenClassification, AutoProcessor
import torch


def get_label_list():
    return ['O', 'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'B-TAX_CODE', 'I-TAX_CODE', 'B-DATE', 'I-DATE', 'B-TOTAL_AMOUNT', 'I-TOTAL_AMOUNT']


def evaluate(model_dir, val_records, image_dir=''):
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    model.cuda()
    label_list = get_label_list()
    all_preds = []
    all_labels = []
    for rec in val_records:
        from PIL import Image
        image = Image.open(rec['image_path']).convert('RGB')
        words = rec['words']
        boxes = rec['bboxes']
        labels = rec['labels']
        enc = processor(image, words, boxes=boxes, word_labels=labels, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        batch = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits.argmax(-1).cpu().numpy()[0]
        lbls = batch['labels'].cpu().numpy()[0]
        all_preds.append(preds)
        all_labels.append(lbls)
    true_preds = []
    true_labels = []
    for pred, lbl in zip(all_preds, all_labels):
        tp, tl = [], []
        for p_i, l_i in zip(pred, lbl):
            if l_i == -100:
                continue
            tp.append(label_list[p_i])
            tl.append(label_list[l_i])
        true_preds.append(tp)
        true_labels.append(tl)
    report = {
        'precision': precision_score(true_labels, true_preds),
        'recall': recall_score(true_labels, true_preds),
        'f1': f1_score(true_labels, true_preds),
    }
    out_path = Path(model_dir) / 'eval_report.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--val_records', required=True)
    args = parser.parse_args()
    import yaml
    with open(args.val_records, 'r', encoding='utf-8') as f:
        val_records = yaml.safe_load(f)
    evaluate(args.model_dir, val_records)
