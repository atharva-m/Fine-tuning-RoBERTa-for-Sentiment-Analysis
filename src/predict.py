#!/usr/bin/env python3
# predict.py — Inference for fine-tuned RoBERTa (Hate Speech Classification)

import argparse
import yaml
import re
import string
import emoji
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Preprocess text input for inference
def preprocess_text(text: str) -> str:
    text = emoji.replace_emoji(text, replace="")
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned = string.punctuation + 'Ã±ã¼â»§'
    text = text.translate(str.maketrans('', '', banned))
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    words = [w for w in text.split() if '$' not in w and '&' not in w]
    return re.sub(r"\s+", " ", ' '.join(words)).strip()

# Load YAML config and cast types
def load_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    cfg['max_seq_length'] = int(cfg.get('max_seq_length', 256))
    return cfg

# Predict single text input
def predict_text(model, tokenizer, text: str, device, max_length: int):
    clean = preprocess_text(text)
    inputs = tokenizer(
        clean,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy()[0]
    idx = probs.argmax()
    label = model.config.id2label[idx]
    return label, float(probs[idx])

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned RoBERTa model')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'),
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint directory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Single text input to classify')
    group.add_argument('--input', type=Path, help='CSV file with a "text" column for batch inference')
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'),
                        help='Path to save batch predictions CSV')
    args = parser.parse_args()

    # Load model config and set device
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load model and tokenizer from checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Inference for a single input string
    if args.text:
        label, prob = predict_text(model, tokenizer, args.text, device, cfg['max_seq_length'])
        print(f'Prediction: {label} (probability: {prob:.4f})')
    else:
        # Batch inference from a CSV file
        df = pd.read_csv(args.input)
        df = df.rename(columns={'tweet': 'text'})
        df = df.drop(columns=['id'])
        results = []
        for _, row in df.iterrows():
            label, prob = predict_text(
                model, tokenizer, row['text'], device, cfg['max_seq_length']
            )
            results.append({**row.to_dict(), 'prediction': label, 'probability': prob})
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f'Saved predictions to {args.output}')

if __name__ == '__main__':
    main()
