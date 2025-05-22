#!/usr/bin/env python3
# evaluate.py — Evaluate fine-tuned RoBERTa model on held-out test set

import argparse
import yaml
import re
import string
import emoji
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Clean raw text to match training preprocessing
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

# Compute accuracy, precision, recall, f1
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML config path")
    args = parser.parse_args()

    # Load config values
    cfg = yaml.safe_load(args.config.read_text())
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["max_seq_length"] = int(cfg.get("max_seq_length", 256))

    # Detect CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load and preprocess dataset
    df = pd.read_csv(cfg["train_file"])
    df = df.rename(columns={"tweet": "text"}).drop(columns=["id"])
    df["text"] = df["text"].astype(str).apply(preprocess_text)

    # Split test set only
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=cfg["seed"])
    ds = DatasetDict({"test": Dataset.from_pandas(test_df.reset_index(drop=True))})

    # Tokenize test set
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=cfg["max_seq_length"])
    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load model checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(Path(cfg["output_dir"])).to(device)

    # Evaluation setup
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir="reports",
        per_device_eval_batch_size=cfg.get("batch_size", 16),
        do_train=False,
        do_eval=True,
        eval_strategy="no",
        no_cuda=(device.type != "cuda"),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Print metrics and save
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics_eval.json").write_text(json.dumps(eval_metrics, indent=2))

    # Plot confusion matrix
    preds_output = trainer.predict(ds["test"])
    y_true = preds_output.label_ids
    y_pred = preds_output.predictions.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hate Speech", "Hate Speech"])
    disp.plot(cmap="Blues")
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved confusion matrix to reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
