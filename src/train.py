#!/usr/bin/env python3
# train.py — Fine-tune RoBERTa for Sentiment Analysis (Hate Speech Classification)

import argparse
import yaml
import re
import string
import emoji
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Preprocessing pipeline for tweet cleaning
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
    
# Classification metrics used by Trainer during eval
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load and parse training configuration
    cfg = yaml.safe_load(args.config.read_text())
    cfg["epochs"] = int(cfg["epochs"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"] = float(cfg.get("weight_decay", 0.0))
    cfg["fp16"] = bool(cfg.get("fp16", False))
    cfg["seed"] = int(cfg["seed"])
    cfg["max_seq_length"] = int(cfg["max_seq_length"])
    cfg["warmup_steps"] = int(cfg["warmup_steps"])

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load and preprocess dataset
    df = pd.read_csv(cfg["train_file"])
    df = df.rename(columns={"tweet": "text"})
    df = df.drop(columns=["id"])
    df["text"] = df["text"].astype(str).apply(preprocess_text)

    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=cfg["seed"],
    )

    # Convert to HuggingFace Dataset
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg["max_seq_length"],
        )
    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Padding collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Model setup and label mappings
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
    )
    model.config.label2id = {"Not Hate Speech": 0, "Hate Speech": 1}
    model.config.id2label = {0: "Not Hate Speech", 1: "Hate Speech"}
    model.to(device)

    # Training configuration
    training_args = TrainingArguments(
        output_dir="models",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=1,
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        fp16=(cfg["fp16"] and device.type == "cuda"),
        dataloader_pin_memory=(device.type == "cuda"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=cfg["seed"],
        no_cuda=False,
    )

    # Train and evaluate with HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    eval_results = trainer.evaluate()

    # Save model and tokenizer
    final_dir = Path(training_args.output_dir) / "best_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save metrics and logs
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

if __name__ == "__main__":
    main()
