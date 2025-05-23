# 🧠 Hate Speech Detection using RoBERTa (Fine-Tuned)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

This project demonstrates an end-to-end pipeline to fine-tune a `roberta-base` model for binary **hate speech detection**. It covers custom preprocessing, training, evaluation, inference, and visualization — optimized for high recall on sensitive inputs.

---

## 🚀 Quickstart

```bash
# 1. Clone and enter the project
$ git clone https://github.com/atharva-m/fine-tuning-roberta-for-sentiment-analysis.git
$ cd fine-tuning-roberta-for-sentiment-analysis

# 2. Setup virtual environment
$ python -m venv .venv && source .venv/bin/activate  

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Train model
$ python src/train.py --config config.yaml

# 5. Evaluate on test set
$ python src/evaluate.py --config config.yaml

# 6. Predict a single text
$ python src/predict.py --checkpoint models/best_model --text "Some tweet here"

# 7. Batch prediction
$ python src/predict.py --checkpoint models/best_model --input data/test.csv --output predictions.csv
```

---

## 📊 Key Features

| Stage           | Description                                                           |
|----------------|-----------------------------------------------------------------------|
| ⚙ Preprocessing | Emoji removal, hashtag cleaning, punctuation stripping, lowercasing  |
| 🧠 Model        | `roberta-base`, fine-tuned for 5 epochs using `Trainer` API          |
| 🎯 Objective     | Binary classification: Hate Speech (1) vs Not Hate Speech (0)        |
| 📈 Metrics       | Accuracy: **97.07%**, Precision: 96.95%, F1: 96.99%, Recall: 97.07%  |
| 📊 Confusion Matrix | Exported to `reports/confusion_matrix.png` for visual inspection |
| 💾 Inference     | Script supports both single-sentence and batch CSV classification    |

---

## 🗂 Project Structure

```
├── data/              # Raw and preprocessed CSV files
├── models/            # Final saved transformer model and tokenizer
├── reports/           # Evaluation metrics and confusion matrix
├── src/
│   ├── train.py       # Training and fine-tuning script
│   ├── evaluate.py    # Test evaluation and confusion matrix
│   └── predict.py     # CLI inference for single/batch mode
├── config.yaml        # Model config: learning rate, batch size, epochs
├── requirements.txt   # pip dependencies
└── predictions.csv    # Output from batch prediction
```

---

## 🧠 Model Training

- Model: `roberta-base` from Hugging Face
- Loss: CrossEntropy
- Optimizer: AdamW
- Scheduler: Linear with warmup
- Max Sequence Length: 256
- Batch Size: 50
- Epochs: 5
- Device: CUDA (if available)

---

## 🔬 Evaluation Metrics (from `evaluate.py`)

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 97.07%    |
| Precision  | 96.95%    |
| Recall     | 97.07%    |
| F1-Score   | 96.99%    |
| Confusion Matrix | ✅ See `reports/confusion_matrix.png` |

---

## 📝 License

This project is licensed under the **Creative Commons Zero v1.0 Universal (CC0 1.0)** — you may use, modify, and distribute freely without attribution.

---

### Contact

Atharva Mokashi · atharvamokashi01@gmail.com · [LinkedIn](https://www.linkedin.com/in/atharva-m)
