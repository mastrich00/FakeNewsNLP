# 📘 Training Script Overview — `train.py`

This script is used to train a **DoubleInputBERT** model on the LIAR-PLUS dataset for **multi-class fake news classification**. The model takes both a **statement** and its **justification** as input.

---

## 📂 Input Data

- **train_plus.csv**: Preprocessed training data with:
  - `cleaned_statement`
  - `cleaned_justification`
  - `label` (class label from 0 to 5)

- **valid_plus.csv**: Preprocessed validation data (same format)

---

## 🧠 Model

- **Architecture**: `DoubleInputBERTModel`
  - A custom PyTorch model that uses shared-weight BERT encoders for two inputs:
    - `cleaned_statement` → `input_ids`, `attention_mask`
    - `cleaned_justification` → `input_ids_2`, `attention_mask_2`

- **Base model**: `distilbert-base-uncased`

---

## 🔁 Processing Steps

### 1. Load Data
- Reads preprocessed CSVs and ensures label column is integer.
- Prints class distribution.

### 2. Tokenization
- Uses `AutoTokenizer` to tokenize both inputs (`statement` and `justification`).
- Padding & truncation to max length of 128.
- Applies to both train and validation sets.
- Keeps only tokenized columns and label.

### 3. Dataset Conversion
- Converts `pandas` DataFrames to Hugging Face `Dataset` objects.
- Applies formatting for PyTorch tensors.

---

## ⚙️ Training Configuration

Uses `transformers.TrainingArguments`:
- `num_train_epochs=12`
- `batch_size=8`
- `eval_strategy="epoch"`
- Uses FP16 if CUDA is available
- Logs every 10 steps
- Saves the best model based on `eval_loss`

---

## 🏋️ Training

- Uses `Trainer` from Hugging Face 🤗
- Passes model, datasets, tokenizer, and training args
- Starts `.train()` to begin training

---

## 💾 Saving

After training:
- Saves model weights to: `models/fake-news-double-bert/`
- Saves tokenizer config to the same folder

---

## ✅ Output

- Trained model weights and tokenizer config
- Log directory: `logs/`
- Optional: can be used for further fine-tuning or prediction

---

## 📎 Notes

- Custom loss function is **not overridden** to avoid conflicts with `Trainer`.
- Uses standard Trainer loop.
