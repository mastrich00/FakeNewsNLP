# 📘 Continued Training Script Overview — `more_train.py`

This script allows you to **resume training** of a transformer-based model (DistilBERT, BERT, RoBERTa, DeBERTa) from the latest available checkpoint saved by `train.py`. It is designed to **extend training** without restarting from scratch.

---

## 📚 Purpose
- Resume training from **last saved checkpoint**.
- Specify a new target epoch for extended training.
- Optional: Apply class weights to address imbalance.

---

## 🔧 How It Works

### 1. User Input
- Prompt for:
  - Model choice (1 to 4)
  - Confirmation of detected checkpoint path
  - Desired total epoch to continue training to

### 2. Checkpoint Discovery
- Scans `models/<model_key>/checkpoints/` for existing checkpoints.
- Uses the **latest checkpoint** by default (or allows override).
- Parses `trainer_state.json` to identify `last_epoch`.

### 3. Training Resumption
- Loads model weights from: `checkpoint-*/model/pytorch_model.bin`
- Loads LIAR-PLUS data and applies the same tokenizer setup as `train.py`
- Creates a `Trainer` or `WeightedTrainer` instance
- Continues training using `resume_from_checkpoint`

---

## 🔄 Input Data
- `data/processed/train_plus.csv`
- `data/processed/valid_plus.csv`

---

## ⚖️ Optional Argument
- `--weighted`: Recalculates class weights from training distribution and applies them via a custom loss function using `F.cross_entropy()`.

---

## 📆 Tokenization & Formatting
- Tokenizes `cleaned_statement` and `cleaned_justification`.
- Pads to max length 128.
- Supports `token_type_ids` for BERT-like models.
- Sets output to PyTorch tensors using Hugging Face `Dataset` API.

---

## 💡 Output
- Updated model weights saved to: `models/<model_key>/model/pytorch_model.bin`
- Tokenizer configuration saved with model.
- Logs saved to a unique log file per run: `logs/continue_train_<n>.log`

---

## 🚀 Example
```bash
$ python more_train.py --weighted

Choose model to continue training:
  1 - DistilBERT
  2 - BERT
  3 - RoBERTa
  4 - DeBERTa
Enter model number (1-4): 3
📍 Latest checkpoint detected: checkpoint-1800
Use latest checkpoint? (press Enter to accept):
Until which total epoch should training continue? (must be > 3.0): 5.0
...
✅ Updated model saved to models/roberta-base/model
```

---

## 🚧 Dependencies
- `transformers`, `torch`, `datasets`, `pandas`, `sklearn`, `argparse`, `re`, `json`
- Model-specific classes in: `src.models.<model_module>`

