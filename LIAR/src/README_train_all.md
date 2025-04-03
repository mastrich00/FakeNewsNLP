# 📘 Training Script Overview — `train_all.py`

This script automates the training of **multiple DoubleInput Transformer models** (DistilBERT, BERT, RoBERTa, DeBERTa) on the LIAR-PLUS dataset by calling the `train.py` script programmatically using Python's `subprocess` module.

---

## 🔄 Purpose
- Streamline **batch training** of all model variants in one run.
- Allow user-specific configuration for each model (epochs, class weighting).

---

## 🔧 How It Works

### 1. Model Definitions
- Four models supported:
  - `1` → `distilbert-base-uncased`
  - `2` → `bert-base-uncased`
  - `3` → `roberta-base`
  - `4` → `microsoft/deberta-v3-base`

### 2. User Interaction
- For each model, the script prompts:
  - Number of training epochs
  - Whether to use **class weights** for handling class imbalance (`--weighted`)

### 3. Automated Training
- Constructs the inputs to mimic interactive mode of `train.py`.
- Launches subprocesses to execute training one model at a time:
  ```bash
  python src/train.py --weighted
  ```
  - Input: `<model_number>\n<epochs>\n`

---

## 📅 Output

For each model, `train.py` will create:
- Trained model folder: `models/<model_name>/model/`
- Checkpoints: `models/<model_name>/checkpoints/`
- Log files: `models/<model_name>/logs/train.log`

---

## ⚠️ Notes
- Script uses `subprocess.run()` with `input` to simulate CLI inputs.
- Will **run sequentially**, one model at a time.
- Errors are caught and reported per model.
- To stop execution midway, use keyboard interrupt (Ctrl+C).

---

## 📁 File Structure

```
project-root/
├── src/
│   └── train.py
├── train_all.py
├── data/
│   ├── processed/
│   │   ├── train_plus.csv
│   │   └── valid_plus.csv
├── models/
│   ├── distilbert-base-uncased/
│   ├── bert-base-uncased/
│   ├── roberta-base/
│   └── microsoft/
```

---

## 👍 Best Practices
- Run this script **after testing** that `train.py` works for each model.
- Ensure `train.py` has interactive prompts compatible with this script.
- Check CUDA/GPU availability before launching for large-scale training.

---

## 🚀 Example

```
$ python train_all.py

🔁 Batch training setup for all models...
🕒 How many epochs to train DistilBERT? (e.g., 10.0): 10
📊 Use class weights for DistilBERT? (y/n, default: n): y
...
🚀 Starting training for all selected models...
```

All logs and outputs will be stored under `models/`.

