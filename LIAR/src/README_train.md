# 📘 Training Script Overview — `train.py`

This script is used to train a **DoubleInput Transformer** model on the LIAR-PLUS dataset for **multi-class fake news classification**. The model takes both a **statement** and its **justification** as input.

---

## 📂 Input Data

- **train_plus.csv**: Preprocessed training data containing:
  - `cleaned_statement`
  - `cleaned_justification`
  - `label` (integer from 0 to 5)

- **valid_plus.csv**: Preprocessed validation data (same structure)

---

## 🧠 Model

- **Architecture**: Dynamic import of model class based on user input:
  - Options include:
    - `distilbert-base-uncased` (DoubleInputBERTModel)
    - `bert-base-uncased` (DoubleInputBERTModel)
    - `roberta-base` (DoubleInputRoBERTaModel)
    - `microsoft/deberta-v3-base` (DoubleInputDebertaModel)

- **Design**: Two encoder branches with shared or model-specific layers to process `statement` and `justification` independently.

---

## ♻️ Processing Pipeline

### 1. Argument Parsing and Model Choice
- Parses optional `--weighted` argument to apply class weighting.
- Prompts user to choose a model and number of epochs interactively.

### 2. Load Dataset
- Loads preprocessed training and validation CSV files.
- Converts `label` column to integer.
- Prints class distribution in training data.

### 3. Tokenization
- Tokenizes `cleaned_statement` and `cleaned_justification` separately.
- Applies truncation, padding, and optional token type IDs (for BERT).
- Prepares HuggingFace `Dataset` format with PyTorch tensors.

### 4. Class Weights (Optional)
- If `--weighted` is passed:
  - Calculates class weights using inverse frequency.
  - Defines custom `WeightedTrainer` with modified `compute_loss` using `F.cross_entropy` and weights.

---

## ⚙️ Training Configuration

- Uses `transformers.TrainingArguments` with:
  - `num_train_epochs` from user input (default: 10)
  - `batch_size = 8`
  - Evaluation & checkpointing per epoch
  - Logging every 10 steps
  - FP16 training if CUDA available
  - Optimizer: `adamw_torch`
  - Metric to track: `eval_loss`

- Model and tokenizer are loaded dynamically and moved to the selected device (CPU/GPU).

---

## 🏋️️ Training Execution

- Trainer class (either `Trainer` or `WeightedTrainer`) is instantiated.
- Includes a callback to save model weights inside each checkpoint folder.
- Starts training with `trainer.train()`.

---

## 🗂️ Output & Saving

- Saves the best model weights and tokenizer to:
  - `models/{model_key}/model/pytorch_model.bin`
- Logs are saved to: `models/{model_key}/logs/train.log`
- Checkpoints in: `models/{model_key}/checkpoints/`

---

## 📌 Notes

- CLI-based interaction (model selection & epoch count).
- Logs are duplicated to a file using a custom `Tee` class.
- Modular and scalable for any HuggingFace-compatible transformer model.
- Final model is saved after training for downstream evaluation or inference.

