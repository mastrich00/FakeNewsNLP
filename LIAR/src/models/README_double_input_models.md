# 🧠 Model Architectures Overview — `DoubleInput*Model`

This document describes the **shared architecture** and **model-specific details** of the four DoubleInput transformer-based models used in the fake news classification project.

---

## 📐 General Structure (Common to All)

All models follow a **Double Input architecture**, where each sample has two fields:
- `cleaned_statement`
- `cleaned_justification`

### 🧱 Pipeline
1. Each input is passed independently through a **pretrained transformer encoder**.
2. The `[CLS]` (or first token) embedding is extracted from each.
3. The two embeddings are **concatenated**.
4. A dropout layer is applied.
5. A linear classifier maps the combined vector to **6 output classes**.

### 🧾 Forward Inputs
All models accept the following inputs:
- `input_ids`, `attention_mask` — for `statement`
- `input_ids_2`, `attention_mask_2` — for `justification`
- Optionally: `token_type_ids` if required (BERT only)
- Optionally: `labels` to compute training loss

---

## 🔍 Individual Model Details

### 1️⃣ `DoubleInputBERTModel`
- **Backbone**: `BertModel`
- **Supports**: `token_type_ids`
- **File**: `double_input_bert.py`
- **Tokenizer**: BERT-style (uncased)

### 2️⃣ `DoubleInputDistilBERTModel` (same class name)
- **Backbone**: `AutoModel` initialized with `distilbert-base-uncased`
- **No token_type_ids**
- **Lighter** and faster than full BERT
- **File**: `double_input_distilbert.py`
- **Dropout**: 0.3 (higher than others)

### 3️⃣ `DoubleInputRoBERTaModel`
- **Backbone**: `RobertaModel`
- **No token_type_ids**
- **File**: `double_input_roberta.py`
- Tokenizer handles spacing and BPE more robustly

### 4️⃣ `DoubleInputDebertaModel`
- **Backbone**: `DebertaV2Model`
- **Advanced contextual encoding** (disentangled attention)
- **File**: `double_input_deberta.py`
- Slightly different tokenizer behavior (no fast tokenizer recommended)

---

## 🎯 Output
All models return a dictionary:
```python
{
  "logits": logits,           # Always present
  "loss": loss (optional)     # Only if labels are passed
}
```

---

## ⚠️ Notes
- All models use `CrossEntropyLoss` for classification.
- `token_type_ids` must be passed **only** for BERT.
- Model class names are consistent across implementations to simplify dynamic loading.

---

## 🧩 Compatibility
- Fully compatible with HuggingFace tokenizers and `Trainer` API.
- Used in training via `train.py`, continued training via `more_train.py`, and evaluation via `evaluate.py`.

---

## 📎 Model Files Summary
| Model     | Class File                    | Class Name               |
|-----------|-------------------------------|--------------------------|
| BERT      | `double_input_bert.py`        | `DoubleInputBERTModel`   |
| DistilBERT| `double_input_distilbert.py`  | `DoubleInputBERTModel`   |
| RoBERTa   | `double_input_roberta.py`     | `DoubleInputRoBERTaModel`|
| DeBERTa   | `double_input_deberta.py`     | `DoubleInputDebertaModel`|

