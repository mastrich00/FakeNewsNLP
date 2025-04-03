# 📘 Preprocessing Script Overview — `preprocess.py`

This script prepares the **LIAR** and **LIAR-PLUS** datasets for training transformer-based models (e.g., BERT, RoBERTa, DeBERTa) for **fake news classification**.

---

## 📂 Input Datasets

- **LIAR**
  - `train.tsv`, `valid.tsv`, `test.tsv`
- **LIAR-PLUS**
  - `train2.tsv`, `val2.tsv`, `test2.tsv`

---

## 🔄 What It Does

1. **Loads** the dataset files from raw `.tsv` format.
2. **Cleans text** fields (`statement` and optionally `justification`) by:
   - Lowercasing
   - Removing digits and punctuation
   - Removing English stopwords
3. **Maps labels** (e.g., `true`, `pants-fire`) to integer class IDs (0–5).
4. **Filters out** rows with missing or empty text/labels.
5. **Saves** the processed splits as CSVs:
   - `data/processed/train.csv`, `valid.csv`, `test.csv` (LIAR)
   - `data/processed/train_plus.csv`, `val_plus.csv`, `test_plus.csv` (LIAR-PLUS)

---

## 💡 Output Format

- **LIAR**
  - Columns: `cleaned_statement`, `label`
- **LIAR-PLUS**
  - Columns: `cleaned_statement`, `cleaned_justification`, `label`

---

## 📆 Label Mapping

| Original Label     | Encoded ID |
|--------------------|-------------|
| true               | 0           |
| mostly-true        | 1           |
| half-true          | 2           |
| barely-true        | 3           |
| false              | 4           |
| pants-fire         | 5           |

---

## 🔧 Cleaning Pipeline

The `clean_text()` function:
- Converts text to lowercase
- Removes digits (`\d+`), punctuation (`[^\w\s]`), and extra spaces
- Filters out stopwords using NLTK's English list

---

## 💡 Usage

Just run the script:
```bash
python preprocess.py
```
It processes all six files (3 from LIAR, 3 from LIAR-PLUS) and stores them under `data/processed/`.

---

## 🔹 Notes
- Based on LIAR dataset README, column headers are corrected during loading.
- Files that don’t match expected column count raise an error.
- Empty or non-informative rows are dropped.
- NLTK stopwords are downloaded automatically if not available.

---

## 📁 File Structure

```
project-root/
├── data/
│   ├── liar/          # Raw LIAR dataset (train.tsv, etc.)
│   ├── liar_plus/     # Raw LIAR-PLUS (train2.tsv, etc.)
│   └── processed/     # Cleaned outputs as CSV
├── preprocess.py
```

---

## 🚀 Example Output Preview

```
📅 Processing train_plus ...
✅ Saved to: data/processed/train_plus.csv — Shape: (8200, 3)
  cleaned_statement              cleaned_justification        label
0    obamacare working...       fact check shows...          0
1    tax breaks for rich...     analysis shows...            4
```

---

## 🚧 Dependencies
- `pandas`, `nltk`, `sklearn`, `re`
- Ensure `nltk.download('stopwords')` has access to download initially

