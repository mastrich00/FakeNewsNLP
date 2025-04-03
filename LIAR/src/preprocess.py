# ================================================================
# Script: preprocess_datasets.py
# Description:
#   This script preprocesses the LIAR and LIAR-PLUS datasets 
#   for fake news classification using transformer-based models.
#
# What it does:
#  1. Loads train/validation/test splits of both datasets.
#  2. Cleans the 'statement' text (lowercase, remove digits, 
#     punctuation, and English stopwords).
#  3. If using LIAR-PLUS, it also cleans the 'justification' field.
#  4. Maps textual labels ('true', 'false', etc.) to numerical IDs (0–5).
#  5. Saves cleaned datasets into CSV format for modeling.
#
# Output:
#   data/processed/
#       - train.csv, valid.csv, test.csv         (from LIAR)
#       - train_plus.csv, val_plus.csv, test_plus.csv (from LIAR-PLUS)
#
# Use case:
#   These preprocessed files are ready for input into transformer models 
#   like BERT or multi-input architectures (e.g., Double or Triple BERT) 
#   as proposed by Mehta et al.
# ================================================================


# ================================================================
# Script: preprocess_datasets.py
# Description:
#   Preprocesses LIAR and LIAR-PLUS datasets for fake news classification.
#   Corrects for missing headers based on official dataset README.
# ================================================================

import pandas as pd
import re
import os
import nltk
from sklearn.model_selection import train_test_split

# Download NLTK stopwords if not already
nltk.download('stopwords')
from nltk.corpus import stopwords

# === Paths and config ===
DATASET_BASE_PATHS = {
    "liar": "data/liar/",
    "liar_plus": "data/liar_plus/"
}

DATASET_FILES = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "test": "test.tsv",
    "train_plus": "train2.tsv",
    "valid_plus": "val2.tsv",
    "test_plus": "test2.tsv"
}

PROCESSED_PATH = "data/processed/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# === Label mapping ===
LABEL_MAP = {
    "true": 0,
    "mostly-true": 1,
    "half-true": 2,
    "barely-true": 3,
    "false": 4,
    "pants-fire": 5
}

# === Text cleaning ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in text.split() if w not in stop_words])

# === Load dataset (with fixed columns from README) ===
def load_data(file_path, is_plus=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # From official LIAR README
    base_columns = [
        "id", "label", "statement", "subject", "speaker", "job", "state",
        "party", "barely_true", "false", "half_true", "mostly_true",
        "pants_fire", "context"
    ]
    if is_plus:
        base_columns.insert(0, "liar_index")  # extra index column at the start
        base_columns.append("justification")

    df = pd.read_csv(file_path, delimiter="\t", header=None)

    if df.shape[1] < len(base_columns):
        raise ValueError(f"❌ {file_path} has only {df.shape[1]} columns, expected at least {len(base_columns)}.")

    df.columns = base_columns[:df.shape[1]]
    return df

# === Preprocess and save ===
def preprocess_and_save(dataset_type, use_plus=False):
    base = "liar_plus" if use_plus else "liar"
    file_key = f"{dataset_type}_plus" if use_plus else dataset_type
    file_path = os.path.join(DATASET_BASE_PATHS[base], DATASET_FILES[file_key])

    print(f"\n📥 Processing {file_key} ...")
    df = load_data(file_path, is_plus=use_plus)

    df = df.dropna(subset=["label", "statement"])
    df["cleaned_statement"] = df["statement"].apply(clean_text)

    if use_plus and "justification" in df.columns:
        df = df.dropna(subset=["justification"])
        df["cleaned_justification"] = df["justification"].apply(clean_text)
        df = df[df["cleaned_justification"].str.strip() != ""]

    df = df[df["cleaned_statement"].str.strip() != ""]

    df = df[df["label"].isin(LABEL_MAP.keys())]
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    save_cols = ["cleaned_statement", "label"]
    if use_plus and "cleaned_justification" in df.columns:
        save_cols.append("cleaned_justification")

    output_file = f"{file_key}.csv"
    output_path = os.path.join(PROCESSED_PATH, output_file)
    df[save_cols].to_csv(output_path, index=False)

    print(f"✅ Saved to: {output_path} — Shape: {df[save_cols].shape}")
    print(df[save_cols].head(2))

# === Run all ===
if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        preprocess_and_save(split, use_plus=False)   # LIAR
        preprocess_and_save(split, use_plus=True)    # LIAR-PLUS
    print("\n🎉 All LIAR and LIAR-PLUS datasets processed successfully!")