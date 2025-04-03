# === Import necessary libraries for model evaluation and visualization ===
import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

# === Suppress UndefinedMetricWarnings ===
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# === Add project root to Python path for module imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Define available model configurations (name, module, class) ===
model_options = {
    "1": ("distilbert-base-uncased", "double_input_distilbert", "DoubleInputBERTModel"),
    "2": ("bert-base-uncased", "double_input_bert", "DoubleInputBERTModel"),
    "3": ("roberta-base", "double_input_roberta", "DoubleInputRoBERTaModel"),
    "4": ("microsoft/deberta-v3-base", "double_input_deberta", "DoubleInputDebertaModel")
}

print("📦 Choose model to evaluate:")
print("  1 - DistilBERT")
print("  2 - BERT")
print("  3 - RoBERTa")
print("  4 - DeBERTa")

# === Prompt user to select which model to evaluate ===
choice = input("Enter model number (1-4): ").strip()


# === Define available model configurations (name, module, class) ===
if choice not in model_options:
    print("❌ Invalid model choice. Exiting.")
    sys.exit(1)


# === Define available model configurations (name, module, class) ===
MODEL_NAME, MODULE_NAME, CLASS_NAME = model_options[choice]
MODEL_KEY = MODEL_NAME.split("/")[-1]
MODEL_PATH = f"models/{MODEL_KEY}/model"
TEST_FILE = "data/processed/test_plus.csv"

# === Import necessary libraries for model evaluation and visualization ===
import importlib

# === Dynamically import selected model module and class ===
model_module = importlib.import_module(f"src.models.{MODULE_NAME}")
ModelClass = getattr(model_module, CLASS_NAME)

# === Load Model and Tokenizer ===
print(f"\n📥 Loading model: {MODEL_NAME}")

# === Load model and its trained weights from disk ===
model = ModelClass(model_name=MODEL_NAME, num_labels=6)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu"))

# === Load tokenizer associated with the model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# === Load test dataset for evaluation ===
test_df = pd.read_csv(TEST_FILE)
test_df["label"] = test_df["label"].astype(int)

# === Define tokenization method for dual input fields (statement & justification) ===
def tokenize_double_input(examples):
    statements = [str(x) if pd.notnull(x) else "" for x in examples.get("cleaned_statement", [])]
    justifications = [str(x) if pd.notnull(x) else "" for x in examples.get("cleaned_justification", [])]
    tok1 = tokenizer(statements, truncation=True, padding="max_length", max_length=128)
    tok2 = tokenizer(justifications, truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": tok1["input_ids"],
        "attention_mask": tok1["attention_mask"],
        "input_ids_2": tok2["input_ids"],
        "attention_mask_2": tok2["attention_mask"]
    }

# === Tokenize Dataset ===
print("🔄 Tokenizing test dataset...")

# === Apply tokenization and prepare the dataset for inference ===
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize_double_input, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "input_ids_2", "attention_mask_2"])

# === Set device (CPU/GPU) for model inference ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Function to perform batch-wise prediction using the model ===
def predict(model, dataset, batch_size=16):
    model.eval()
    all_preds = []
    dataloader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# === Run Predictions ===
print("🚀 Running model on test data...")

# === Run the model to generate predictions on the test dataset ===
y_pred = predict(model, test_dataset)
y_true = test_df["label"].values

# === Evaluation ===
print("✅ Model Evaluation:")

# === Display overall classification accuracy ===
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# === Class Distribution Insight ===
print("\n🔍 Predicted class distribution:")

# === Show distribution of predicted classes for additional insight ===
print(Counter(y_pred))

# === Classification Report (zero_division=0 to suppress warnings) ===

# === Detailed precision, recall, F1-score for each class ===
print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

# === Confusion Matrix Plot ===
plt.figure(figsize=(8, 6))

# === Plot the confusion matrix for visualization of prediction performance ===
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
