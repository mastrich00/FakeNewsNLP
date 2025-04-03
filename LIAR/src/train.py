import sys
import os
import argparse
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--weighted", action="store_true", help="Use class weights for imbalanced dataset")
args = parser.parse_args()

# === Select Model ===
print("Choose model to train:")
print("  1 - DistilBERT")
print("  2 - BERT")
print("  3 - RoBERTa")
print("  4 - DeBERTa")
choice = input("Enter model number (1-4): ").strip()

# Ask for number of epochs
epochs_input = input("How many epochs? (default: 10): ").strip()
try:
    num_epochs = float(epochs_input)
    if num_epochs <= 0:
        raise ValueError
except:
    print("⚠️ Invalid input. Using default: 10 epochs.")
    num_epochs = 10.0

model_configs = {
    "1": {"name": "distilbert-base-uncased", "model_class": "DoubleInputBERTModel", "module": "double_input_distilbert"},
    "2": {"name": "bert-base-uncased", "model_class": "DoubleInputBERTModel", "module": "double_input_bert"},
    "3": {"name": "roberta-base", "model_class": "DoubleInputRoBERTaModel", "module": "double_input_roberta"},
    "4": {"name": "microsoft/deberta-v3-base", "model_class": "DoubleInputDebertaModel", "module": "double_input_deberta"}
}

if choice not in model_configs:
    print("❌ Invalid choice. Exiting.")
    sys.exit(1)

config = model_configs[choice]
MODEL_NAME = config["name"]

# === Dynamic import ===
import importlib
model_module = importlib.import_module(f"src.models.{config['module']}")
ModelClass = getattr(model_module, config["model_class"])

# === Paths ===
model_key = MODEL_NAME.split("/")[-1]
base_output_dir = f"models/{model_key}"
log_dir = os.path.join(base_output_dir, "logs")
ckpt_dir = os.path.join(base_output_dir, "checkpoints")
final_model_dir = os.path.join(base_output_dir, "model")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# === Logging ===
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file_path = os.path.join(log_dir, "train.log")
log_file = open(log_file_path, "w", encoding="utf-8", errors="ignore")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"\n📥 Loading LIAR-PLUS dataset for model: {MODEL_NAME}")
train_df = pd.read_csv("data/processed/train_plus.csv").dropna(subset=["label"])
valid_df = pd.read_csv("data/processed/valid_plus.csv").dropna(subset=["label"])
train_df["label"] = train_df["label"].astype(int)
valid_df["label"] = valid_df["label"].astype(int)

print("📊 Class Distribution (Training Set):")
print(train_df["label"].value_counts(normalize=True))

# === Tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")
use_fast = False if "deberta" in MODEL_NAME else True
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=use_fast)

# === Tokenization ===
def tokenize_function(examples):
    statements = [str(x) if pd.notnull(x) else "" for x in examples["cleaned_statement"]]
    justifications = [str(x) if pd.notnull(x) else "" for x in examples["cleaned_justification"]]
    tok1 = tokenizer(statements, truncation=True, padding="max_length", max_length=128)
    tok2 = tokenizer(justifications, truncation=True, padding="max_length", max_length=128)
    result = {
        "input_ids": tok1["input_ids"],
        "attention_mask": tok1["attention_mask"],
        "input_ids_2": tok2["input_ids"],
        "attention_mask_2": tok2["attention_mask"],
        "label": examples["label"]
    }
    if "token_type_ids" in tok1:
        result["token_type_ids"] = tok1["token_type_ids"]
        result["token_type_ids_2"] = tok2["token_type_ids"]
    return result

def tokenize_and_filter(dataset):
    dataset = dataset.map(tokenize_function, batched=True)
    keep_cols = list(tokenize_function(dataset[0]).keys())
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in keep_cols])
    dataset.set_format(type="torch", columns=keep_cols)
    return dataset

print("🔄 Tokenizing datasets...")
train_dataset = tokenize_and_filter(Dataset.from_pandas(train_df))
valid_dataset = tokenize_and_filter(Dataset.from_pandas(valid_df))

# === Optional Weighted Trainer ===
if args.weighted:
    print("🔹 Using class weights")
    label_counts = Counter(train_df["label"])
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    weights = {label: total / (num_classes * count) for label, count in label_counts.items()}
    class_weights = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float).to(device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            return (loss, outputs) if return_outputs else loss

    TrainerClass = WeightedTrainer
else:
    TrainerClass = Trainer

# === Save PyTorch model inside each checkpoint subfolder ===
class SaveModelInCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model_dir = os.path.join(ckpt, "model")
        os.makedirs(model_dir, exist_ok=True)

        model = kwargs["model"]
        torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(model_dir)
        return control

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=ckpt_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir=log_dir,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    report_to="none",
    max_grad_norm=1.0,
)

model = ModelClass(model_name=MODEL_NAME, num_labels=6).to(device)

trainer = TrainerClass(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[SaveModelInCheckpointCallback()]
)

print("🚀 Starting training...")
trainer.train()

print("💾 Saving final model...")
torch.save(model.state_dict(), os.path.join(final_model_dir, "pytorch_model.bin"))
tokenizer.save_pretrained(final_model_dir)
print(f"✅ Final model saved to {final_model_dir}")
