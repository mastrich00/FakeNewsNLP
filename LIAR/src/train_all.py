# import subprocess
# import sys
# import os

# # Path to train.py (relative to root)
# train_script = os.path.join("src", "train.py")

# # Model options: key = menu number, value = (name, display label)
# models_to_train = {
    # "1": ("distilbert-base-uncased", "DistilBERT"),
    # "2": ("bert-base-uncased", "BERT"),
    # "3": ("roberta-base", "RoBERTa"),
    # "4": ("microsoft/deberta-v3-base", "DeBERTa")
# }

# # Dictionary to hold user input for each model's epoch count
# epoch_schedule = {}

# print("🔁 Batch training setup for all models...\n")

# # === Ask user once per model how many epochs ===
# for option, (model_name, label) in models_to_train.items():
    # while True:
        # user_input = input(f"⏱️  How many epochs to train {label}? (e.g., 10.0): ").strip()
        # try:
            # num_epochs = float(user_input)
            # if num_epochs <= 0:
                # raise ValueError
            # epoch_schedule[option] = num_epochs
            # break
        # except:
            # print("❌ Invalid input. Please enter a number greater than 0.")

# # === Now run all trainings in sequence ===
# print("\n🚀 Starting training for all selected models...\n")

# for option, (model_name, label) in models_to_train.items():
    # epochs = epoch_schedule.get(option)
    # print(f"\n▶️ Training [{label}] for {epochs} epochs...\n")
    
    # simulated_input = f"{option}\n{epochs}\n"
    
    # try:
        # result = subprocess.run(
            # [sys.executable, train_script],
            # input=simulated_input,
            # text=True
        # )
        # if result.returncode != 0:
            # print(f"❌ Error during training of {label}.")
        # else:
            # print(f"✅ Finished training for {label}.\n")
    # except Exception as e:
        # print(f"⚠️ Failed to train {label}: {e}")

# print("\n🎉 All batch trainings completed.")



import subprocess
import sys
import os

# Path to train.py (relative to root)
train_script = os.path.join("src", "train.py")

# Model options: key = menu number, value = (name, display label)
models_to_train = {
    "1": ("distilbert-base-uncased", "DistilBERT"),
    "2": ("bert-base-uncased", "BERT"),
    "3": ("roberta-base", "RoBERTa"),
    "4": ("microsoft/deberta-v3-base", "DeBERTa")
}

# Store user input per model
training_schedule = {}

print("🔁 Batch training setup for all models...\n")

# === Ask for training configuration per model ===
for option, (model_name, label) in models_to_train.items():
    while True:
        epoch_input = input(f"⏱️  How many epochs to train {label}? (e.g., 10.0): ").strip()
        try:
            epochs = float(epoch_input)
            if epochs <= 0:
                raise ValueError
            break
        except:
            print("❌ Invalid input. Please enter a number greater than 0.")
    
    weight_input = input(f"⚖️  Use class weights for {label}? (y/n, default: n): ").strip().lower()
    weighted_flag = "--weighted" if weight_input == "y" else ""
    
    training_schedule[option] = (epochs, weighted_flag)

# === Run training for each model ===
print("\n🚀 Starting training for all selected models...\n")

for option, (model_name, label) in models_to_train.items():
    epochs, weighted_flag = training_schedule[option]
    print(f"\n▶️ Training [{label}] for {epochs} epochs... {'(weighted)' if weighted_flag else '(unweighted)'}\n")
    
    simulated_input = f"{option}\n{epochs}\n"
    
    try:
        result = subprocess.run(
            [sys.executable, train_script, weighted_flag],
            input=simulated_input,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Error during training of {label}.")
        else:
            print(f"✅ Finished training for {label}.\n")
    except Exception as e:
        print(f"⚠️ Failed to train {label}: {e}")

print("\n🎉 All batch trainings completed.")
