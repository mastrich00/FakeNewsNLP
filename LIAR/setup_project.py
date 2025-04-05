import requests
import zipfile
import os

# === Define folders and base structure ===
folders = [
    "data",
    "notebooks",
    "src",
    "models",
    "logs"
]

files = {
    "README.md": "",
    "requirements.txt": "",
    ".gitignore": "",
    "src/preprocess.py": "",
    "src/train.py": "",
    "src/evaluate.py": ""
}

# === Create project folder structure ===
print("🔧 Creating project structure...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file_path in files:
    if not os.path.exists(file_path):
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("")
        print(f"Created: {file_path}")
    else:
        print(f"Skipped (already exists): {file_path}")
print("✅ Project structure verified.")

# === LIAR Dataset (UCSB) ===
liar_url = "http://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
liar_output_path = "data/liar_dataset.zip"
liar_extract_path = "data/liar"

# === LIAR-PLUS Dataset (raw GitHub) ===
liar_plus_base_url = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/"
liar_plus_files = ["train2.tsv", "val2.tsv", "test2.tsv"]
liar_plus_output_path = "data/liar_plus"

# === Ensure dataset directories exist ===
os.makedirs(liar_extract_path, exist_ok=True)
os.makedirs(liar_plus_output_path, exist_ok=True)

# === Download LIAR ===
if not os.path.exists(liar_output_path):
    print("⬇️  Downloading LIAR dataset...")
    response = requests.get(liar_url, stream=True)
    if response.status_code == 200:
        with open(liar_output_path, "wb") as f:
            f.write(response.content)
        print("✅ LIAR download complete.")
    else:
        print("❌ Error: Could not download LIAR dataset.")
else:
    print("Skipped (already exists): LIAR zip file")

# === Extract LIAR ===
if os.path.exists(liar_output_path) and os.path.getsize(liar_output_path) > 10000:
    print("📦 Extracting LIAR dataset...")
    with zipfile.ZipFile(liar_output_path, "r") as zip_ref:
        zip_ref.extractall(liar_extract_path)
    print("✅ LIAR extraction complete.")
else:
    print("❌ Error: Downloaded LIAR file is too small or corrupted.")

# === Download LIAR-PLUS ===
print("⬇️  Verifying LIAR-PLUS dataset...")
for file_name in liar_plus_files:
    file_path = os.path.join(liar_plus_output_path, file_name)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
        print(f"Skipped (already exists): {file_name}")
        continue

    print(f"⬇️  Downloading {file_name}...")
    file_url = liar_plus_base_url + file_name
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {file_name}")
    else:
        print(f"❌ Error downloading {file_name} from {file_url}")

print("🏁 All setup complete.")
