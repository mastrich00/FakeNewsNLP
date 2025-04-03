# Fake News Detection with LIAR and LIAR-PLUS Datasets

This project focuses on detecting fake news using transformer-based models such as BERT, RoBERTa, DistilBERT, and DeBERTa. It supports both the original LIAR dataset and the enhanced LIAR-PLUS dataset, with functionality for training, evaluation, and visualization.

---

## 📁 Project Structure

```
├── data/
│   ├── liar/
│   │   ├── train.tsv
│   │   ├── valid.tsv
│   │   └── test.tsv
│   └── liar_plus/
│       ├── train2.tsv
│       ├── val2.tsv
│       └── test2.tsv
│
├── models/
│   └── <model-name>/
│       ├── checkpoints/
│       ├── logs/
│       └── model/
│           ├── pytorch_model.bin
│           └── tokenizer config files
│
├── notebooks/
│   └── *.ipynb
│
├── src/
│   ├── models/
│   │   ├── double_input_bert.py
│   │   ├── double_input_deberta.py
│   │   ├── double_input_distilbert.py
│   │   └── double_input_roberta.py
│   ├── preprocess.py
│   ├── train.py
│   ├── more_train.py
│   ├── evaluate.py
│   └── train_all.py
│
├── setup_project.py
├── requirements.txt
└── README.md
```

---

## 📦 Setup Instructions

### Create Environment & Install Dependencies
```bash
conda create -n NLP-env python=3.12
conda activate NLP-env
pip install -r requirements.txt
```

### Requirements
Key packages include:
- torch (CUDA-enabled)
- transformers
- datasets
- scikit-learn
- pandas
- matplotlib
- seaborn
- jupyter

To install PyTorch with CUDA (11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔧 Dataset Preparation

Run the setup script to:
- Create folder structure
- Download and extract the LIAR dataset
- Download the LIAR-PLUS dataset

```bash
python setup_project.py
```

This downloads the TSV files and places them into `data/liar/` and `data/liar_plus/` respectively.

---

## ✂️ Preprocessing

Before training, the raw TSV files must be preprocessed and converted into cleaned CSV files used by the training scripts. This includes:
- Cleaning text fields
- Merging statement and justification
- Formatting labels

Run the following:
```bash
python src/preprocess.py
```
This generates `train.csv`, `valid.csv`, and `test.csv` inside `data/processed/`.

---

## 🚀 Training

### Train a single model interactively:
```bash
python src/train.py              # Unweighted training
python src/train.py --weighted  # Weighted loss for class imbalance
```
Follow prompts to choose model and set number of epochs.

### Train all models in batch:
```bash
python src/train_all.py
```
You'll be prompted to enter epochs and whether to use weighted training per model.

### Stored models

Finetuned models can be found in our Nextcloud share, because of big sizes:

https://cloud.technikum-wien.at/s/t9PtKHKDbpk248r

---

## 🔁 Resume Training

Continue training from a saved checkpoint:
```bash
python src/more_train.py             # Unweighted continuation
python src/more_train.py --weighted  # Continue training with class weights
```

---

## 📊 Evaluation

Run evaluation and view confusion matrix:
```bash
python src/evaluate.py
```
You will be prompted to select the model path to load.

---

## 📓 Jupyter Notebooks

Jupyter notebooks inside `notebooks/` are used for:
- Exploratory data analysis
- Visualizing class distribution
- Viewing model predictions
- Confusion matrices and performance comparison

---

## 💡 Features

- ✅ Support for both LIAR and LIAR-PLUS datasets
- ✅ Token-level double input modeling (statement + justification)
- ✅ Model selection: DistilBERT, BERT, RoBERTa, DeBERTa
- ✅ Balanced training with class weights
- ✅ Checkpoint saving and resuming
- ✅ Evaluation with classification report and confusion matrix

---

## 🔍 Credits

Inspired by the LIAR dataset (UCSB) and LIAR-PLUS extensions.
GitHub source for LIAR-PLUS: https://github.com/Tariq60/LIAR-PLUS

Mehta, D., Dwivedi, A., Patra, A., & Kumar, M. A. (2021). 
A transformer-based architecture for fake news classification. Social Network Analysis and Mining, 11(1), 39. 
https://doi.org/10.1007/s13278-021-00738-y

---

## ✅ License
This project is released for academic use only.

