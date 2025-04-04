**FEVER Dataset**

This folder contains the FEVER (Fact Extraction and VERification) dataset, located in the `./input` directory. The dataset is structured in a single CSV file:

- `fever.csv`: Contains fact-checked claims labeled as either *SUPPORTS*, *REFUTES*, or *NOT ENOUGH INFO*.

### **About the Dataset**

- **Content:**  
  The FEVER dataset consists of short textual claims and corresponding verifications based on Wikipedia evidence. It is a **multi-class classification** problem with three possible labels:
  - **SUPPORTS**: The claim is supported by Wikipedia evidence.
  - **REFUTES**: The claim is contradicted by Wikipedia evidence.
  - **NOT ENOUGH INFO**: There is insufficient evidence to verify the claim.

- **Used Model:**  
  DistilBERT, BERT, and RoBERTa are fine-tuned for classification on this dataset.

### **Setup & Execution**

- **Python Version:**  
  Python 3.12 is used for this dataset.

- **Dependencies:**  
  This project primarily utilizes the `transformers` library. GPU acceleration is automatically utilized if available.

  Install the required packages with:
  ```bash
  pip install -r requirements.txt
  ```

- **Running the Code:**  
  To execute the analysis and model training, simply open and run the `fever_dataset.ipynb` notebook.

- **Fine-tuned Models & Tokenizers:**  
  - Models are stored in the `./saved_model/` directory using `save_pretrained()`.
  - Tokenizers are also stored in `./saved_model/`, ensuring compatibility when reloading models.
  - Structure:
    - `./saved_model/DistilBERT/`
    - `./saved_model/BERT/`
    - `./saved_model/RoBERTa/`

- **Training Checkpoints:**  
  - Intermediate training checkpoints are stored in the `./results/` directory.  
  - Each model has its own results folder based on the `output_dir` parameter in `TrainingArguments`.
  - Checkpoints allow resuming training and tracking progress over epochs.
  - Structure:
    - `./results/distilBERT/`
    - `./results/BERT/`
    - `./results/RoBERTa/`



