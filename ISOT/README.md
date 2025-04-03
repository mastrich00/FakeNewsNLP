# ISOT Dataset

This folder contains the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset/), located in the `./input` directory. The dataset is divided into two CSV files:

- **True.csv:** Contains approximately 21.4k real news articles.
- **False.csv:** Contains approximately 23.5k fake news articles.

## About the Dataset

- **Content:**  
The ISOT dataset consists of full-length news articles. Each article is labeled as either true or fake, forming a binary classification task.

- **Used Model**
  DistilBERT is used.

## Setup & Execution

- **Python Version:**  
  Python 3.12 is used for this dataset.

- **Dependencies:**  
  This project uses mainly Tensorflow. It should automatically recognize available GPUs.
  
  Install the required packages with:
  ```sh
  pip install -r requirements.txt
  ```

- **Running the Code:**  
  To execute the analysis and model training, simply open and run the `main.ipynb` notebook.

- **Finetuned Models:**  
  Finetuned models are saved in the `./model` folder for further evaluation or deployment.

  Use the code in loadModels.ipynb to load the trained models.