# Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project for binary sentiment classification of movie reviews using multiple supervised machine learning models and feature extraction techniques.

## 1 Task overview
This project focuses on **binary sentiment analysis** of movie reviews. The goal is to train and evaluate machine learning models capable of predicting the overall sentiment of a user-provided movie review. Each review is classified into one of two categories:
- **Positive**
- **Negative**

The training data is constructed by combining two publicly available datasets:
- **Stanford's Large Movie Review Dataset (IMDb)**
- **SAR14 Dataset**

The merged dataset contains:
| Sentiment | Number of Reviews |
| --------- | ----------------: |
| Positive  |           192,378 |
| Negative  |            96,222 |

The project investigates the influence of different text representations and machine learning algorithms on sentiment classification performance. Two feature extraction techniques (TF-IDF and Word2Vec) are compared across several supervised learning models.  
The final system is also capable of predicting the sentiment of arbitrary user-provided movie reviews through a graphical interface.  

## Datasets
For convenience, the processed datasets used in this project are [available on Google Drive](https://drive.google.com/drive/folders/1ACDrihk3dvMMEIhsVKuf3jsO6Rv7DDbP?usp=drive_link). Simply download them and place them into the project's **`code`** directory before running the application. The Google Drive folder also contains the generated evaluation artifacts, including classification reports and confusion matrices.

## Original Data Sources
The project is based on the following publicly available datasets:
1. [**Stanford's Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/)  
   For convenience, this project uses the equivalent CSV version (IMDb Dataset of 50K Movie Reviews) [**published on Kaggle**](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
3. [**SAR14 Dataset**](https://github.com/daiquocnguyen/SAR14)

## 2 Program installation
## Requirements 
**Language**: Python 3.11.3+  
**Python Modules** can be found in the **"requirements.txt"** file. 

Install all required dependencies:  
```bash
pip install -r requirements.txt
```

## 3 How to use
The project consists of three main scripts that cover the complete machine learning pipeline, from data preparation to model training and evaluation.
- **SAR14ConverterToCSV.py**: Run this script before executing the main application. The original SAR14 dataset is distributed in .txt format, while the project expects a .csv file for efficient processing. This script converts the original dataset into the required format.
- **config.yml**: This configuration file contains all parameters required by the project, including dataset locations, preprocessing options, feature extraction settings, model training parameters, and evaluation options. A detailed description of each parameter is provided in Section 4.2 – Configuration File.
- **app.py**: This is the project's main entry point. It executes the complete machine learning pipeline:
1. loads the datasets;
2. performs optional dataset analysis;
3. preprocesses the text data;
4. extracts features using TF-IDF and/or Word2Vec;
5. trains the selected supervised learning models;
6. evaluates model performance;
7. generates classification reports, confusion matrices, comparison plots, and summary tables.

After the models have been trained, PredictCustomReviewApp.py can be executed. This script launches a graphical user interface that allows users to enter a custom movie review and obtain its predicted sentiment (positive or negative) using one of the trained models.  

## 4 Project structure
```text
├── app.py                       # Main pipeline
├── config.yml                   # Project configuration
├── RawDataAnalyzer.py           # Exploration of the initial dataset
├── DataPreprocessor.py          # Text preprocessing
├── FeatureExtractor.py          # TF-IDF and Word2Vec feature extraction
├── NewModelTrainer.py           # Supervised model training
├── ModelEvaluator.py            # Model evaluation and visualization
├── PredictCustomReviewApp.py    # GUI for custom review prediction
├── SAR14ConverterToCSV.py       # SAR14 dataset conversion
├── datasets/                    # Input datasets
├── models/                      # Saved vectorizers and trained models
└── results/                     # Evaluation reports and plots
```

# 5 Implementation
## 5.1 Data Preprocessing
Text preprocessing is performed in two stages.  
1. lightweight manual preprocessing removes:
- HTML tags;
- punctuation;
- digits;
- stop words;
- extra whitespace.
All text is converted to lowercase before further processing.
2. Reviews are lemmatized using the **spaCy** English language model.

This hybrid approach is approximately **twice as fast** as performing the entire preprocessing pipeline solely through spaCy while producing nearly identical preprocessing quality.  
To improve stop-word removal, the default spaCy stop-word list is extended with additional custom tokens originating from split English contractions (e.g. *wouldn't → wouldn + t*).


## 5.2 Feature Extraction
Two feature extraction techniques are implemented and compared: TF-IDF and Word2Vec

## 5.3 Machine Learning Models
Four supervised learning algorithms are evaluated:
- Logistic Regression
- Linear SVM
- Random Forest
- Naive Bayes

An important implementation detail concerns the Naive Bayes classifier. Multinomial Naive Bayes assumes non-negative feature values, making it appropriate for TF-IDF representations. Word2Vec embeddings, however, naturally contain both positive and negative values. Therefore, Gaussian Naive Bayes is used instead when training on Word2Vec features. This allows fair comparisons while respecting the assumptions of each probabilistic model.

## 5.4 Evaluation Methodology
Model performance is assessed using:
- Accuracy
- Precision
- Recall
- F1-score

Each experiment is repeated multiple times using different random seeds. Instead of reporting the metrics from a single train/test split, the presented results correspond to the **average performance across all experimental runs**. Additionally, **95% confidence intervals** are calculated for every evaluation metric to estimate the variability of the obtained results.  
The evaluation pipeline automatically generates:
- averaged classification reports;
- averaged confusion matrices;
- metric comparison plots;
- confidence interval summary tables.

# 6. Results
## 6.1 Average Performance over 10 Runs (20/80 train/test split)
<table>
  <tr>
    <td width="50%">
      <img width="2955" height="1762" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/56d8043b-71a1-4926-ac3b-476159d40799" />
    </td>
    <td width="50%">
      <img width="2955" height="1762" alt="f1-score_comparison" src="https://github.com/user-attachments/assets/81c11964-61a8-4612-95f5-89262709731b" />
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img width="2955" height="1762" alt="precision_comparison" src="https://github.com/user-attachments/assets/65b9706c-3c2e-4c2e-90df-e75475eef720" />
    </td>
    <td width="50%">
      <img width="2955" height="1762" alt="recall_comparison" src="https://github.com/user-attachments/assets/2d4548d7-0976-49e6-9611-50d0829b414d" />
    </td>
  </tr>
</table> 

<img width="2850" height="1056" alt="model_metrics_confidence_table" src="https://github.com/user-attachments/assets/ba2b9e3c-da16-4646-9297-b7143f07ecd8" />


## 6.2 Average Performance over 10 Runs (70/30 train/test split)
<table>
  <tr>
    <td width="50%">
      <img width="2955" height="1762" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/d40cd2a5-88ec-4971-823f-fae68b4a2476" />
    </td>
    <td width="50%">
      <img width="2955" height="1762" alt="f1-score_comparison" src="https://github.com/user-attachments/assets/7681cc0c-8be0-47b8-8534-818efae1217f" />
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img width="2955" height="1762" alt="precision_comparison" src="https://github.com/user-attachments/assets/a17646fe-c2e9-46a2-a496-f5aff149c92f" />
    </td>
    <td width="50%">
      <img width="2955" height="1762" alt="recall_comparison" src="https://github.com/user-attachments/assets/3564b6ac-6ac5-484f-88e1-036342a979dd" />
    </td>
  </tr>
</table> 

<img width="2850" height="1056" alt="model_metrics_confidence_table" src="https://github.com/user-attachments/assets/a2dfdf8b-f3cd-4dba-8a38-88b1667ed5b1" />

# 7. Conclusions





