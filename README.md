## 1. Task overview
This project focuses on **binary sentiment analysis** of movie reviews. The objective is to train and evaluate machine learning models capable of predicting the overall sentiment of a user-provided movie review. Each review is classified into one of two categories:
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

### Datasets
For convenience, the processed datasets used in this project are [available on Google Drive](https://drive.google.com/drive/folders/1ACDrihk3dvMMEIhsVKuf3jsO6Rv7DDbP?usp=drive_link). Simply download them and place them into the project's **`code`** directory before running the application. The Google Drive folder also contains the generated evaluation artifacts, including classification reports and confusion matrices.

### Original Data Sources
The project is based on the following publicly available datasets:
1. [**Stanford's Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/)  
   For convenience, this project uses the equivalent CSV version (IMDb Dataset of 50K Movie Reviews) [**published on Kaggle**](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
3. [**SAR14 Dataset**](https://github.com/daiquocnguyen/SAR14)

## 2. Program installation
### Requirements 
You need to intall Python with the version 3.11.3 and higher. All required modules to install you can find in the **"requirements.txt"** file. Download the folder "code" from github and the folder "datasets" from google drive. The latter folder put into the first one.  

## 3. How to use
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

## 4. Project structure
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

### Models
Logistic Regression, Naive Bayes, Random Forest, and Linear SVM  

### 4.1. Implementation specifics
**Data preprocessing:**  
Data preprocessing is accomplished in two steps:  
1. Manual cleaning from the noise. Text is lowercased and cleaned from the html tags, numbers, punctuation, and stop words.  
2. Lemmatiztion via spacy nlp model.  
This approach is appriximately **two times faster** than cleaning data via the spacy nlp model only, while the quality of cleaning process is almost preserved.  

To provide an accurate remove of stop words the set of custom stop words is added. The choice of words can be explained by the following logic. The first thing that takes place is cleaning from the punctuation, therefore contractions, such as wouldn't, he'll, etc. are divided into two parts which are not read as stop words by the original static set from the spacy library. Thus, we expand the original list by possible beginnings and endings (see **self.__custom_stop_words** variable).  

```python
  def __init__(self, spacy_batch_size: int, spacy_n_process: int):
      """
      Initializes the DataPreprocessor with the provided batch size for lemmatization 
      and the number of processes for spacy model.

      Args:
          spacy_batch_size (int): The batch size for faster lemmatization.
          spacy_n_process (int): The number of processes for spacy model.
      """
      self.__nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
      self.__stop_words = set(spacy.load("en_core_web_sm").Defaults.stop_words)  
      self.__custom_stop_words = {"t", "ll", "s", "d", 
                                  "couldn", "wouldn", "mightn", "mayn", 
                                  "don", "doesn"}
      self.__stop_words.update(self.__custom_stop_words)
      ... # The remained part of the function
```

**Data training:**  
There are four trained models: Linear Regression, Linear SVM, Random Forest, and Naive Bayes. Additionally, there are two algorithms utilized to extract features: TF-IDF and Word2Vec. This is done to train model on different features and compare these models in their accuracy and generalization performance.  

However, an issue arised during the training process. Namely, Word2Vec features may contain negative values, while TF-IDF features can not. It affects the training of the Naive Bayes model. As a result, to process TF-IDF features Multinomial Naive Bayes is used, while for Word2Vec features Gaussian Naive Bayes is utilized.  

## 5. Results of the work
### Models' performance 10 runs. split 20/80
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



### Models' performance 10 runs. split 70/30
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







