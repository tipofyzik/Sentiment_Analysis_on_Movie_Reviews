## 1. Task formulation
There are movie reviews collected from the internet. Basically, there are 2 datasets: stanford's Large Movie Review Dataset and SAR14 dataset. The goal is to train and evaluate the model(s) that will analyze the custom movie review and determine its overall sentiment. An outcome of a model should be either positive or negative, i.e., this is a binary classification task.  

I uploaded datasets to google drive, so they can be simply put into the **code** folder for their processing. You can also access confusion matrices and classification reports for various cases through this link:  
https://drive.google.com/drive/folders/1ACDrihk3dvMMEIhsVKuf3jsO6Rv7DDbP?usp=drive_link  



The references to the original datasets:  
1. Stanford's Large Movie Review Dataset: https://ai.stanford.edu/~amaas/data/sentiment/  
**However, for convenience, I used its .csv version (IMBD dataset) published on kaggle**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews   
2. SAR14 dataset: https://github.com/daiquocnguyen/SAR14?tab=readme-ov-file  

Overall, there are two types (classes) of review with the following number of samples:  
1. positive: 192,378 reviews  
2. negative: 96,222 reviews  


## 2. Program installation
### Requirements 
You need to intall Python with the version 3.11.3 and higher. All required modules to install you can find in the **"requirements.txt"** file. Download the folder "code" from github and the folder "datasets" from google drive. The latter folder put into the first one.  



## 3. How to use
There are three key files for the program to work: **SAR14ConverterToCSV.py**, **config.json**, and **app.py**.  
**SAR14ConverterToCSV.py** — This file should be run before any other ones. It converts the original SAR14 dataset from .txt format to .csv one, which is more convenient to process by Python.  
**config.json** — This file contains all crucial parameters for the code to work. See **"4.2. Config file"** section to look at what parameters are responsible for.  
**app.py** — This file rins the entire program which, in turn, cleans data, encodes it, trains various supervised learning models and evaluates their quality.  

Additionally, there is a file **PredictCustomReviewApp.py**, which might be run once all models are trained. It creates a window where the user can enter any custom review and get its sentiment as an outcome.  

## 4. Implementation
### 4.1. Classes
There are 4 classes ensure the program works:  
**· RawDataAnalyzer** analyzes the raw dataset to give a hint where the preprocess should start first. It looks for null values and prints the size of the initial datasets and their column names.  
**· DataPreprocessor** cleans the data from stop words and punctuation. Additionally, makes all the text lowercase and lemmatizes it.  
**· FeatureExtractor** extracts textual features from a review dataset. It utilizes two extraction algorithms: TF-IDF and Word2Vec.  
**· ModelTrainer** trains various models to determine sentiment of the give review. Namely, there are four models: Logistic Regression, Linear SVM, Naive Bayes, and Random Forest.  

**· PredictCustomReviewApp** is a separate class which is a GUI application for the sentiment prediction of a custom movie review. It can be run individually after models training.  

### 4.2. Config file
Config file contains settings for different stages of data analysis and training. For further information, read the **config_explanation.txt**. 
### 4.3. Launch file
The **"app.py"** file analyzes data, preprocesses it, extracts features and trains models on these features. Let's go through a step-by-step explanation of what happens there. The program:  
1. Imports all the custom classes and reads parameters from **"config.json"**.  
2. Reads the original dataset and outputs basic information about it to the console.  
3. Preprocesses it by removing noise (punctuation, stopwords, lemmatize data). Preprocessed texts are saved to .csv file to avoid repetiotion of preprocessing.  
4. Extracts features from reviews. There are two feature extraction algorithms: TF-IDF and Word2Vec.   
5. Teaches four models on two types of extracted features (Logistic Regression, Naive Bayes, Random Forest, and Linear SVM).  
  

### 4.4. Implementation specifics
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
10 runs. split 20/80


### Models' accuracy  
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






