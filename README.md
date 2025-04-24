# Sentiment Analysis of Movie Reviews

## 1. Task formulation
There are movie reviews collected from the internet. Basically, there are 2 datasets: Standford's Large Movie Review Dataset and SAR14 dataset. The goal is to train and evaluate the model(s) that will analyze the custom movie review and determine its overall sentiment. An outcome of a model should be either positive or negative, i.e., this is a binary classification task.  

I uploaded datasets to google drive, so they can be simply put into the **code** folder for their processing. Datasets are available through this link: https://drive.google.com/drive/folders/1ACDrihk3dvMMEIhsVKuf3jsO6Rv7DDbP?usp=drive_link  

The references to the original datasets:  
1. Standford's Large Movie Review Dataset: https://ai.stanford.edu/~amaas/data/sentiment/  
**However, for convenience, I used its .csv version (IMBD dataset) published on kaggle**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews   
2. SAR14 dataset: https://github.com/daiquocnguyen/SAR14?tab=readme-ov-file  

You can get acquainted with the results on Kaggle: https://www.kaggle.com/code/tipofyzik/sentiment-analysis-of-movie-reviews  

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
Config file contains settings for different stages of data analysis and training. There are 6 parameter categories in this .json file:  
_**· DatasetsPaths:**_  
&emsp;&emsp; — path_to_standford_movie_reviews_dataset: Path to the Standford's Movie Review dataset in .csv format.   
&emsp;&emsp; — path_to_sar14_dataset_txt: Path to the SAR14 dataset in .txt format.   
&emsp;&emsp; — path_to_sar14_dataset_csv: Path to the SAR14 dataset in .csv format.  
_**· DataPreprocessingParameters:**_  
&emsp;&emsp; — path_to_standford_movie_reviews_dataset_cleaned: Path to the cleaned from noise Standford's Movie Review dataset in .csv format (to avoid repetition of the cleaning process).   
&emsp;&emsp; — path_to_sar14_dataset_cleaned: Path to the cleaned from noise SAR14 dataset in .csv format (to avoid repetition of the cleaning process).  
&emsp;&emsp; — data_is_preprocessed: Shows whether data was preprocessed or not. If the value is 0, then the original data have not been preprocessed yet, otherwise it have been (to avoid repetition of the cleaning process).  
&emsp;&emsp; — spacy_batch_size: Determines how many rewiews should be preprocessed simultaneously. It is used to speed up preprocessing.  
&emsp;&emsp; — spacy_n_process: Multiprocessing to speed up preprocessing. If the value is -1, then all cpu cores are used. In this case, I left 1 core to for preprocessing.  
_**· FeatureExtractorParameters:**_  
&emsp;&emsp; — path_to_features: Path to save extracted features (to avoid repetition of the feature extraction process).    
&emsp;&emsp; — path_to_tfidf_vectorizer: Path to the TF-IDF vectorizer that will be used further to prepare custom review for evaluation.   
&emsp;&emsp; — path_to_x_train_tfidf: Path to the train part of the TF-IDF features (to avoid repetition of the feature extraction process).  
&emsp;&emsp; — path_to_x_test_tfidf: Path to the test part of the TF-IDF features (to avoid repetition of the feature extraction process).    
&emsp;&emsp; — path_to_w2v_model: Path to the Word2Vec that will be used further to prepare custom review for evaluation.    
&emsp;&emsp; — path_to_w2v_features: Path to the Word2Vec features (to avoid repetition of the feature extraction process).      
&emsp;&emsp; — tfidf_features_extracted: Shows whether TF-IDF features are extracted or not. If the value is 0, then they have not been extracted yet, otherwise they have been (to avoid repetition of the feature extraction process).     
&emsp;&emsp; — word2vec_features_extracted: Shows whether Word2Vec features are extracted or not. If the value is 0, then they have not been extracted yet, otherwise they have been (to avoid repetition of the feature extraction process).         
_**· ModelTrainerParameters:**_  
&emsp;&emsp; — logistic_regression_trained: Shows whether logistic regression model is trained or not. If the value is 0, then it has not been trained yet, otherwise it has been (to avoid repetition of the training process).           
&emsp;&emsp; — naive_bayes_trained: Shows whether naive bayes model is trained or not. If the value is 0, then it has not been trained yet, otherwise it has been (to avoid repetition of the training process).  
&emsp;&emsp; — random_forest_trained: Shows whether random forest model is trained or not. If the value is 0, then it has not been trained yet, otherwise it has been (to avoid repetition of the training process).  
&emsp;&emsp; — linear_svc_trained: Shows whether linear SVM is trained or not. If the value is 0, then it has not been trained yet, otherwise it has been (to avoid repetition of the training process).  
_**· GraphPlotterParameters:**_  
&emsp;&emsp; —    
_**· ResultSaver:**_  
&emsp;&emsp; — path_to_results: Path to results folder.   
&emsp;&emsp; — path_to_tfidf_result_models: Path to models trained on TF-IDF features.   
&emsp;&emsp; — path_to_w2v_result_models: Path to models trained on Word2Vec features.   

### 4.3. Launch file

### 4.4. Implementation specifics



## 5. Results of the work
### 5.1 Data preparation
<table>
  <tr>
  </tr>
  <tr>
  </tr>
  <tr>
  </tr>
  <tr>
  </tr>
</table>  





