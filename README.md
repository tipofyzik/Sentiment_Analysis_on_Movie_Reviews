# Sentiment Analysis of Movie Reviews

## 1. Task formulation
There are movie reviews collected from the internet. Basically, there are 2 datasets: Standford's Large Movie Review Dataset and SAR14 dataset. The goal is to train and evaluate the model(s) that will analyze the custom movie review and determine its overall sentiment. An outcome of a model should be either positive or negative, i.e., it is a binary classification task.  

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
**config.json** — This file contains all crucial parameters for the code to work. See **"4.1. Config file"** section to look at what parameters are responsible for.  
**app.py** — This file rins the entire program which, in turn, cleans data, encodes it, trains various supervised learning models and evaluates their quality.  

Additionally, there is a file **PredictCustomReviewApp.py**, which might be run once all models are trained. It creates a window where the user can enter any custom review and get its sentiment as an outcome.  

## 4. Implementation
### 4.1. Config file

### 4.2. Launch file

### 4.3. Implementation specifics



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





