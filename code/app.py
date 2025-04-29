from RawDataAnalyzer import RawDataAnalyzer
from DataPreprocessor import DataPreprocessor
from FeatureExtractor import FeatureExtractor
from ModelTrainer import ModelTrainer

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from gensim.models import Word2Vec

import pandas as pd
import numpy as np
import json
import os



# Accessing parameters for application to work
with open('config.json', 'r') as f:
    config = json.load(f)

path_to_features = config["FeatureExtractorParameters"]["path_to_features"]
os.makedirs(path_to_features, exist_ok=True)

path_to_results = config["ResultSaver"]["path_to_results"]
os.makedirs(path_to_results, exist_ok=True)
path_to_tfidf_result_models = config["ResultSaver"]["path_to_tfidf_result_models"]
path_to_w2v_result_models = config["ResultSaver"]["path_to_w2v_result_models"]
os.makedirs(path_to_tfidf_result_models, exist_ok=True)
os.makedirs(path_to_w2v_result_models, exist_ok=True)



# Accessing original data.
path_to_standford_movie_reviews_dataset = config["DatasetsPaths"]["path_to_standford_movie_reviews_dataset"]
path_to_sar14_dataset = config["DatasetsPaths"]["path_to_sar14_dataset_csv"]

# Accessing processing parameters and already preprocessed data if exists.
path_to_standford_movie_reviews_dataset_cleaned = config["DataPreprocessingParameters"]["path_to_standford_movie_reviews_dataset_cleaned"]
path_to_sar14_dataset_cleaned = config["DataPreprocessingParameters"]["path_to_sar14_dataset_cleaned"]
data_is_preprocessed = config["DataPreprocessingParameters"]["data_is_preprocessed"]
spacy_batch_size = config["DataPreprocessingParameters"]["spacy_batch_size"]
spacy_n_process = config["DataPreprocessingParameters"]["spacy_n_process"]

# To save vectorizers and models for tokenizing custom reviews.
balance_dasets = bool(config["FeatureExtractorParameters"]["balance_dasets"])
path_to_tfidf_vectorizer = config["FeatureExtractorParameters"]["path_to_tfidf_vectorizer"]
path_to_w2v_model = config["FeatureExtractorParameters"]["path_to_w2v_model"]

#Accessing parameters for model training
sample_test_size = config["ModelTrainerParameters"]["sample_test_size"]
training_number = config["ModelTrainerParameters"]["training_number"]
model_random_state = config["ModelTrainerParameters"]["model_random_state"]
logistic_regression_trained = bool(config["ModelTrainerParameters"]["logistic_regression_trained"])
naive_bayes_trained = bool(config["ModelTrainerParameters"]["naive_bayes_trained"])
random_forest_trained = bool(config["ModelTrainerParameters"]["random_forest_trained"])
linear_svc_trained = bool(config["ModelTrainerParameters"]["linear_svc_trained"])



# To avoid the repetition of preprocessing
if data_is_preprocessed:
    path_to_standford_movie_reviews_dataset = path_to_standford_movie_reviews_dataset_cleaned
    path_to_sar14_dataset = path_to_sar14_dataset_cleaned



def analyze_dataset(dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Analyzes the initial dataset to print its shape, column names and to highlight null values if they are.

    Args:
    dataset (pd.DataFrame): The dataset to analyze.
    dataset_name (str): The name of dataset to be analyzed for clear output.
    """
    dataset_analyzer = RawDataAnalyzer(dataset = dataset, dataset_name = dataset_name)
    dataset_analyzer.print_dataset_shape()
    dataset_analyzer.print_have_null()
    dataset_analyzer.print_column_names()
    print("\n")

def preprocess_dataset(data_preprocessor: DataPreprocessor, dataset: pd.DataFrame, 
                       column_to_preprocess: str, path_to_save: str,
                       dataset_name: str) -> pd.DataFrame:
    """
    Cleans column in the dataset from noise such as punctuation, html tags, numbers, etc. 
    After preprocessing, this function saves cleaned dataset to .csv table.

    Args:
        data_preprocessor (DataPreprocessor): The object of the custom DataPreprocessor class that performs data cleaning.
        dataset (pd.DataFrame): The dataset to preprocess.
        column_to_preprocess (str): Column to be cleaned.
        path_to_save (str): The cleaned dataset is saved at this file path. 
        dataset_name (str): The name of dataset to be analyzed for clear output.

    Returns:
        pd.DataFrame: Cleaned from the noise dataset.
    """
    cleaned_dataset = data_preprocessor.preprocess_data_batch(dataset = dataset, column_to_preprocess = column_to_preprocess)
    cleaned_dataset.to_csv(path_to_save, index=False, encoding="utf-8")
    print(f"{dataset_name} is preprocessed!")
    return cleaned_dataset



if __name__ == "__main__":
    standford_dataset = pd.read_csv(path_to_standford_movie_reviews_dataset)
    sar14_dataset = pd.read_csv(path_to_sar14_dataset)



    # Analysis of datasets
    analyze_dataset(dataset = standford_dataset, dataset_name = "Stanford's Movie Review dataset")
    analyze_dataset(dataset = sar14_dataset, dataset_name = "SAR14 dataset")
    print("Data analysis accomplished.")



    # Noise cleaning from data
    data_preprocessor = DataPreprocessor(spacy_batch_size = spacy_batch_size, spacy_n_process = spacy_n_process)
    if not data_is_preprocessed:
        standford_dataset = preprocess_dataset(data_preprocessor = data_preprocessor, 
                                               dataset = standford_dataset, 
                                               column_to_preprocess = "review", 
                                               path_to_save = path_to_standford_movie_reviews_dataset_cleaned,
                                               dataset_name = "Stanford's Movie Review dataset")
        sar14_dataset = preprocess_dataset(data_preprocessor = data_preprocessor, 
                                           dataset = sar14_dataset, 
                                           column_to_preprocess = "review", 
                                           path_to_save = path_to_sar14_dataset_cleaned,
                                           dataset_name = "SAR14 dataset")
        config["DataPreprocessingParameters"]["data_is_preprocessed"] = 1
    print("Data preprocessing accomplished.")

    data = pd.concat([standford_dataset, sar14_dataset], ignore_index=True)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

    if balance_dasets:
        data_positive = data[data['sentiment'] == 1]
        data_negative = data[data['sentiment'] == 0]

        data_positive_downsampled = resample(data_positive,
                                        replace=False, 
                                        n_samples=len(data_negative), 
                                        random_state=0)
        balanced_data = pd.concat([data_positive_downsampled, data_negative])
        balanced_data = balanced_data.sample(frac=1, random_state=0)

        x, y = balanced_data["review"], balanced_data["sentiment"]
    else:
        x, y = data["review"], data["sentiment"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=sample_test_size, 
                                                        random_state = model_random_state, 
                                                        stratify = y)



    # Feature extraction process
    feature_extractor = FeatureExtractor()

    # TF-IDF FEATURES
    x_train_tfidf, x_test_tfidf = feature_extractor.extract_tfidf_features(x_train = x_train, 
                                                                            x_test = x_test)
    feature_extractor.save_tfidf_vectorizer(path_to_vectorizer = path_to_tfidf_vectorizer)

    # WORD2VEC FEATURES
    reviews_train = x_train.tolist()
    reviews_test = x_test.tolist()
    word2vec_model = Word2Vec(sentences=reviews_train, vector_size=150, window=3, min_count=2, workers=4, seed = 0)
    x_train_w2v = np.array([feature_extractor.extract_word2vec_features(word2vec_model = word2vec_model, 
                                                                        review = review) 
                                                                        for review in reviews_train])
    x_test_w2v = np.array([feature_extractor.extract_word2vec_features(word2vec_model = word2vec_model, 
                                                                        review = review) 
                                                                        for review in reviews_test])
    word2vec_model.save(path_to_w2v_model)
    print("Feature extraction accomplished.")



    # Training of various models
    model_trainer = ModelTrainer()

    tfidf_parameters_for_training = [x_train_tfidf, x_test_tfidf, y_train, y_test, 
                                     path_to_tfidf_result_models, model_random_state]
    word2vec_parameters_for_training = [x_train_w2v, x_test_w2v, y_train, y_test, 
                                        path_to_w2v_result_models, model_random_state]

    for iteration in range(training_number):
        if not logistic_regression_trained:
            model_trainer.train_log_reg(*tfidf_parameters_for_training, i = iteration)
            model_trainer.train_log_reg(*word2vec_parameters_for_training, i = iteration)
        if not naive_bayes_trained:
            model_trainer.train_naive_bayes(*tfidf_parameters_for_training, i = iteration)
            model_trainer.train_gauss_naive_bayes(*word2vec_parameters_for_training, i = iteration)
        if not random_forest_trained:
            model_trainer.train_random_forest(*tfidf_parameters_for_training, i = iteration)
            model_trainer.train_random_forest(*word2vec_parameters_for_training, i = iteration)
        if not linear_svc_trained:
            model_trainer.train_linear_svc(*tfidf_parameters_for_training, i = iteration)
            model_trainer.train_linear_svc(*word2vec_parameters_for_training, i = iteration)

    config["ModelTrainerParameters"]["logistic_regression_trained"] = 1
    config["ModelTrainerParameters"]["naive_bayes_trained"] = 1
    config["ModelTrainerParameters"]["random_forest_trained"] = 1
    config["ModelTrainerParameters"]["linear_svc_trained"] = 1
    model_trainer.save_models()
    print("Models training accomplished.")



    with open("./config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
