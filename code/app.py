from RawDataAnalyzer import RawDataAnalyzer
from DataPreprocessor import DataPreprocessor
from FeatureExtractor import FeatureExtractor
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from gensim.models import Word2Vec

import pandas as pd
import numpy as np
import yaml
import os

# Accessing parameters for application to work
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

path_to_features = config["paths"]["features"]
os.makedirs(path_to_features, exist_ok=True)

path_to_results = config["paths"]["results"]
os.makedirs(path_to_results, exist_ok=True)

path_to_tfidf_result_models = config["paths"]["result_models"]["tfidf"]
path_to_w2v_result_models = config["paths"]["result_models"]["word2vec"]
os.makedirs(path_to_tfidf_result_models, exist_ok=True)
os.makedirs(path_to_w2v_result_models, exist_ok=True)

# Accessing original data
path_to_stanford_movie_reviews_dataset = config["paths"]["datasets"][
    "stanford_movie_reviews"
]
path_to_sar14_dataset_txt = config["paths"]["datasets"]["sar14_txt"]
path_to_sar14_dataset = config["paths"]["datasets"]["sar14_csv"]

# Accessing preprocessing parameters
path_to_stanford_movie_reviews_dataset_cleaned = config["paths"]["cleaned_datasets"][
    "stanford_movie_reviews"
]
path_to_sar14_dataset_cleaned = config["paths"]["cleaned_datasets"]["sar14"]

data_is_analyzed = config["analysis"]["data_is_analyzed"]
data_is_preprocessed = config["preprocessing"]["data_is_preprocessed"]
spacy_batch_size = config["preprocessing"]["spacy"]["batch_size"]
spacy_n_process = config["preprocessing"]["spacy"]["n_process"]

# Feature extraction parameters
balance_datasets = config["feature_extraction"]["balance_datasets"]

tfidf_enabled = config["feature_extraction"]["tfidf"]["enabled"]
word2vec_enabled = config["feature_extraction"]["word2vec"]["enabled"]

path_to_tfidf_vectorizer = config["paths"]["models"]["tfidf_vectorizer"]
path_to_w2v_model = config["paths"]["models"]["word2vec"]

# Model training parameters
sample_test_size = config["training"]["test_size"]
training_number = config["training"]["training_number"]
model_random_state = config["training"]["random_state"]

logistic_regression_trained = config["training"]["models"][
    "logistic_regression_trained"
]
naive_bayes_trained = config["training"]["models"]["naive_bayes_trained"]
random_forest_trained = config["training"]["models"]["random_forest_trained"]
linear_svc_trained = config["training"]["models"]["linear_svc_trained"]

# Evaluation
evaluation_enabled = config["evaluation"]["enabled"]

# To avoid the repetition of preprocessing
if data_is_preprocessed:
    path_to_stanford_movie_reviews_dataset = (
        path_to_stanford_movie_reviews_dataset_cleaned
    )
    path_to_sar14_dataset = path_to_sar14_dataset_cleaned


def analyze_dataset(dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Analyzes the initial dataset to print its shape, column names and to highlight null values if they are.

    Args:
    dataset (pd.DataFrame): The dataset to analyze.
    dataset_name (str): The name of dataset to be analyzed for clear output.
    """
    dataset_analyzer = RawDataAnalyzer(dataset=dataset, dataset_name=dataset_name)
    dataset_analyzer.print_dataset_shape()
    dataset_analyzer.print_have_null()
    dataset_analyzer.print_column_names()
    print("\n")


def preprocess_dataset(
    data_preprocessor: DataPreprocessor,
    dataset: pd.DataFrame,
    column_to_preprocess: str,
    path_to_save: str,
    dataset_name: str,
) -> pd.DataFrame:
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
    cleaned_dataset = data_preprocessor.preprocess_data_batch(
        dataset=dataset, column_to_preprocess=column_to_preprocess
    )
    cleaned_dataset.to_csv(path_to_save, index=False, encoding="utf-8")
    print(f"{dataset_name} is preprocessed!")
    return cleaned_dataset


if __name__ == "__main__":
    stanford_dataset = pd.read_csv(path_to_stanford_movie_reviews_dataset)
    sar14_dataset = pd.read_csv(path_to_sar14_dataset)

    # Analysis of datasets
    if not data_is_analyzed:
        analyze_dataset(
            dataset=stanford_dataset, dataset_name="Stanford's Movie Review dataset"
        )
        analyze_dataset(dataset=sar14_dataset, dataset_name="SAR14 dataset")
        print("Data analysis accomplished.")

    # Noise cleaning from data
    data_preprocessor = DataPreprocessor(
        spacy_batch_size=spacy_batch_size, spacy_n_process=spacy_n_process
    )
    if not data_is_preprocessed:
        stanford_dataset = preprocess_dataset(
            data_preprocessor=data_preprocessor,
            dataset=stanford_dataset,
            column_to_preprocess="review",
            path_to_save=path_to_stanford_movie_reviews_dataset_cleaned,
            dataset_name="Stanford's Movie Review dataset",
        )
        sar14_dataset = preprocess_dataset(
            data_preprocessor=data_preprocessor,
            dataset=sar14_dataset,
            column_to_preprocess="review",
            path_to_save=path_to_sar14_dataset_cleaned,
            dataset_name="SAR14 dataset",
        )
        config["preprocessing"]["data_is_preprocessed"] = True
    print("Data preprocessing accomplished.")

    data = pd.concat([stanford_dataset, sar14_dataset], ignore_index=True)
    data["sentiment"] = data["sentiment"].map({"positive": 1, "negative": 0})
    # data = data[:5000]

    if balance_datasets:
        data_positive = data[data["sentiment"] == 1]
        data_negative = data[data["sentiment"] == 0]

        data_positive_downsampled = resample(
            data_positive, replace=False, n_samples=len(data_negative), random_state=0
        )
        balanced_data = pd.concat(
            [pd.DataFrame(data_positive_downsampled), data_negative]
        )
        balanced_data = balanced_data.sample(frac=1, random_state=0)

        x, y = balanced_data["review"], balanced_data["sentiment"]
    else:
        x, y = data["review"], data["sentiment"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=sample_test_size, random_state=model_random_state, stratify=y
    )

    # Feature extraction process & model training
    feature_extractor = FeatureExtractor()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()

    # TF-IDF FEATURES
    if tfidf_enabled:
        # Feature extraction
        x_train_tfidf, x_test_tfidf = feature_extractor.extract_tfidf_features(
            x_train=x_train, x_test=x_test
        )
        feature_extractor.save_tfidf_vectorizer(
            path_to_vectorizer=path_to_tfidf_vectorizer
        )
        print("TF-IDF features extracted.")

        # Model training

        for iteration in range(training_number):
            tfidf_training_parameters = [
                x_train_tfidf,
                y_train,
                model_random_state + iteration,
            ]
            models = {}
            if not logistic_regression_trained:
                models["logistic_regression"] = model_trainer.train_log_reg(
                    *tfidf_training_parameters
                )
                model_trainer.save_model(
                    model=models["logistic_regression"],
                    path=path_to_tfidf_result_models,
                    filename="logistic_regression",
                )
            if not naive_bayes_trained:
                models["multinomial_naive_bayes"] = model_trainer.train_multinomial_nb(
                    *tfidf_training_parameters
                )
                model_trainer.save_model(
                    model=models["multinomial_naive_bayes"],
                    path=path_to_tfidf_result_models,
                    filename="multinomial_naive_bayes",
                )
            if not random_forest_trained:
                models["random_forest"] = model_trainer.train_random_forest(
                    *tfidf_training_parameters
                )
                model_trainer.save_model(
                    model=models["random_forest"],
                    path=path_to_tfidf_result_models,
                    filename="random_forest",
                )
            if not linear_svc_trained:
                models["linear_svc"] = model_trainer.train_linear_svc(
                    *tfidf_training_parameters
                )
                model_trainer.save_model(
                    model=models["linear_svc"],
                    path=path_to_tfidf_result_models,
                    filename="linear_svc",
                )

            # Evaluation
            for model_name, model in models.items():
                model_evaluator.evaluate_model(
                    model=model,
                    x_test=x_test_tfidf,
                    y_test=y_test,
                    model_name=model_name,
                    feature_type="tf_idf",
                )
        print("Model training on TF-IDF features accomplished.")

    # WORD2VEC FEATURES
    if word2vec_enabled:
        reviews_train = x_train.tolist()
        reviews_test = x_test.tolist()
        word2vec_model = Word2Vec(
            sentences=reviews_train,
            vector_size=100,
            window=8,
            min_count=1,
            workers=4,
            seed=model_random_state,
        )
        x_train_w2v = np.array(
            [
                feature_extractor.extract_word2vec_features(
                    word2vec_model=word2vec_model, review=review
                )
                for review in reviews_train
            ]
        )
        x_test_w2v = np.array(
            [
                feature_extractor.extract_word2vec_features(
                    word2vec_model=word2vec_model, review=review
                )
                for review in reviews_test
            ]
        )
        word2vec_model.save(path_to_w2v_model)
        print("Word2Vec features extracted.")

        # Model training
        for iteration in range(training_number):
            word2vec_training_parameters = [
                x_train_w2v,
                y_train,
                model_random_state + iteration,
            ]
            models = {}
            if not logistic_regression_trained:
                models["logistic_regression"] = model_trainer.train_log_reg(
                    *word2vec_training_parameters
                )
                model_trainer.save_model(
                    model=models["logistic_regression"],
                    path=path_to_w2v_result_models,
                    filename="logistic_regression",
                )
            if not naive_bayes_trained:
                models["gaussian_naive_bayes"] = model_trainer.train_gaussian_nb(
                    *word2vec_training_parameters
                )
                model_trainer.save_model(
                    model=models["gaussian_naive_bayes"],
                    path=path_to_w2v_result_models,
                    filename="gaussian_naive_bayes",
                )
            if not random_forest_trained:
                models["random_forest"] = model_trainer.train_random_forest(
                    *word2vec_training_parameters
                )
                model_trainer.save_model(
                    model=models["random_forest"],
                    path=path_to_w2v_result_models,
                    filename="random_forest",
                )
            if not linear_svc_trained:
                models["linear_svc"] = model_trainer.train_linear_svc(
                    *word2vec_training_parameters
                )
                model_trainer.save_model(
                    model=models["linear_svc"],
                    path=path_to_w2v_result_models,
                    filename="linear_svc",
                )
            # Evaluation
            for model_name, model in models.items():
                model_evaluator.evaluate_model(
                    model=model,
                    x_test=x_test_w2v,
                    y_test=y_test,
                    model_name=model_name,
                    feature_type="word2vec",
                )

    # Saving evaluation results
    # model_evaluator.save_final_reports(path_to_results)
    # model_evaluator.save_confusion_matrices(path_to_results)
    model_evaluator.save_metric_comparison_plots(path_to_results)
    model_evaluator.save_metric_confidence_table(path_to_results)
    print("Model evaluation accomplished.")
