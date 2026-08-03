from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

from scipy.sparse import csr_matrix
from typing import Any

import pandas as pd
import joblib
import os


class ModelTrainer:
    """
    Trains machine learning models for sentiment classification.

    This class is responsible only for model training and saving.
    Evaluation and visualization are handled by ModelEvaluator.
    """

    def __init__(self):
        """
        Initializes ModelTrainer.
        """
        self.models = {}

    def train_log_reg(
        self,
        x_train: csr_matrix,
        y_train: pd.Series,
        random_state: int,
        model_name: str = "logistic_regression",
    ):
        """
        Trains Logistic Regression model.

        Args:
            x_train (csr_matrix): Training features.
            y_train (pd.Series): Training labels.
            random_state (int): Random state for reproducibility.
            model_name (str): Name used for saving.

        Returns:
            LogisticRegression: Trained model.
        """

        model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state, solver="liblinear"
        )

        model.fit(x_train, y_train)
        self.models[model_name] = model
        return model

    def train_multinomial_nb(
        self,
        x_train: csr_matrix,
        y_train: pd.Series,
        random_state: int,
        model_name: str = "multinomial_naive_bayes",
    ):
        """
        Trains Multinomial Naive Bayes model.

        This model is intended for TF-IDF features.

        Args:
            x_train (csr_matrix): TF-IDF features.
            y_train (pd.Series): Training labels.

        Returns:
            MultinomialNB: Trained model.
        """

        model = MultinomialNB()
        model.fit(x_train, y_train)
        self.models[model_name] = model
        return model

    def train_gaussian_nb(
        self,
        x_train: csr_matrix,
        y_train: pd.Series,
        random_state: int,
        model_name: str = "gaussian_naive_bayes",
    ):
        """
        Trains Gaussian Naive Bayes model.

        This model is intended for dense numerical features,
        such as Word2Vec embeddings.

        Args:
            x_train (np.ndarray): Dense feature vectors.
            y_train (pd.Series): Training labels.

        Returns:
            GaussianNB: Trained model.
        """
        model = GaussianNB()
        model.fit(x_train, y_train)
        self.models[model_name] = model
        return model

    def train_random_forest(
        self,
        x_train: csr_matrix,
        y_train: pd.Series,
        random_state: int,
        model_name: str = "random_forest",
    ):
        """
        Trains Random Forest classifier.

        Args:
            x_train (csr_matrix): Training features.
            y_train (pd.Series): Training labels.
            random_state (int): Random state for reproducibility.
            model_name (str): Name used for saving.

        Returns:
            RandomForestClassifier: Trained model.
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
        )
        model.fit(x_train, y_train)
        self.models[model_name] = model
        return model

    def train_linear_svc(
        self,
        x_train: csr_matrix,
        y_train: pd.Series,
        random_state: int,
        model_name: str = "linear_svc",
    ):
        """
        Trains Linear Support Vector Classifier.

        Args:
            x_train (csr_matrix): Training features.
            y_train (pd.Series): Training labels.
            random_state (int): Random state for reproducibility.
            model_name (str): Name used for saving.

        Returns:
            LinearSVC: Trained model.
        """
        model = LinearSVC(random_state=random_state)
        model.fit(x_train, y_train)
        self.models[model_name] = model
        return model

    def save_model(self, model: Any, path: str, filename: str) -> None:
        """
        Saves trained model to disk.

        Args:
            model (Any): Trained sklearn model.
            path (str): Directory where model will be saved.
            filename (str): File name without extension.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(model, f"{path}/{filename}.pkl")

    def save_all_models(self, path: str) -> None:
        """
        Saves all trained models.

        Args:
            path (str): Directory where models are stored.
        """
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
