import tkinter as tk
from tkinter import ttk, messagebox

from DataPreprocessor import DataPreprocessor
from gensim.models import Word2Vec

from scipy.sparse import csr_matrix
import sklearn

import numpy as np
import joblib
import json



with open('config.json', 'r') as f:
    config = json.load(f)

spacy_batch_size = config["DataPreprocessingParameters"]["spacy_batch_size"]
spacy_n_process = config["DataPreprocessingParameters"]["spacy_n_process"]
path_to_tfidf_vectorizer = config["FeatureExtractorParameters"]["path_to_tfidf_vectorizer"]
path_to_w2v_model = config["FeatureExtractorParameters"]["path_to_w2v_model"]

path_to_tfidf_result_models = config["ResultSaver"]["path_to_tfidf_result_models"]
path_to_w2v_result_models = config["ResultSaver"]["path_to_w2v_result_models"]

data_preprocessor = DataPreprocessor(spacy_batch_size = spacy_batch_size, spacy_n_process = spacy_n_process)
tfidf_vectorizer = joblib.load(path_to_tfidf_vectorizer)

tfidf_logistic_regression_model = joblib.load(f"{path_to_tfidf_result_models}/logistic_regression_model.pkl")
tfidf_naive_bayes_model = joblib.load(f"{path_to_tfidf_result_models}/naive_bayes_model.pkl")
tfidf_random_forest_model = joblib.load(f"{path_to_tfidf_result_models}/random_forest_model.pkl")
tfidf_linear_svc_model = joblib.load(f"{path_to_tfidf_result_models}/linear_svc_model.pkl")

w2v_logistic_regression_model = joblib.load(f"{path_to_w2v_result_models}/logistic_regression_model.pkl")
w2v_gauss_naive_bayes_model = joblib.load(f"{path_to_w2v_result_models}/gauss_naive_bayes_model.pkl")
w2v_random_forest_model = joblib.load(f"{path_to_w2v_result_models}/random_forest_model.pkl")
w2v_linear_svc_model = joblib.load(f"{path_to_w2v_result_models}/linear_svc_model.pkl")



class PredictCustomReviewApp:
    """
    A GUI application for predicting the sentiment of a custom movie review.

    Allows the user to select a machine learning model and feature extraction method
    (TF-IDF or Word2Vec), enter a review, and receive a sentiment prediction (positive/negative).
    """

    def __init__(self, root):
        """
        Initializes the GUI components and layout.

        Args:
            root (tk.Tk): The root window for the Tkinter interface.
        """
        self.root = root
        self.root.title("Movie Review Sentiment Classifier")
        self.root.geometry("600x400")
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))

        self.model_var = tk.StringVar()
        self.feature_var = tk.StringVar()

        # Кэш для загруженных моделей
        self.models = {}
        self.vectorizers = {}

        self.create_widgets()
        
        self.review_entry.bind("<Control-a>", self.select_all)
        self.review_entry.bind("<Control-A>", self.select_all)

    def select_all(self, event) -> str:
        """
        Selects all text in the review entry widget when Ctrl+A is pressed.

        Args:
            event: The key event.

        Returns:
            str: "break" to prevent default behavior.
        """
        self.review_entry.tag_add("sel", "1.0", "end")  # выделение всего текста
        return "break"

    def create_widgets(self) -> None:
        """
        Creates and places the GUI widgets including dropdowns, input field,
        run button, and result display.
        """
        tk.Label(self.root, text="Select Model:").pack(pady=(10, 0))
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=[
            "LogisticRegression", "Linear SVM", "RandomForest", "NaiveBayes"
        ])
        self.model_dropdown.pack()

        tk.Label(self.root, text="Select Feature Method:").pack(pady=(10, 0))
        self.feature_dropdown = ttk.Combobox(self.root, textvariable=self.feature_var, values=[
            "TF-IDF", "Word2Vec"
        ])
        self.feature_dropdown.pack()

        tk.Label(self.root, text="Write a Review:").pack(pady=(10, 0))
        self.review_entry = tk.Text(self.root, height=6, width=50)
        self.review_entry.pack(pady=(0, 10))

        self.run_button = tk.Button(self.root, text="Run", command=self.run_prediction)
        self.run_button.pack(pady=5)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)



    def run_prediction(self) -> None:
        """
        Handles the full review processing pipeline:
        preprocessing, feature extraction, model prediction,
        and displaying the sentiment result.

        Displays a warning if the review field is empty or if feature extraction fails.
        """
        model_name = self.model_var.get()
        feature_method = self.feature_var.get()
        review_text = [self.review_entry.get("1.0", tk.END).strip()]
        review_text = data_preprocessor.preprocess_custom_reviews(review_text)

        if not review_text:
            messagebox.showwarning("Warning", "Please enter a review.")
            return

        model = self.load_model(model_name, feature_method)
        features = self.extract_features(review_text, feature_method)

        if features is None:
            self.result_label.config(text="Error in feature extraction", fg="red")
            return
        
        prediction = model.predict(features)[0]
        prediction = "negative" if prediction == 0 else "positive"
        self.result_label.config(text=f"Sentiment: {prediction.capitalize()}", fg="blue")

    def load_model(self, model_name, feature_method) -> sklearn.base.BaseEstimator:
        """
        Loads a pre-trained classification model based on the selected name and feature method.

        Args:
            model_name (str): The name of the model (e.g., "LogisticRegression").
            feature_method (str): The selected feature extraction method ("TF-IDF" or "Word2Vec").

        Returns:
            sklearn.base.BaseEstimator: The loaded machine learning model.
        """
        model_files = {
            "LogisticRegression": "logistic_regression_model.pkl", 
            "NaiveBayes" : "naive_bayes_model.pkl",
            "GaussNaiveBayes" : "gauss_naive_bayes_model.pkl",
            "RandomForest": "random_forest_model.pkl", 
            "Linear SVM": "linear_svc_model.pkl"
            }
        feature_method_paths = {
            "TF-IDF" : path_to_tfidf_result_models, 
            "Word2Vec" : path_to_w2v_result_models
        }
        if feature_method == "Word2Vec" and model_name == "NaiveBayes":
            model_name = "GaussNaiveBayes"

        prediction_model = joblib.load(f"{feature_method_paths[feature_method]}/{model_files[model_name]}")
        return prediction_model

    def extract_features(self, text, method):
        """
        Extracts feature vectors from the input text using the selected method.

        Args:
            text (list of str): List of processed review texts (usually with one element).
            method (str): The feature extraction method ("TF-IDF" or "Word2Vec").

        Returns:
            np.ndarray or sparse matrix: The extracted feature vector(s).
        """
        if method == "TF-IDF":
            return self.extract_tfidf(text)
        elif method == "Word2Vec":
            return self.extract_word2vec(text)
        else:
            return None

    def extract_tfidf(self, review: str) -> csr_matrix:
        """
        Transforms the input text into a feature vector using a TF-IDF vectorizer.

        Args:
            review (str): A single preprocessed review.

        Returns:
            sparse matrix: TF-IDF feature vectors.
        """
        features = tfidf_vectorizer.transform(review)
        return features

    def extract_word2vec(self, text: list[str]) -> np.ndarray:
        """
        Transforms the input text into a feature vector using a Word2Vec model.

        Args:
            text (list[str]): List of processed review texts (usually with one element).

        Returns:
            np.ndarray: A 1D averaged Word2Vec feature vector.
        """
        w2v_model = Word2Vec.load(path_to_w2v_model)
        features = self.extract_word2vec_features(w2v_model, text)
        return features.reshape(1, -1)

    def extract_word2vec_features(self, word2vec_model: Word2Vec, review: str) -> np.ndarray:
        """
        Computes the averaged Word2Vec vector from the tokens of a single review.

        Args:
            word2vec_model (gensim.models.Word2Vec): The loaded Word2Vec model.
            review (str): A single preprocessed review.

        Returns:
            np.ndarray: The averaged Word2Vec feature vector.
        """
        review_tokens = review[0].split()
        vectors = [word2vec_model.wv[word] for word in review_tokens if word in word2vec_model.wv]
        if len(vectors) == 0:
            return np.zeros(word2vec_model.vector_size)
        return np.mean(vectors, axis=0)



if __name__ == "__main__":
    root = tk.Tk()
    app = PredictCustomReviewApp(root)
    root.mainloop()