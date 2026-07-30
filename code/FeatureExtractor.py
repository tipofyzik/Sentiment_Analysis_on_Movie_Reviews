from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from scipy.sparse import spmatrix
import pandas as pd
import numpy as np
import joblib



class FeatureExtractor:
    """
    A class responsible for extracting numerical features from textual data.

    The class supports two feature extraction approaches:
        - TF-IDF vectorization for representing text based on word importance.
        - Word2Vec embeddings for representing text as dense numerical vectors.

    Extracted features can be used for training machine learning models for
    text classification tasks.

    Attributes:
        __tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer used
            to transform text data into sparse feature matrices.
    """
    
    def __init__(self):
        """
        Initializes the FeatureExtractor object.

        The TF-IDF vectorizer is initialized during feature extraction because
        it requires training data to learn the vocabulary and calculate weights.
        """
        pass
    
    def extract_tfidf_features(self, x_train: pd.Series, 
                               x_test: pd.Series) -> tuple[spmatrix, spmatrix]:
        """
        Extracts TF-IDF features from training and testing text data.

        Args:
            x_train (pd.Series): Training text samples.
            x_test (pd.Series): Testing text samples.

        Returns:
            tuple[spmatrix, spmatrix]:
                A tuple containing:
                    - TF-IDF sparse matrix for training data.
                    - TF-IDF sparse matrix for testing data.
        """
        self.__tfidf_vectorizer = TfidfVectorizer(stop_words='english', 
                                           max_features = 5000,
                                           ngram_range=(1, 3),  #(1,2) gives almost the same result
                                           min_df=5,
                                           max_df=0.8,
                                           token_pattern = r'\b\w[\w\'-]*\b')
        
        x_train_tfidf = self.__tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = self.__tfidf_vectorizer.transform(x_test)    
        return x_train_tfidf, x_test_tfidf

    def extract_word2vec_features(self, word2vec_model: Word2Vec,
                                  review: str) -> np.ndarray:
        """
        Extracts the most important n-grams from data via Bag-of-Words method

        Args:
            word2vec_model (Word2Vec): A trained Word2Vec model used to obtain
                word embeddings.
            review (str): A preprocessed text review.

        Returns:
            np.ndarray:
                A numerical vector representing the semantic information of
                the input review. The vector size corresponds to the size of
                the Word2Vec embeddings.
        """
        review_tokens = review.split()
        vectors = [word2vec_model.wv[word] for word in review_tokens if word in word2vec_model.wv]
        if len(vectors) == 0:
            return np.zeros(word2vec_model.vector_size)
        return np.mean(vectors, axis=0)



    def save_tfidf_vectorizer(self, path_to_vectorizer: str) -> None:
        """
        Saves the trained TF-IDF vectorizer to a file.

        The saved vectorizer can later be loaded and used to transform new
        custom reviews into the same feature space used during model training.

        Args:
            path_to_vectorizer (str):
                Path where the TF-IDF vectorizer will be stored.
        """
        joblib.dump(self.__tfidf_vectorizer, path_to_vectorizer)

