from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec

from scipy.sparse import csr_matrix
from scipy import sparse

import pandas as pd
import numpy as np
import joblib



class FeatureExtractor:
    """
    A class for extracting textual features from a review dataset.

    Attributes:
        __dataset_for_feature_extraction (pd.DataFrame): The dataset used for feature extraction.
        __nlp (spacy.language.Language): The NLP model for text processing.
    """
    
    def __init__(self):
        """
        Initializes the TextFeatureExtractor object.
        """
    
    def extract_tfidf_features(self, x_train: pd.Series, x_test: pd.Series) -> tuple[pd.DataFrame]:
        """
        Extracts the most important n-grams from data via tf-idf metric.

        Args:

        Returns:
        """  
        self.__tfidf_vectorizer = TfidfVectorizer(stop_words='english', 
                                           max_features = 5000,
                                           ngram_range=(1, 3),  #(1,2) gives almost the same result
                                        #    min_df=5,
                                        #    max_df=0.8,
                                        #    binary=False, 
                                           token_pattern = r'\b\w[\w\'-]*\b')
        
        x_train_tfidf = self.__tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = self.__tfidf_vectorizer.transform(x_test)    
        return x_train_tfidf, x_test_tfidf

    def extract_word2vec_features(self, word2vec_model: Word2Vec,
                                  review: str) -> tuple[pd.DataFrame]:
        """
        Extracts the most important n-grams from data via Bag-of-Words method

        Args:

        Returns:
        """
        review_tokens = review.split()
        vectors = [word2vec_model.wv[word] for word in review_tokens if word in word2vec_model.wv]
        if len(vectors) == 0:
            return np.zeros(word2vec_model.vector_size)
        return np.mean(vectors, axis=0)



    def save_tfidf_vectorizer(self, path_to_vectorizer: str) -> None:
        """
        Saves tf-idf vectorizer for further utilization, e.g., for evaluation of custom review(s).

        Args:
            path_to_vectorizer (str): The path where the vectorizer will be saved.
        """
        joblib.dump(self.__tfidf_vectorizer, path_to_vectorizer)

    def save_tfidf_features(self, x_train: csr_matrix, 
                      x_test: csr_matrix,
                      path_to_x_train: str,
                      path_to_x_test: str) -> None:
        """
        Saves features etracted via TF-IDF algorithm. They are used to avoid repetition of the feature extraction process.

        Args:
            x_train (csr_matrix): Matrix with features extracted from the train part of the dataset. 
            x_text (csr_matrix): Matrix with features extracted from the test part of the dataset.
            path_to_x_train (str): The path where features of the train part of the dataset will be saved.
            path_to_x_test (str): The path where features of the test part of the dataset will be saved.
        """
        sparse.save_npz(path_to_x_train, x_train)
        sparse.save_npz(path_to_x_test, x_test)

    def load_tfidf_features(self, path_to_x_train: str,
                      path_to_x_test: str) -> tuple[csr_matrix]:
        """

        path_to_x_train (str): The path from which features of the train part of the dataset will be loaded.
        path_to_x_test (str): The path from which features of the test part of the dataset will be loaded.

        Retuns:
            tuple[csr_matrix]: Features for the training and testing of various ML models.
        """
        x_train = sparse.load_npz(path_to_x_train)
        x_test = sparse.load_npz(path_to_x_test)
        return x_train, x_test
