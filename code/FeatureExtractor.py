from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec

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

