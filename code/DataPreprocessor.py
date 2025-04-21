from bs4 import BeautifulSoup
import pandas as pd
import spacy
import re

import time



class DataPreprocessor:
    """
    Cleans the data from stop words and punctuation. Additionally, makes all the text lowercase and lemmatizes it.

    Attributes:
        __nlp (spacy.lang.en.English): The spaCy model for preprocessing the dataset.
        __stop_words (set): The set of stop words from spaCy model used in a custom data cleaning. 
                            Being used separately, it helps to speed up data processing.
        __custom_stop_words (set): Custom stop words derived by practice to make custom preprocessing more accurate.
        __spacy_batch_size (int): The batch size for faster lemmatization.
        __spacy_n_process (int): The number of processes for spacy model.
    """

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
        self.__spacy_batch_size = spacy_batch_size
        self.__spacy_n_process = spacy_n_process



    def __preprocess_data_spacy(self, texts: list[str]) -> list[str]:
        """
        Preprocesses data by making text lowercase, removing html tags, stop words, and punctuation. 
        Stop words and punctuation is determined only by spacy model.  

        Args:
            texts (list[str]): List of texts, i.e., rows in the table, to preprocess.

        Returns:
            list[str]: Preprocessed texts.
        """
        for text in texts:
            text = text.lower()
            text = BeautifulSoup(text, "html.parser").get_text()

        lemmatized_texts = []
        for doc in self.__nlp.pipe(texts, batch_size=self.__spacy_batch_size, n_process=self.__spacy_n_process):
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            lemmatized_texts.append(' '.join(tokens))

        return lemmatized_texts
    


    def __remove_noise_from_data(self, text: str) -> str:
        """
        Makes all the text lowercase and removes html tags, punctuation, and stop words from the dataset.

        Args:
            text(str): Text to clean noise from. 

        Returns:
            str: Cleaned text.
        """
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove numbers and punctuation.
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        tokens = text.split()
        tokens = [word for word in tokens if word not in self.__stop_words]
        text = " ".join(tokens)

        return text

    def __lemmatize_data_batch(self, texts: list[str]) -> list[str]:
        """
        Lemmatizes data via nlp.pipe. 

        Args:
            texts (list[str]): List of texts, i.e., rows in the table, to lemmatize.

        Returns:
            list[str]: Lemmatized texts.
        """
        lemmatized_texts = []
        for doc in self.__nlp.pipe(texts, batch_size=self.__spacy_batch_size, n_process=self.__spacy_n_process):
            lemmatized = " ".join([token.lemma_ for token in doc])
            lemmatized_texts.append(lemmatized)

        return lemmatized_texts



    def preprocess_data_batch(self, dataset: pd.DataFrame, column_to_preprocess: str) -> pd.DataFrame:
        """
        Preprocesses dataset by removing noise from data and lemmatizing it. 
        Additionally, computes time spend for preprocessing the entire dataset.

        Args:
            dataset (pd.DataFrame): Dataset to preprocess.
            column_to_preprocess (str): The name of the column to be preprocessed by class.

        Returns:
            pd.DataFrame: Result dataset.
        """
        start_time = time.time()

        # processed_texts = self.__preprocess_data_spacy(dataset[column_to_preprocess].tolist())

        dataset[column_to_preprocess] = dataset[column_to_preprocess].apply(self.__remove_noise_from_data)
        processed_texts = self.__lemmatize_data_batch(dataset[column_to_preprocess].tolist())

        dataset[column_to_preprocess] = processed_texts

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}m {int(execution_time % 60)}s")

        return dataset
    
    def preprocess_custom_reviews(self, texts: list[str]) -> list[str]:
        """
        Preprocesses custom review(s) for sentiment evaluation by an already trained model.

        Args:
            texts (list[str]): The list of custom review(s) to be preprocessed. 

        Returns: 
            list[str]: The list of cleaned for sentiment evaluation reviews.
        """
        cleaned_texts = [self.__remove_noise_from_data(text = text) for text in texts]
        cleaned_texts = self.__lemmatize_data_batch(cleaned_texts)
        return cleaned_texts
    
