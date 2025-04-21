from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import joblib



class ModelTrainer:
    """
    Trains various models to determine sentiment of the give review.
    """

    def __init__(self):
        """
        Initializes the ModelTrainer object.
        """

    def train_log_reg(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str) -> None: 
        """
        Trains ligistic regression model and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
        """
        logistic_regression = LogisticRegression(C=1.0, 
                                                 max_iter=1000, 
                                                 random_state=0, 
                                                 solver='liblinear')
        logistic_regression.fit(x_train, y_train)
        y_pred = logistic_regression.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                          path_to_results, 
                                          filename = "logistic_regression_report")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred),
                                     path_to_results,
                                     filename = "logistic_regression_confusion_matrix")
        joblib.dump(logistic_regression, f"{path_to_results}/logistic_regression_model.pkl")

    def train_naive_bayes(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str) -> None:
        """
        Trains Naive Bayes model and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc. It is used for features extracted via TF-IDF algorithm.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
        """
        naive_bayes = MultinomialNB()
        naive_bayes.fit(x_train, y_train)
        y_pred = naive_bayes.predict(x_test)

        report = classification_report(y_test, y_pred,output_dict=True)
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                          path_to_results, 
                                          filename = "naive_bayes_report")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred), 
                                     path_to_results,
                                     filename = "naive_bayes_confusion_matrix")
        joblib.dump(naive_bayes, f"{path_to_results}/naive_bayes_model.pkl")

    def train_gauss_naive_bayes(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str) -> None:
        """
        Trains Gaussian Naive Bayes model and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc. It is used for features extracted via Word2Vec algorithm.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
        """
        naive_bayes = GaussianNB()        
        naive_bayes.fit(x_train, y_train)
        y_pred = naive_bayes.predict(x_test)

        report = classification_report(y_test, y_pred,output_dict=True)
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                          path_to_results, 
                                          filename = "gauss_naive_bayes_report")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred), 
                                     path_to_results,
                                     filename = "gauss_naive_bayes_confusion_matrix")
        joblib.dump(naive_bayes, f"{path_to_results}/gauss_naive_bayes_model.pkl")

    def train_random_forest(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str) -> None:
        """
        Trains random forest model and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
        """
        random_forest_model = RandomForestClassifier(n_estimators=100, 
                                                     max_depth=10, 
                                                     n_jobs=-1,
                                                     random_state=0,
                                                     min_samples_leaf=2,
                                                     max_features='sqrt')
        random_forest_model.fit(x_train, y_train)
        y_pred = random_forest_model.predict(x_test)

        report = classification_report(y_test, y_pred,output_dict=True)
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                          path_to_results, 
                                          filename = "random_forest_report")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred), 
                                     path_to_results,
                                     filename = "random_forest_confusion_matrix")
        joblib.dump(random_forest_model, f"{path_to_results}/random_forest_model.pkl")

    def train_linear_svc(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str) -> None:
        """
        Trains Linear SVM and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
        """
        linear_svc_model = LinearSVC(random_state = 0)
        linear_svc_model.fit(x_train, y_train)
        y_pred = linear_svc_model.predict(x_test)

        report = classification_report(y_test, y_pred,output_dict=True)
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                          path_to_results, 
                                          filename = "linear_svc_report")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred), 
                                     path_to_results,
                                     filename = "linear_svc_confusion_matrix")
        joblib.dump(linear_svc_model, f"{path_to_results}/linear_svc_model.pkl")

    def __save_classification_report(self, report: pd.DataFrame, path_to_results: str, filename: str):
        """
        Saves the classification report as an image.

        Args: 
            report (pd.DataFrame): The report that contains the following metrics: presicion, recall, f1-score,
            accuracy, macro and weighted average scores.
            path_to_results (str): The path where the classification report will be saved.
            filename (str): The name of the final image.
        """
        _, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table = ax.table(cellText=report.values, colLabels=report.columns, rowLabels=report.index, loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.text(0.5, 0.7, filename, fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

        plt.savefig(f"{path_to_results}/{filename}.jpg", bbox_inches='tight', dpi=300)
        plt.close()

    def __save_confusion_matrix(self, confusion_matrix: np.ndarray, path_to_results: str, filename: str):
        """
        Saves the confusion matrix as an image.

        Args: 
            confusion_matrix (np.ndarray): The confusion_matrix.
            path_to_results (str): The path where the confusion_matrix will be saved.
            filename (str): The name of the final image.
        """
        _, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=False, fmt='d', cmap='Blues',
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    cbar=False, ax=ax)

        # Adding annotation:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j + 0.5, i + 0.5, str(confusion_matrix[i, j]),
                        ha='center', va='center', color='black', fontsize=12)

        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(filename)
        ax.tick_params(length=0)

        plt.savefig(f"{path_to_results}/{filename}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

