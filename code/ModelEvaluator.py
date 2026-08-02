from sklearn.metrics import classification_report, confusion_matrix

from matplotlib.container import BarContainer
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from collections import defaultdict
from typing import Any
import joblib
import os


class ModelEvaluator:
    """
    Evaluates trained machine learning models.

    This class is responsible for:
    - generating predictions;
    - calculating evaluation metrics;
    - saving reports;
    - plotting confusion matrices;
    - comparing feature extraction methods.

    Model training is handled by ModelTrainer.
    """

    def __init__(self):
        """
        Initializes ModelEvaluator.
        """
        self.reports = defaultdict(list)
        self.confusion_matrices = defaultdict(list)

    def evaluate_model(
        self, model, x_test, y_test, model_name: str, feature_type: str
    ) -> None:
        """
        Evaluates a trained model.

        Args:
            model: Trained sklearn model.
            x_test: Test features.
            y_test: True labels.
            model_name (str): Name of the model.
            feature_type (str): Feature extraction method.
                Example:
                "tf-idf", "word2vec"
        """
        predictions = model.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True)
        key = (model_name, feature_type)

        if key not in self.reports:
            self.reports[key] = []
        if key not in self.confusion_matrices:
            self.confusion_matrices[key] = []

        self.reports[key].append(report)
        self.confusion_matrices[key].append(confusion_matrix(y_test, predictions))

    def load_and_evaluate(
        self, model_path: str, x_test, y_test, model_name: str, feature_type: str
    ) -> None:
        """
        Loads saved model and evaluates it.

        Args:
            model_path (str):
                Path to .pkl model.

            x_test:
                Test features.

            y_test:
                True labels.

            model_name (str):
                Model name.

            feature_type (str):
                Feature type.
        """
        model = joblib.load(model_path)
        self.evaluate_model(model, x_test, y_test, model_name, feature_type)

    def calculate_metric_statistics(
        self,
        reports: list,
        cls: str,
        metric: str,
    ) -> tuple[float | None, float | None]:
        """
        Calculates mean and standard deviation for a metric.
        """

        values = [
            report[cls][metric]
            for report in reports
            if cls in report and isinstance(report[cls], dict) and metric in report[cls]
        ]

        if not values:
            return None, None

        return float(np.mean(values)), float(np.std(values))

    def create_report_row(
        self,
        reports: list,
        cls: str,
        metrics: list[str],
    ) -> dict[str, Any]:
        """
        Creates one row of averaged classification report.
        """
        row: dict[str, Any] = {"class": cls}

        for metric in metrics:
            mean, std = self.calculate_metric_statistics(
                reports,
                cls,
                metric,
            )

            if mean is not None:
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std

        return row

    def save_final_reports(self, path_to_results: str) -> None:
        """
        Saves averaged classification reports.

        Creates one CSV file per:
        model + feature type combination.
        """

        os.makedirs(path_to_results, exist_ok=True)

        metrics = [
            "precision",
            "recall",
            "f1-score",
            "accuracy",
        ]
        classes = [
            "0",
            "1",
            "macro avg",
            "weighted avg",
        ]

        for (model_name, feature_type), reports in self.reports.items():

            rows = [
                self.create_report_row(
                    reports=reports,
                    cls=cls,
                    metrics=metrics,
                )
                for cls in classes
            ]

            accuracy_values = [report["accuracy"] for report in reports]

            rows.append(
                {
                    "class": "accuracy",
                    "accuracy_mean": np.mean(accuracy_values),
                    "accuracy_std": np.std(accuracy_values),
                }
            )

            dataframe = pd.DataFrame(rows)

            filename = f"{model_name}_{feature_type}_summary.csv"

            dataframe.to_csv(
                os.path.join(
                    path_to_results,
                    filename,
                ),
                index=False,
            )

    def save_confusion_matrices(self, path_to_results: str) -> None:
        """
        Saves averaged confusion matrices.
        """
        os.makedirs(path_to_results, exist_ok=True)

        for (model_name, feature_type), matrices in self.confusion_matrices.items():
            average_matrix = np.mean(matrices, axis=0)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                average_matrix,
                annot=True,
                fmt=".1f",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            num_runs = len(matrices)
            plt.title(
                f"Average Confusion Matrix\n"
                f"{model_name} ({feature_type}), n={num_runs}"
            )
            plt.savefig(
                f"{path_to_results}/"
                f"{model_name}_{feature_type}_confusion_matrix.jpg",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def calculate_ci(self, values):
        """
        Calculates 95% confidence interval.
        """
        values = np.array(values)
        mean = np.mean(values)

        if len(values) < 2:
            return mean, 0.0, 0.0

        std = np.std(values, ddof=1)
        n = len(values)
        ci95 = 1.96 * (std / np.sqrt(n))

        return mean, std, ci95

    def save_metric_comparison_plots(self, path_to_results: str) -> None:
        """
        Creates comparison bar plots.

        Compares TF-IDF and Word2Vec
        for the same models.
        """
        os.makedirs(path_to_results, exist_ok=True)

        metrics = ["precision", "recall", "f1-score", "accuracy"]
        model_labels = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "linear_svc": "Linear SVC",
            "multinomial_naive_bayes": "Naive Bayes (Multinomial)",
            "gaussian_naive_bayes": "Naive Bayes (Gaussian)",
        }
        model_order = list(model_labels.values())
        data = []

        for (model_name, feature_type), reports in self.reports.items():
            data.append(
                {
                    "model": model_labels.get(model_name, model_name),
                    "feature": feature_type,
                    "precision": np.mean(
                        [r["weighted avg"]["precision"] for r in reports]
                    ),
                    "recall": np.mean([r["weighted avg"]["recall"] for r in reports]),
                    "f1-score": np.mean(
                        [r["weighted avg"]["f1-score"] for r in reports]
                    ),
                    "accuracy": np.mean([r["accuracy"] for r in reports]),
                }
            )

        dataframe = pd.DataFrame(data)
        dataframe["model"] = pd.Categorical(
            dataframe["model"],
            categories=model_order,
            ordered=True,
        )
        dataframe["feature"] = dataframe["feature"].replace(
            {
                "tf_idf": "TF-IDF features",
                "word2vec": "Word2Vec features",
            }
        )

        num_runs = len(next(iter(self.reports.values())))
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plot = sns.barplot(
                data=dataframe,
                x="model",
                y=metric,
                hue="feature",
                order=model_order,
            )
            for container in plot.containers:
                if hasattr(container, "datavalues") and isinstance(
                    container, BarContainer
                ):
                    plot.bar_label(container, fmt="%.3f", padding=3)

            plt.ylim(0, 1.1)
            plt.title(
                f"Comparison of Average {metric.capitalize()} over {num_runs} runs"
            )
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.legend(bbox_to_anchor=(0.5, 1.15), loc="center", ncol=2)
            plt.tight_layout()
            plt.savefig(
                f"{path_to_results}/{metric}_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def save_metric_confidence_table(self, path_to_results: str) -> None:
        """
        Saves model metrics with 95% confidence intervals as an image table.
        """
        os.makedirs(path_to_results, exist_ok=True)

        model_labels = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "linear_svc": "Linear SVC",
            "multinomial_naive_bayes": "Naive Bayes (Multinomial)",
            "gaussian_naive_bayes": "Naive Bayes (Gaussian)",
        }
        rows = []

        for (model_name, feature_type), reports in self.reports.items():
            accuracy_values = [r["accuracy"] for r in reports]
            precision_values = [r["weighted avg"]["precision"] for r in reports]
            recall_values = [r["weighted avg"]["recall"] for r in reports]
            f1_values = [r["weighted avg"]["f1-score"] for r in reports]
            _, _, accuracy_ci = self.calculate_ci(accuracy_values)
            accuracy_mean = np.mean(accuracy_values)
            _, _, precision_ci = self.calculate_ci(precision_values)
            precision_mean = np.mean(precision_values)
            _, _, recall_ci = self.calculate_ci(recall_values)
            recall_mean = np.mean(recall_values)
            _, _, f1_ci = self.calculate_ci(f1_values)
            f1_mean = np.mean(f1_values)
            rows.append(
                [
                    model_labels.get(model_name, model_name),
                    feature_type,
                    f"{accuracy_mean:.3f} ± {accuracy_ci:.3f}",
                    f"{precision_mean:.3f} ± {precision_ci:.3f}",
                    f"{recall_mean:.3f} ± {recall_ci:.3f}",
                    f"{f1_mean:.3f} ± {f1_ci:.3f}",
                ]
            )

        columns = [
            "Model",
            "Features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ]
        dataframe = pd.DataFrame(rows, columns=columns)
        dataframe["Features"] = dataframe["Features"].replace(
            {
                "tf_idf": "TF-IDF",
                "word2vec": "Word2Vec",
            }
        )
        _, ax = plt.subplots(figsize=(12, 0.5 * len(dataframe)))
        ax.axis("off")

        table = ax.table(
            cellText=dataframe.to_numpy().tolist(),
            colLabels=dataframe.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        # Setting column widths and text alignment for better readability
        for (row, col), cell in table.get_celld().items():
            if col == 0:
                cell.set_width(0.24)
            elif col == 1:
                cell.set_width(0.1)
            else:
                cell.set_width(0.16)
            if col in [0, 1]:
                cell.set_text_props(ha="left")

        table.scale(1, 1.8)
        num_runs = len(next(iter(self.reports.values())))
        plt.title(
            f"Average Model Performance over {num_runs} runs\n"
            "Mean ± 95% Confidence Interval",
            y=0.95,
        )
        plt.savefig(
            f"{path_to_results}/model_metrics_confidence_table.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
