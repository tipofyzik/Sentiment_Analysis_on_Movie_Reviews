import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ResultsVisualizer:

    def __init__(self, results_path: str):
        self.results_path = results_path


    def load_results(self):
        """
        Loads all model summary csv files.
        Expected format:
        model_feature_summary.csv

        Example:
        logistic_regression_tfidf_summary.csv
        logistic_regression_word2vec_summary.csv
        """

        results = {}

        for filename in os.listdir(self.results_path):

            if not filename.endswith("_summary.csv"):
                continue

            path = os.path.join(self.results_path, filename)

            df = pd.read_csv(path)

            parts = filename.replace("_summary.csv", "").split("_")

            feature = parts[-1]
            model = "_".join(parts[:-1])

            results[(model, feature)] = df

        return results


    def plot_metric_comparison(
            self,
            metric: str,
            save_path: str
    ):

        """
        Creates comparison plot for one metric.

        Example:
        precision:
            Logistic Regression   TF-IDF vs Word2Vec
            Random Forest         TF-IDF vs Word2Vec
        """

        results = self.load_results()

        models = sorted(
            set(
                model
                for model, feature in results.keys()
            )
        )


        tfidf_values = []
        w2v_values = []


        for model in models:

            tfidf_df = results.get(
                (model, "tf-idf")
            )

            w2v_df = results.get(
                (model, "word2vec")
            )


            def extract_value(df):

                if df is None:
                    return 0

                row = df[df["class"] == "weighted avg"]

                return float(
                    row[f"{metric}_mean"].iloc[0]
                )


            tfidf_values.append(
                extract_value(tfidf_df)
            )

            w2v_values.append(
                extract_value(w2v_df)
            )


        x = np.arange(len(models))

        width = 0.35


        fig, ax = plt.subplots(
            figsize=(10,6)
        )


        tfidf_bars = ax.bar(
            x - width/2,
            tfidf_values,
            width,
            label="TF-IDF"
        )


        w2v_bars = ax.bar(
            x + width/2,
            w2v_values,
            width,
            label="Word2Vec"
        )


        ax.set_ylabel(metric.capitalize())

        ax.set_title(
            f"{metric.capitalize()} comparison: TF-IDF vs Word2Vec"
        )


        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                model.replace("_", " ").title()
                for model in models
            ],
            rotation=30,
            ha="right"
        )


        ax.set_ylim(
            0,
            1
        )


        # легенда выше
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2
        )


        # числа над столбцами
        self.__add_labels(
            ax,
            tfidf_bars
        )

        self.__add_labels(
            ax,
            w2v_bars
        )


        plt.tight_layout()


        plt.savefig(
            f"{save_path}/{metric}_comparison.jpg",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()



    def __add_labels(self, ax, bars):

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )


visualizer = ResultsVisualizer("./results")

os.makedirs("./results/report", exist_ok=True)
visualizer.plot_metric_comparison(
    metric="precision",
    save_path="./results/report"
)

visualizer.plot_metric_comparison(
    metric="recall",
    save_path="./results/report"
)

visualizer.plot_metric_comparison(
    metric="f1-score",
    save_path="./results/report"
)



# visualizer = ModelVisualizer()


# models_tfidf = [
#     "logistic_regression",
#     "naive_bayes",
#     "random_forest",
#     "linear_svc"
# ]


# tfidf_results = visualizer.load_results(
#     "./results",
#     "tf-idf",
#     models_tfidf
# )


# visualizer.plot_metrics(
#     tfidf_results,
#     "TF-IDF",
#     "./results/tfidf_comparison.jpg"
# )