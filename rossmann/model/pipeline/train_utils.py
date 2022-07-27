import csv
from pathlib import Path
from typing import Callable, Dict, List, Text
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore


def seed(s: int):
    """Sets seed for numpy"""
    np.random.seed(s)


def write_kaggle_evaluation_file(
    path: Path, predicted_X: pd.DataFrame, ids: pd.Series
) -> None:
    """Writes the predictions to a Kaggle evaluation file.

    Args:
        path: The path to the Kaggle evaluation file.
         Create file "path/test_predictions.csv.
        predicted_X: The predictions to write to the file.
        ids: The ids of the test data.
    """
    y_pred_ids = predicted_X.join(ids)
    y_pred_ids["Sales"] = y_pred_ids["PredictedSales"].round().astype(int)
    y_pred_ids[["Id", "Sales"]].to_csv(
        path / "test_predictions.csv",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def predict(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predicts and add the results as a new colum PredictedSales"""
    y_pred = pd.Series(data=pipeline.predict(X), index=X.index)
    y_pred.name = "PredictedSales"
    X_pred = X.join(y_pred)
    return X_pred


def evaluate(
    pipeline: Pipeline,
    Xs: Dict[Text, pd.DataFrame],
    metrics: Dict[Text, Callable],
) -> List[Dict[Text, float]]:
    """Evaluates the pipeline on the given data sets and metrics.

    Args:
        pipeline: The pipeline to evaluate.
        Xs: A dictionary of data sets to evaluate the pipeline on.
        metrics: A dictionary of metrics to evaluate the pipeline on.
            Must follow the signature of the sklearn.metric function.

    Returns:
        A list of dictionaries with the metrics and their values.
    """

    def _evaluate(
        name: Text, X: pd.DataFrame, metrics: Dict[Text, Callable]
    ) -> Dict[Text, float]:
        x_pred = predict(pipeline, X)
        report = {
            metric_name: metric(x_pred["Sales"], x_pred["PredictedSales"])
            for metric_name, metric in metrics.items()
        }
        report["split"] = name
        return report

    return [_evaluate(name, X, metrics) for name, X in Xs.items()]
