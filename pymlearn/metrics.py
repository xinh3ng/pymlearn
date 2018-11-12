from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
)

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def regression_perf_scores(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str] = ["mae", "mean_error", "median_absolute_error", "rmse"]
) -> pd.DataFrame:
    """Regression performance scores"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert y_true.shape == y_pred.shape
    assert all([x in ["mae", "mean_error", "median_absolute_error", "rmse"] for x in metrics])

    perf = dict()
    if "mae" in metrics:
        perf["mae"] = mean_absolute_error(y_true, y_pred)

    if "mean_error" in metrics:
        # A good measure to see if positive and negative errors cancel out
        perf["mean_error"] = np.mean(y_pred - y_true)

    if "rmse" in metrics:
        perf["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

    if "median_absolute_error" in metrics:
        perf["median_absolute_error"] = median_absolute_error(y_true, y_pred)

    perf_row = pd.DataFrame([perf])
    return perf_row


def clf_perf_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"],
    average: str = "binary",
) -> pd.DataFrame:
    """Classification performance scores
    Args:
        y_true: True y values
        y_pred: Predicted y values
        average: String of averaging schemes
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert y_true.shape == y_pred.shape
    assert all([x in ["accuracy", "precision", "recall", "f1"] for x in metrics])

    perf = dict()
    if "accuracy" in metrics:
        perf["accuracy"] = accuracy_score(y_true, y_pred)

    if "precision" in metrics:
        perf["precision"] = precision_score(y_true, y_pred, average=average)

    if "recall" in metrics:
        perf["recall"] = recall_score(y_true, y_pred, average=average)

    if "f1" in metrics:
        perf["f1"] = fbeta_score(y_true, y_pred, beta=1, average=average)

    perf_row = pd.DataFrame([perf])
    return perf_row
