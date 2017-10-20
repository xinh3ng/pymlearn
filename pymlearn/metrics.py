"""Scoring of performance metrics

"""
from pdb import set_trace as debug
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


def clf_perf_scores(y_true, y_pred, average="weighted"):

    # A single row of several performance metrics
    perf_row = pd.DataFrame([{
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": fbeta_score(y_true, y_pred, beta=1, average=average)
    }])
    return perf_row
