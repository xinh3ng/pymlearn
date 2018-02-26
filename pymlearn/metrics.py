"""Performance metrics functions

"""
from pdb import set_trace as debug
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


def clf_perf_scores(y_true, y_pred, metrics=['accuracy'], average='weighted'):
    """Classification performance scores

    Args:
        y_true: True y values
        y_pred: Predicted y values
        average: String of averaging schemes
    """
    # A single row of several performance metrics
    perf = dict()
    if 'accuracy' in metrics:
        perf['accuracy'] = accuracy_score(y_true, y_pred)

    if 'precision' in metrics:
        perf['precision'] = precision_score(y_true, y_pred, average=average)

    if 'recall' in metrics:
        perf['recall'] = recall_score(y_true, y_pred, average=average)

    if 'f1' in metrics:
        perf['f1'] = fbeta_score(y_true, y_pred, beta=1, average=average)

    perf_row = pd.DataFrame([perf])
    return perf_row
