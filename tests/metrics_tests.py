import pytest
from pymlearn import metrics


def test_regression_perf_scores():
    # Functional tests mostly. No need to delve too much time checking scikit-learn metrics's correctness

    y_true = [1, 1]

    y_pred = [1, 2, 3]
    with pytest.raises(AssertionError):  # Different length
        metrics.regression_perf_scores(y_true, y_pred)

    y_pred = [1, 1]
    with pytest.raises(AssertionError):
        metrics.regression_perf_scores(y_true, y_pred, metrics=[""])

    perf_row = metrics.regression_perf_scores(y_true, y_pred, metrics=["mae", "median_absolute_error"])
    assert len(perf_row) == 1
    assert perf_row["mae"][0] == 0
    assert perf_row["median_absolute_error"][0] == 0

    y_pred = [2, 0]
    perf_row = metrics.regression_perf_scores(y_true, y_pred, metrics=["mae", "mean_error", "rmse"])
    assert perf_row["mae"][0] == 1.0
    assert perf_row["mean_error"][0] == 0
    assert round(perf_row["rmse"][0], 4) == 1.0


def test_clf_perf_scores():
    y_true = [1, 0]

    y_pred = [1, 0, 0]
    with pytest.raises(AssertionError):  # Different length
        metrics.regression_perf_scores(y_true, y_pred)

    y_pred = [1, 1]
    with pytest.raises(AssertionError):
        metrics.clf_perf_scores(y_true, y_pred, metrics=[""])

    perf_row = metrics.clf_perf_scores(y_true, y_pred, metrics=["precision", "recall"])
    assert len(perf_row) == 1
    assert perf_row["precision"][0] == 0.5
    assert perf_row["recall"][0] == 1

    y_pred = [0, 0]
    perf_row = metrics.clf_perf_scores(y_true, y_pred, metrics=["precision", "recall"])
    assert perf_row["precision"][0] == 0
    assert perf_row["recall"][0] == 0
