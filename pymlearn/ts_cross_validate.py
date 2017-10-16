"""Time series cross validate

"""
from pdb import set_trace as debug
import copy
import pandas as pd


def ts_cross_validate(estimator, data, cv_splitter, perf_score_fn, verbose=False):
    """Cross validate for time series models

    :param estimator: The classification or regression model
    :param data: Data to be tested on
    :param cv_splitter: The splitter that can split the data for CV
    :return:
    """
    orig_estimator = copy.deepcopy(estimator)
    data = orig_estimator.process_data(data)

    y = pd.DataFrame()
    for train_data, val_data in cv_splitter.split(data):

        # Fit the model and
        estimator = copy.deepcopy(orig_estimator)  # NB xheng: estimator is mutable
        estimator.fit(train_data, verbose=verbose)
        y_pred = estimator.predict(val_data)
        y_true = val_data[estimator.get_label_col()]

        y = pd.concat([y, pd.DataFrame.from_dict({
            "n_ahead": 1 + range(len(val_data))
            "true": y_true,
            "pred_num": y_pred["num"]})
                       ])

    performance = pd.DataFrame()
    for y in y.groupby("n_ahead"):
        performance = pd.concat([performance, perf_score_fn(y["true"], y["pred_num"])0)

    return performance
