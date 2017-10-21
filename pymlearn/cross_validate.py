"""ML related util functions

"""
from pdb import set_trace as debug
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pymlearn.metrics import clf_perf_scores


def train_validate(estimator, train_data, val_data,
                   perf_score_fn=clf_perf_scores, verbose=False):
    estimator = copy.deepcopy(estimator)  # NB xheng: must do this because estimator is mutable
    estimator.fit(train_data, verbose=verbose)
    y_pred = estimator.predict(val_data)
    y_true = val_data[estimator.get_label_col()]

    y = pd.DataFrame.from_dict({
        "true": y_true,
        "pred_cat": y_pred["pred_cat"].values
    })
    perf_row = perf_score_fn(y["true"], y["pred_cat"])

    if verbose:
        print("Report of out-of-sample performance")
        print(confusion_matrix(y["true"], y["pred_cat"]))
        print(classification_report(y["true"], y["pred_cat"]))

    return y, perf_row


def cross_validate(estimator, data, cv_splitter, perf_score_fn=clf_perf_scores, verbose=False):
    """Cross validate
    
    :param estimator:
    :param data:
    :param cv_folds:
    :return:
    """
    data = estimator.process_data(data)
    performance = pd.DataFrame()  # holds performance metrics from each fold
    fold = 0
    for train_data, val_data in cv_splitter.split(data):
        assert len(train_data) >= 3 * len(val_data)

        # 1 row of performance scores
        _, perf_row = train_validate(estimator, train_data, val_data,
                                  perf_score_fn=perf_score_fn,
                                  verbose=verbose)
        performance = pd.concat([performance, perf_row])
        if verbose:
            print("Completed fold: %s" % fold)
        fold += 1

    mean_perf = pd.DataFrame([{"average": "mean"}])
    for metric in performance.columns:
        mean_perf[metric] = np.mean(performance[metric])

    sd_perf = pd.DataFrame([{"average": "sd"}])
    for metric in performance.columns:
        sd_perf[metric] = np.std(performance[metric])

    if verbose:
        perf = pd.concat([mean_perf, sd_perf]).reset_index(drop=True)
        print("cross_validate(), performance report: \n%s" % perf.to_string(line_width=144))

    return mean_perf, sd_perf
