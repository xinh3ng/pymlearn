"""Time series sampling functions

"""
from pdb import set_trace as debug
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from pymlearn.metrics import clf_perf_scores
from pydsutils.generic import create_logger

logger = create_logger(__name__, 'info')

##############################################################
# Splitter's output must be a pair of train and validation sets
###############################################################

class CVSplitter(object):
    def __init__(self, folds, shuffle=False):
        self.folds = folds
        self.shuffle = shuffle

    def split(self, data):
        """Split the data into a series of train and validate sets

        Args:
            data: Data to be split on

        Returns:
        """
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle)
        for train_idx, test_idx in  kf.split(data):
            yield data.loc[train_idx].reset_index(drop=True),\
                  data.loc[test_idx].reset_index(drop=True)


def train_validate(estimator, train_data, test_data,
                   perf_score_fn=clf_perf_scores, verbose=0):
    estimator = copy.deepcopy(estimator)  # NB xheng: must do this because estimator is mutable
    estimator.fit(train_data, verbose=verbose)
    y_pred = estimator.predict(test_data)
    y_true = test_data[estimator.get_ycol()]

    y = pd.DataFrame.from_dict({
        'true': y_true,
        'pred_cat': y_pred['pred_cat'].values
    })
    perf_row = perf_score_fn(y['true'], y['pred_cat'])

    if verbose >= 1:
        logger.info('Report of out-of-sample performance')
        logger.info(confusion_matrix(y['true'], y['pred_cat']))
        logger.info(classification_report(y['true'], y['pred_cat']))

    return y, perf_row


def cross_validate(data, estimator, cv_splitter, perf_score_fn=clf_perf_scores, verbose=0):
    """Cross validate

    Args:
        data:
        estimator:
        cv_splitter:
    Returns:

    """
    data = estimator.process_data(data)
    performance = pd.DataFrame()  # holds performance metrics from each fold
    fold = 0
    for train_data, test_data in cv_splitter.split(data):
        assert len(train_data) >= 3 * len(test_data)

        # 1 row of performance scores
        _, perf_row = train_validate(estimator, train_data, test_data,
                                     perf_score_fn=perf_score_fn,
                                     verbose=verbose)
        performance = pd.concat([performance, perf_row])
        if verbose:
            logger.info('Completed fold: %s' % fold)
        fold += 1

    mean_perf = pd.DataFrame([{'average': 'mean'}])
    for metric in performance.columns:
        mean_perf[metric] = np.mean(performance[metric])

    sd_perf = pd.DataFrame([{'average': 'sd'}])
    for metric in performance.columns:
        sd_perf[metric] = np.std(performance[metric])

    if verbose >= 1:
        perf = pd.concat([mean_perf, sd_perf]).reset_index(drop=True)
        logger.info('cross_validate(), performance report: \n%s' % perf.to_string(line_width=144))

    return mean_perf, sd_perf
