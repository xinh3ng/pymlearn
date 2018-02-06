"""Time series cross validate

"""
from pdb import set_trace as debug
import copy
import pandas as pd


class TimeSeriesSplitter(object):
    def __init__(self, train_size, n_ahead=1):
        self.train_size = train_size
        self.n_ahead = n_ahead

    def split(self, data):
        """Split the data into a series of train and validate sets

        :param data: Data to be split on
        :return:
        """
        data.reset_index(drop=True, inplace=True)
        length = len(data)
        assert length >= self.train_size + self.n_ahead, 'Data size is too small'

        for idx in range(self.train_size, length - self.n_ahead + 1):
            train_rows = range(idx - self.train_size, idx)
            val_rows = range(idx, idx + self.n_ahead)

            yield data.loc[train_rows].reset_index(drop=True),\
                  data.loc[val_rows].reset_index(drop=True)


def ts_cross_validate(estimator, data,
                      cv_splitter=TimeSeriesSplitter,
                      perf_score_fn=lambda x: 1,
                      verbose=False):
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
            'n_ahead': 1 + range(len(val_data)),
            'true': y_true,
            'pred_num': y_pred['num']})
                       ])

    performance = pd.DataFrame()
    for y in y.groupby('n_ahead'):
        performance = pd.concat([performance, perf_score_fn(y['true'], y['pred_num'])])

    return performance
