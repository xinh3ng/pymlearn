"""Time series cross validate

"""
from pdb import set_trace as debug
import copy
import pandas as pd
from .generic import create_logger

logger = create_logger(__name__)


class TimeSeriesSplitter(object):
    def __init__(self, train_size, n_ahead=1):
        self.train_size = train_size
        self.n_ahead = n_ahead

    def split(self, data):
        """Split the data into a series of train and test sets

        Args:
            data: Data (pandas) to be split on
        Returns:
        """
        data.reset_index(drop=True, inplace=True)
        length = len(data)
        assert length >= self.train_size + self.n_ahead, 'Data size is too small'

        # Rolling window type of data splitting
        for idx in range(self.train_size, length - self.n_ahead + 1):
            train_rows = range(idx - self.train_size, idx)
            test_rows = range(idx, idx + self.n_ahead)
            yield data.loc[train_rows, :], \
                  data.loc[test_rows, :]


def ts_cross_validate(data, estimator, cv_splitter,
                      perf_score_fn=lambda x: 1,
                      verbose=0):
    """Cross validate for time series models

    Args:
        data: Data to be tested on
        estimator: The classification or regression model. Estimator must have process_data, fit, predict and get_ycol
        cv_splitter: The splitter that can split the data for CV
    Returns:

    """
    orig_estimator = copy.deepcopy(estimator)
    data = orig_estimator.process_data(data)

    y = pd.DataFrame()
    for train_data, test_data in cv_splitter.split(data):

        estimator = copy.deepcopy(orig_estimator)  # NB xheng: estimator is mutable
        estimator.fit(train_data)
        y_pred = estimator.predict(test_data)
        y_true = test_data[estimator.get_ycol()]

        y = pd.concat([y, pd.DataFrame.from_dict({
            'n_ahead': range(1, 1 + len(test_data)),
            'true': y_true,
            'pred': y_pred})
                       ])
    if verbose >= 1:
        logger.info("Showing first 10 rows: \n%s" % y.head(10).to_string(line_width=120))

    # Measure performance
    performance = pd.DataFrame()
    for n_ahead, y in y.groupby('n_ahead'):
        df = perf_score_fn(y['true'], y['pred'])
        df['n_ahead'] = n_ahead
        performance = pd.concat([performance, df])
    return performance
