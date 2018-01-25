"""Time series sampling functions

"""
from pdb import set_trace as debug
from sklearn.model_selection import KFold


##############################################################
# Splitter's output must be a pair of train and validation sets
###############################################################

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


class CVSplitter(object):
    def __init__(self, folds, shuffle=False):
        self.folds = folds
        self.shuffle = shuffle

    def split(self, data):
        """Split the data into a series of train and validate sets

        :param data: Data to be split on
        :return:
        """
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle)
        for train_idx, val_idx in  kf.split(data):
            yield data.loc[train_idx].reset_index(drop=True),\
                  data.loc[val_idx].reset_index(drop=True)
