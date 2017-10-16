from pdb import set_trace as debug
import numpy as np
import pandas as pd


class BaseClassifierRegressor(object):
    """Base classifier / regressor for the entire library

    """
    def __init__(self, label_col, feature_cols, model_params={}, n_cores=1):
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.model_params = model_params
        self.n_cores = n_cores

    def get_label_col(self):
        return self.label_col

    def get_feature_cols(self):
        return self.feature_cols

    def set_feature_cols(self, feature_cols):
        self.feature_cols = feature_cols

    def fit(self, data):
      raise NotImplementedError("Not implemented")

    def predict(self, data):
        raise NotImplementedError("Not implemented")

    def summary(self):
        return None

    def process_data(self, data):
        return data


class EnsembleClassifierRegressor(object):
    def __init__(self, estimators, label_col, avg_fn=np.mean):
        self.estimators = estimators
        self.label_col = label_col
        self.avg_fn = avg_fn

    def get_label_col(self):
        return self.label_col

    def process_data(self, data):
        return data

    def fit(self, data, verbose=False):
        for estimator in self.estimators:
            estimator.fit(data, verbose=verbose)
        return self

    def summary(self):
        for estimator in self.estimators:
            estimator.summary()
        return


class EnsembleRegressor(EnsembleClassifierRegressor):
    """Ensemble several regressors

    """
    def __init__(self,  estimators, label_col, avg_fn=np.mean):
        super(EnsembleRegressor, self).__init__(estimators, label_col, avg_fn)


    def predict(self, data):
        pred = pd.DataFrame()
        for estimator in self.estimators:
            pred = pd.concat(pred[estimator.predict(data)["pred_num"]], axis=1)
        # Tale avg across rows
        debug()
        self.avg_fn


class EnsembleClassifier(EnsembleClassifierRegressor):
    """Ensemble several regressors

    """
    def __init__(self,  estimators, label_col, avg_fn=np.mean):
        super(EnsembleRegressor, self).__init__(estimators, label_col, avg_fn)


    def predict(self, data):
        pred = pd.DataFrame()
        for estimator in self.estimators:
            pred = pd.concat(pred[estimator.predict(data)["pred_num"]], axis=1)
        # Tale avg across rows
        debug()
        self.avg_fn
