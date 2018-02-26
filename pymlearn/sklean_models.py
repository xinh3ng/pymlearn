from pdb import set_trace as debug
import numpy as np
import pandas as pd


class BaseClassifierRegressor(object):
    """Base classifier / regressor for the entire library

    """
    def __init__(self, ycol, xcols, model_params={}, n_cores=1):
        self.ycol = ycol
        self.xcols = xcols
        self.model_params = model_params
        self.n_cores = n_cores

    def get_ycol(self):
        return self.ycol

    def get_xcols(self):
        return self.xcols

    def set_xcols(self, xcols):
        self.xcols = xcols

    def get_model_params(self):
        return self.model_params

    def process_data(self, data):
        return data

    def fit(self, data):
      raise NotImplementedError('Not implemented')

    def predict(self, data):
        raise NotImplementedError('Not implemented')

    def summary(self):
        return None


class EnsembleClassifierRegressor(object):
    def __init__(self, estimators, ycol, avg_fn=np.mean):
        self.estimators = estimators
        self.ycol = ycol
        self.avg_fn = avg_fn

    def get_ycol(self):
        return self.ycol

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
    def __init__(self,  estimators, ycol, avg_fn=np.mean):
        super(EnsembleRegressor, self).__init__(estimators, ycol, avg_fn)


    def predict(self, data):
        pred = pd.DataFrame()
        for estimator in self.estimators:
            pred = pd.concat(pred[estimator.predict(data)['pred_num']], axis=1)
        # Tale avg across rows
        self.avg_fn


class EnsembleClassifier(EnsembleClassifierRegressor):
    """Ensemble several regressors

    """
    def __init__(self,  estimators, ycol, avg_fn=np.mean):
        super(EnsembleRegressor, self).__init__(estimators, ycol, avg_fn)


    def predict(self, data):
        pred = pd.DataFrame()
        for estimator in self.estimators:
            pred = pd.concat(pred[estimator.predict(data)['pred_num']], axis=1)
        # Tale avg across rows
        self.avg_fn
