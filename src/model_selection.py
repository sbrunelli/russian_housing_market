import pandas as pd
import numpy as np
from feature_engineering import Featurizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from datetime import datetime

class ModelSelector(object):

    def __init__(self):
        self._X_train = None
        self._y_train = None
        self._estimators = [RandomForestRegressor(n_jobs=-1),
                            GradientBoostingRegressor()
                            ]
        self._best_estimator_index = None
        self._scores = np.repeat(0., len(self._estimators))
        self._parameters_grid = {'RandomForestRegressor':
                                    {'n_estimators': [100, 200]},
                                'GradientBoostingRegressor':
                                    {'n_estimators': [100, 200]}}
        self._grid_search_results = defaultdict()
        self._best_retrained = False

    def _now(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_estimator_name(self, estimator):
        return str(estimator.__class__).split('.')[-1].replace("'>","")

    def grid_search_cv(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        print '\n\n'
        for estimator in self._estimators:
            estimator_name = self._get_estimator_name(estimator)
            print ' # {:s} | Grid searching for estimator: {:s}'.format(self._now(), estimator_name)
            parameters = self._parameters_grid[estimator_name]
            gs = GridSearchCV(estimator, parameters, cv=3)
            gs.fit(X_train, y_train)
            self._grid_search_results[estimator_name] = gs

    def retrain_with_best_params(self):
        for estimator in self._estimators:
            estimator_name = self._get_estimator_name(estimator)
            best_parameters = self._grid_search_results[estimator_name].best_params_
            print '\n # {:s} | Retraining estimator: {:s}'.format(self._now(), estimator_name)
            print ' # {:s} | Best parameters: {:s}'.format(self._now(), best_parameters)
            estimator.set_params(**best_parameters)
            print estimator
            estimator.fit(self._X_train, self._y_train)
        self._best_retrained = True

    def _rmlse(self, estimator, X_test, y_test):
        y_predicted = estimator.predict(X_test)
        score = np.sqrt(np.sum((np.log(y_predicted+1) - np.log(y_test+1))**2) / len(y_predicted))
        return score

    def score(self, estimator, X_test, y_test):
        return self._rmlse(estimator, X_test, y_test)

    def score_best_estimators(self, X_test, y_test):
        if not self._best_retrained:
            self.retrain_with_best_params()
        for idx, estimator in enumerate(self._estimators):
            self._scores[idx] = self.score(estimator, X_test, y_test)
        self._set_best_estimator_index()

    def report_score_best_estimators(self):
        print
        for idx in xrange(len(self._scores)):
            estimator_name = self._get_estimator_name(self._estimators[idx])
            estimator_score = self._scores[idx]
            print ' # {:s} | RMLSE for estimator: {:s} ==> {:.5f}'.format(self._now(), estimator_name, estimator_score)

    def _set_best_estimator_index(self):
        self._best_estimator_index = np.argmin(self._scores)

    def get_best_estimator(self):
        if not self._best_estimator_index:
            self._set_best_estimator_index()
        estimator_name = self._get_estimator_name(self._estimators[self._best_estimator_index])
        print '\n # {:s} | Best estimator: {:s}'.format(self._now(), estimator_name)
        print self._estimators[self._best_estimator_index]
        return self._estimators[self._best_estimator_index]
