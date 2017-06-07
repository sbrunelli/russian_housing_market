import pandas as pd
import numpy as np
from feature_engineering import Featurizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelSelector(object):

    def __init__(self):
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._predictors = [#LinearRegression(),
                            RandomForestRegressor(n_estimators=1000),
                            GradientBoostingRegressor(n_estimators=100,
                                                        learning_rate=0.01,
                                                        max_depth=3)
                            ]
        self._best_predictor_index = None
        self._scores = np.repeat(0., len(self._predictors))

    def _fit_all(self):
        for predictor in self._predictors:
            predictor.fit(self._X_train, self._y_train)

    def _score_predictors(self, X_test, y_test):
        for i, predictor in enumerate(self._predictors):
            y_predicted = predictor.predict(X_test)
            score = np.sqrt(np.sum((np.log(y_predicted+1) - np.log(y_test+1))**2) / len(y_predicted))
            self._scores[i] = score
            print '{:s} has score: {:2.5f}'.format(str(predictor.__class__), score)

    def _select_best_predictor(self):
        self._score_predictors(self._X_train, self._y_train)
        best_score = self._scores[0]
        self._best_predictor_index = 0
        for i in xrange(len(self._predictors)):
            if self._scores[i] < best_score:
                best_score = self._scores[i]
                self._best_predictor_index = i

    def fit(self, X, y):
        self._X_train = X
        self._y_train = y
        self._fit_all()
        self._select_best_predictor()

    def best_predictor(self):
        print 'The best predictor is : {:s}'.format(str(self._predictors[self._best_predictor_index].__class__))
        return self._predictors[self._best_predictor_index]

    def predict(self, X):
        self._X_test = X
        y_predicted = self._predictors[self._best_predictor].predict(self._X_test)
        return y_predicted
