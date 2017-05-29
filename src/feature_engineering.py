import pandas as pd
import numpy as np

from data_cleaning import DataCleaner

class Featurizer(object):

    def __init__(self, train_test='train'):
        self._data = None
        self._X = None
        self._y = None
        self._train_test = train_test
        self._features_names = None
        self._house_ids = None

    def _import_features_mask(self):
        with open('/Users/stefanobrunelli/kaggle/sberbank_russian_housing_market/params/features_mask.txt') as features_mask:
            self._features_names = features_mask.read().splitlines()

    def get_features_names(self):
        return self._features_names

    def get_house_ids(self):
        return self._house_ids

    def _create_design_matrix_y_split(self):
        if self._train_test=='train':
            self._y = np.array(self._data.pop('price_doc'))
        self._X = self._data[self._features_names].values

    def featurize(self):
        dc = DataCleaner(train_test=self._train_test)
        self._data = dc.clean()
        self._house_ids = self._data['id']
        self._import_features_mask()
        self._create_design_matrix_y_split()
        return (self._X, self._y)
