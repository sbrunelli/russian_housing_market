import pandas as pd

class DataCleaner(object):

    def __init__(self, data, train_test='train'):
        self._data = data
        self._features_names = None
        self._train_test = train_test

    def _import_features_mask(self):
        with open('/Users/stefanobrunelli/github/russian_housing_market/params/features_mask.txt') as features_mask:
            self._features_names = features_mask.read().splitlines()
        if self._train_test=='test':
            self._features_names.remove('price_doc')

    def _apply_features_mask(self):
        self._data = self._data[self._features_names]

    def get_features_mask(self):
        return self._features_names

    def clean(self):
        self._import_features_mask()
        self._apply_features_mask()
        if self._train_test=='train':
            y = self._data.pop('price_doc')
        X = self._data
        return (X, y)

    def kaggle(self):
        self._import_features_mask()
        self._apply_features_mask()
        X = self._data
        return X
