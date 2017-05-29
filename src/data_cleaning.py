import pandas as pd

class DataCleaner(object):

    def __init__(self, train_test='train'):
        self._data = None
        self._macro = None
        self._train_test = train_test

    def _read_data(self):
        if self._train_test=='train':
            self._data = pd.read_csv('/Users/stefanobrunelli/kaggle/sberbank_russian_housing_market//data/train.csv')
        else:
            self._data = pd.read_csv('/Users/stefanobrunelli/kaggle/sberbank_russian_housing_market//data/test.csv')

    def clean(self):
        self._read_data()
        return (self._data)
