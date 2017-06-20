import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaner(object):

    def __init__(self, data, train_test='train', sample_rate=1.0):
        self._data = data
        self._features_names = None
        self._train_test = train_test
        self._sample_rate = sample_rate

    def _now(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _sample(self):
        n = int(self._data.shape[0] * self._sample_rate)
        print ' # {:s} | Sample rate: {:.2f}'.format(self._now(), self._sample_rate)
        print ' # {:s} | Data set size before sampling: {:d}'.format(self._now(), self._data.shape[0])
        self._data = self._data.sample(n, random_state=77)
        print ' # {:s} | Data set size after sampling: {:d}'.format(self._now(), self._data.shape[0])

    def _import_features_mask(self):
        with open('/Users/stefanobrunelli/github/russian_housing_market/params/features_mask.txt') as features_mask:
            self._features_names = features_mask.read().splitlines()
        if self._train_test=='test':
            self._features_names.remove('price_doc')

    def _apply_features_mask(self):
        self._data = self._data[self._features_names]

    def get_features_mask(self):
        return self._features_names

    def _clean_build_year(self):
        build_year_clean_dict = {0.0: 2000, 1.0: 2001, 3.0: 2003, 20.0: 1920, 71.0: 1971, 215.0: 2015, 4965.0: 1965, 20052009.0: 2005}
        self._data['build_year'] = self._data.build_year.map(lambda x: build_year_clean_dict[x] if x in build_year_clean_dict else x)

    def trim_price_extremes(self, data, y):
        data['price_doc'] = y
        percentiles = np.linspace(0, 100, 1001)
        breaks = np.percentile(a=data.price_doc, q=percentiles)
        cuts = pd.cut(data.price_doc, np.unique(breaks), include_lowest=True)
        data['cuts'] = cuts
        print ' # {:s} | Before trim {:s}'.format(self._now(), data.shape)
        data = data[data.cuts < data.cuts.max()]
        print ' # {:s} | After trim {:s}'.format(self._now(), data.shape)
        y = data.pop('price_doc')
        data.pop('cuts')
        return data, y

    def _clean_full_sq(self):
        # mask = ~(self._data.full_sq > 1000).values
        # self._data = self._data[mask]
        # mask = ~((self._data.full_sq == 1.0) & (self._data.life_sq == 1.0)).values
        # self._data = self._data[mask]
        # mask = ~((self._data.full_sq == 0.0) & (self._data.life_sq == 0.0)).values
        # self._data = self._data[mask]
        # mask = (self._data.full_sq <= 1.0).values
        # self._data.loc[mask, 'full_sq'] = self._data.loc[mask, 'life_sq']
        full_sq_mean = self._data.full_sq.mean()
        full_sq_std = self._data.full_sq.std()
        full_sq_z = self._data.full_sq.map(lambda x: (x-full_sq_mean)/full_sq_std)
        mask = abs(full_sq_z) <= 10.0
        self._data = self._data[mask]
        mask = ~(self._data.full_sq <= 1.0)
        self._data = self._data[mask]
        mask = ~(self._data.full_sq < self._data.life_sq)
        self._data = self._data[mask]

    def clean(self):
        self._sample()
        self._clean_build_year()
        self._clean_full_sq()
        if self._train_test=='train':
            y = self._data.pop('price_doc')
        return (self._data, y)

    def kaggle(self):
        self._clean_build_year()
        # self._clean_full_sq()
        return self._data
