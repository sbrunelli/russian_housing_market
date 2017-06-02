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
        with open('/Users/stefanobrunelli/github/russian_housing_market/params/features_mask.txt') as features_mask:
            self._features_names = features_mask.read().splitlines()
        if self._train_test == 'test':
            self._features_names.remove('price_doc')
        self._data = self._data[self._features_names]

    def get_features_names(self):
        return self._features_names

    def get_house_ids(self):
        return self._house_ids

    def _create_design_matrix_y_split(self):
        if self._train_test=='train':
            self._y = np.array(self._data.pop('price_doc'))
        self._X = self._data.values

    def _create_missing_flag(self, feature):
        self._data[feature] = self._data[feature].isnull() * 1

    def _impute_nas_with_mean_by_category(self, feature, category):
        category_means = self._data.groupby(category)[feature].mean()
        feature_values = self._data[feature].tolist()
        category_levels = self._data[category].tolist()
        for idx, feature_value in enumerate(feature_values):
            if np.isnan(feature_value):
                category_mean = category_means[category_values[idx]]
                feature_values[idx] = category_mean
        self._data[feature] = feature_values

    def _impute_nas_with_mean(self, feature):
        self._data[feature].fillna(np.mean(self._data[feature]), inplace=True)

    def _impute_nas_with_mode_by_category(self, feature, category):
        pass

    def _impute_nas_with_mode(self, feature):
        pass

    def _dummify(self, feature):
        dummies = pd.get_dummies(self._data[feature], prefix=feature)
        self._data.drop(feature, axis=1, inplace=True)
        self._data = pd.concat([self._data, dummies], axis=1)

    def featurize(self):
        dc = DataCleaner(train_test=self._train_test)
        self._data = dc.clean()
        self._house_ids = self._data['id']
        self._import_features_mask()
        self._create_missing_flag('life_sq')
        self._impute_nas_with_mean_by_category('life_sq', 'sub_area')
        self._impute_nas_with_mean('life_sq')
#        self._dummify('sub_area')
        self._data.pop('sub_area')
        self._create_design_matrix_y_split()
        return (self._X, self._y)
