import pandas as pd
import numpy as np
from collections import Counter
from data_cleaning import DataCleaner

class Featurizer(object):

    def __init__(self):
        self._X = None

    def get_features_names(self):
        return self._X.columns.values.tolist()

    def get_house_ids(self):
        return self._house_ids

    def _create_missing_flag(self, feature):
        new_col_name = feature + '_missing'
        self._X[new_col_name] = self._X[feature].isnull() * 1

    def _impute_nas_with_mean_by_category(self, feature, category):
        category_means = self._X.groupby(category)[feature].mean()
        feature_values = self._X[feature].tolist()
        category_levels = self._X[category].tolist()
        for idx, feature_value in enumerate(feature_values):
            if np.isnan(feature_value):
                category_mean = category_means[category_levels[idx]]
                feature_values[idx] = category_mean
        self._X[feature] = feature_values

    def _impute_nas_with_mean(self, feature):
        self._X[feature].fillna(np.mean(self._X[feature]), inplace=True)

    def _impute_nas_with_mode_by_category(self, feature, category):
        feature_values = self._X[feature].tolist()
        category_levels = self._X[category].tolist()
        for idx, feature_value in enumerate(feature_values):
            if np.isnan(feature_value):
                cat = category_levels[idx]
                mask = category_levels == cat
                print '\n\n\n\n\n', cat, mask
        self._X[feature] = feature_values

    def _impute_nas_with_mode(self, feature):
        mask = self._X[feature].notnull()
        mode = Counter(self._X[mask][feature]).most_common(1)[0][0]
        self._X[feature].fillna(mode, inplace=True)

    def _dummify(self, feature):
        dummies = pd.get_dummies(self._X[feature], prefix=feature)
        self._X.pop(feature)
        self._X = pd.concat([self._X, dummies], axis=1)

    def _convert_to_categorical(self, feature):
        self._X[feature] = self._X[feature].astype('category')

    def _code_flags_as_10(self, feature):
        self._X[feature] = self._X[feature].map(lambda x: 1 if x=='yes' else 0)

    def featurize(self, X):
        self._X = X
        self._house_ids = self._X.pop('id')
        self._code_flags_as_10('culture_objects_top_25')
        self._code_flags_as_10('shopping_centers_raion')
        self._code_flags_as_10('incineration_raion')
        self._code_flags_as_10('radiation_raion')
        self._code_flags_as_10('big_market_raion')
        self._create_missing_flag('num_room')
        self._impute_nas_with_mode('num_room')
        return self._X.values
