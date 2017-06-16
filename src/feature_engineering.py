import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from data_cleaning import DataCleaner
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class Featurizer(object):

    def __init__(self):
        self._train_test = None
        self._data = None
        self._X = None
        self._scalers = defaultdict()
        self._principal_components = defaultdict()
        self._first_pca = True

    def get_features_names(self):
        return self._X.columns.values.tolist()

    def get_house_ids(self):
        return self._house_ids

    def _create_missing_flag(self, feature):
        new_col_name = feature + '_missing'
        self._X[new_col_name] = self._X[feature].isnull() * 1

    def _impute_nas_with_mean(self, feature):
        self._X[feature].fillna(np.mean(self._X[feature]), inplace=True)

    def _impute_nas_with_median(self, feature):
        self._X[feature].fillna(self._X[feature].median(), inplace=True)

    def _impute_nas_with_mode(self, feature):
        mask = self._X[feature].notnull()
        mode = Counter(self._X[mask][feature]).most_common(1)[0][0]
        self._X[feature].fillna(mode, inplace=True)

    def _impute_NA_with_KNN(self, feature, predictors, n_neighbors=10):
        # Build X_train, y_train
        X_train = self._X[self._data[feature].notnull()][predictors]
        y_train = self._X[self._data[feature].notnull()][feature]

        # Fit knn
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        # Make predictions for entire data frame
        X = self._X[predictors]
        y_predicted = knn.predict(X)

        # Attach predictions coalesced
        feature_predicted_name = feature + '_predicted'
        self._X[feature_predicted_name] = y_predicted
        self._X[feature] = self._X[feature].combine_first(self._X[feature_predicted_name])
        self._X.pop(feature_predicted_name)

    def _dummify(self, feature):
        dummies = pd.get_dummies(self._X[feature], prefix=feature)
        self._X.pop(feature)
        self._X = pd.concat([self._X, dummies], axis=1)

    # def _convert_to_categorical(self, feature):
    #     self._X[feature] = self._X[feature].astype('category')
    #
    def _code_flags_as_10(self, feature):
        self._X[feature] = self._X[feature].map(lambda x: 1 if x=='yes' else 0)

    def _parse_timestamp(self):
        self._X['year'] = self._X.timestamp.map(lambda x: pd.to_datetime(x).year)
        self._X['month'] = self._X.timestamp.map(lambda x: pd.to_datetime(x).month)
        # self._X['week'] = self._X.timestamp.map(lambda x: pd.to_datetime(x).week)
        self._X['year_month'] = (self._X['year']*100) + self._X['month']
        # self._X['year_week'] = (self._X['year']*100) + self._X['week']
        # self._X['year_month'] = self._X.timestamp.map(lambda x: int(x.split('-')[0] + x.split('-')[1]))
        self._X.pop('timestamp')

    def _attach_main_section(self):
        # variables = ['full_sq', 'life_sq', 'material', 'build_year', 'num_room', 'kitch_sq', 'state', 'culture_objects_top_25', 'shopping_centers_raion', 'incineration_raion', 'radiation_raion', 'big_market_raion', 'product_type', 'ecology', 'timestamp']
        variables = ['full_sq', 'life_sq', 'kitch_sq', 'num_room', 'state', 'zd_vokzaly_avto_km', 'nuclear_reactor_km', 'university_km', 'office_km', 'office_sqm_5000', 'trc_count_5000',
        'cafe_count_5000_price_1000', 'sport_count_5000', 'sadovoe_km',
        'floor', 'school_education_centers_raion', 'leisure_count_5000', 'deposits_value', 'salary']

        self._X = pd.concat([self._X, self._data[variables]], axis=1)

    def _pca(self, section_name, indices, num_components):
        section = self._data[indices].copy()
        section = section.apply(lambda x: x.fillna(np.mean(x)))
        matrix = section.values
        if self._train_test=='train':
            self._scalers[section_name] = StandardScaler()
            self._scalers[section_name].fit(matrix)
        matrix_scaled = self._scalers[section_name].transform(matrix)
        if self._train_test=='train':
            self._principal_components[section_name] = PCA(n_components=num_components)
            self._principal_components[section_name].fit(matrix_scaled)
        principal_components = self._principal_components[section_name].transform(matrix_scaled)
        return principal_components

    def _pca_feature_extraction(self, section_name, num_components=5):
        sect2idx = {'general': range(12, 40) + range(151, 152),
                    'demographics': range(40, 67),
                    'buildings': range(67, 83),
                    'distances': range(84, 151),
                    'ring500': range(152, 175),
                    'ring1000': range(175, 198),
                    'ring1500': range(198, 221),
                    'ring2000': range(221, 244),
                    'ring3000': range(244, 267),
                    'ring5000': range(267, 290),
                    'economy': range(290, 389)}
        principal_components_values = self._pca(section_name, sect2idx[section_name], num_components)
        principal_components_names = [section_name + '_pc' + str(val) for val in range(num_components)]
        pc = pd.DataFrame(principal_components_values, columns=principal_components_names)
        if self._first_pca:
            self._X.reset_index(inplace=True)
            self._first_pca = False
        self._X = pd.concat([self._X, pc], axis=1)

    def _bin_categorical(self, feature, min_level_size=10):
        # Identify levels < min_level_size
        mask = self._X.sub_area.value_counts() < 10
        low_card_lvl_set = set(self._X.sub_area.value_counts()[mask].index)

        # Rename them to others
        feat_list = list(self._X[feature])
        feat_renamed = ['Others' if el_name in low_card_lvl_set else el_name for el_name in feat_list]
        self._X[feature] = feat_renamed

    def _create_ratios(self, baseline, features):
        for feature in features:
            new_feat_name = feature + '_ratio'
            self._X[new_feat_name] = self._X[feature] / self._X[baseline]
            self._X[new_feat_name].fillna(0, inplace=True)
            self._X[new_feat_name] = self._X[new_feat_name].map(lambda x: 0 if x == np.Inf else x)

    def _create_sqm_per_room(self):
        self._X['sqm_per_room'] = self._X.num_room / self._X.full_sq

    def featurize(self, data, train_test='train'):
        self._data = data
        self._X = None
        self._train_test = train_test
        if self._train_test=='test':
            self._first_pca = True
        self._house_ids = self._data.pop('id')
        self._attach_main_section()
        self._create_missing_flag('num_room')
        self._create_missing_flag('state')
        self._create_missing_flag('kitch_sq')
        self._create_missing_flag('life_sq')
        self._create_missing_flag('floor')
        self._create_missing_flag('salary')
        self._X['floor'] = self._X.floor.fillna(0.0)
        # self._X.pop('build_year')
        # self._impute_nas_with_mode('num_room')
        self._impute_NA_with_KNN(feature='num_room', predictors=['full_sq'], n_neighbors=40)
        self._impute_nas_with_mode('state')
        self._impute_nas_with_median('kitch_sq')
        self._impute_nas_with_median('life_sq')
        self._impute_nas_with_median('salary')
        self._create_sqm_per_room()
#        self._bin_categorical('sub_area')
#        self._dummify('sub_area')
        # # self._pca_feature_extraction('general')
        # self._pca_feature_extraction('demographics')
        # self._pca_feature_extraction('buildings')
        # # self._pca_feature_extraction('distances')
        # self._pca_feature_extraction('ring500')
        # self._pca_feature_extraction('ring1000')
        # self._pca_feature_extraction('ring1500')
        # self._pca_feature_extraction('ring2000')
        # self._pca_feature_extraction('ring3000')
        # self._pca_feature_extraction('ring5000')
        # self._pca_feature_extraction('economy')
        self._create_ratios(baseline='full_sq', features=['life_sq', 'kitch_sq'])
        # self._X.pop('life_sq')
        # self._parse_timestamp()
        return self._X.values
