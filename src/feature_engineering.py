import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from data_cleaning import DataCleaner
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

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

    def _code_flags_as_10(self, feature):
        self._X[feature] = self._X[feature].map(lambda x: 1 if x=='yes' else 0)

    def _parse_timestamp(self):
        self._X['year'] = self._X.timestamp.dt.year
        self._X['month'] = self._X.timestamp.dt.month
        self._X['dow'] = self._X.timestamp.dt.dayofweek

        # Adding counts
        # Add month-year

        month_year = (self._X.timestamp.dt.month + self._X.timestamp.dt.year * 100)
        month_year_cnt_map = month_year.value_counts().to_dict()
        self._X['month_year_cnt'] = month_year.map(month_year_cnt_map)

        # Add week-year count
        week_year = (self._X.timestamp.dt.weekofyear + self._X.timestamp.dt.year * 100)
        week_year_cnt_map = week_year.value_counts().to_dict()
        self._X['week_year_cnt'] = week_year.map(week_year_cnt_map)

        self._X.pop('timestamp')


    def _attach_main_section(self):
        variables = ['full_sq', 'life_sq', 'kitch_sq', 'num_room', 'state', 'zd_vokzaly_avto_km', 'nuclear_reactor_km', 'university_km', 'office_km', 'office_sqm_5000', 'trc_count_5000',
        'cafe_count_5000_price_1000', 'sport_count_5000', 'sadovoe_km', 'build_year', 'sub_area', 'timestamp',
        'floor', 'school_education_centers_raion', 'leisure_count_5000', 'deposits_value', 'salary']#,
        # 'longitude', 'latitude']

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

    def _create_sq_per_room(self):
        self._X['sq_per_room'] = self._X.num_room / self._X.life_sq
        self._X['sq_per_room'] = self._X.sq_per_room.map(lambda x: 0 if x == np.Inf else x)

    def _create_extra_sq(self):
        self._X['extra_sq'] = self._X['full_sq'] - self._X['life_sq']

    def _create_cafe_avg_price_index(self):
        distances = [500, 1000, 1500, 2000, 3000, 5000]
        weights = [(1./d)/np.sum(1./np.array(distances)) for d in distances]
        variables = ['cafe_avg_price_500', 'cafe_avg_price_1000', 'cafe_avg_price_1500', 'cafe_avg_price_2000', 'cafe_avg_price_3000', 'cafe_avg_price_5000']
        cafe_df = self._data[variables]
        # We might want to get a lot more intelligent in doing this imputation
        cafe_df = cafe_df.apply(lambda x: x.fillna(x.median()))
        cafe_weighted_avgs = []
        for _, row in cafe_df.iterrows():
            cafe_weighted_avg = np.dot(row.values, weights)
            cafe_weighted_avgs.append(cafe_weighted_avg)
        self._X['cafe_weighted_avg'] = cafe_weighted_avgs

    def _create_cafe_density_index(self):
        distances = [500, 1000, 1500, 2000, 3000, 5000]
        weights = [(1./d)/np.sum(1./np.array(distances)) for d in distances]
        variables = ['cafe_count_500', 'cafe_count_1000', 'cafe_count_1500', 'cafe_count_2000', 'cafe_count_3000', 'cafe_count_5000']
        cafe_df = self._data[variables]
        # We might want to get a lot more intelligent in doing this imputation
        cafe_df = cafe_df.apply(lambda x: x.fillna(x.median()))
        cafe_densities = []
        for _, row in cafe_df.iterrows():
            densities = [(1./cnt) if (cnt!=0) else 0. for cnt in row]
            cafe_density = np.dot(densities, weights)
            cafe_densities.append(cafe_density)
        self._X['cafe_densities'] = cafe_densities

    def _divide_by_zero(self, N, D):
        if D==0:
            return 0.
        else:
            return float(N) / D

    def _create_cafe_under_over_price_share(self):
        distances = [500, 1000, 1500, 2000, 3000, 5000]
        weights = [(1./d)/np.sum(1./np.array(distances)) for d in distances]
        variables = ['cafe_count_500', 'cafe_count_500_price_500', 'cafe_count_500_price_high',
        'cafe_count_1000', 'cafe_count_1000_price_500', 'cafe_count_1000_price_high',
        'cafe_count_1500', 'cafe_count_1500_price_500', 'cafe_count_1500_price_high',
        'cafe_count_2000', 'cafe_count_2000_price_500', 'cafe_count_2000_price_high',
        'cafe_count_3000', 'cafe_count_3000_price_500', 'cafe_count_3000_price_high',
        'cafe_count_5000', 'cafe_count_5000_price_500', 'cafe_count_5000_price_high']
        cafe_df = self._data[variables]
        # We might want to get a lot more intelligent in doing this imputation
        cafe_df = cafe_df.apply(lambda x: x.fillna(x.median()))
        cafe_count_500_price_500_shares = []
        cafe_count_500_price_high_shares = []
        cafe_price_500_shares = []
        cafe_price_high_shares = []

        for _, row in cafe_df.iterrows():
            if row[0] > 0:
                cafe_count_500_price_500_share = row[1] / row[0]
                cafe_count_500_price_high_share = row[2] / row[0]
            else:
                cafe_count_500_price_500_share = 0
                cafe_count_500_price_high_share = 0

            cafe_price_500_share = self._divide_by_zero(row[1], row[0])*weights[0] + self._divide_by_zero(row[4], row[3])*weights[1] + self._divide_by_zero(row[7], row[6])*weights[2] + self._divide_by_zero(row[10], row[9])*weights[3] + self._divide_by_zero(row[13], row[12])*weights[5] + self._divide_by_zero(row[16], row[15])*weights[5]

            cafe_price_high_share = self._divide_by_zero(row[2], row[0])*weights[0] + self._divide_by_zero(row[5], row[3])*weights[1] + self._divide_by_zero(row[8], row[6])*weights[2] + self._divide_by_zero(row[11], row[9])*weights[3] + self._divide_by_zero(row[14], row[12])*weights[5] + self._divide_by_zero(row[17], row[15])*weights[5]

            cafe_count_500_price_500_shares.append(cafe_count_500_price_500_share)
            cafe_count_500_price_high_shares.append(cafe_count_500_price_high_share)
            cafe_price_500_shares.append(cafe_price_500_share)
            cafe_price_high_shares.append(cafe_price_high_share)

        self._X['cafe_count_500_price_shares'] = cafe_count_500_price_500_shares
        self._X['cafe_count_500_price_high_shares'] = cafe_count_500_price_high_shares
        self._X['cafe_price_500_shares'] = cafe_price_500_shares
        self._X['cafe_price_high_shares'] = cafe_price_high_shares

    def _impute_nas_life_sq(self):
        self._X['life_sq'] = self._X.life_sq.map(lambda x: np.nan if x <= 1.0 else x)
        mask = (self._X.full_sq == self._X.life_sq)
        self._X.loc[mask, 'life_sq'] = self._X.life_sq.map(lambda x: np.nan)
        self._X['extra_sq'] = self._X.full_sq - self._X.life_sq
        mask = (self._X.extra_sq <= 2.0)
        self._X.loc[mask, 'life_sq'] = np.nan
        self._X.pop('extra_sq')
        train = self._X[self._X.life_sq.notnull()][['full_sq','life_sq']]
        full_sq_mean = train.full_sq.mean()
        full_sq_std = train.full_sq.std()
        life_sq_mean = train.life_sq.mean()
        life_sq_std = train.life_sq.std()
        train['full_sq_z'] = train.full_sq.map(lambda x: (x-full_sq_mean)/full_sq_std)
        train['life_sq_z'] = train.life_sq.map(lambda x: (x-life_sq_mean)/life_sq_std)
        mask = ~(abs(train.full_sq_z) > 3.0)
        train = train[mask]
        mask = ~(abs(train.life_sq_z) > 3.0)
        train = train[mask]
        train.pop('full_sq_z')
        train.pop('life_sq_z')
        y = np.array(train.pop('life_sq'))
        X = train.values.reshape(-1, 1)
        life_sq_imputer = LinearRegression(normalize=True)
        life_sq_imputer.fit(X, y)
        X_new = self._X.full_sq.values.reshape(-1, 1)
        y_new = life_sq_imputer.predict(X_new)
        self._X['life_sq_predicted'] = y_new
        self._X['life_sq'] = self._X['life_sq'].combine_first(self._X['life_sq_predicted'])
        self._X.pop('life_sq_predicted')

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
        self._create_missing_flag('build_year')
        self._X['floor'] = self._X.floor.fillna(0.0)
        # self._impute_nas_with_mode('num_room')
        self._impute_NA_with_KNN(feature='num_room', predictors=['full_sq'], n_neighbors=40)
        self._impute_nas_with_mode('state')
        self._impute_nas_with_mode('build_year')
        self._impute_nas_with_median('kitch_sq')
        self._impute_nas_with_median('life_sq')
        self._impute_nas_life_sq()
        self._impute_nas_with_median('salary')
        self._create_sq_per_room()
        self._create_extra_sq()
        self._dummify('sub_area')
        # # self._pca_feature_extraction('general')
        self._create_ratios(baseline='full_sq', features=['life_sq', 'kitch_sq'])
        self._create_cafe_avg_price_index()
        self._create_cafe_density_index()
        self._create_cafe_under_over_price_share()
        # self._X.pop('life_sq')
        self._parse_timestamp()
        return self._X.values
