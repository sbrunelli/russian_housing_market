import pandas as pd
import numpy as np
from data_cleaning import DataCleaner
from feature_engineering import Featurizer
from model_selection import ModelSelector
from datetime import datetime
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Turn off warnings
    pd.options.mode.chained_assignment = None

    # Read train data set
    train = pd.read_csv('/Users/stefanobrunelli/github/russian_housing_market/data/train.csv')

    # Clean
    dc = DataCleaner(train)
    X, y = dc.clean()
    y = np.array(y)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Featurize training data set
    feat_train = Featurizer()
    X_train = feat_train.featurize(X_train)

    # Grid search tune all estimators
    ms = ModelSelector()
    ms.grid_search_cv(X_train, y_train)

    # Retrain estimators with best parameters
    ms.retrain_with_best_params()

    # Featurize test data set
    feat_test = Featurizer()
    X_test = feat_test.featurize(X_test)

    # Score all estimators with best parameters against test data set
    ms.score_best_estimators(X_test, y_test)

    # Select best estimator
    best_estimator = ms.get_best_estimator()

    # Train / test merge

    # Featurize train / test merge
    featurizer = Featurizer()
    X = featurizer.featurize(X)

    # Train best estimator against merged train / test
    best_estimator.fit(X, y)

    # Read Kaggle test set
    test = pd.read_csv('/Users/stefanobrunelli/github/russian_housing_market/data/test.csv')

    # Featurize Kaggle test set
    dc_kaggle = DataCleaner(test, train_test='test')
    test = dc_kaggle.kaggle()
    feat_kaggle = Featurizer()
    X_kaggle = feat_kaggle.featurize(test)

    # Create prediction file to submit to Kaggle
    tmstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
    house_ids = feat_kaggle.get_house_ids()
    y_predicted = best_estimator.predict(X_kaggle)
    kaggle = pd.DataFrame({'id': house_ids, 'price_doc': y_predicted})
    kaggle.to_csv('/Users/stefanobrunelli/github/russian_housing_market/predictions/predictions'+tmstmp+'.csv'
        , index=False)
