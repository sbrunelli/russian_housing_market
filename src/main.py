import pandas as pd
import numpy as np
from data_cleaning import DataCleaner
from feature_engineering import Featurizer
from model_selection import ModelSelector
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.interactive(True)

def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    # Turn off warnings
    pd.options.mode.chained_assignment = None

    # Read train data set
    train = pd.read_csv('data/train.csv')
    macro = pd.read_csv('data/macro.csv')
    train = train.merge(macro, how='left', on='timestamp', suffixes=('_train', '_macro'))

    # Clean
    dc = DataCleaner(data=train, sample_rate=0.3)
    data, y = dc.clean()
    y = np.array(y)
    y = np.log(y+1)

    # Train / test split
    data_train, data_test, y_train, y_test = train_test_split(data, y, random_state=77)

    # Featurize training data set
    feat_train = Featurizer()
    X_train = feat_train.featurize(data_train)

    # Grid search tune all estimators
    ms = ModelSelector()
    print ' # {:s} | X_train shape: {:s}'.format(now(), X_train.shape)
    print ' # {:s} | y_train size: {:d}'.format(now(), y_train.shape[0])
    ms.grid_search_cv(X_train, y_train)

    # Retrain estimators with best parameters
    ms.retrain_with_best_params()

    # Featurize test data set
    X_test = feat_train.featurize(data_test, train_test='test')
    print ' # {:s} | X_test shape: {:s}'.format(now(), X_test.shape)
    print ' # {:s} | y_test size: {:d}'.format(now(), y_test.shape[0])

    # Score all estimators with best parameters against test data set
    ms.score_best_estimators(X_test, y_test)
    ms.report_score_best_estimators()

    # Select best estimator
    best_estimator = ms.get_best_estimator()

    # Featurize train / test merge
    featurizer = Featurizer()
    X = featurizer.featurize(data)

    # Train best estimator against merged train / test
    best_estimator.fit(X, y)

    # Plot y_test vs y_predicted for best estimator
    fig1 = plt.figure(figsize=(12,9))
    plt.scatter((np.exp(best_estimator.predict(X_test)) - 1), (np.exp(y_test)-1), s=100, alpha=0.10)
    plt.plot((np.exp(y_test)-1), (np.exp(y_test)-1), color='chocolate', alpha=0.20)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Predicted vs actuals plot')
    plt.show()

    # Plot feature importances
    feats = dict()
    feat_names = featurizer.get_features_names()
    feat_importances = best_estimator.feature_importances_
    for feature, importance in zip(feat_names, feat_importances):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini'})
    importances.sort_values(by='Gini').plot(kind='bar', rot=45, figsize=(12,9))
    # plt.show()

    # Read Kaggle test set
    test = pd.read_csv('data/test.csv')
    test = test.merge(macro, how='left', on='timestamp', suffixes=('_train', '_macro'))

    # Featurize Kaggle test set
    dc_kaggle = DataCleaner(data=test, train_test='test')
    test = dc_kaggle.kaggle()
    # feat_kaggle = Featurizer()
    X_kaggle = featurizer.featurize(test, train_test='test')

    # Create prediction file to submit to Kaggle
    tmstmp = now()
    house_ids = featurizer.get_house_ids()
    y_predicted = np.exp(best_estimator.predict(X_kaggle)) - 1
    kaggle = pd.DataFrame({'id': house_ids, 'price_doc': y_predicted})
    kaggle.to_csv('predictions/predictions'+tmstmp+'.csv'
        , index=False)
