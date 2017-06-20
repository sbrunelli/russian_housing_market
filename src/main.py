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

    # Read data sets
    train = pd.read_csv('./data/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('./data/test.csv', parse_dates=['timestamp'])
    gps = pd.read_csv('./data/Longitud_Latitud.csv')
    # Create sub_area categorical with all levels shared
    # between train and test to avoid errors
    test['price_doc'] = -99
    merged = pd.concat([train, test], axis=0)
    merged = merged.merge(gps, how='left', on='sub_area')
    merged['sub_area'] = merged.sub_area.astype('category')
    train = merged[merged.price_doc != -99]
    test = merged[merged.price_doc == -99]
    test.pop('price_doc')

    macro = pd.read_csv('data/macro.csv', parse_dates=['timestamp'])
    train = train.merge(macro, how='left', on='timestamp', suffixes=('_train', '_macro'))

    # Clean
    dc = DataCleaner(data=train, sample_rate=0.75)
    data, y = dc.clean()
    y = np.array(y)
    y = np.log(y+1)

    # Train / test split
    data_train, data_test, y_train, y_test = train_test_split(data, y, random_state=77)
    house_ids_test = data_test.id

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

    # Diagnostic: dump X_test with y_test, y_predicted attached for deep analysis of errors
    features = featurizer.get_features_names()
    labels = np.array(np.exp(y_test) - 1).reshape(-1,1)
    labels_predicted = np.array(np.exp(best_estimator.predict(X_test)) - 1).reshape(-1,1)
    X_test = np.hstack((X_test, labels, labels_predicted))
    features.extend(['price_doc', 'price_doc_predicted'])
    df_test = pd.DataFrame(X_test, columns=features)
    df_test['prediction_error'] = df_test.price_doc - df_test.price_doc_predicted
    prediction_error_mean = df_test.prediction_error.mean()
    prediction_error_stddev = df_test.prediction_error.std()
    df_test['price_doc_z_score'] = df_test.price_doc.map(lambda x: (x-prediction_error_mean)/prediction_error_stddev)
    df_test.to_csv('X_test_diagnostic.csv', index=False)

    # Diagnistic: dump original test data with predicted_price
    data_test['id'] = house_ids_test
    data_test['price_doc'] = np.exp(y_test) - 1
    test_predictions_df = pd.DataFrame()
    test_predictions_df['id'] = house_ids_test
    test_predictions_df['price_doc_predicted'] = labels_predicted
    data_test = data_test.merge(test_predictions_df, how='left', on='id', suffixes=('_data', '_predictions'))
    data_test['prediction_error'] = data_test.price_doc - data_test.price_doc_predicted
    data_test['price_doc_z_score'] = data_test.price_doc.map(lambda x: (x-prediction_error_mean)/prediction_error_stddev)
    data_test.to_csv('data_test.csv', index=False)

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
    test = test.merge(macro, how='left', on='timestamp', suffixes=('_train', '_macro'))

    # Featurize Kaggle test set
    dc_kaggle = DataCleaner(data=test, train_test='test')
    test = dc_kaggle.kaggle()
    X_kaggle = featurizer.featurize(test, train_test='test')
    print '\n # {:s} | Test file after feature engineering'.format(now())
    print ' # {:s} | Shape : {:s}'.format(now(), X_kaggle.shape)
    print ' # {:s} | Nr. NA: {:d}'.format(now(), np.sum(np.isnan(X_kaggle)))
    print ' # {:s} | Nr. Inf: {:d}'.format(now(), np.sum(X_kaggle == np.Inf))


    # Create prediction file to submit to Kaggle
    tmstmp = now()
    house_ids = featurizer.get_house_ids()
    y_predicted = np.exp(best_estimator.predict(X_kaggle)) - 1
    print '\n # {:s} | Predictions'.format(now())
    print ' # {:s} | Nr. rows: {:d}'.format(now(), len(y_predicted))
    print ' # {:s} | Nr. negative values: {:d}'.format(now(), np.sum(y_predicted < 0.0))
    print ' # {:s} | Nr. NA: {:d}'.format(now(), np.sum(np.isnan(y_predicted)))
    print ' # {:s} | Nr. Inf: {:d}'.format(now(), np.sum(y_predicted == np.Inf))

    kaggle = pd.DataFrame({'id': house_ids, 'price_doc': y_predicted})
    kaggle.to_csv('predictions/predictions'+tmstmp+'.csv'
        , index=False)

    ########################
    # DELETE ME
    ########################
    import xgboost as xgb
    dtrain_all = xgb.DMatrix(X, y, feature_names=feat_names)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feat_names)
    dval = xgb.DMatrix(X_test[:, :(X_test.shape[1]-2)], y_test, feature_names=feat_names)
    dkaggle = xgb.DMatrix(X_kaggle, feature_names=feat_names)

    xgb_params = {
        'eta': 0.0001,
        'max_depth': 3,
        'subsample': 1,
        'colsample_bytree': 0.75,
        'eval_metric': 'rmse',
        'silent': 1
    }

    watchlist = [(dval, 'eval'), (dtrain, 'train')]

    num_round=int(1e5)

    partial_model = xgb.train(xgb_params, dtrain, num_round, watchlist, early_stopping_rounds=20)

    num_boost_round = partial_model.best_iteration

    model = xgb.train(xgb_params, dtrain_all, num_round)
    y_predicted_xgb = model.predict(dkaggle)
    y_predicted_xgb = np.exp(y_predicted_xgb) - 1

    print '\n # {:s} | Predictions (xgboost)'.format(now())
    print ' # {:s} | Nr. rows: {:d}'.format(now(), len(y_predicted_xgb))
    print ' # {:s} | Nr. negative values: {:d}'.format(now(), np.sum(y_predicted_xgb < 0.0))
    print ' # {:s} | Nr. NA: {:d}'.format(now(), np.sum(np.isnan(y_predicted_xgb)))
    print ' # {:s} | Nr. Inf: {:d}'.format(now(), np.sum(y_predicted_xgb == np.Inf))

    kaggle_xgb = pd.DataFrame({'id': house_ids, 'price_doc': y_predicted_xgb})
    kaggle_xgb.to_csv('predictions/predictions_xgb_'+tmstmp+'.csv', index=False)
