import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
plt.interactive(True)

def rmsle_score(y_true, y_predicted):
    return np.sqrt(np.sum((np.log1p(y_predicted) - np.log1p(y_true))**2) / len(y_true))

def impute_num_room(data_valid, data_train):
    data_valid = data_valid[data_valid.num_room.notnull()]
    train = data_train[data_train.num_room.notnull()]
    y_train = np.array(train.pop('num_room'))
    X_train = train['full_sq'].values.reshape(-1, 1)
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    X = data_train['full_sq'].values.reshape(-1, 1)
    y_predicted = knn.predict(X)
    data_train['num_room_predicted'] = y_predicted
    data_train['num_room'] = data_train.num_room.combine_first(data_train['num_room_predicted'])
    data_train.pop('num_room_predicted')
    return data_valid, data_train

def impute_life_sq(data_valid, data_train):
    mask = (data_valid.life_sq <= 1.0)#.values
    data_valid.loc[mask, 'life_sq'] = np.nan
    mask = (data_train.life_sq <= 1.0)
    data_train.loc[mask, 'life_sq'] = np.nan

    train = data_train[data_train.life_sq.notnull()]

    # Data cleaning
    full_sq_mean = train.full_sq.mean()
    full_sq_std = train.full_sq.std()
    full_sq_z = train.full_sq.map(lambda x: (x-full_sq_mean)/full_sq_std)
    life_sq_mean = train.life_sq.mean()
    life_sq_std = train.life_sq.std()
    life_sq_z = train.life_sq.map(lambda x: (x-life_sq_mean)/life_sq_std)
    train = train[(abs(full_sq_z) < 5.0) & (abs(life_sq_z) < 5.0)]


    # Model fitting
    y_train = np.array(train.pop('life_sq'))
    X_train = train['full_sq'].values.reshape(-1, 1)
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    X = data_train['full_sq'].values.reshape(-1, 1)
    y_predicted = lr.predict(X)
    data_train['life_sq_predicted'] = y_predicted
    data_train['life_sq'] = data_train.life_sq.combine_first(data_train.life_sq_predicted)
    data_train.pop('life_sq_predicted')
    X = data_valid['full_sq'].values.reshape(-1, 1)
    y_predicted = lr.predict(X)
    data_valid['life_sq_predicted'] = y_predicted
    data_valid['life_sq'] = data_valid.life_sq.combine_first(data_valid.life_sq_predicted)
    return data_valid, data_train

if __name__ == "__main__":
    ###################
    # Read data
    ###################
    train = pd.read_csv('./data/train.csv')
    data = train[['timestamp', 'full_sq', 'life_sq', 'num_room', 'price_doc']]
    nrows = data.shape[0]
    train_idx = np.random.choice([True, False], size=nrows, replace=True, p=[0.7, 0.3])
    valid_idx = ~(train_idx)
    data_train = data[train_idx]
    data_valid = data[valid_idx]
    print 'Training data set size:', data_train.shape
    print 'Validation data set size:', data_valid.shape

    ###################
    # Cleaning
    ###################
    # Remove first and last percentiles
    # price_doc = data_train.price_doc
    # cat = pd.cut(price_doc, np.percentile(price_doc, q=[0, 1, 99, 100]), include_lowest=True)
    # mask = ((cat > cat.min()) & (cat < cat.max()))
    # data_train = data_train[mask]
    print 'Remove first and last percentiles:', data_train.shape

    # Drop full_sq with z_score greater than 5
    full_sq_mean = data_train.full_sq.mean()
    full_sq_std = data_train.full_sq.std()
    full_sq_z = data_train.full_sq.map(lambda x: (x-full_sq_mean)/full_sq_std)
    mask = ~(abs(full_sq_z) > 5.0)
    data_train = data_train[mask]
    print 'Drop full_sq with z_score greater than 5:', data_train.shape

    mask = ~(data_train.full_sq < data_train.life_sq)
    data_train = data_train[mask]
    print 'Drop full_sq < life_sq:', data_train.shape

    mask = ~(data_train.full_sq <= 1.0)
    data_train = data_train[mask]
    print 'Drop full_sq <= 1.0:', data_train.shape

    # data_train['full_sq'] = data_train.full_sq.map(lambda x: np.log1p(x))
    # data_valid['full_sq'] = data_valid.full_sq.map(lambda x: np.log1p(x))

    ###################
    # Featurization
    ###################
    data_valid, data_train = impute_num_room(data_valid, data_train)
    # data_valid, data_train = impute_life_sq(data_valid, data_train)

    y = np.array(data_train.pop('price_doc'))
    X = data_train[['full_sq', 'num_room']].values#.reshape(-1, 1)

    ###################
    # Model fitting
    ###################
    lr = LinearRegression(normalize=True)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=3)
    num_folds = 10
    num_iter = 100
    nrows = X.shape[0]

    # # Linear model
    # fig = plt.figure()
    # scores = []
    # for iter in range(num_iter):
    #     folds = np.random.choice(range(num_folds), size=nrows, replace=True)
    #     iter_scores = []
    #     for fold in range(10):
    #         X_train, X_test = X[folds!=fold], X[folds==fold]
    #         y_train, y_test = y[folds!=fold], y[folds==fold]
    #         lr.fit(X_train, np.log1p(y_train))
    #         y_predicted = np.exp(lr.predict(X_test)) - 1
    #         score = rmsle_score(y_test, y_predicted)
    #         iter_scores.append(score)
    #     scores.append(iter_scores)
    # plt.plot(range(num_iter), np.mean(scores, axis=1))
    # plt.show()
    # rmsle_cv = np.mean(scores)
    # print 'RMSLE linear (CV): {:.5f}'.format(rmsle_cv)

    # Tree-based model
    fig2 = plt.figure()
    num_iter=10
    scores = []
    for iter in range(num_iter):
        folds = np.random.choice(range(num_folds), size=nrows, replace=True)
        iter_scores = []
        for fold in range(10):
            X_train, X_test = X[folds!=fold], X[folds==fold]
            y_train, y_test = y[folds!=fold], y[folds==fold]
            rf.fit(X_train, np.log1p(y_train))
            y_predicted = np.exp(rf.predict(X_test)) - 1
            score = rmsle_score(y_test, y_predicted)
            iter_scores.append(score)
        scores.append(iter_scores)
    plt.plot(range(num_iter), np.mean(scores, axis=1))
    plt.show()
    rmsle_cv = np.mean(scores)
    print 'RMSLE tree-based (CV): {:.5f}'.format(rmsle_cv)

    ###################
    # Scoring
    ###################
    # # Linear model
    # y_valid = np.array(data_valid.pop('price_doc'))
    # X_valid = data_valid[['full_sq', 'num_room']].values#.reshape(-1, 1)
    # y_predicted = np.exp(lr.predict(X_valid)) - 1
    # rmsle = rmsle_score(y_valid, y_predicted)
    # print 'RMSLE linear: {:.5f}'.format(rmsle)
    # fig3 = plt.figure(figsize=(9,6))
    # plt.scatter(y_valid, y_predicted, alpha=0.2)
    # plt.plot(y, y, color='chocolate', linestyle='dotted', alpha=0.2)
    # plt.xlim(0, 1e8)
    # plt.ylim(0, 1e8)
    # plt.xlabel('Actual')
    # plt.ylabel('Prediction')
    # plt.show()
    # y_predicted_lr = y_predicted[::]

    # Tree-base model
    y_valid = np.array(data_valid.pop('price_doc'))
    X_valid = data_valid[['full_sq', 'num_room']].values#.reshape(-1, 1)
    y_predicted = np.exp(rf.predict(X_valid)) - 1
    rmsle = rmsle_score(y_valid, y_predicted)
    print 'RMSLE tree-based: {:.5f}'.format(rmsle)
    fig4 = plt.figure(figsize=(9,6))
    plt.scatter(y_valid, y_predicted, alpha=0.2)
    plt.plot(y, y, color='chocolate', linestyle='dotted', alpha=0.2)
    plt.xlim(0, 1e8)
    plt.ylim(0, 1e8)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.show()
    y_predicted_rf = y_predicted[::]

    fig5 = plt.figure(figsize=(9,6))
    # plt.scatter(y_valid, y_predicted_lr, alpha=0.2, color='green', label='Linear regression')
    plt.scatter(y_valid, y_predicted_rf, alpha=0.2, color='chocolate', label='Random forest')
    plt.xlim(0, 1e8)
    plt.ylim(0, 1e8)
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()
    # plt.scatter(y_predicted_lr, y_predicted_rf, alpha=0.2)
    # plt.xlim(0, 1e8)
    # plt.ylim(0, 1e8)
    # plt.xlabel('Linear regression')
    # plt.ylabel('Random forest')
    # plt.show()

    data_valid['price_doc'] = y_valid
    data_valid['price_doc_predicted'] = y_predicted
    data_valid['transaction_year'] = data_valid.timestamp.map(lambda x: pd.to_datetime(x).year)
    colors = {2013: 'red', 2014: 'blue', 2015: 'yellow'}
    col = [colors[el] for el in data_valid.transaction_year.values]
    plt.scatter(data_valid.price_doc, data_valid.price_doc_predicted, color=col, alpha=0.5)
    price_doc_max = data_valid.price_doc.max()
    price_doc_predicted_max = data_valid.price_doc_predicted.max()
    axis_lim = 1.1*max(price_doc_max, price_doc_predicted_max)
    plt.xlim(0, axis_lim)
    plt.ylim(0, axis_lim)
    plt.plot(data_valid.price_doc, data_valid.price_doc, linestyle='dotted', color='grey', alpha=0.3)
    plt.xlabel('actual')
    plt.ylabel('predicted')
