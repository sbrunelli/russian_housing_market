import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tools.tools import add_constant
from sklearn import linear_model
from time import sleep
import matplotlib.pyplot as plt

plt.interactive(True)

def read_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    macro = pd.read_csv('../data/macro.csv')
    data_train = train.merge(macro, how='left', on='timestamp', suffixes=('_data', '_macro'))
    data_kaggle = test.merge(macro, how='left', on='timestamp', suffixes=('_data', '_macro'))
    return data_train, data_kaggle


if __name__ == "__main__":
    # Read data
    data, kaggle = read_data()
    data.pop('timestamp')
    data.pop('id')
    kaggle.pop('timestamp')

    # Get number of rows of respective data frames


    # Split quantitative and categorical variables
    data['modern_education_share'] = data.modern_education_share.map(lambda x: float(str(x).replace(',','.')))
    data['old_education_build_share'] = data.modern_education_share.map(lambda x: float(str(x).replace(',','.')))
    data['child_on_acc_pre_school'] = data.child_on_acc_pre_school.map(lambda x: np.nan if x=='#!' else x)
    data['child_on_acc_pre_school'] = data.child_on_acc_pre_school.map(lambda x: float(str(x).replace(',','')))
    categorical = data.dtypes == 'object'
    quantitative = ~(categorical)
    data_categorical = data.loc[:, categorical].copy()
    data_quantitative = data.loc[:, quantitative].copy()

    # Exclude from analysis columns with too many NAs to begin with

    # Dummify categorical variables
    # data_categorical = pd.get_dummies(data_categorical, drop_first=True)
    # data = pd.concat([data_quantitative, data_categorical], axis=1)
    data = data_quantitative
    target = data.pop('price_doc')

    # Train test split
    train_df, test_df, target_train, target_test = train_test_split(data, target)

    # Fill quantitative NAs with means
    train_df = train_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    test_df = test_df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Create X_train, y_train
    train_df['constant'] = 1.0
    # X_train = train_df.values
    y_train = np.array(target_train)

    # Create X_test, y_test
    test_df['constant'] = 1.0
    X_test = test_df.values
    y_test = np.array(target_test)

    # Forward stepwise selection
    attributeList = []
    attributeCandidates = train_df.columns.values.tolist()
    oosError = []

    print
    print
    print ' # Starting stepwise forward selection'
    for i in range(100):
        print
        print
        print ' # Iteration nr. {:d}'.format(i+1)
        # print ' # Model length so far: {:d}'.format(len(set(attributeList)))
        print ' # Nr. of attributes to try: {:d}'.format(len(attributeCandidates))
        # sleep(5)
        attrSet = set(attributeList)
        attributeCandidatesSet = set(attributeCandidates)
        attrTrySet = attributeCandidatesSet - attrSet
        attrTry = list(attrTrySet)
        rmsle_iter = []
        attTemp = []
        for attr in attrTry:
            # print ' # Trying attribute: {:s}'.format(attr)
            attTemp = [] + attributeList
            attTemp.append(attr)
            # print ' # Used features: {:s}'.format(attTemp)
            X_train = train_df.loc[:, attTemp].values
            X_test = test_df.loc[:, attTemp].values
            # print ' # X_train shape: {:s}'.format(X_train.shape)
            # print ' # X_test shape: {:s}'.format(X_test.shape)
            housePricesModel = linear_model.LinearRegression()
            housePricesModel.fit(X_train, y_train)
            y_predicted = housePricesModel.predict(X_test)
            rmsle = np.sqrt(np.sum((np.log(y_predicted+1) - np.log(y_test+1))**2)/len(y_predicted))
            if np.isnan(rmsle):
                rmsle_iter.append(np.Inf)
            else:
                rmsle_iter.append(rmsle)
        iBest = np.argmin(rmsle_iter)
        attributeList.append(attrTry[iBest])
        # attributeCandidates.remove(attrTry[iBest])
        oosError.append(rmsle_iter[iBest])

    print
    print
    print ' # Selected attributes: '
    print attributeList

    # Plot results
    x = range(len(oosError))
    plt.plot(x, oosError, 'k')
    plt.xlabel('Number of attributes')
    plt.ylabel('RMSLE')
    plt.show()
