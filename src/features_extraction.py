import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LassoCV, lasso_path, ElasticNetCV, enet_path
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
plt.interactive(True)

if __name__ == '__main__':
    # Read the data
    train = pd.read_csv('./data/train.csv')
    macro = pd.read_csv('./data/macro.csv')
    train.pop('id')
    train = train.merge(macro, how='left', on='timestamp', suffixes=('_train', '_macro'))
    print 'train shape', train.shape

    # Remove all categorical variables for now
    mask = ~(train.dtypes == 'object').values
    train = train.iloc[:, mask]
    print 'train shape with only numerical features', train.shape

    # Print NAs proportions for features with NA values
    # print train.apply(lambda x: np.mean(x.isnull())).sort_values(ascending=False)

    # Convert all columns to float
    train = train.apply(lambda x: x.astype('float'))

    # Impute NAs with median values
    train = train.apply(lambda x: x.fillna(x.median()))

    # Print NAs proportions for features with NA values
    # print train.apply(lambda x: np.mean(x.isnull())).sort_values(ascending=False)

    #---------------------------------
    # Run LASSO to estimate RMSE
    #---------------------------------
    alphaMin = 10000
    alphaGrid = np.linspace(alphaMin, alphaMin*(1/1e-3), 100)
    y = train.pop('price_doc')
    X = train.values
    X = preprocessing.scale(X)
    PriceModel = LassoCV(cv=10, max_iter=100000, alphas=alphaGrid).fit(X, y)
    bestAlpha = PriceModel.alpha_
    MSE = min(PriceModel.mse_path_.mean(axis=-1))
    print 'Best alpha', bestAlpha
    print 'RMSE', np.sqrt(MSE)
    # Add RMSLE

    # Run lasso_path to estimate variable importance
    alphas, coefs, _ = lasso_path(X, y, alphas=alphaGrid, return_models=False)

    # Find coefficients ordering
    nattr, nalpha = coefs.shape

    # Non zero coefficients in the order they enter the model as model
    # complexity increases (variance over bias)
    nzList = []
    for iAlpha in range(nalpha):
        # Coefficients with current value of alpha (lambda)
        coefList = list(coefs[:, iAlpha])
        # Indices for non zero coefficients
        nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
        # Add coefficient index if not already in nzList
        for q in nzCoef:
            if not (q in nzList):
                nzList.append(q)

    names = train.columns.values.tolist()
    nameList = [names[nzList[i]] for i in range(len(nzList))]
    print '\n\n\nAttributes ordered by how early they enter the model\n'
    for idx, n in enumerate(nameList):
        print str(idx+1), n

    # Find coefficients corresponding to best value of alpha
    indexLTbestAlpha = [index for index in range(nalpha) if alphas[index] > bestAlpha]
    indexStar = max(indexLTbestAlpha)
    coefStar = list(coefs[:, indexStar])
    # print 'Best coefficients', coefStar

    # Sort by magnitude
    absCoef = [abs(a) for a in coefStar]
    coefSorted = sorted(absCoef, reverse=True)
    idxCoefSize = [absCoef.index(a) for a in coefSorted if not (a==0.0)]
    namesList2 = [names[idxCoefSize[i]] for i in range(len(idxCoefSize))]
    print '\n\n\nAttributes ordered by coefficient size at best alpha\n',
    for idx, n in enumerate(namesList2):
        print str(idx+1), n

    #---------------------------------
    # Run Elastinet to estimate RMSE
    #---------------------------------


    # Get Feature importance from Random Forest
    # rf = RandomForestRegressor(n_estimators=100)
    # rf.fit(X, y)
    # importances = rf.feature_importances_
    # indexRFfeatureImportances = np.argsort(importances)[::-1]
    # nameListRF = [names[indexRFfeatureImportances[i]] for i in range(len(indexRFfeatureImportances))]
    # print '\n\n\nRandom Forest feature importances\n', nameListRF
