import pandas as pd
from feature_engineering import Featurizer
from model_selection import ModelSelector
from datetime import datetime
from sklearn.model_selection import train_test_split

def main():
    # Prepare data
    featurizer_train = Featurizer(train_test='train')
    X, y = featurizer_train.featurize()
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train model
    ms = ModelSelector()
    ms.fit(X_train, y_train)
    predictor = ms.best_predictor()

    # Predict test data
    featurizer_test = Featurizer(train_test='test')
    X_test, _ = featurizer_test.featurize()
    y_predicted = predictor.predict(X_test)

    # Create prediction file for Kaggle submission
    tmstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
    house_ids = featurizer_test.get_house_ids()
    kaggle = pd.DataFrame({'id': house_ids, 'price_doc': y_predicted})
    kaggle.to_csv('/Users/stefanobrunelli/github/russian_housing_market/predictions/predictions'+tmstmp+'.csv'
        , index=False)

if __name__ == "__main__":
    main()



from feature_engineering import Featurizer
f_train = Featurizer(train_test='train')
X_train, y_train = f_train.featurize()
f_test = Featurizer(train_test='test')
X_test, y_test = f_test.featurize()
