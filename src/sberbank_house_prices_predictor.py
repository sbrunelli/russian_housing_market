from model_selection import ModelSelector
from datetime import datetime

class HousePricesPredictor(object):

    def __init__(self):
        pass

    def create_kaggle_submission_file(self):
        ms = ModelSelector()
        ms._get_features()
        tmstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_data = ms.create_kaggle_submission_data()
        submission_data.to_csv('./predictions/predictions'+tmstmp+'.csv'
        , index=False)
