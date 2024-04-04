import joblib
import numpy as np
import  pandas as pd
from datetime import datetime as dt
import category_encoders as ce

class FraudPrediction:
    def __init__(self):
        self.model = joblib.load('../models/gboost_ru.pkl')
        
        
    def preprocess(self, data):
        data['age'] = int(data['age'])
        # data['amount'] = int(data['amount'])
        data['amount'] = 1000000000000
        data['hour'] = dt.strptime(data['timeOfTransaction'], '%H:%M').hour
        data['amt_log'] = np.log(data['amount'])
        
        woe = ce.WOEEncoder()
        print(data['category'])
        data['category_WOE'] = 1
        data['city_WOE'] = 1
        data['job_WOE'] = 1
        # data['category_WOE'] = woe.fit_transform(data['category'], 1)
        # data['city_WOE'] = woe.fit_transform(data['city'], 1)
        # data['job_WOE'] = woe.fit_transform(data['job'], 1)

        data['job_WOE'] =  31
        data['cc_num_frequency'] = 10000000
        data['merch_lat'] = 40.7128    # NYC lat
        
        columns = ['merch_lat', 'age', 'hour', 'amt_log', 'category_WOE', 'city_WOE', 'job_WOE', 'cc_num_frequency']

        model_data = np.array([[data[column] for column in columns]])
        
        return model_data

    def predict(self, data):
        preproccess_data = self.preprocess(data)
        prediction = self.model.predict(preproccess_data)
        return prediction[0]