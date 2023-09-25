import os
import numpy as np
from datetime import timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from breeze_connect import BreezeConnect
import urllib

class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = stock
        #self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def get_stock_short_name(self):
        return self._sec

    def get_stock_currency(self):
        return "INR"

    def get_min_max(self):
        return self._min_max

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        original_price = latest_close_price

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            original_price = original_price + (original_price * random_slope)
            #print(original_price)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)

        test_data = pd.DataFrame({'date': x_future, 'close': y_future})
        test_data = test_data.set_index('date')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        #print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            #print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data


    # Callback to receive ticks.
    def on_ticks(ticks):
        print("Ticks: {}".format(ticks))

    def download_transform_to_numpy(self, time_steps, project_folder):
        breeze = BreezeConnect(api_key="0475177197l66221Bz93*23ts5`5461#")
        print("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus("0475177197l66221Bz93*23ts5`5461#"))
        # Generate Session
        breeze.generate_session(api_secret="13214FH!314215`812iP39@o69A4wt23",session_token="22427835")
        end_date = datetime.today()
        end_date = datetime.strptime(end_date.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'
        print('End Date: ' + end_date)
        frm_date = self._stock.get_start_date() #"2023-08-17T07:00:00.000Z"
        frm_date = datetime.strptime(frm_date.strftime("%d/%m/%Y %H:%M:%S"),"%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'
        # get historic data NSE/NFO 
        data = breeze.get_historical_data(interval="1minute",
                            from_date= frm_date,
                            to_date= end_date,
                            stock_code="NIFTY",
                            exchange_code="NSE",
                            product_type="cash")
        data = pd.DataFrame(data['Success'])
        cols_to_keep = ['date','open', 'high', 'low', 'close','volume']
        data = pd.DataFrame(data).reset_index()
        data.columns = ['index','date','stock_code','exchange_code','product_type','expiry_date','right','strike_price','open', 'high', 'low', 'close', 'volume','open_interest','count']
        data = data[cols_to_keep]
        data = data[['date','close']]
        data['date'] = pd.to_datetime(data['date'])
        #remove weekends data
        data = data[data['date'].dt.dayofweek < 5]
        #Set "Date" as index column
        data = data.set_index('date')\
        #remove non trading hours data        
        data = data.between_time(start_time='9:14', end_time='15:31')
        #store into a csv 
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        data = pd.read_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        data['date'] = pd.to_datetime(data['date'])
        training_data = data[data['date'] < self._stock.get_validation_date()].copy()
        test_data = data[data['date'] >= self._stock.get_validation_date()].copy()
        training_data = training_data.set_index('date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('date')
        #print(test_data)

        train_scaled = self._min_max.fit_transform(training_data)
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = self._min_max.fit_transform(inputs)

        # Testing Data Transformation
        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

