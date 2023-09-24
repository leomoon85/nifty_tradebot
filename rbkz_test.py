import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np 
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from finta import TA
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import holoviews as hv 
import json 
import requests
from sentipy.sentipy import Sentipy 
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVC
from breeze_connect import BreezeConnect
import urllib
# going to use show() to open plot in browser
from bokeh.plotting import show
#from imblearn import under_sampling, over_sampling
#from imblearn.over_sampling import SMOTE
#---------------------------------------------------
# Initialize ICICI breeze SDK
#---------------------------------------------------
breeze = BreezeConnect(api_key="<API KEY>")
print("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus("<API KEY>"))

# Generate Session
breeze.generate_session(api_secret="<SECRET KEY>",
                        session_token="<SESSION KEY>")
# Generate ISO8601 Date/DateTime String
iso_date_string = datetime.strptime("28/02/2021","%d/%m/%Y").isoformat()[:10] + 'T05:30:00.000Z'
iso_date_time_string = datetime.strptime("28/02/2021 23:59:59","%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'
breeze.get_customer_details(api_session="your_api_session") 
#hist data from and to dates 
frm_date = "2023-08-17T07:00:00.000Z"
to_date =  "2023-08-23T07:00:00.000Z"
yf_frm_date = frm_date[:10]; 
yf_to_date = to_date[:10];
cols_to_keep = ['date','open', 'high', 'low', 'close','volume']
#---------------------------------------------------
# get data from icici
#---------------------------------------------------
# Connect to websocket(it will connect to tick-by-tick data server)
breeze.ws_connect()

# Callback to receive ticks.
def on_ticks(ticks):
    print("Ticks: {}".format(ticks))


##############################################################################################
# ## **STEP 2:Fundamental Analysis**

# # **Sentiment Indicators** TODO

# *** Calling in Sentiment Analysis Indicator Data from Sentiment Investor API ***



##############################################################################################
# get data 
# Assign the callbacks.
breeze.on_ticks = on_ticks
# get historic data NSE/NFO 
data = breeze.get_historical_data(interval="1minute",
                            from_date= frm_date,
                            to_date= to_date,
                            stock_code="NIFTY",
                            exchange_code="NSE",
                            product_type="cash")

amc_stock = pd.DataFrame(data['Success'])
amc_close = pd.DataFrame(amc_stock).reset_index()
#Name columns
#amc_close.columns = ['date','open', 'high', 'low', 'close', 'adj_close', 'volume']
amc_close.columns = ['index','date','stock_code','exchange_code','product_type','expiry_date','right','strike_price','open', 'high', 'low', 'close', 'volume','open_interest','count']
amc_close = amc_close[cols_to_keep]
# extra manipulations 
amc_close['adj_close'] = amc_close['close']
amc_close['volume'] = 0
# only trading time data , remaining remove
amc_close['date'] = pd.to_datetime(amc_close['date'])
#remove weekends data
amc_close = amc_close[amc_close['date'].dt.dayofweek < 5]
#Set "Date" as index column
amc_close = amc_close.set_index('date')\
#remove non trading hours data        
amc_close = amc_close.between_time(start_time='8:59', end_time='15:31')
#store into a csv 
amc_close.to_csv('nifty_history.csv',index=True);
##############################################################################################

#Create new DataFrame using only 'close' price
amc_close = pd.DataFrame(amc_close['close'])
#move the amc to merge_df - TODO merge sentimets to this 
merged_df = amc_close 

#Creating new DataFrame to manipulate for plot visualization
mean_sa_df = merged_df
#Groupby day and mean the values to smooth out line plots
mean_sa_df = mean_sa_df.groupby('date').mean()
#Changing scale of Daily Closing Prices to better scale with other indicators
mean_sa_df['close'] = mean_sa_df['close'] / 10

#print (mean_sa_df)
#Creating individual plots to analyze how indicators trend together. 
daily_close_plot = mean_sa_df["close"].hvplot(
    color='black',
    width=1000,
    height=500,
    label = 'Daily Closing Price'
)
#Overlay all plots to visualize correlation between indicators and closing prices. 

hv.extension('bokeh')
#TODO sa_plot = (rhi_plot * ahi_plot * sgp_plot * sentiment_plot * daily_close_plot).opts(title='AMC Daily Close Prices vs. Sentiment Analysis Indicators 2021-04-15 to 2021-07-15', xlabel="Date", ylabel="AMC Closing Price (10% scale)")
sa_plot = daily_close_plot.opts(title='AMC Daily Close Prices vs. Sentiment Analysis Indicators ', xlabel="Date", ylabel="AMC Closing Price (10% scale)")
show(hv.render(sa_plot))

##############################################################################################

# # **STEP 3: Technical Analysis**

# # **Technical Indicators**

stock_ohlcav = pd.read_csv("nifty_history.csv")
stock_df = stock_ohlcav

#Name the columns of the new DataFrame
#stock_df.columns = ['index','date','stock_code','exchange_code','product_type','expiry_date','right','strike_price','open', 'high', 'low', 'close', 'volume','open_interest','count']
print ("####################################################")
#cols_to_keep = ['date','open', 'high', 'low', 'close']
##stock_df = stock_df[cols_to_keep]
#Set the Date as the index
stock_df = stock_df.set_index('date')
print (stock_df)

##Calling in technical indicators from Yahoo Finance
## #Make API call to Yahoo Finance to pull in ticker data
## stock_ohlcav = yf.download('^NSEI', start= yf_frm_date, end= yf_to_date, interval = '1h')
## #Create DataFrame from ticker data
## stock_df = pd.DataFrame(stock_ohlcav).reset_index()
## #Name the columns of the new DataFrame
## stock_df.columns = ['date','open', 'high', 'low', 'close', 'adj_close', 'volume']
## #Set the Date as the index
## stock_df = stock_df.set_index('date')
## 
## 
## print ("####################################################")
## print (stock_df)



stock_df['rsi'] = TA.RSI(stock_df)
stock_df['cci'] = TA.CCI(stock_df)
stock_df[['macd','macd_signal']] = TA.MACD(stock_df)

stock_df['signal'] = 0.0


stock_df.loc[(stock_df['rsi'] <= 43) & (stock_df['cci'] >= -100) & (stock_df['macd'] > stock_df['macd_signal'] ), 'signal'] = 1

stock_df.loc[(stock_df['rsi'] >= 70) & (stock_df['cci'] >= 100) & (stock_df['macd'] < stock_df['macd_signal'] ), 'signal'] = -1
# temp print 
stock_df.to_csv('nifty_history_ta.csv',index=True);

##############################################################################################
## Ploting the tech analysis buy/sell chart 
hv.extension('bokeh')

entry = stock_df[stock_df['signal'] == 1]['close'].hvplot.scatter(
    color='green',
    marker='^',
    width=1000,
    height=500,
    label = 'buy')

exit = stock_df[stock_df['signal'] == -1]['close'].hvplot.scatter(
    color='red',
    marker='v',
    width=1000,
    height=500,
    label = 'sell')

closing_price = stock_df[['close']].hvplot(
    line_color = 'yellow',
    width=1000,
    height=500,
    label = 'closing price',
    title = 'RSI/CCI/MACD Technical Indicators Entry And exit Points')


hv.extension('bokeh')

entry_exit_graph = entry * exit * closing_price
# use show() from bokeh
show(hv.render(entry_exit_graph))
##############################################################################################
# sentiments data need to take TODO 

sa_ti_df = stock_df

##############################################################################################
# # **STEP 4: Machine Learning**

# *** Preparing the DataFrame for the Machine Learning process***
# 

#Filling any NaN values with 0 and seeing the info on the DataFrame
sa_ti_df = sa_ti_df.fillna(0)
sa_ti_df.info()
#Split the training and testing data
#TODO X = sa_ti_df.drop(columns = 'strategy_signal')
X = sa_ti_df.drop(columns = 'signal')
#TODO y = sa_ti_df['strategy_signal']
y = sa_ti_df['signal']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#Scaling the X_train and X_test data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
from imblearn.over_sampling import RandomOverSampler
# Use RandomOverSampler to resample the dataset using random_state=1
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
# *** Fit the Data to the Machine Learning Models ***
# 
#Creating the SVC model instance and fitting to the resampled data
svc_model = SVC()
svc_model = svc_model.fit(X_resampled, y_resampled)
#Creating a LogisticRegression model for comparison
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
lr = lr.fit(X_resampled, y_resampled)


# *** Predict the Signals Using the ML Models ***
# 
#Have the SVC model make predictions on the resampled data
svc_predictions = svc_model.predict(X_resampled)
svc_training_report = classification_report(y_resampled, svc_predictions)
#Have the LogisticRegression model make predictions on the resampled data
lr_predictions = lr.predict(X_resampled)
lr_training_report = classification_report(y_resampled, lr_predictions)
#View the training reports
print(svc_training_report + lr_training_report)
#Backtest the algorithm against the SVC and LogisticRegression model predictions
svc_test_predictions = svc_model.predict(X_test_scaled)
svc_test_report = classification_report(y_test, svc_test_predictions)

lr_test_predictions = lr.predict(X_test_scaled)
lr_test_report = classification_report(y_test, lr_test_predictions)
#Print the test prediction reports

print (svc_test_report + lr_test_report)
# Create a new empty predictions DataFrame using code provided below.
predictions_df = pd.DataFrame(index=X_test.index)
predictions_df['svc_predicted_signals'] = svc_test_predictions
predictions_df['lr_predicted_signals'] = lr_test_predictions
predictions_df = predictions_df.sort_index()
print(predictions_df)

#Plotting the prediction signals
hv.extension('bokeh')

predictions_df_plot = predictions_df.hvplot().opts(title='Predicted Signals - SVC vs. Multinomial Logistic Regression', xlabel="Date", ylabel="Signal")
show(hv.render(predictions_df_plot))
# **Option Returns Analysis**
amc_ticker = breeze.get_option_chain_quotes(stock_code="NIFTY",
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date="2023-09-28T06:00:00.000Z",
                    right="call")
#print(amc_ticker)
#Use Yahoo Finance to bring in option chains
#amc_ticker = yf.Ticker('^NSEI')
#amc_ticker.options
#Choose option chain for 9/17/2021 Expiration Date
amc_options = pd.DataFrame(amc_ticker['Success'])
## amc_ticker = breeze.get_option_chain_quotes(stock_code="NIFTY",
##                     exchange_code="NFO",
##                     product_type="options",
##                     expiry_date="2023-09-28T06:00:00.000Z",
##                     right="put")
## amc_options_put = pd.DataFrame(amc_ticker['Success'])
#amc_options.loc[len(amc_options.index)] = pd.DataFrame(amc_ticker['Success'])

#amc_options = amc_options.dropna(subset='ltt')
amc_options = amc_options[(amc_options['ltp'] > 0)]
print(amc_options)
#print(amc_options_put)
#Create Dataframe from the option chain
amc_calls = pd.DataFrame(amc_options).set_index('ltt')
amc_calls.tail()

#Create a new column for purchase price for easier calculation of returns
amc_calls['Purchase Price'] = ((amc_calls['best_bid_price'] + amc_calls['best_offer_price']) / 2) * 100
#print(amc_calls)
#Create a new DataFrame for calculation AMC option returns by dropping columns not necessary for calculation
#amc_call_returns = amc_calls.drop(columns = ['contractSymbol', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency'])
amc_call_returns = amc_calls.drop(columns = ['total_buy_qty','total_sell_qty','right','ltp','best_bid_quantity','best_offer_quantity','open_interest','exchange_code','product_type','stock_code','expiry_date','best_bid_price','best_offer_price','open','high','low','previous_close','ltp_percent_change','upper_circuit','lower_circuit','total_quantity_traded','spot_price','ltq','chnge_oi'])
print(amc_call_returns)
#Create new columns for returns on hypothetical scenarios on if it reached $40, $100, $150
#Only using increasing values since the model predicted an upward trend, and only buying calls

amc_call_returns['40 Returns'] = ((40000 - amc_call_returns['strike_price']) * 100) - amc_call_returns['Purchase Price']
amc_call_returns['100 Returns'] = ((100000 - amc_call_returns['strike_price']) * 100) - amc_call_returns['Purchase Price']
amc_call_returns['150 Returns'] = ((150000 - amc_call_returns['strike_price']) * 100) - amc_call_returns['Purchase Price']

print(amc_call_returns)
#Create new columns for the percent returns on the option calls for each scenario
amc_call_returns['40 PReturns'] = amc_call_returns['40 Returns']/amc_call_returns['Purchase Price']
amc_call_returns['100 PReturns'] = amc_call_returns['100 Returns']/amc_call_returns['Purchase Price']
amc_call_returns['150 PReturns'] = amc_call_returns['150 Returns']/amc_call_returns['Purchase Price']

amc_call_returns[20:60]
hv.extension('bokeh')

amc_returns_plot = amc_call_returns.hvplot(title = 'AMC Call Options Returns - ', x = 'strike_price', xlabel = 'Strike Price', y = ['40 Returns', '100 Returns', '150 Returns'], ylabel = 'Returns')

show(hv.render(amc_returns_plot))



amc_call_returns.to_csv('nifty_opt_call.csv',index=True);


# # **STEP 5: Auto Trading Algorithm**





##############################################################################################
# ws_disconnect (it will disconnect from all actively connected servers)
breeze.ws_disconnect()
