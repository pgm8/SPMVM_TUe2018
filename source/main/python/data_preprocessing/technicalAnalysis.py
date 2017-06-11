# Imports
import numpy as np
import pandas as pd
import sys
from pandas_datareader import data
import datetime as dt
from pandas import read_csv

import talib as ta
import matplotlib.pyplot as plt
import os


# ^GSPC: S&P500 Index (large cap stock market index) but SPY is ETF that follows the index.
# ^RUT : Russell 2000 Index (small cap stock market index) It is the most widely quoted measure of the overall performance of the small-cap to mid-cap company shares
#  EEM : iShares MSCI Emerging Markets ETF
# ^SPGSCI: S&P GS commodity index (but unfortunately does not work)


# Download data directly from finance.yahoo
"""
start = dt.datetime(1992, 1, 1)
end = dt.datetime(2016, 12, 31)
spx = data.DataReader('SPY', 'yahoo', start, end)

# Delete Adjusted Close Price column
df = spx.drop('Adj Close', axis=1) # Wont use Adjusted Close Price as feature
#print(spx.head())
spx = pd.DataFrame(df)  # web.DataReader returns Pandas panel object not a DataFrame
spx = spx.reset_index()
print(spx.head())
print(spx.tail())
"""

# Load SP500 data from csv file downloaded from finance.yahoo
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/SPY.csv')
spx = pd.read_csv(filename)
spx = spx.drop('Adj Close', axis=1)
column_names = ['Open', 'High', 'Low', 'Close', 'Volume']
for name in column_names:
    spx[name] = pd.to_numeric(spx[name], errors='coerce')

# Basic Data preprocessing operations
# 1) Scan dataset on NaN values
pd.isnull(spx).any(1).nonzero()[0] # No rows with missing values.



# 2) Feature Engineering
################### Lagging Technical Indicators #######################

# Simple Moving Average SMA(5):
# Technical Indicator Calculation
spx['sma5'] = ta.SMA(np.asarray(spx['Close']), 5)
spx['sma10'] = ta.SMA(np.asarray(spx['Close']), 10)
spx['sma20'] = ta.SMA(np.asarray(spx['Close']), 20)

# Exponential Moving Average EMA(5, 10 , 20)
# Technical Indicator Calculation
spx['ema5'] = ta.EMA(np.asarray(spx['Close']), 5)
spx['ema10'] = ta.EMA(np.asarray(spx['Close']), 10)
spx['ema20'] = ta.EMA(np.asarray(spx['Close']), 20)
spx['ema100'] = ta.EMA(np.asarray(spx['Close']), 100)
spx['ema200'] = ta.EMA(np.asarray(spx['Close']), 200)

# Bollinger Bands BB(20,2): SMA(20) +- 2*std(20)
# Technical Indicator Calculation (matype=0: simple MA)
spx['BBhigh'], spx['BBmid'], spx['BBlow'] = ta.BBANDS(np.asarray(spx['Close']),
                                                      timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
# Moving average Divergence/Convergence MACD(12,26,9): macdHist is what we are mainly interested in
# Technical Indicator Calculation
spx['macd'], spx['macdSignal'], spx['macdHist'] = ta.MACD(np.asarray(spx['Close']), fastperiod=12, slowperiod=26,
                                                          signalperiod=9)


################# Leading Technical Indicators #################

# Average Directional Movement Index ADX(14)
# Technical Indicator Calculation
spx['adx14'] = ta.ADX(np.asarray(spx['High']), np.asarray(spx['Low']), np.asarray(spx['Close']), timeperiod=14)
spx['di+'] = ta.PLUS_DI(np.asarray(spx['High']), np.asarray(spx['Low']), np.asarray(spx['Close']), timeperiod=14)
spx['di-'] = ta.MINUS_DI(np.asarray(spx['High']), np.asarray(spx['Low']), np.asarray(spx['Close']), timeperiod=14)

# Commodity Channel Index CCI(20,0.015)
# technical Indicator Calculation
spx['cci'] = ta.CCI(np.asarray(spx['High']), np.asarray(spx['Low']), np.asarray(spx['Close']),timeperiod=20)

# Rate of Change ROC(21): one month rate of change
# Technical Indicator Calculation
spx['roc21'] = ta.ROC(np.asarray(spx['Close']), timeperiod=21)

# Relative Strength Index RSI(14): 14 is commonly used
# Technical Indicator Calculation
spx['rsi14'] = ta.RSI(np.asarray(spx['Close']), timeperiod=14)

# Stochastic Oscillator Full (slow) STOCH(14,3,3): Need to convert data to double float (TA-Lib doesnt like real data)
# Technical Indicator Calculation
spx['slow%K'], spx['slow%D'] = ta.STOCH(
    np.asarray([float(x) for x in spx['High']]), np.asarray([float(x) for x in spx['Low']]),
    np.asarray([float(x) for x in spx['Close']]), fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
    slowd_matype=0)
# SO Chart
plt.subplot(2, 1, 1)
plt.title('S&P500 Close Prices & Slow Stochastic Oscillator STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
spx.plot(x=['Date'], y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
spx.plot(x=['Date'], y=['slow%K'], color='r', linestyle='--')
spx.plot(x=['Date'], y=['slow%D'], color='g')
plt.legend(loc='upper left')
# plt.show()


# Williams %R WILLR(14)
# Technical Indicator Calculation
spx['%r'] = ta.WILLR(np.asarray(spx['Low']), np.asarray(spx['High']), np.asarray(spx['Close']), timeperiod=14)

# On Balance Volume OBV
# Technical Indicator Calculation
spx['obv'] = ta.OBV(np.asarray(spx['Close']), np.asarray([float(x) for x in spx['Volume']]))



# RESPONSE VARIABLE y_t = P_{t+1}^C
# Response Variable Calculation
for i in range(len(spx['Close'])-1):
    spx.loc[i, 'response'] = np.asarray(spx['Close'])[i+1]


# 200 rows with NaN values (as expected: consequence from feature engineering)
sum([True for idx, row in spx.iterrows() if any(row.isnull())])
#print(pd.isnull(spx).any(1).nonzero()[0])


# Drop rows with NaN values
spx = spx.dropna()
#sum([True for idx, row in spx.iterrows() if any(row.isnull())])

# Write data into CSV file
spx.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/SP500_data.csv'))

