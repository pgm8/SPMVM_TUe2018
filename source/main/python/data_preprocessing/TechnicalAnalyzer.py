import numpy as np
import talib as ta


class TechnicalAnalyzer(object):
    """Technical analysis class. This class has the responsibility to engineer the feature space.
    It does so by the computation of several technical indicators."""

    def __init__(self, data, full=True):
        """Initializer TechnicalAnalyzer object.
        :param data: dataset containing historical price and volume information."""
        self.data = data
        self.dailyReturn(data)
        """If statement to decide whether or not extensive feature engineering or
        only computation of response variable."""
        if full is True:
            self.simpleMovingAverage(data)
            self.exponentialMovingAverage(data)
            self.bollingerBands(data)
            self.movingAverageConvergenceDivergence(data)
            self.averageDirectionalMovementIndex(data)
            self.commodityChannelIndex(data)
            self.rateOfChange(data)
            self.relativeStrengthIndex(data)
            self.stochasticOscillatorFull(data)
            self.williamsR(data)
            self.onBalanceVolume(data)
        self.responseVariable(data)

    def dailyReturn(self, data):
        """Method for computing daily asset return.
        :param data:
        :return:
        """
        for i in range(len(data['Close'])-1):
            data.loc[i+1, 'Return'] = ((np.asarray(data['Close'])[i+1]) - (np.asarray(data['Close'])[i])) / \
                                      np.asarray(data['Close'])[i]
        return data

    def simpleMovingAverage(self, data):
        """Method for computing simple moving average technical indicator.
        :param data:
        :return:
        """
        data['sma5'] = ta.SMA(np.asarray(data['Close']), 5)
        data['sma10'] = ta.SMA(np.asarray(data['Close']), 10)
        data['sma20'] = ta.SMA(np.asarray(data['Close']), 20)
        return data

    def exponentialMovingAverage(self, data):
        """Method for computing exponential moving average technical indicator.
        :param data:
        :return:
        """
        data['ema5'] = ta.EMA(np.asarray(data['Close']), 5)
        data['ema10'] = ta.EMA(np.asarray(data['Close']), 10)
        data['ema20'] = ta.EMA(np.asarray(data['Close']), 20)
        data['ema100'] = ta.EMA(np.asarray(data['Close']), 100)
        data['ema200'] = ta.EMA(np.asarray(data['Close']), 200)
        return data

    def bollingerBands(self, data):
        """Method for computing bollinger bands technical indicator.
        :param data:
        :return:
        """
        data['BBhigh'], data['BBmid'], data['BBlow'] = ta.BBANDS(np.asarray(data['Close']),
                                                                 timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        return data

    def movingAverageConvergenceDivergence(self, data):
        """Method for computing moving average convergence divergence technical indicator.
        :param data:
        :return:
        """
        data['macd'], data['macdSignal'], data['macdHist'] = ta.MACD(np.asarray(data['Close']), fastperiod=12,
                                                                     slowperiod=26, signalperiod=9)
        return data

    def averageDirectionalMovementIndex(self, data):
        """Method for computing average directional movement index technical indicator.
        :param data:
        :return:
        """
        data['adx14'] = ta.ADX(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']),
                               timeperiod=14)
        data['di+'] = ta.PLUS_DI(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']),
                                 timeperiod=14)
        data['di-'] = ta.MINUS_DI(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']),
                                  timeperiod=14)
        return data

    def commodityChannelIndex(self, data):
        """Method for computing commodity channel index technical indicator.
        :param data:
        :return:
        """
        data['cci'] = ta.CCI(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']),
                             timeperiod=20)
        return data

    def rateOfChange(self, data):
        """Method for computing rate of change technical indicator.
        :param data:
        :return:
        """
        data['roc21'] = ta.ROC(np.asarray(data['Close']), timeperiod=21)
        return data

    def relativeStrengthIndex(self, data):
        """Method for computing relative strength index technical indicator.
        :param data:
        :return:
        """
        data['rsi14'] = ta.RSI(np.asarray(data['Close']), timeperiod=14)
        return data

    def stochasticOscillatorFull(self, data):
        """Method for computing full stochastic oscillator technical indicator.
        :param data:
        :return:
        """
        data['slow%K'], data['slow%D'] = ta.STOCH(
            np.asarray(map(lambda x: x, data['High'])), np.asarray(map(lambda x: x, data['Low'])),
            np.asarray(map(lambda x: x, data['Close'])), fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0)
        return data

    def williamsR(self, data):
        """Method for computing williams % R technical indicator.
        :param data:
        :return:
        """
        data['%r'] = ta.WILLR(np.asarray(data['Low']), np.asarray(data['High']), np.asarray(data['Close']),
                              timeperiod=14)
        return data

    def onBalanceVolume(self, data):
        """Method for computing on balance volume technical indicator.
        :param data:
        :return:
        """
        data['obv'] = ta.OBV(np.asarray(data['Close']), np.asarray(map(lambda x: float(x), data['Volume'])))
        return data

    def responseVariable(self, data):
        """Method for computing the response variable: next day's percentage asset return.
        :param data:
        :return:
        """
        for i in range(len(data['Close'])-1):
            data.loc[i, 'response'] = np.asarray(data['Return'])[i+1]
        return data


"""    
# 200 rows with NaN values (as expected: consequence from feature engineering)
sum([True for idx, row in spx.iterrows() if any(row.isnull())])

# Drop rows with NaN values
spx = spx.dropna()

# Reset row indices starting at zero
spx = spx.reset_index(drop=True)

# Write data into CSV file
spx.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/Data/SP500_data.csv'))
"""