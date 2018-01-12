import numpy as np
import os.path
import re
import time

from ModuleManager import ModuleManager

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

"""
The "canonical" way to do time-series cross-validation is to roll through the dataset. Basically, the training set
should not contain information that occurs after the test set, hence k-fold cross-validation is not appropiate.

In other words: When the data are not independent, cross-validation becomes more difficult as leaving out an observation
does not remove all the associated information due to the correlations with other observations. For time series 
forecasting, a cross-validation statistic may be obtained as follows:

1. Fit the model to the data y_1,...,y_t and let y^_t+1 denote the forecast of the next observation. Then compute the 
error (e_t+1 = y_t+1 - y^_t+1) for the forecast observation.
2. Repeat step 1 for t=m,..., T-1 where m is the minimmum number of observations needed for fitting the model and T is
the length of the time serie dataset. 
3. Compute the MAE/MSE from e_m+1,...,e*_T.

In our case: m = 1000, T = 1500
script running time: 133 seconds
"""


#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)  # 42:The answer to life, the universe and everything.

mm = ModuleManager()

# Load data into dataframe
files_list = os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        'resources/Data/bivariate_analysis/proxy_cor/mw/'))
n_neighbors_vec = [5, 25]  # [5, 10, 25, 50, 100]
start_time = time.time()

for n_neighbors in n_neighbors_vec:
    mse_knn_vec = np.full(252, np.nan)  # Initialisation vector containing MSE for all window sizes
    for filename in files_list:
        i = [int(s) for s in re.findall(r'\d+', filename)]
        data_cor_true = mm.load_data('bivariate_analysis/proxy_cor/mw/' + filename)
        # Drop first m_max = 251 rows to ensure same training and test set for all values of m
        data_cor_true.drop(data_cor_true.head(251).index, inplace=True)
        data_cor_true.reset_index(drop=True, inplace=True)
        # Separate data into feature and response components
        X = np.asarray(data_cor_true.iloc[:, 0:-1])  # feature matrix (vectorize data for speed up)
        y = np.asarray(data_cor_true.iloc[:, -1])    # response vector
        t_start = 1000
        T = len(y)
        y_hat_knn = np.full(T - t_start, np.nan)    # Initialisation vector containing y_hat_t for t = m+1,...,T
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)  # Default settings: n_neighbors=5, weights=’uniform’

        for j, t in enumerate(range(t_start, T)):
            X_train = X[0:t, :]
            y_train = y[0:t]
            x_test = X[t]  # This is in fact x_t+1
            y_test = y[t]  # This is in fact y_t+1
            # Obtain estimation uncertainty in Pearson correlation estimation rho_t using bootstrap resampling:
            # randomly extract 1000 samples of size delta_t (mv window length)
            y_hat = knn.fit(X_train, y_train).predict(x_test.reshape(1, -1))
            y_hat_knn[j] = y_hat

        mse_knn_vec[i] = mean_squared_error(y[t_start:], y_hat_knn)

    mm.save_data('/bivariate_analysis/proxy_cor/mse_knn%i_mw_proxy_corr.pkl' % n_neighbors, mse_knn_vec)

print("%s: %f" % ('Execution time script', (time.time() - start_time)))







