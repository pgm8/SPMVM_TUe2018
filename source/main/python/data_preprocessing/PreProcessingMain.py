from pandas_datareader import data as dt
from ModuleManager import ModuleManager
from TechnicalAnalyzer2 import TechnicalAnalyzer2
from FeatureNormalizer import FeatureNormalizer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from math import sqrt, exp

from sklearn.metrics import mean_absolute_error

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(30)  # globally set random seed  (30 is a good option) 21 days


class PreProcesser(object):
    """Preprocesser class. This class has the responsibility to preprocess the data. More specifically, the class
    has the task of simulating random correlated asset paths in the bivariate case."""

    def __init__(self):
        """Initializer PreProcesser object."""

    def data_retrieval(self):
        """Method to pull financial data from yahoo.finance. Note: API does not seem to work anymore
        due to changes of webpage structure at server's end."""
        tickers = ['SPY']  # ETF SPY as proxy for S&P 500
        data_source = 'google'
        start = '1992-01-01'
        end = '2016-12-31'
        data_panel = dt.DataReader(tickers, data_source=data_source, start=start, end=end)

    def simulate_random_correlation_ar(self, T, a0, a1):
        """Simulate a random correlation process with highly persistent time-varying correlations following an
           auto-regressive process. Add noise with ar process
        :param T: simulation length
        :param a0:
        :param a1:
        :return: random_corr: correlation process following specified dynamics."""
        eps = 1e-5  # eq to 10^-5
        random_corr = np.empty(T)
        random_corr[0] = a0 / (1 - a1)  # initialise random correlation process
        for t in range(1, T):
            eta = np.random.normal(0, 0.2)
            random_corr[t] = np.maximum(-1 + eps, np.minimum(1 - eps, a0 + a1 * random_corr[t-1] + eta))
        return random_corr

    def simulate_random_correlation_garch(self, T, a0, a1, b1):
        """Simulate a random correlation process following a GARCH(1,1)-like process. The parameter values used are in
        line with those found empirically in stock return series.
        :param T: simulation length
        :param a0:
        :param a1:
        :param b1:
        :return: random_corr_garch: correlation process following specified dynamics.
        :return: sigma: volatility process."""
        eps = 1e-5  # equivalent to 10^-5
        random_corr_garch = np.empty(T)
        sigma = np.empty(T)
        #sigma[0] = sqrt(a0 / (1 - a1 - b1))  # parameter initialisation
        sigma[0] = 0.03  # parameter initialisation
        for t in range(1, T):
            eta = np.random.normal(0, 0.2)
            # Draw next correlation_t
            random_corr_garch[t-1] = np.maximum(-1 + eps, np.minimum(1 - eps, sigma[t-1] * eta))
            # Draw next sigma_t
            sigma_squared = a0 * sigma[0]**2 + a1 * random_corr_garch[t-1]**2 + b1 * sigma[t-1]**2
            sigma[t] = sqrt(sigma_squared)
        return random_corr_garch, sigma

    def simulate_correlated_asset_paths(self, corr_vector, vol_matrix, T):
        """Simulate asset paths with specified time-varying correlation dynamics.
        :param corr_vector: time-varying correlation vector
        :param vol_matrix: volatility matrix
        :param T: simulation length
        :return: correlated_asset_paths: simulated asset paths with specified correlation dynamics."""
        if corr_vector.ndim == 1:
            size = 2
        else:
            size = corr_vector.shape[1]  # no of columns, i.e. no of assets
        z = np.random.normal(0, 1, (T, size))  # T-by-number of assets draws from N(0,1) random variable
        correlated_asset_paths = np.empty([T, size])  # initialise Txsize dimensional array for correlated asset paths
        for t, rho in enumerate(corr_vector):
            corr_matrix = self.construct_correlation_matrix(rho, size)
            cov_matrix = self.construct_covariance_matrix(vol_matrix, corr_matrix)
            cholesky_factor = self.cholesky_factorization(cov_matrix)  # Cholesky decomposition
            correlated_asset_paths[t] = np.dot(cholesky_factor, z[t].transpose())  # Generating Y_t = H_t^(0.5) * z_t
        return correlated_asset_paths

    def construct_correlation_matrix(self, corr_vec, n):
        """Method for constructing time-varying correlation matrix given a time-varying correlations vector.
        :param corr_vec: time-varying correlation vector
        :param n: dimension correlation matrix
        :return corr_matrix: time-varying correlation matrix"""
        corr_triu = np.zeros((n, n))
        iu1 = np.triu_indices(n, 1)  # returns indices for upper-triangular matrix with diagonal offset of 1
        corr_triu[iu1] = corr_vec    # Assign vector correlations to corresponding upper-triangle matrix indices
        corr_matrix = corr_triu + corr_triu.T + np.eye(n)  # Transform upper-triangular matrix into symmetric matrix
        return corr_matrix

    def construct_covariance_matrix(self, vol_matrix, corr_matrix):
        """Method for constructing time-varying covariance matrix given a time-varying correlations matrix and asset
        volatility vector.
        :param vol_matrix: diagonal matrix containing asset volatilities
        :param corr_matrix: time-varying correlation matrix
        :return: cov_matrix: time-varying covariance matrix."""
        cov_matrix = np.dot(vol_matrix, np.dot(corr_matrix, vol_matrix))
        return cov_matrix

    def cholesky_factorization(self, cov_matrix):
        """Method for matrix decomposition through Cholesky factorization. The Cholesky factorization states that every
        symmetric positive definite matrix A has a unique factorization A = LL' where L is a lower-triangular matrix and
        L' is its conjugate transpose.
        :param cov_matrix: time-varying positive definite covariance matrix
        :return: cholesky_factor: cholesky decomposition lower-triangular matrix L such that LL' = cov_matrix"""
        cholesky_factor = np.linalg.cholesky(cov_matrix)
        return cholesky_factor

    def generate_bivariate_dataset(self, ta, simulated_data_process, m):
        """Method for generating a dataset with proxies (exponentially weighted moving window correlation estimates
        for feature set and true correlation as the response variables.
        :param ta: technical analyzer object
        :param simulated_data_process: bivariate asset process with predefined correlation dynamics.
        :param m: window length
        :return: dataset (datastructure: dataframe)."""
        #mw_estimates = simulated_data_process[0].rolling(window=m).corr(other=simulated_data_process[1])
        emw_estimates = ta.pearson_weighted_correlation_estimation(simulated_data_process[0], simulated_data_process[1],
                                                                   m)
        # Dataset with true correlations as response variable
        dataset = pd.DataFrame(emw_estimates, columns=['EMW_t-1'])
        # Ensure feature is past emw correlation estimate, i.e. x_t = EMW_t-1
        dataset['EMW_t-1'] = dataset['EMW_t-1'].shift(periods=1, axis='index')
        # Add output/ response variable to dataframe
        dataset['rho_true'] = simulated_data_process['rho']
        return dataset








def main():

    preprocesser = PreProcesser()
    mm = ModuleManager()
    ta = TechnicalAnalyzer2()
    ft = FeatureNormalizer()

    """
    # Create dataframe with Adjusted Close prices SP500 index and Russel2000 index
    data = pd.DataFrame()
    mm.transform_csv_to_pickle(filename='RUT.csv')
    sp500_data = mm.load_data(filename='GSPC_csv.pkl')
    russel2000_data = mm.load_data(filename='RUT_csv.pkl')
    data['Date'] = sp500_data['Date']
    data['Adj Close SP500'] = sp500_data['Adj Close']
    data['Adj Close Rus2000'] = russel2000_data['Adj Close']

    # Compute sample bivariate correlations using moving window estimates
    m = 120
    data['RhoSP500Rus2000'] = data['Adj Close SP500'].rolling(window=m).corr(other=data['Adj Close Rus2000'])
    #plt.plot(data['RhoSP500Rus2000'])
    #plt.show()

    # Generated data
    a0 = 0.02
    a1 = 0.2
    b1 = 0.78
    random_corr_garch, _ = simulate_random_correlation_garch(500, a0, a1, b1)  # results in non positive definite matrices
    """

    T = 1700
    a0 = 0.1
    a1 = 0.8
    random_corr = preprocesser.simulate_random_correlation_ar(T, a0, a1)
    vol_matrix = np.array([[0.08, 0],  # Simple volatility matrix with unit variances for illustration purposes
                           [0, 0.1]])

    #correlated_asset_paths = preprocesser.simulate_correlated_asset_paths(random_corr, vol_matrix, T)

    """
    plt.title('Simulated data using Cholesky decomposition and time-varying correlations')
    plt.plot(correlated_asset_paths[200:1699, 0], label='$y_{1,t}$')
    plt.plot(correlated_asset_paths[200:1699, 1], label='$y_{2,t}$')
    plt.plot(random_corr, label='$\\rho_t$')
    plt.legend(fontsize='small', bbox_to_anchor=(1, 0.22), fancybox=True)
    plt.xlim(0, 1500)
    plt.show()

    data = pd.DataFrame(correlated_asset_paths)
    data['rho'] = random_corr
    mm.save_data('correlated_sim_data.pkl', data)
    
    """

    ## MAE for (weighted) moving window estimates with varying window size
    #One idea in order to be consistent with later ml comparison. Take random corr process of length 1500 and
    #take MAE measures over last 500 values. Then with ml we can train the models on first 1000 observations and
    #compare MAE measures over last 500 values. SO out-of-sample MAE.



    simulated_data_process = mm.load_data('correlated_sim_data.pkl')
    mae_knn_vec = mm.load_data('mae_knn_true_corr.pkl')
    mae_rf1_vec = mm.load_data('mae_rf_true_corr.pkl')
    mae_rf10_vec = mm.load_data('mae_rf_true_corr_default.pkl')
    mae_mw_vec = mm.load_data('mae_mw_true_corr.pkl')
    mae_emw_vec = mm.load_data('mae_emw_true_corr.pkl')

    """
    window_min = 3
    window_max = 201
    mae_mw_vec = np.full(window_max, np.nan)
    mae_emw_vec = np.full(window_max, np.nan)

    for m in range(window_min, window_max):
        mw_estimates = simulated_data_process[0].rolling(window=m).corr(other=simulated_data_process[1])
        emw_estimates = ta.pearson_weighted_correlation_estimation(simulated_data_process[0], simulated_data_process[1],
                                                                   m)
        mae_mw_vec[m-1] = mean_absolute_error(random_corr[1200:], mw_estimates[1200:])
        mae_emw_vec[m-1] = mean_absolute_error(random_corr[1200:], emw_estimates[1200:])

    mm.save_data('mae_mw_true_corr.pkl', mae_mw_vec)
    mm.save_data('mae_emw_true_corr.pkl', mae_emw_vec)

    plt.figure(0)
    plt.plot(mae_mw_vec[0:101], label='Moving Window')
    plt.plot(mae_emw_vec[0:101], label='Exp. Weighted Moving Window')
    #plt.plot(mae_knn_vec, label='KNN')
    #plt.plot(mae_rf_vec, label='RF')
    plt.title('MAE for MW and EMW')
    plt.xlabel('window length'); plt.ylabel('MAE'); plt.legend(loc='upper right', fancybox=True)
    plt.ylim(0, 0.6)
    plt.show()
    
    """


    plt.figure(1)
    plt.plot(mae_mw_vec[0:101], label='Moving Window')
    plt.plot(mae_emw_vec[0:101], label='Exp. Weighted Moving Window')
    plt.plot(mae_knn_vec[0:101], label='KNN')
    #plt.plot(mae_rf1_vec[0:101], label='RF(1)')
    plt.plot(mae_rf10_vec[0:101], label='RF(10)')
    plt.plot()
    plt.title('MAE for MW, EMW, KNN and RF')
    plt.xlabel('window length')
    plt.ylabel('MAE')
    plt.legend(loc='upper right', fancybox=True)
    plt.ylim(0, 0.6)
    plt.show()


    """

    ################################### Data set creation ###############################
    simulated_data_process = mm.load_data('correlated_sim_data.pkl')
    window_min = 3
    window_max = 201
    start_time = time.time()
    for m in range(window_min, window_max):
        dataset = preprocesser.generate_bivariate_dataset(ta, simulated_data_process, m)
        mm.save_data('/bivariate_analysis/dataset_emw_%d.pkl' % m, dataset)
    print("%s: %f" % ('Execution time', (time.time() - start_time)))

    dt = 200
    ## Dataset with true correlations as target variable
    dataset_cor_true = pd.DataFrame(
        ta.pearson_weighted_correlation_estimation(simulated_data_process[0],
                                                   simulated_data_process[1], dt), columns=['EMW_t-1'])
    # Ensure feature is past correlation mw estimate, i.e. x_t = EMW_t-1
    dataset_cor_true['EMW_t-1'] = dataset_cor_true['EMW_t-1'].shift(periods=1, axis='index')
    # Add output/ target variable to dataframe
    dataset_cor_true['rho_true'] = simulated_data_process['rho']
    ## Dataset with proxies for correlation as target variable
    dataset_cor_proxies = pd.DataFrame(
        ta.pearson_weighted_correlation_estimation(simulated_data_process[0],
                                                   simulated_data_process[1], dt), columns=['rho_proxy'])
    dataset_cor_proxies.insert(loc=0, column='EMW_t-1', value=dataset_cor_proxies['rho_proxy'].shift(periods=1,
                                                                                                  axis='index'))
    # Write datasets to pickle objects
    mm.save_data('dataset_cor_true.pkl', dataset_cor_true)
    mm.save_data('dataset_cor_proxy.pkl', dataset_cor_proxies)
    
    """



    """
    a0 = 0.1
    a1 = 0.8
    random_corr = simulate_random_correlation_ar(500, a0, a1)
    plt.title('Persistent time-varying correlation process following auto-regressive process')
    plt.xlabel('t')
    plt.plot(random_corr)
    #plt.show()

    a0 = 1.0
    a1 = 0.1
    b1 = 0.8
    random_corr_garch, _ = simulate_random_corr_garch(500, a0, a1, b1)
    plt.title('Time-varying correlation process following GARCH-like process')
    plt.plot(random_corr_garch)
    plt.show()

    filenames = ['GSPC.csv']
    for filename in filenames:
        # Transform csv files to pickled objects
        mm.transform_csv_to_pickle(filename)
        # Load pickled files
        data = mm.load_data(filename[:-4] + '_csv.pkl')
        # Feature Engineering and Normalization:
        # Create pickled dataframes obtained form technical analysis
        # 1a) base: price and volume data + return + response
        data_ta_base = ta.ta_base(data)
        mm.save_data(filename[:-4] + '_base.pkl', data_ta_base)
        # 1b) normalize base dataset:
        data_base_normalized = ft.normalize_feature_matrix(data_ta_base)
        mm.save_data(filename[:-4] + '_base_norm.pkl', data_base_normalized)
        # 2a) full: price and volume data + feature set + return + response
        data_ta_full = ta.ta_full(data)
        mm.save_data(filename[:-4] + '_full.pkl', data_ta_full)
        # 2b) normalize full dataset:
        data_full_normalized = ft.normalize_feature_matrix(data_ta_full)
        mm.save_data(filename[:-4] + '_full_norm.pkl', data_full_normalized)
        # Write pickled dataframe to csv file
        mm.transform_pickle_to_csv('GSPC_base.pkl')
        mm.transform_pickle_to_csv('GSPC_base_norm.pkl')
        
        """




















###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()

