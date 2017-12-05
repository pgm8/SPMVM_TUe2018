#from pandas_datareader import data as dt

import PreProcessor as PreProcessor
import ModuleManager as ModuleManager
import TechnicalAnalyzer2 as TechnicalAnalyzer2


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from math import sqrt, exp

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error # use mse to penalize outliers more

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
#np.random.seed(30)  # globally set random seed  (30 is a good option) 21 days
np.random.seed(42)





def main():

    preprocesser = PreProcessor()
    mm = ModuleManager.ModuleManager()
    ta = TechnicalAnalyzer2.TechnicalAnalyzer2()
    # ft = FeatureNormalizer()

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

    """
    T = 1751
    a0 = 0.1
    a1 = 0.8
    random_corr = preprocesser.simulate_random_correlation_ar(T, a0, a1)
    vol_matrix = np.array([[0.08, 0],  # Simple volatility matrix with unit variances for illustration purposes
                           [0, 0.1]])

    #vol_matrix = np.array([[1, 0],  # Simple volatility matrix with unit variances for illustration purposes
     #                      [0, 1]])

    correlated_asset_paths = preprocesser.simulate_correlated_asset_paths(random_corr, vol_matrix, T)


    plt.title('Simulated data using Cholesky decomposition and time-varying correlations')
    plt.plot(correlated_asset_paths[1200:, 0], label='$y_{1,t}$', linewidth=1, color='black')
    plt.plot(correlated_asset_paths[1200:, 1], label='$y_{2,t}$', linewidth=1, linestyle='--', color='blue')
    plt.plot(random_corr[1200:], label='$\\rho_t$', linewidth=1, color='red')
    plt.legend(fontsize='small', bbox_to_anchor=(1, 0.22), fancybox=True)
    plt.xlim(0, 500)
    plt.ylim(-0.5, 1)
    plt.show()

    data = pd.DataFrame(correlated_asset_paths)
    data['rho'] = random_corr
    mm.save_data('correlated_sim_data.pkl', data)
    """

    ## MAE for (weighted) moving window estimates with varying window size
    # One idea in order to be consistent with later ml comparison. Take random corr process of length 1500 and
    # take MAE measures over last 500 values. Then with ml we can train the models on first 1000 observations and
    # compare MAE measures over last 500 values. SO out-of-sample MAE.


    """
    simulated_data_process = mm.load_data('correlated_sim_data.pkl')
    mae_knn_vec = mm.load_data('mae_knn_true_corr.pkl')
    mae_rf1_vec = mm.load_data('mae_rf_true_corr.pkl')
    mae_rf10_vec = mm.load_data('mae_rf_true_corr_default.pkl')
    mae_mw_vec = mm.load_data('mae_mw_true_corr.pkl')
    mae_emw_vec = mm.load_data('mae_emw_true_corr.pkl')


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


    """
        ################################### Data set creation ###############################
    simulated_data_process = mm.load_data('correlated_sim_data.pkl')
    window_min = 21
    window_max = 22
    start_time = time.time()
    for m in range(window_min, window_max):
        dataset = preprocesser.generate_bivariate_dataset(ta, simulated_data_process, m)
        mm.save_data('/bivariate_analysis/mw/dataset_mw_%d.pkl' % m, dataset)
    print("%s: %f" % ('Execution time', (time.time() - start_time)))


    dt = 200
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
    """







###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()