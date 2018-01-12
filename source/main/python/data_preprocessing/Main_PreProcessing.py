#from pandas_datareader import data as dt

from PreProcessor import PreProcessor
from ModuleManager import ModuleManager
from TechnicalAnalyzer import TechnicalAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import precision_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error # use mse to penalize outliers more

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
#np.random.seed(30)  # globally set random seed  (30 is a good option) 21 days
np.random.seed(30)


def main():

    preprocesser = PreProcessor()
    mm = ModuleManager()
    ta = TechnicalAnalyzer()
    # ft = FeatureNormalizer()


    ##################################################################################################################
    ###     Asset path simulation using Cholesky Factorization and predefined time-varying correlation dynamics    ###
    ################## ################################################################################################
    """
    T = 1751
    a0 = 0.1
    a1 = 0.8
    random_corr = preprocesser.simulate_random_correlation_ar(T, a0, a1)
    vol_matrix = np.array([[0.08, 0],  # Simple volatility matrix with unit variances for illustration purposes
                           [0, 0.1]])

    correlated_asset_paths = preprocesser.simulate_correlated_asset_paths(random_corr, vol_matrix, T)
    data = pd.DataFrame(correlated_asset_paths)
    data['rho'] = random_corr
    mm.save_data('/bivariate_analysis/correlated_sim_data.pkl', data)
    """
    """
    # Figure
    correlated_asset_paths = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    #plt.title('Simulated data using Cholesky decomposition and time-varying correlations')
    correlated_asset_paths = correlated_asset_paths.tail(500);
    correlated_asset_paths.reset_index(drop=True, inplace=True)
    plt.plot(correlated_asset_paths.iloc[:, 0], label='$y_{1,t}$', linewidth=1, color='black')
    plt.plot(correlated_asset_paths.iloc[:, 1], label='$y_{2,t}$', linewidth=1, linestyle='--', color='blue')
    plt.plot(correlated_asset_paths.iloc[:, -1], label='$\\rho_t$', linewidth=1, color='red')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                   edgecolor='black')
    plt.xlim(0, 500)
    plt.ylim(-0.5, 1)
    plt.show()
    """

    ##################################################################################################################
    ###     Estimation uncertainty in (weighted) Pearson correlation coefficient using moving window estimates     ###
    ##################################################################################################################
    """
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    delta_t = [21, 251]
    proxy_type = ['mw', 'emw']
    ciw = 99
    start_time = time.time()
    for dt, proxy_type in [(x, y) for x in delta_t for y in proxy_type]:
        print(dt); print(proxy_type)
        rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates = \
        preprocesser.bootstrap_moving_window_estimate(data=simulated_data_process, delta_t=dt, T=T, ciw=ciw,
                                                      proxy_type=proxy_type)
        data_frame = pd.DataFrame({'Percentile_low': lower_percentiles, 'Percentile_up': upper_percentiles,
                                   'std rho estimate': sd_rho_estimates, 'Rho_estimate': rho_estimates})
        filename = '%s_%i_estimate_uncertainty.pkl' % (proxy_type, dt)
        mm.save_data('bivariate_analysis/' + filename, data_frame)
    print("%s: %f" % ('Execution time:', (time.time() - start_time)))
    """
    """
    # Figures
    for dt, proxy_type in [(x, y) for x in delta_t for y in proxy_type]:
        data = mm.load_data('bivariate_analysis/%s_%i_estimate_uncertainty.pkl' % (proxy_type, dt))
        rho_estimates = data['Rho_estimate']
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure()
        plt.plot(rho_estimates, label='%s correlation' % proxy_type.upper(), linewidth=1, color='red')
        plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
        plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
        plt.xlabel('observation')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
                   edgecolor='black')
        plt.xlim(0, T)
        plt.yticks(np.arange(-1, 1.00000001, 0.2))
        plt.ylim(-1, 1)
        plt.show()
    """
    ##################################################################################################################
    ###       Mean squared error of (weighted) Pearson correlation coefficient using moving window estimates       ###
    ##################################################################################################################
    """
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    delta_t_min = 3
    delta_t_max = 252
    mse_mw_vec = np.full(delta_t_max-1, np.nan)
    mse_emw_vec = np.full(delta_t_max-1, np.nan)

    for dt in range(delta_t_min, delta_t_max):
        mw_estimates = simulated_data_process.tail(T+dt-1).iloc[:, 0].rolling(window=dt).corr(
            other=simulated_data_process.tail(T+dt-1)[1])
        emw_estimates = ta.pearson_weighted_correlation_estimation(simulated_data_process.tail(T+dt-1).iloc[:, 0],
                                                                   simulated_data_process.tail(T+dt-1)[1], dt)
        mse_mw_vec[dt - 1] = mean_squared_error(rho_true, mw_estimates.tail(T))
        mse_emw_vec[dt - 1] = mean_squared_error(rho_true, emw_estimates[-T:])

    mm.save_data('mse_mw_true_corr.pkl', mse_mw_vec)
    mm.save_data('mse_emw_true_corr.pkl', mse_emw_vec)
    """
    """
    mse_mw_vec = mm.load_data('bivariate_analysis/mse_mw_true_corr.pkl')
    mse_emw_vec = mm.load_data('bivariate_analysis/mse_emw_true_corr.pkl')

    sd_mse_mw = np.nanstd(mse_mw_vec)
    sd_mse_emw = np.nanstd(mse_emw_vec)

    print(sd_mse_mw); print(sd_mse_mw**2)
    print(sd_mse_emw); print(sd_mse_emw**2)
    #mse_knn_mw_vec = mm.load_data('/bivariate_analysis/mse_knn_mw_true_corr.pkl')
    #mse_knn_emw_vec = mm.load_data('/bivariate_analysis/mse_knn_emw_true_corr.pkl')
    
   
    

    # Figure
    plt.figure(1)
    plt.plot(mse_mw_vec, label='MW', color='blue')
    plt.plot(mse_emw_vec, label='EMW', color='red')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 0.51, 0.1))
    plt.ylim(0, 0.5)
    plt.show()
    """


    ##################################################################################################################
    ###                                          Dataset creation                                                  ###
    ##################################################################################################################
    # Pearson correlation moving window estimates as covariates and true correlation as response variable
    """
    simulated_data_process = mm.load_data('correlated_sim_data.pkl')
    delta_t_min = 3
    delta_t_max = 4
    start_time = time.time()
    for dt in range(delta_t_min, delta_t_max):
        dataset = preprocesser.generate_bivariate_dataset(ta, simulated_data_process, dt, weighted=False)
        mm.save_data('/bivariate_analysis/emw/dataset_emw_%d.pkl' % dt, dataset)

    print("%s: %f" % ('Execution time:', (time.time() - start_time)))
    """

    """
    mse_knn_mw_vec = mm.load_data('mse_knn_mw_true_corr.pkl')
    mse_knn_emw_vec = mm.load_data('mse_knn_emw_true_corr.pkl')
    mse_mw_vec = mm.load_data('mse_mw_true_corr.pkl')

    #mse_mw_vec = mm.load_data('mse_mw_true_corr.pkl')
    #plt.plot(mse_mw_vec, label='Moving Window')
    #plt.plot(mse_emw_vec, label='Exp. Weighted Moving Window')
    plt.plot(mse_knn_mw_vec, label='KNN_mw')
    plt.plot(mse_knn_emw_vec, label='KNN_emw')
    plt.plot(mse_mw_vec, label='MW')
    plt.title('MSE for KNN')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(loc='lower right', fancybox=True)
    plt.ylim(0.06, 0.10)
    plt.xlim(0, 250)
    plt.show()
    """

    ##################################################################################################################
    ###    Estimation uncertainty in (weighted) Pearson correlation coefficient using machine learner estimates    ###
    ##################################################################################################################
    T = 500
    ciw = 99
    reps = 1000
    delta_t = [21, 251]
    model = ['knn', 'rf']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['mw', 'emw']
    """
    for dt, proxy_type, model in [(x, y, z) for x in delta_t for y in proxy_type for z in model]:
        start_time = time.time()
        dataset = mm.load_data('bivariate_analysis/%s/dataset_%s_%i.pkl' % (proxy_type, proxy_type, dt))
        rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates = \
        preprocesser.bootstrap_learner_estimate(data=dataset, reps=reps, model=model)
        data_frame = pd.DataFrame({'Percentile_low': lower_percentiles, 'Percentile_up': upper_percentiles,
                                   'std rho estimate': sd_rho_estimates, 'Rho_estimate': rho_estimates})
        filename = '%s_%s_%i_estimate_uncertainty.pkl' % (model, proxy_type, dt)
        mm.save_data('bivariate_analysis/' + filename, data_frame)
        print("%s: %f" % ('Execution time', (time.time() - start_time)))
    
    # Execution time rf10, 21, emw: 9514 seconds
    # Execution time rf10, 251, mw: 9594 seconds
    # Execution time rf10, 251, emw: 9696 seconds
    # Execution time rf, 21, mw: 95593 seconds (roughly 26.5 hours)
    # Execution time rf, 251, mw: 162240 seconds (roughly 45 hours)
    # Execution time rf, 21, emw: 274400 seconds (roughly 76 hours)
    """
    """
        # Figures
        for dt, proxy_type, model in [(x, y, z) for x in delta_t for y in proxy_type for z in model]:
            data = mm.load_data('bivariate_analysis/%s10_%s_%i_estimate_uncertainty.pkl' % (model, proxy_type, dt))
            rho_estimates = data['Rho_estimate']
            lower_percentiles = data['Percentile_low']
            upper_percentiles = data['Percentile_up']
            plt.figure()
            plt.plot(rho_estimates, label='%s30 correlation' % model.upper(), linewidth=1, color='red')
            plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
            plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
            plt.xlabel('observation')
            plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
                       edgecolor='black')
            plt.xlim(0, T)
            plt.yticks(np.arange(-1, 1.00000001, 0.2))
            plt.ylim(-1, 1)
            plt.show()
        """
    """
    n_neighbours = [10, 25]
    for k in n_neighbours:
        data = mm.load_data('bivariate_analysis/knn%i_mw_21_estimate_uncertainty.pkl' % k)
        rho_estimates = data['Rho_estimate']
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure()
        plt.plot(rho_estimates, label='KNN(%i) correlation' % k, linewidth=1, color='red')
        plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
        plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
        plt.xlabel('observation')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
                   edgecolor='black')
        plt.xlim(0, T)
        plt.yticks(np.arange(-1, 1.00000001, 0.2))
        plt.ylim(-1, 1)
        plt.show()
    """
    """
    # Figure
    data = mm.load_data('bivariate_analysis/rf10_emw_251_estimate_uncertainty.pkl')
    rho_estimates = data['Rho_estimate']
    lower_percentiles = data['Percentile_low']
    upper_percentiles = data['Percentile_up']
    plt.figure(1)
    plt.plot(rho_estimates, label='RF correlation', linewidth=1, color='red')
    plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
    plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
    plt.xlabel('observation')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
               edgecolor='black')
    plt.xlim(0, T)
    plt.yticks(np.arange(-1, 1.1, 0.2))
    plt.ylim(-1, 1)
    plt.show()
    """

    ##################################################################################################################
    ###       Mean squared error of (weighted) Pearson correlation coefficient using machine learner estimates     ###
    ##################################################################################################################
    """
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    T = 500
    model = ['knn', 'rf']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['mw', 'emw']
    """
    # Individual scripts for obtaining mse knn, rf
    """
    mse_mw_vec = mm.load_data('/bivariate_analysis/mse_mw_true_corr.pkl')
    mse_emw_vec = mm.load_data('/bivariate_analysis/mse_emw_true_corr.pkl')
    # KNN
    mse_knn5_mw_vec = mm.load_data('/bivariate_analysis/mse_knn5_mw_true_corr.pkl')
    mse_knn10_mw_vec = mm.load_data('/bivariate_analysis/mse_knn10_mw_true_corr.pkl')
    mse_knn25_mw_vec = mm.load_data('/bivariate_analysis/mse_knn25_mw_true_corr.pkl')
    mse_knn50_mw_vec = mm.load_data('/bivariate_analysis/mse_knn50_mw_true_corr.pkl')
    mse_knn100_mw_vec = mm.load_data('/bivariate_analysis/mse_knn100_mw_true_corr.pkl')

    mse_knn5_emw_vec = mm.load_data('/bivariate_analysis/mse_knn5_emw_true_corr.pkl')
    mse_knn10_emw_vec = mm.load_data('/bivariate_analysis/mse_knn10_emw_true_corr.pkl')
    mse_knn25_emw_vec = mm.load_data('/bivariate_analysis/mse_knn25_emw_true_corr.pkl')
    mse_knn50_emw_vec = mm.load_data('/bivariate_analysis/mse_knn50_emw_true_corr.pkl')
    mse_knn100_emw_vec = mm.load_data('/bivariate_analysis/mse_knn100_emw_true_corr.pkl')
    """
    """
    # Figure
    plt.figure(1)
    plt.plot(mse_knn5_mw_vec, label='KNN_mw', color='blue')
    plt.plot(mse_knn5_emw_vec, label='KNN_emw', color='red')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 0.51, 0.1))
    plt.ylim(0, 0.5)

    plt.figure(2)
    plt.plot(mse_mw_vec, label='MW', color='blue')
    plt.plot(mse_emw_vec, label='EMW', color='red')
    plt.plot(mse_knn5_mw_vec, label='KNN(5)', linewidth=1, linestyle='--', color='black')
    #plt.plot(mse_knn10_mw_vec, label='KNN(10)', color='red')
    plt.plot(mse_knn25_mw_vec, label='KNN(25)', color='black')
    #plt.plot(mse_knn50_mw_vec, label='KNN(50)', color='magenta')
    #plt.plot(mse_knn100_mw_vec, label='KNN(100)', color='grey')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0.06, 0.21, 0.02))
    plt.ylim(0.06, 0.20)
    plt.show()
    """
    # Variance in MSE window sizes
    #sd_mse_knn5_mw = np.nanstd(mse_knn5_mw_vec); print(np.power(sd_mse_knn5_mw, 2))
    #sd_mse_knn5_emw = np.nanstd(mse_knn5_emw_vec); print(np.power(sd_mse_knn5_emw, 2))
    #sd_mse_knn25_mw = np.nanstd(mse_knn25_mw_vec); print(np.power(sd_mse_knn25_mw, 2))
    # Max-min in MSE window sizes
    #print('knn5_mw_min: %f' % np.nanmin(mse_knn5_mw_vec));
    #print('knn5_mw_max: %f' % np.nanmax(mse_knn5_mw_vec));
    #print('knn5_emw_min: %f' % np.nanmin(mse_knn5_emw_vec));
    #print('knn5_emw_max: %f' % np.nanmax(mse_knn5_emw_vec));


    # Random Forest
    mse_rf10_mw_vec = mm.load_data('/bivariate_analysis/mse_rf10_mw_true_corr.pkl')
    mse_rf10_emw_vec = mm.load_data('/bivariate_analysis/mse_rf10_emw_true_corr.pkl')
    mse_rf100_mw_vec = mm.load_data('/bivariate_analysis/mse_rf100_mw_true_corr.pkl')
    # Variance in MSE window sizes
    sd_mse_rf10_mw = np.nanstd(mse_rf10_mw_vec); print(np.power(sd_mse_rf10_mw, 2))
    sd_mse_rf10_emw = np.nanstd(mse_rf10_emw_vec);print(np.power(sd_mse_rf10_emw, 2))
    sd_mse_rf100_mw = np.nanstd(mse_rf100_mw_vec); print(np.power(sd_mse_rf100_mw, 2))

    # Max-min in MSE window sizes
    print('rf10_mw_min: %f' % np.nanmin(mse_rf10_mw_vec));
    print('rf10_mw_max: %f' % np.nanmax(mse_rf10_mw_vec));
    print('rf10_emw_min: %f' % np.nanmin(mse_rf10_emw_vec));
    print('rf10_emw_max: %f' % np.nanmax(mse_rf10_emw_vec));

    """
    # Figure
    plt.figure(3)
    #plt.plot(mse_rf10_mw_vec, label='RF_mw', color='blue')
    #plt.plot(mse_rf10_emw_vec, label='RF_emw', color='red')
    plt.plot(mse_mw_vec, label='MW', color='blue')
    plt.plot(mse_emw_vec, label='EMW', color='red')
    plt.plot(mse_rf10_mw_vec, label='RF(10)', linewidth=1, linestyle='--', color='black')
    plt.plot(mse_rf100_mw_vec, label='RF(100)', color='black')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0.06, 0.21, 0.02))
    plt.ylim(0.06, 0.2)
    plt.show()
    """






###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()