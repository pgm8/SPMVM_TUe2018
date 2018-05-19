from PreProcessor import PreProcessor
from ModuleManager import ModuleManager
from TechnicalAnalyzer import TechnicalAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, kendalltau
import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error  # use mse to penalize outliers more

import os.path

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(30)


def main():

    preprocesser = PreProcessor()
    mm = ModuleManager()
    ta = TechnicalAnalyzer()


    ##################################################################################################################
    ###     Asset path simulation using Cholesky Factorization and predefined time-varying correlation dynamics    ###
    ################## ################################################################################################
    """
    T = 1751
    a0 = 0.1
    a1 = 0.8
    random_corr = preprocesser.simulate_random_correlation_ar(T, a0, a1)
    #random_corr, _ = preprocesser.simulate_random_correlation_garch(T, 0.02, 0.2, 0.78)
    vol_matrix = np.array([[0.08, 0],# Simple volatility matrix with randomly chosen variances for illustration purposes
                           [0, 0.1]])
    correlated_asset_paths = preprocesser.simulate_correlated_asset_paths(random_corr, vol_matrix, T)

    data = pd.DataFrame(correlated_asset_paths)
    data['rho'] = random_corr
    mm.save_data('/bivariate_analysis/correlated_sim_data.pkl', data)
    """
    """
    # Figure
    correlated_asset_paths = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    correlated_asset_paths = data.tail(500)
    data.reset_index(drop=True, inplace=True)
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
    ###     Estimation uncertainty in Pearson and Kendall correlation coefficient using moving window estimates    ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    delta_t = np.arange(3, 252)        # 3, 4, 5, 6, 7, 8, 9, 10, 21, 42, 63, 84, 126, 251
    proxy_type = ['kendall']  # kendall ['mw', 'emw', 'kendall']
    ciw = 99

    """
    for dt, proxy_type in [(x, y) for x in delta_t for y in proxy_type]:
        start_time = time.time()
        print('(%s, %i)' % (proxy_type, dt))
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
        print(rho_estimates)
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure()
        plt.plot(simulated_data_process['rho'], label='true correlation', linewidth=1, color='black')
        plt.plot(rho_estimates, label='%s correlation' % proxy_type.upper(), linewidth=1, color='red')
        plt.plot((upper_percentiles-lower_percentiles)-1, label='%d%% interval (bootstrap)'
                                                                % ciw, linewidth=1, color='magenta')
        #plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
        #plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
        plt.xlabel('observation')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                   edgecolor='black')
        plt.xlim(0, T)
        plt.yticks(np.arange(-1, 1.00000001, 0.2))
        plt.ylim(-1, 1)
        plt.show()
    """
    ##################################################################################################################
    ###       Mean squared error of Pearson and Kendall correlation coefficient using moving window estimates      ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    delta_t_min, delta_t_max = 3, 252
    delta_t = np.arange(3, 252)  # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    proxy_type = ['pearson', 'emw', 'kendall']  # run proxies individually otherwise one saves dataframe over other.
    rho_bias_squared = np.full(delta_t_max, np.nan)
    rho_var_vec = np.full(delta_t_max, np.nan)
    """
    # Create dataframe with (interpolated) mse results, squared bias, variance for varying window sizes
    for proxy_type, dt in [(x, y) for x in proxy_type for y in delta_t]:
        print('%s, %i' % (proxy_type, dt))
        data = mm.load_data('bivariate_analysis/%s_%i_estimate_uncertainty.pkl'
                            % (proxy_type, dt))
        rho_estimates = data['Rho_estimate']
        rho_bias_squared[dt] = np.mean(np.power(rho_estimates - rho_true, 2))
        rho_var_vec[dt] = np.power(np.mean(data['std rho estimate']), 2)

    rho_mse_vec = np.array([np.sum(pair) for pair in zip(rho_bias_squared, rho_var_vec)])
    data_frame = pd.DataFrame({'bias_squared': rho_bias_squared, 'variance': rho_var_vec,
                               'MSE': rho_mse_vec})
    filename = 'mse_%s.pkl' % proxy_type
    mm.save_data('bivariate_analysis/' + filename, data_frame)
    """
    """
    # Kendall correlation estimate 
        for col1, col2, in IT.combinations(simulated_data_process.columns[:-1], 2):
            def my_tau(idx):
                df_tau = simulated_data_process[[col1, col2]].iloc[idx+len(simulated_data_process)-T-dt+1]
                return kendalltau(df_tau[col1], df_tau[col2])[0]
            kendall_estimates = pd.rolling_apply(np.arange(T+dt-1), dt, my_tau)
        mse_kendall_vec[dt - 1] = mean_squared_error(rho_true, kendall_estimates[-T:])
    mm.save_data('/bivariate_analysis/mse_kendall_true_corr.pkl', mse_kendall_vec)
    print("%s: %f" % ('Execution time:', (time.time() - start_time)))
    """
    """
    # Load MSE data Pearson/ Kendall
    mse_pearson_vec = mm.load_data('bivariate_analysis/mse_pearson.pkl')
    mse_kendall_vec = mm.load_data('bivariate_analysis/mse_kendall.pkl')
    """
    """
    # Figure without interpolation MSE 
    plt.figure(1)
    plt.plot(mse_pearson_vec['MSE'], label='Pearson', color='indigo', linewidth=1)
    plt.plot(mse_kendall_vec['MSE'], label='Kendall', color='aquamarine', linewidth=1, linestyle='--')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 0.61, 0.1))
    plt.ylim(0, 0.6)
    plt.show()
    """
    """
    # Figure without interpolation MSE decomposition 
    plt.figure(2)
    plt.plot(mse_kendall_vec['bias_squared'], label='Squared Bias', color='blue', linewidth=1)
    plt.plot(mse_kendall_vec['variance'], label='Variance', color='red', linewidth=1)
    plt.plot(mse_kendall_vec['MSE'], label='MSE', color='black', linestyle='--', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 0.61, 0.1))
    plt.ylim(0, 0.6)
    plt.show()
    """
    """
    # Variance in MSE window sizes
    var_mse_pearson = np.nanvar(mse_pearson_vec['MSE']); print('mse_pearson_var: %f' % var_mse_pearson)
    var_mse_kendall = np.nanvar(mse_kendall_vec['MSE']); print('mse_kendall_var: %f' % var_mse_kendall)

    # Max-min in MSE window sizes
    print('mse_pearson_min_max: (%f, %f)' % (np.nanmin(mse_pearson_vec['MSE']), np.nanmax(mse_pearson_vec['MSE'])))
    print('mse_kendall_min_max: (%f, %f)' % (np.nanmin(mse_kendall_vec['MSE']), np.nanmax(mse_kendall_vec['MSE'])))
    """

    ##################################################################################################################
    ###                         Minimum Determinant Pearson and Kendall Moving Window                              ###
    ##################################################################################################################
    # Get information on the minimum determinants over all corrlation estimates for all window sizes [3, 100]
    delta_t = range(3, 101)
    det_min_vec = np.full(101, np.nan)
    proxy_type = 'pearson'
    """
    for dt in delta_t:
        # Load data Pearson/ Kendall
        det_data_vec = np.full(501, np.nan)
        filename = '%s_%i_estimate_uncertainty.pkl' % (proxy_type, dt)
        data = mm.load_data('bivariate_analysis/results_%s/%s' % (proxy_type, filename))
        # Compute determinants for every dataset
        for i, rho in enumerate(data['Rho_estimate']):
            det_data_vec[i+1] = preprocesser.determinant_LU_factorization(rho, 2)
        det_min_vec[dt] = np.nanmin(det_data_vec)
    mm.save_data('bivariate_analysis/determinant_min_%s.pkl' % proxy_type, det_min_vec)
    """
    """
    # Plot minimum determinants of Pearson and Kendal Moving Window estimates of correlation
    det_min_pearson = mm.load_data('bivariate_analysis/determinant_min_pearson.pkl')
    det_min_kendall = mm.load_data('bivariate_analysis/determinant_min_kendall.pkl')
    plt.figure(1)
    plt.plot(det_min_pearson, label='Pearson', linewidth=1, color='orange')
    plt.plot(det_min_kendall, label='Kendall', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('minimum det($R_t)$')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(-0.1, 1.1, 0.1))
    plt.ylim(-0.1, 1)
    plt.show()
    """

    
    
    
    
    ##################################################################################################################
    ###                                          Dataset creation                                                  ###
    ##################################################################################################################
    # Pearson and Kendall correlation moving window estimates as covariate and true correlation or moving window
    # estimate as proxy for output variable
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    delta_t_min = 5
    delta_t_max = 6
    proxy_type = ['kendall']     # ['pearson', 'emw', 'kendall']
    """
    start_time = time.time()
    for dt, proxy_type in [(x,y) for x in range(delta_t_min, delta_t_max) for y in proxy_type]:
        print('(%i, %s)' % (dt, proxy_type))
        dataset, dataset_proxy = \
            preprocesser.generate_bivariate_dataset(ta, simulated_data_process, dt, proxy_type=proxy_type)
        mm.save_data('/bivariate_analysis/true_cor/%s/data/dataset_%s_%d.pkl' % (proxy_type, proxy_type, dt), dataset)
        mm.save_data('/bivariate_analysis/proxy_cor/%s/data/dataset_%s_%d.pkl' % (proxy_type, proxy_type, dt), dataset_proxy)
    print("%s: %f" % ('Execution time:', (time.time() - start_time)))
    """
    ##################################################################################################################
    ###    Estimation uncertainty in Pearson and Kendall correlation coefficient using machine learner estimates   ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    ciw = 99
    reps = 1000
    delta_t = [21]   # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    model = ['knn']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['pearson', 'kendall']
    output_type = ['true', 'proxy']
    n_neighbours = [5]

    """
    for dt, proxy_type, model, k, output_type in [(x, y, z, k, o) for x in delta_t for y in proxy_type
                                     for z in model for k in n_neighbours for o in output_type]:
        start_time = time.time()
        print('(%i, %s, %s, %i)' % (dt, proxy_type, model, k))
        dataset = mm.load_data('bivariate_analysis/%s_cor/%s/data/dataset_mw_%i.pkl' % (output_type, proxy_type, dt))
        rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates = \
        preprocesser.bootstrap_learner_estimate(data=dataset, reps=reps, model=model, n_neighbors=k)
        data_frame = pd.DataFrame({'Percentile_low': lower_percentiles, 'Percentile_up': upper_percentiles,
                                   'std rho estimate': sd_rho_estimates, 'Rho_estimate': rho_estimates})
        filename = '%s5_%s_%i_estimate_uncertainty_%s_corr.pkl' % (model, proxy_type, dt, output_type)
        mm.save_data('bivariate_analysis/%s_cor/%s/results_%s_%s_%s_cor/' % (output_type, proxy_type, model, proxy_type,
                                                                             output_type) + filename, data_frame)
        print("%s: %f" % ('Execution time', (time.time() - start_time)))
    """

    """
    # Figure with bootstrap uncertainty Nearest Neighbors
    for dt, proxy_type in [(x, y) for x in delta_t for y in proxy_type]:
        print('(%s, %i)' % (proxy_type, dt))
        data = mm.load_data('bivariate_analysis/proxy_cor/%s/results_knn_%s_proxy_cor/'
                            'knn5_%s_%i_estimate_uncertainty_proxy_corr.pkl' % (proxy_type, proxy_type, proxy_type, dt))
        rho_estimates = data['Rho_estimate']
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure()
        plt.plot(simulated_data_process['rho'], label='true correlation', linewidth=1, color='black')
        plt.plot(rho_estimates, label='KNN correlation', linewidth=1, color='red')
        plt.plot((upper_percentiles - lower_percentiles) - 1, label='%d%% interval (bootstrap)' % ciw,
                 linewidth=1, color='magenta')
        plt.xlabel('observation')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                   edgecolor='black')
        plt.xlim(0, T)
        plt.yticks(np.arange(-1, 1.00000001, 0.2))
        plt.ylim(-1, 1)
        plt.show()
    """
    """
    # Figure with bootstrap uncertainty Random Forest
    for proxy_type, output_type in [(x, y) for x in proxy_type for y in output_type]:
        filename = 'rf10_%s_21_estimate_uncertainty_rep_1000_%s_corr.pkl' % (proxy_type, output_type)
        print(filename)
        data = mm.load_data('bivariate_analysis/%s_cor/%s/results_rf_%s_%s_cor/%s' % (output_type, proxy_type,
                                                                                      proxy_type, output_type, filename))
        rho_estimates = data['Rho_estimate']
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure(1)
        plt.plot(simulated_data_process['rho'], label='true correlation', linewidth=1, color='black')
        plt.plot(rho_estimates, label='RF correlation', linewidth=1, color='red')
        plt.plot((upper_percentiles - lower_percentiles) - 1, label='%d%% interval (bootstrap)' % ciw,
                 linewidth=1, color='magenta')
        #plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
        #plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
        plt.xlabel('observation')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                   edgecolor='black')
        plt.xlim(0, T)
        plt.yticks(np.arange(-1, 1.1, 0.2))
        plt.ylim(-1, 1)
        plt.show()
     """

    ##################################################################################################################
    ###        Mean squared error of Pearson/Kendall correlation coefficient using machine learner estimates       ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    ciw = 99
    reps = 1000
    delta_t = range(3, 101)   # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    model = ['rf']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['pearson']
    output_type = ['true']
    n_neighbour = [10]  # 5, 10, 25, 50, 100, len_train, IDW
    rho_bias_squared = np.full(101, np.nan)
    rho_var_vec = np.full(101, np.nan)
    rho_mse_vec = np.full(101, np.nan)

    """
    # Create dataframe with (interpolated) mse results, squared bias, variance for varying window lengths
    for model, n_neighbour, proxy_type, dt, output_type in [(w, k, x, y, z) for w in model for k in n_neighbour for
                                                            x in proxy_type for y in delta_t for z in output_type]:
        filename = '%s%i_%s_%i_estimate_uncertainty_rep_1000_%s_corr.pkl' % (model, n_neighbour, proxy_type, dt, output_type)
        print(filename)
        data = mm.load_data('bivariate_analysis/%s_cor/%s/results_%s_%s_%s_cor/' % (output_type, proxy_type, model,
                                                                                    proxy_type, output_type) + filename)
        rho_estimates = data['Rho_estimate']
        rho_bias_squared[dt] = np.mean(np.power(rho_estimates-rho_true, 2))
        rho_var_vec[dt] = np.power(np.mean(data['std rho estimate']), 2)

    rho_mse_vec = np.array([np.sum(pair) for pair in zip(rho_bias_squared, rho_var_vec)])
    data_frame = pd.DataFrame({'bias_squared': rho_bias_squared, 'variance': rho_var_vec,
                               'MSE': rho_mse_vec})
    filename_save = 'mse_%s%i_%s_%s_cor.pkl' % (model, n_neighbour, proxy_type, output_type)
    print(filename_save)
    mm.save_data('bivariate_analysis/%s_cor/mse_results_%s_cor/' % (output_type, output_type) + filename_save, data_frame)
    """
    """
    # Decision tree mse computation
    for dt in delta_t:
        filename = 'dt_pearson_%i_estimate_uncertainty_true_corr.pkl' % dt
        data = mm.load_data('bivariate_analysis/true_cor/pearson/dt/' + filename)
        rho_estimates = data['Rho_estimate']
        rho_bias_squared[dt] = np.mean(np.power(rho_estimates - rho_true, 2))
    data_frame = pd.DataFrame({'MSE': rho_bias_squared})
    filename = 'mse_dt_pearson_true_cor.pkl' 
    mm.save_data('bivariate_analysis/true_cor/mse_results_true_cor/' + filename, data_frame)
    """


        
        


    ## Load MSE data Pearson/ Kendall
    mse_pearson_vec = mm.load_data('bivariate_analysis/mse_pearson.pkl')
    mse_kendall_vec = mm.load_data('bivariate_analysis/mse_kendall.pkl')

    ## Load MSE data KNN
    # True Correlation
    mse_knn5_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn5_pearson_true_cor.pkl')
    mse_knn10_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn10_pearson_true_cor.pkl')
    mse_knn25_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn25_pearson_true_cor.pkl')
    mse_knn50_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn50_pearson_true_cor.pkl')
    mse_knn100_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn100_pearson_true_cor.pkl')
    mse_knn_len_train_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_len_train_pearson_true_cor.pkl')
    mse_knn_IDW_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_IDW_pearson_true_cor.pkl')

    mse_knn5_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn5_kendall_true_cor.pkl')
    mse_knn10_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn10_kendall_true_cor.pkl')
    mse_knn25_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn25_kendall_true_cor.pkl')
    mse_knn50_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn50_kendall_true_cor.pkl')
    mse_knn100_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn100_kendall_true_cor.pkl')
    mse_knn_len_train_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_len_train_kendall_true_cor.pkl')
    mse_knn_IDW_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_IDW_kendall_true_cor.pkl')

    # Proxy Correlation
    mse_knn5_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn5_pearson_proxy_cor.pkl')
    mse_knn10_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn10_pearson_proxy_cor.pkl')
    mse_knn25_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn25_pearson_proxy_cor.pkl')
    mse_knn50_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn50_pearson_proxy_cor.pkl')
    mse_knn100_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn100_pearson_proxy_cor.pkl')

    mse_knn_len_train_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_len_train_pearson_proxy_cor.pkl')
    mse_knn_IDW_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_IDW_pearson_proxy_cor.pkl')

    mse_knn5_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn5_kendall_proxy_cor.pkl')
    mse_knn_len_train_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_len_train_kendall_proxy_cor.pkl')
    mse_knn_IDW_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_IDW_kendall_proxy_cor.pkl')

    ## Load MSE data RF
    # True Correlation
    mse_rf10_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf10_pearson_true_cor.pkl')
    mse_rf100_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf100_pearson_true_cor.pkl')
    mse_rf300_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf300_pearson_true_cor.pkl')
    mse_rf1000_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf1000_pearson_true_cor.pkl')


    mse_rf10_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf10_kendall_true_cor.pkl')
    mse_rf100_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf100_kendall_true_cor.pkl')
    mse_rf300_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf300_kendall_true_cor.pkl')
    mse_rf1000_kendall_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf1000_kendall_true_cor.pkl')

    # Proxy Correlation
    mse_rf10_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_rf10_pearson_proxy_cor.pkl')

    mse_rf10_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_rf10_kendall_proxy_cor.pkl')


    """
    # Figure without interpolation MSE
    plt.figure(1)
    plt.plot(mse_pearson_vec['MSE'], label='Pearson', color='indigo', linewidth=1)
    plt.plot(mse_kendall_vec['MSE'], label='Kendall', color='cyan', linestyle='--', linewidth=1)
    plt.plot(mse_knn5_pearson_proxy['MSE'], label='KNN(5)_pearson', linewidth=1, color='brown')
    plt.plot(mse_knn5_kendall_proxy['MSE'], label='KNN_kendall', linewidth=1, color='xkcd:azure')
    #plt.plot(mse_knn10_pearson_proxy['MSE'], label='KNN(10)', linewidth=1)
    #plt.plot(mse_knn25_pearson_proxy['MSE'], label='KNN(25)', linewidth=1)
    #plt.plot(mse_knn50_pearson_proxy['MSE'], label='KNN(50)', linewidth=1)
    #plt.plot(mse_knn100_pearson_proxy['MSE'], label='KNN(100)', linewidth=1)
    #plt.plot(mse_knn_IDW_pearson_true['MSE'], label='KNN_pearson_IDW', color='black', linewidth=1)
    #plt.plot(mse_knn_IDW_kendall_true['MSE'], label='KNN_kendall_idw', linewidth=1, color='xkcd:azure')
    #plt.plot(mse_knn_len_train_pearson_true['MSE'], label='KNN_pearson_len_train', linewidth=1)
    #plt.plot(mse_knn_len_train_pearson_proxy['MSE'], label='KNN_pearson_len_train', color='black', linewidth=1)
    #plt.plot(mse_knn_IDW_pearson_proxy['MSE'], label='KNN_pearson_IDW', color='black', linewidth=1)

    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 0.61, 0.1))
    plt.ylim(0, 0.60)
    plt.show()
    """
    # Figure without interpolation MSE decomposition
    """
    plt.figure(2)
    plt.plot(mse_knn_IDW_kendall_true['bias_squared'], label='Squared Bias', color='blue', linewidth=1)
    plt.plot(mse_knn_IDW_kendall_true['variance'], label='Variance', color='red', linewidth=1)
    plt.plot(mse_knn_IDW_kendall_true['MSE'], label='MSE', color='black', linestyle='--', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 0.31, 0.02))
    plt.ylim(0, 0.2)
    plt.show()

    """
    # Figure with interpolation MSE decomposition sensitivity analysis
    """
    mse_knn_pearson_true_cor_sa = preprocesser.mse_knn_sensitivity_analysis()
    mm.save_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_pearson_true_cor_sensitivity_analysis.pkl',
                 mse_knn_pearson_true_cor_sa)
    mse_knn_kendall_true_cor_sa = preprocesser.mse_knn_sensitivity_analysis(proxy_type='kendall')
    mm.save_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_knn_kendall_true_cor_sensitivity_analysis.pkl',
                 mse_knn_kendall_true_cor_sa)
    """
    """
    mse_knn_pearson_proxy_cor_sa = preprocesser.mse_knn_sensitivity_analysis(output_type='proxy')
    mm.save_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_pearson_proxy_cor_sensitivity_analysis.pkl',
                 mse_knn_pearson_proxy_cor_sa)
    mse_knn_kendall_proxy_cor_sa = preprocesser.mse_knn_sensitivity_analysis(proxy_type='kendall', output_type='proxy')
    mm.save_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_kendall_proxy_cor_sensitivity_analysis.pkl',
                 mse_knn_kendall_proxy_cor_sa)

    """
    """
    plt.figure(3)
    xs = np.arange(1001)
    s1mask = np.isfinite(mse_knn_pearson_proxy_cor_sa['bias_squared'])
    s2mask = np.isfinite(mse_knn_pearson_proxy_cor_sa['variance'])
    s3mask = np.isfinite(mse_knn_pearson_proxy_cor_sa['MSE'])
    plt.plot(xs[s1mask], mse_knn_pearson_proxy_cor_sa['bias_squared'][s1mask], label='Squared Bias', color='blue', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s2mask], mse_knn_pearson_proxy_cor_sa['variance'][s2mask], label='Variance', color='red', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s3mask], mse_knn_pearson_proxy_cor_sa['MSE'][s3mask], label='MSE', color='black', linestyle='--', linewidth=1, marker='.')

    plt.xlabel('number of neighbours')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                  edgecolor='black')
    plt.xlim(0, 100)
    plt.xticks([5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.yticks(np.arange(0, 0.21, 0.02))
    plt.ylim(0, 0.2)
    plt.show()
    """
    """
    # Variance in MSE window sizes for KNN with Pearson/ Kendall as covariates.
    # True Correlation
    #var_mse_knn5_pearson_true = np.nanvar(mse_knn5_pearson_true['MSE']); print('mse_knn5_pearson_var: %.8f' % var_mse_knn5_pearson_true)
    #var_mse_knn5_kendall_true = np.nanvar(mse_knn5_kendall_true['MSE']); print('mse_knn5_kendall_var: %.8f' % var_mse_knn5_kendall_true)
    #var_mse_knn_len_train_pearson_true = np.nanvar(mse_knn_len_train_pearson_true['MSE']); print('mse_knn_len_train_pearson_var: %.13f' % var_mse_knn_len_train_pearson_true)
    #var_mse_knn_IDW_pearson_true = np.nanvar(mse_knn_IDW_pearson_true['MSE']); print('mse_knn_IDW_pearson_var: %.9f' % var_mse_knn_IDW_pearson_true)
    #var_mse_knn_len_train_kendall_true = np.nanvar(mse_knn_len_train_kendall_true['MSE']); print('mse_knn_len_train_pearson_var: %f' % var_mse_knn_len_train_kendall_true)
    #var_mse_knn_IDW_kendall_true = np.nanvar(mse_knn_IDW_kendall_true['MSE']); print('mse_knn_IDW_pearson_var: %f' % var_mse_knn_IDW_kendall_true)
    # Proxy Correlation
    #var_mse_knn5_pearson_proxy = np.nanvar(mse_knn5_pearson_proxy['MSE']); print('mse_knn5_pearson_proxy_var: %.6f' % var_mse_knn5_pearson_proxy)
    #var_mse_knn5_kendall_proxy = np.nanvar(mse_knn5_kendall_proxy['MSE']); print('mse_knn5_kendall_proxy_var: %.6f' % var_mse_knn5_kendall_proxy)
    #var_mse_knn_len_train_pearson_proxy = np.nanvar(mse_knn_len_train_pearson_proxy['MSE']); print('mse_knn_len_train_pearson_proxy_var: %.8f' % var_mse_knn_len_train_pearson_proxy)
    #var_mse_knn_len_train_kendall_proxy = np.nanvar(mse_knn_len_train_kendall_proxy['MSE']); print('mse_knn_len_train_kendall_proxy_var: %.9f' % var_mse_knn_len_train_kendall_proxy)
    #var_mse_knn_IDW_pearson_proxy = np.nanvar(mse_knn_IDW_pearson_proxy['MSE']); print('mse_knn_IDW_pearson_proxy_var: %.8f' % var_mse_knn_IDW_pearson_proxy)
    #var_mse_knn_IDW_kendall_proxy = np.nanvar(mse_knn_IDW_kendall_proxy['MSE']); print('mse_knn_IDW_kendall_proxy_var: %.8f' % var_mse_knn_IDW_kendall_proxy)

    # Max-min in MSE window sizes for KNN with Pearson/ Kendall as covariates.
    # True Correlation
    #print('mse_knn5_pearson_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn5_pearson_true['MSE']), np.nanmax(mse_knn5_pearson_true['MSE'])))
    #print('mse_knn5_kendall_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn5_kendall_true['MSE']), np.nanmax(mse_knn5_kendall_true['MSE'])))
    #print('mse_knn_len_train_pearson_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_len_train_pearson_true['MSE']), np.nanmax(mse_knn_len_train_pearson_true['MSE'])))
    #print('mse_knn_IDW_pearson_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_IDW_pearson_true['MSE']), np.nanmax(mse_knn_IDW_pearson_true['MSE'])))
    #print('mse_knn_len_train_kendall_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_len_train_kendall_true['MSE']), np.nanmax(mse_knn_len_train_kendall_true['MSE'])))
    #print('mse_knn_IDW_kendall_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_IDW_kendall_true['MSE']), np.nanmax(mse_knn_IDW_kendall_true['MSE'])))
    # Proxy Correlation
    #print('mse_knn5_pearson_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn5_pearson_proxy['MSE']), np.nanmax(mse_knn5_pearson_proxy['MSE'])))
    #print('mse_knn5_kendall_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn5_kendall_proxy['MSE']), np.nanmax(mse_knn5_kendall_proxy['MSE'])))
    #print('mse_knn_len_train_pearson_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_len_train_pearson_proxy['MSE']), np.nanmax(mse_knn_len_train_pearson_proxy['MSE'])))
    #print('mse_knn_len_train_kendall_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_len_train_kendall_proxy['MSE']), np.nanmax(mse_knn_len_train_kendall_proxy['MSE'])))
    #print('mse_knn_IDW_pearson_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_IDW_pearson_proxy['MSE']), np.nanmax(mse_knn_IDW_pearson_proxy['MSE'])))
    #print('mse_knn_IDW_kendall_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_knn_IDW_kendall_proxy['MSE']), np.nanmax(mse_knn_IDW_kendall_proxy['MSE'])))
    """
    """
    # Variance in MSE window sizes for RF with Pearson/ Kendall as covariates.
    # True Correlation
    #var_mse_rf10_pearson_true = np.nanvar(mse_rf10_pearson_true['MSE']); print('var_mse_rf10_pearson_true: %.8f' % var_mse_rf10_pearson_true)
    #var_mse_rf10_kendall_true = np.nanvar(mse_rf10_kendall_true['MSE']); print('var_mse_rf10_kendall_true: %.8f' % var_mse_rf10_kendall_true)
    # Proxy Correlation
    var_mse_rf10_pearson_proxy = np.nanvar(mse_rf10_pearson_proxy['MSE']); print('var_mse_rf10_pearson_proxy: %.6f' % var_mse_rf10_pearson_proxy)
    var_mse_rf10_kendall_proxy = np.nanvar(mse_rf10_kendall_proxy['MSE']); print('var_mse_rf10_kendall_proxy: %.6f' % var_mse_rf10_kendall_proxy)

    # Max-min in MSE window sizes for RF with Pearson/ Kendall as covariates.
    # True Correlation
    #print('mse_rf10_pearson_min_max: (%.4f, %.4f)' % (np.nanmin(mse_rf10_pearson_true['MSE']), np.nanmax(mse_rf10_pearson_true['MSE'])))
    #print('mse_rf10_kendall_min_max: (%.4f, %.4f)' % (np.nanmin(mse_rf10_kendall_true['MSE']), np.nanmax(mse_rf10_kendall_true['MSE'])))
    
    # Proxy Correlation
    print('mse_rf10_pearson_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_rf10_pearson_proxy['MSE']), np.nanmax(mse_rf10_pearson_proxy['MSE'])))
    print('mse_rf10_kendall_proxy_min_max: (%.4f, %.4f)' % (np.nanmin(mse_rf10_kendall_proxy['MSE']), np.nanmax(mse_rf10_kendall_proxy['MSE'])))
    """
    """
    # Figure without interpolation MSE
    plt.figure(4)
    #plt.plot(mse_pearson_vec['MSE'], label='Pearson', color='indigo', linewidth=1)
    #plt.plot(mse_kendall_vec['MSE'], label='Kendall', color='cyan', linestyle='--', linewidth=1)
    plt.plot(mse_rf10_pearson_proxy['MSE'], label='RF_pearson', color='goldenrod', linewidth=1)
    #plt.plot(mse_rf10_kendall_proxy['MSE'], label='KNN_kendall', color='xkcd:teal', linewidth=1)
    plt.plot(mse_rf10_kendall_proxy['MSE'], label='RF_kendall', color='green', linewidth=1)
    #plt.plot(mse_rf1000_pearson_true['MSE'], label='RF_pearson', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 0.61, 0.1))
    plt.ylim(0, 0.6)
    plt.show()
    """


    # Figure without interpolation MSE decomposition
    """
    mse_dt_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_dt_pearson_true_cor.pkl')
    mse_rf10_2_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf10_2_pearson_true_cor.pkl')
    mse_rf10_3_pearson_true = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf10_3_pearson_true_cor.pkl')
    """
    """
    plt.figure(5)
    plt.plot(mse_rf10_kendall_proxy['bias_squared'], label='Squared Bias', color='blue', linewidth=1)
    plt.plot(mse_rf10_kendall_proxy['variance'], label='Variance', color='red', linewidth=1)
    plt.plot(mse_rf10_kendall_proxy['MSE'], label='MSE', color='black', linestyle='--', linewidth=1)
    #plt.plot(mse_dt_pearson_true, label='dt_squared_bias', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 0.61, 0.02))
    plt.ylim(0, 0.3)
    plt.show()
    """
    """ 
    # Figure with interpolation MSE decomposition sensitivity analysis
    mse_rf_pearson_true_cor_sa = mm.load_data('bivariate_analysis/true_cor/mse_results_true_cor/mse_rf300_1_to_3_pearson_true_cor.pkl')
    plt.figure(3)
    xs = np.arange(4)
    s1mask = np.isfinite(mse_rf_pearson_true_cor_sa['bias_squared'])
    s2mask = np.isfinite(mse_rf_pearson_true_cor_sa['variance'])
    s3mask = np.isfinite(mse_rf_pearson_true_cor_sa['MSE'])
    plt.plot(xs[s1mask], mse_rf_pearson_true_cor_sa['bias_squared'][s1mask], label='Squared Bias', color='blue', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s2mask], mse_rf_pearson_true_cor_sa['variance'][s2mask], label='Variance', color='red', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s3mask], mse_rf_pearson_true_cor_sa['MSE'][s3mask], label='MSE', color='black', linestyle='--', linewidth=1, marker='.')

    plt.xlabel('number of covariates')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                  edgecolor='black')
    plt.xlim(0, 3)
    plt.xticks([0, 1, 2, 3])
    plt.yticks(np.arange(0, 0.21, 0.02))
    plt.ylim(0, 0.2)
    plt.show()
    """
    ##################################################################################################################
    ###                                   Minimum Determinant Learning Algorithms                                       ###
    ##################################################################################################################
    # Rho_estimate
    # Get information on the minimum determinants over all correlation estimates for all window sizes [3, 100]
    delta_t = range(3, 101)
    det_min_vec = np.full(101, np.nan)
    proxy_type = 'pearson'
    output_type = 'true'
    learner = 'rf'

    """                    
    for dt in delta_t:
        # Load data Pearson/ Kendall
        det_data_vec = np.full(501, np.nan)
        filename = '%s10_%s_%i_estimate_uncertainty_rep_1000_%s_corr.pkl' % (learner, proxy_type, dt, output_type)
        print(filename)
        data = mm.load_data('bivariate_analysis/%s_cor/%s/results_%s_%s_%s_cor/%s'
                            % (output_type, proxy_type, learner, proxy_type, output_type, filename))
        # Compute determinants for every dataset
        for i, rho in enumerate(data['Rho_estimate']):
            det_data_vec[i+1] = preprocesser.determinant_LU_factorization(rho, 2)
        det_min_vec[dt] = np.nanmin(det_data_vec)
    filename_save = 'determinant_min_%s10_%s_%s_cor.pkl' % (learner, proxy_type, output_type)
    mm.save_data('bivariate_analysis/%s_cor/det_results_%s_cor/%s' % (output_type, output_type, filename_save), det_min_vec)
    """


    # Plot minimum determinants of KNN estimates of correlation
    # True Cor
    det_min_knn5_pearson = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn5_pearson_true_cor.pkl')
    det_min_knn5_kendall = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn5_kendall_true_cor.pkl')
    det_min_knn_len_train_pearson = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn_len_train_pearson_true_cor.pkl')
    det_min_knn_len_train_kendall = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn_len_train_kendall_true_cor.pkl')
    det_min_knn_IDW_pearson = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn_IDW_pearson_true_cor.pkl')
    det_min_knn_IDW_kendall = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_knn_IDW_kendall_true_cor.pkl')
    # Proxy Cor
    det_min_knn5_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn5_pearson_proxy_cor.pkl')
    det_min_knn5_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn5_kendall_proxy_cor.pkl')
    det_min_knn_len_train_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn_len_train_pearson_proxy_cor.pkl')
    det_min_knn_len_train_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn_len_train_kendall_proxy_cor.pkl')
    det_min_knn_IDW_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn_IDW_pearson_proxy_cor.pkl')
    det_min_knn_IDW_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_knn_IDW_kendall_proxy_cor.pkl')

    """
    plt.figure(1)
    plt.plot(det_min_knn_IDW_pearson_proxy, label='KNN_pearson_IDW', linewidth=1, color='orange')
    plt.plot(det_min_knn_IDW_kendall_proxy, label='KNN_kendall_IDW', linewidth=1)
    plt.plot(det_min_knn_len_train_pearson_proxy, label='KNN_pearson_len_train', linewidth=1)
    plt.plot(det_min_knn_len_train_kendall_proxy, label='KNN_kendall_len_train', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('minimum det($R_t)$')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(-0.1, 1.1, 0.1))
    plt.ylim(-0.1, 1)
    plt.show()
    """

    # Plot minimum determinants of RF estimates of correlation
    # True Cor
    det_min_rf10_pearson_true = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_rf10_pearson_true_cor.pkl')
    det_min_rf10_kendall_true = mm.load_data('bivariate_analysis/true_cor/det_results_true_cor/determinant_min_rf10_kendall_true_cor.pkl')
    """
    # Proxy Cor
    det_min_rf10_pearson_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_rf10_pearson_proxy_cor.pkl')
    det_min_rf10_kendall_proxy = mm.load_data('bivariate_analysis/proxy_cor/det_results_proxy_cor/determinant_min_rf10_kendall_proxy_cor.pkl')

    print(np.nanmin(det_min_rf10_pearson_proxy))
    print(np.nanmin(det_min_rf10_kendall_proxy))

    plt.figure(1)
    plt.plot(det_min_rf10_pearson_proxy, label='RF_pearson', linewidth=1, color='orange')
    plt.plot(det_min_rf10_kendall_proxy, label='RF_kendall', linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('minimum det($R_t)$')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(-0.1, 1.1, 0.1))
    plt.ylim(-0.1, 1)
    plt.show()
    """








###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()