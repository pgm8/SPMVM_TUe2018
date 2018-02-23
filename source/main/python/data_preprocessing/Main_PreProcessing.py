#from pandas_datareader import data as dt

from PreProcessor import PreProcessor
from ModuleManager import ModuleManager
from TechnicalAnalyzer import TechnicalAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as IT
from scipy.stats.stats import pearsonr, kendalltau
import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error  # use mse to penalize outliers more

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
    ###     Estimation uncertainty in (weighted) Pearson correlation coefficient using moving window estimates     ###
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
    ###       Mean squared error of (weighted) Pearson correlation coefficient using moving window estimates       ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    delta_t_min, delta_t_max = 3, 252
    delta_t = np.arange(3, 252)  # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    proxy_type = ['mw', 'emw', 'kendall']  # run proxies individually otherwise one saves dataframe over other.
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
    mse_mw_vec = mm.load_data('bivariate_analysis/mse_mw.pkl')
    mse_emw_vec = mm.load_data('bivariate_analysis/mse_emw.pkl')
    mse_kendall_vec = mm.load_data('bivariate_analysis/mse_kendall.pkl')
    """
    """
    # Figure with interpolation MSE
    plt.figure(1)
    xs = np.arange(252)
    s1mask = np.isfinite(mse_mw_vec['MSE'])
    s2mask = np.isfinite(mse_emw_vec['MSE'])
    plt.plot(xs[s1mask], mse_mw_vec['MSE'][s1mask], label='Moving Window', color='blue', linestyle='-', linewidth=1)
    plt.plot(xs[s2mask], mse_emw_vec['MSE'][s2mask], label='Exp. Weighted Moving Window', color='red', linestyle='-',
             linewidth=1)
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 1.00000001, 0.1))
    plt.ylim(0, 0.6)
    plt.show()
    """
    """
    # Figure without interpolation MSE 
    plt.figure(2)
    plt.plot(mse_mw_vec['MSE'], label='Moving Window', color='blue', linewidth=1)
    plt.plot(mse_emw_vec['MSE'], label='Exp. Weighted Moving Window', color='red', linewidth=1)
    plt.plot(mse_kendall_vec['MSE'], label='Kendall', color='black', linewidth=1, linestyle='--')
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
    plt.figure(3)
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
    var_mse_mw = np.nanvar(mse_mw_vec['MSE']); print('mse_mw_var: %f' % var_mse_mw)
    var_mse_emw = np.nanvar(mse_emw_vec['MSE']); print('mse_emw_var: %f' % var_mse_emw)
    var_mse_kendall = np.nanvar(mse_kendall_vec['MSE']); print('mse_kendall_var: %f' % var_mse_kendall)

    # Max-min in MSE window sizes
    print('mse_mw_min_max: (%f, %f)' % (np.nanmin(mse_mw_vec['MSE']), np.nanmax(mse_mw_vec['MSE'])))
    print('mse_emw_min_max: (%f, %f)' % (np.nanmin(mse_emw_vec['MSE']), np.nanmax(mse_emw_vec['MSE'])))
    print('mse_kendall_min_max: (%f, %f)' % (np.nanmin(mse_kendall_vec['MSE']), np.nanmax(mse_kendall_vec['MSE'])))
    """
    ##################################################################################################################
    ###                                          Dataset creation                                                  ###
    ##################################################################################################################
    # Pearson correlation moving window estimates as covariate and true correlation or moving window
    # estimate as proxy for target variable
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    delta_t_min = 3
    delta_t_max = 252
    proxy_type = ['mw', 'emw']     # ['mw', 'emw']

    start_time = time.time()
    for dt, proxy_type in [(x,y) for x in range(delta_t_min, delta_t_max) for y in proxy_type]:
        print('(%i, %s)' % (dt, proxy_type))
        dataset, dataset_proxy = \
            preprocesser.generate_bivariate_dataset(ta, simulated_data_process, dt, proxy_type=proxy_type)
        mm.save_data('/bivariate_analysis/true_cor/%s/dataset_%s_%d.pkl' % (proxy_type, proxy_type, dt), dataset)
        mm.save_data('/bivariate_analysis/proxy_cor/%s/dataset_%s_%d.pkl' % (proxy_type, proxy_type, dt), dataset_proxy)

    print("%s: %f" % ('Execution time:', (time.time() - start_time)))
    """
    ##################################################################################################################
    ###    Estimation uncertainty in (weighted) Pearson correlation coefficient using machine learner estimates    ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    ciw = 99
    reps = 1000
    delta_t = [4]   # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    model = ['knn']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['mw']
    output_type = ['proxy']
    n_neighbours = [5]

    """
    for dt, proxy_type, model, k, output_type in [(x, y, z, k, o) for x in delta_t for y in proxy_type
                                     for z in model for k in n_neighbours for o in output_type]:
        start_time = time.time()
        print('(%i, %s, %s, %i)' % (dt, proxy_type, model, k))
        dataset = mm.load_data('bivariate_analysis/%s_cor/%s/dataset_%s_%i.pkl' % (output_type, proxy_type, proxy_type, dt))
        rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates = \
        preprocesser.bootstrap_learner_estimate(data=dataset, reps=reps, model=model, n_neighbors=k)
        data_frame = pd.DataFrame({'Percentile_low': lower_percentiles, 'Percentile_up': upper_percentiles,
                                   'std rho estimate': sd_rho_estimates, 'Rho_estimate': rho_estimates})
        filename = '%s5_%s_%i_estimate_uncertainty_%s_corr.pkl' % (model, proxy_type, dt, output_type)
        #mm.save_data('bivariate_analysis/%s_cor/' % output_type + filename, data_frame)
        print("%s: %f" % ('Execution time', (time.time() - start_time)))
  
    # Execution time knn_IDW, 21, mw, proxy:  340 seconds
    # Execution time knn5, 21, mw, proxy:     512 seconds
    # Execution time rf10, 21, emw, true:    9514 seconds
    # Execution time rf10, 251, mw, true:    9594 seconds
    # Execution time rf10, 251, emw, true:   9696 seconds
    # Execution time rf, 21, mw, true:      95593 seconds (roughly 26.5 hours)
    # Execution time rf, 251, mw, true:    162240 seconds (roughly 45 hours)
    # Execution time rf, 21, emw, true:    274400 seconds (roughly 76 hours)
    """
    """
    delta_t = [10, 21]  # [10, 25]
    for dt in delta_t:
        data = mm.load_data('bivariate_analysis/proxy_cor/mw/results_knn_mw_proxy_cor/'
                            'knn5_mw_%i_estimate_uncertainty_proxy_corr.pkl' % dt)
        rho_estimates = data['Rho_estimate']
        lower_percentiles = data['Percentile_low']
        upper_percentiles = data['Percentile_up']
        plt.figure()
        plt.plot(simulated_data_process['rho'], label='true correlation', linewidth=1, color='black')
        plt.plot(rho_estimates, label='KNN(%i) correlation' % dt, linewidth=1, color='red')
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
    # Figure
    data = mm.load_data('bivariate_analysis/true_cor/rf100_mw_21_estimate_uncertainty_true_corr.pkl')
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
    """
    # Figure
    data = mm.load_data('bivariate_analysis/proxy_cor/mw/results_knn_mw_proxy_cor/'
                        'knn_mw_10_IDW_estimate_uncertainty_proxy_corr.pkl')
    rho_estimates = data['Rho_estimate']
    lower_percentiles = data['Percentile_low']
    upper_percentiles = data['Percentile_up']
    plt.figure(1)
    plt.plot(simulated_data_process['rho'], label='true correlation', linewidth=1, color='black')
    plt.plot(rho_estimates, label='KNN_idw correlation', linewidth=1, color='red')
    plt.plot((upper_percentiles - lower_percentiles) - 1, label='%d%% interval (bootstrap)' % ciw,
             linewidth=1, color='magenta')
    # plt.plot(lower_percentiles, label='%d%% interval (bootstrap)' % ciw, linewidth=1, color='magenta')
    # plt.plot(upper_percentiles, label="", linewidth=1, color='magenta')
    plt.xlabel('observation')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
               edgecolor='black')
    plt.xlim(0, T)
    plt.yticks(np.arange(-1, 1.1, 0.2))
    plt.ylim(-1, 1)
    plt.show()
    """
    
    ##################################################################################################################
    ###       Mean squared error of (weighted) Pearson correlation coefficient using machine learner estimates     ###
    ##################################################################################################################
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    T = 500
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)
    ciw = 99
    reps = 1000
    delta_t = range(3, 101)   # dt = {[3, 10], 21, 42, 63, 126, 251}  (and 84 possibly)
    model = ['knn']  # k-nearest neighbour: 'knn', random forest: 'rf'
    proxy_type = ['mw']
    output_type = ['proxy']
    n_neighbour = [5, 10, 25, 50, 100]  # IDW, lenTrain separate
    rho_bias_squared = np.full(101, np.nan)
    rho_var_vec = np.full(101, np.nan)
    rho_mse_vec = np.full(101, np.nan)
    """"
    # Create dataframe with (interpolated) mse results, squared bias, variance for varying window lengths
    for model, n_neighbour, proxy_type, dt, output_type in [(w, k, x, y, z) for w in model for k in n_neighbour for
                                                            x in proxy_type for y in delta_t for z in output_type]:
        filename = '%s%i_%s_%i_estimate_uncertainty_%s_corr.pkl' % (model, n_neighbour, proxy_type, dt, output_type)
        print(filename)
        data = mm.load_data('bivariate_analysis/%s_cor/%s/results_%s_%s_%s_cor/' % (output_type, proxy_type, model,
                                                                                    proxy_type, output_type) + filename)
        rho_estimates = data['Rho_estimate']
        rho_bias_squared[dt] = np.mean(np.power(rho_estimates-rho_true, 2))
        rho_var_vec[dt] = np.power(np.mean(data['std rho estimate']), 2)

    rho_mse_vec = np.array([np.sum(pair) for pair in zip(rho_bias_squared, rho_var_vec)])
    data_frame = pd.DataFrame({'bias_squared': rho_bias_squared, 'variance': rho_var_vec,
                               'MSE': rho_mse_vec})
    filename = 'mse_%s%i_%s_%s_cor.pkl' % (model, n_neighbour, proxy_type, output_type)
    mm.save_data('bivariate_analysis/%s_cor/mse_results_%s_cor/' % (output_type, output_type) + filename, data_frame)
    """
    """
    # Figure with interpolation MSE decomposition
    plt.figure(1)
    xs = np.arange(252)
    s1mask = np.isfinite(rho_bias_squared)
    s2mask = np.isfinite(rho_var_vec)
    s3mask = np.isfinite(rho_mse_vec)
    plt.plot(xs[s1mask], rho_bias_squared[s1mask], label='Squared Bias', color='blue', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s2mask], rho_var_vec[s2mask], label='Variance', color='red', linestyle='-', linewidth=1, marker='.')
    plt.plot(xs[s3mask], rho_mse_vec[s3mask], label='MSE', color='black', linestyle='--', linewidth=1, marker='.')

    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True,
                  edgecolor='black')
    plt.xlim(0, 250)
    plt.yticks(np.arange(0, 1.00000001, 0.1))
    plt.ylim(0, 0.5)
    plt.show()
    """
    
    # Load MSE data Pearson/ Kendall
    mse_mw_vec = mm.load_data('bivariate_analysis/mse_mw.pkl')
    mse_kendall_vec = mm.load_data('bivariate_analysis/mse_kendall.pkl')

    # Load MSE data KNN
    # True Correlation

    # Proxy Correlation
    mse_knn5_mw_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn5_mw_proxy_cor.pkl')
    mse_knn10_mw_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn10_mw_proxy_cor.pkl')
    mse_knn25_mw_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn25_mw_proxy_cor.pkl')
    mse_knn50_mw_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn50_mw_proxy_cor.pkl')
    mse_knn100_mw_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn100_mw_proxy_cor.pkl')
    mse_knn_mw_IDW_proxy_cor = mm.load_data('bivariate_analysis/proxy_cor/mse_results_proxy_cor/mse_knn_mw_IDW_proxy_cor.pkl')

    """
    # Figure without interpolation MSE
    plt.figure(1)
    plt.plot(mse_mw_vec['MSE'], label='MW', color='blue', linewidth=1)
    plt.plot(mse_kendall_vec['MSE'], label='Kendall', color='red', linewidth=1)
    #plt.plot(mse_knn5_mw_proxy_cor['MSE'], label='KNN(5)', linewidth=1, linestyle='--', color='black')
    #plt.plot(mse_knn10_mw_proxy_cor['MSE'], label='knn(10)', color='red')
    #plt.plot(mse_knn25_mw_proxy_cor['MSE'], label='KNN(25)')
    #plt.plot(mse_knn50_mw_proxy_cor['MSE'], label='KNN(50)')
    #plt.plot(mse_knn100_mw_proxy_cor['MSE'], label='KNN(100)', linewidth=1, color='black')
    plt.plot(mse_knn_mw_IDW_proxy_cor['MSE'], label='KNN_inverse_distance', linewidth=1, color='black')

    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 0.51, 0.1))
    plt.ylim(0, 0.54)
    plt.show()
    """

    """
    # Variance in MSE window sizes
    sd_mse_knn5_mw_proxy = np.nanstd(mse_knn5_mw_proxy_cor['MSE']);print(np.power(sd_mse_knn5_mw_proxy, 2))
    sd_mse_knn100_mw_proxy = np.nanstd(mse_knn100_mw_proxy_cor['MSE']);print(np.power(sd_mse_knn100_mw_proxy, 2))
    sd_mse_knn_mw_IDW_proxy = np.nanstd(mse_knn_mw_IDW_proxy_cor['MSE']);print(np.power(sd_mse_knn_mw_IDW_proxy, 2))


    # Max-min in MSE window sizes
    print('knn5_mw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_knn5_mw_proxy_cor['MSE']), np.nanmax(mse_knn5_mw_proxy_cor['MSE'])))
    print('knn100_mw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_knn100_mw_proxy_cor['MSE']), np.nanmax(mse_knn100_mw_proxy_cor['MSE'])))
    print('knn_mw_IDW_proxy_min_max: (%f, %f)' % (np.nanmin(mse_knn_mw_IDW_proxy_cor['MSE']), np.nanmax(mse_knn_mw_IDW_proxy_cor['MSE'])))
    """
    """
    # Random Forest
    mse_rf10_mw_vec = mm.load_data('/bivariate_analysis/true_cor/mse_rf10_mw_true_corr.pkl')
    mse_rf10_emw_vec = mm.load_data('/bivariate_analysis/true_cor/mse_rf10_emw_true_corr.pkl')
    mse_rf100_true_cor = mm.load_data('/bivariate_analysis/true_cor/mse_rf100_mw_true_corr.pkl')
    """
    """
    # Try out mse decomposition
    T = 500
    simulated_data_process = mm.load_data('/bivariate_analysis/correlated_sim_data.pkl')
    rho_true = simulated_data_process.tail(T).iloc[:, -1]
    rho_true.reset_index(drop=True, inplace=True)


    data = mm.load_data('/bivariate_analysis/true_cor/knn25_mw_21_estimate_uncertainty_true_corr.pkl')
    rho_estimate = data['Rho_estimate']
    rho_var = np.power(data['std rho estimate'], 2)
    mse_true_bootstrapped = np.mean(np.power(rho_true-rho_estimate, 2)) + np.mean(rho_var)
    print(mse_true_bootstrapped)
    print(mean_squared_error(rho_estimate, rho_true))
    """
    """
    mse_rf10_mw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf10_mw_proxy_corr.pkl')
    mse_rf10_emw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf10_emw_proxy_corr.pkl')
    mse_rf100_emw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf100_emw_proxy_corr.pkl')
    mse_rf100_mw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf100_mw_proxy_corr.pkl')

    mse_rf300_emw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf300_emw_proxy_corr.pkl')
    #mse_rf300_mw_proxy_vec = mm.load_data('/bivariate_analysis/proxy_cor/mse_rf300_mw_proxy_corr.pkl')

    mse_rf100_mw_vec = mm.load_data('/bivariate_analysis/true_cor/mse_rf100_mw_true_corr.pkl')
    """
    # Variance in MSE window sizes
    """
    sd_mse_rf10_mw = np.nanstd(mse_rf10_mw_vec); print(np.power(sd_mse_rf10_mw, 2))
    sd_mse_rf10_emw = np.nanstd(mse_rf10_emw_vec);print(np.power(sd_mse_rf10_emw, 2))
    sd_mse_rf100_mw = np.nanstd(mse_rf100_mw_vec); print(np.power(sd_mse_rf100_mw, 2))
    """
    """
    sd_mse_rf10_mw_proxy = np.nanstd(mse_rf10_mw_proxy_vec); print(np.power(sd_mse_rf10_mw_proxy, 2))
    sd_mse_rf10_emw_proxy = np.nanstd(mse_rf10_emw_proxy_vec);print(np.power(sd_mse_rf10_emw_proxy, 2))
    sd_mse_rf100_emw_proxy = np.nanstd(mse_rf100_emw_proxy_vec); print(np.power(sd_mse_rf100_emw_proxy, 2))
    sd_mse_rf300_emw_proxy = np.nanstd(mse_rf300_emw_proxy_vec); print(np.power(sd_mse_rf300_emw_proxy, 2))
    sd_mse_rf100_mw_proxy = np.nanstd(mse_rf100_mw_proxy_vec); print(np.power(sd_mse_rf100_mw_proxy, 2))
    """
    # Max-min in MSE window sizes
    """
    print('rf10_mw_min: %f' % np.nanmin(mse_rf10_mw_vec));
    print('rf10_mw_max: %f' % np.nanmax(mse_rf10_mw_vec));
    print('rf10_emw_min: %f' % np.nanmin(mse_rf10_emw_vec));
    print('rf10_emw_max: %f' % np.nanmax(mse_rf10_emw_vec));
    """
    """
    print('rf10_mw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_rf10_mw_proxy_vec), np.nanmax(mse_rf10_mw_proxy_vec)))
    print('rf10_emw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_rf10_emw_proxy_vec), np.nanmax(mse_rf10_emw_proxy_vec)))
    print('rf100_emw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_rf100_emw_proxy_vec), np.nanmax(mse_rf100_emw_proxy_vec)))
    print('rf300_emw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_rf300_emw_proxy_vec), np.nanmax(mse_rf300_emw_proxy_vec)))
    print('rf100_mw_proxy_min_max: (%f, %f)' % (np.nanmin(mse_rf100_mw_proxy_vec), np.nanmax(mse_rf100_mw_proxy_vec)))
    """
    """
    # Figure
    plt.figure(3)
    plt.plot(mse_rf10_mw_proxy_vec, label='RF_mw', color='blue')
    #plt.plot(mse_rf10_emw_proxy_vec, label='RF_emw', color='red')
    #plt.plot(mse_mw_vec, label='MW', color='blue')
    #plt.plot(mse_emw_vec, label='EMW', color='red')
    #plt.plot(mse_rf10_mw_proxy_vec, label='RF(10)', linewidth=1, linestyle='--', color='black')
    #plt.plot(mse_rf100_emw_proxy_vec, label='RF(100)_emw', color='black')
    #plt.plot(mse_rf100_mw_proxy_vec, label='RF(100)', color='black')
    #plt.plot(mse_rf300_mw_proxy_vec, label='RF(300)_mw')
    #plt.plot(mse_rf300_emw_proxy_vec, label='RF(300)_emw')
    plt.xlabel('window length')
    plt.ylabel('MSE')
    plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True,
               edgecolor='black')
    plt.xlim(0, 50)
    plt.yticks(np.arange(0, 0.51, 0.1))
    plt.ylim(0, 0.5)
    plt.show()
    """








###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()