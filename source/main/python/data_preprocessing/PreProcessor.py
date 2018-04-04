import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

from TechnicalAnalyzer import TechnicalAnalyzer
from ModuleManager import ModuleManager
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)


class PreProcessor(object):
    """Preprocessor class. This class has the responsibility to preprocess the data. More specifically, the class
    has the task of simulating random correlated asset paths in the bivariate case. Additionally, the class has the
    responsibility for estimating the uncertainty in the output variable through a bootstrap resampling procedure."""

    def __init__(self):
        """Initializer PreProcessor object."""
        self.ta = TechnicalAnalyzer()
        self.mm = ModuleManager()

    def simulate_random_correlation_ar(self, T, a0, a1):
        """Simulate a random correlation process with highly persistent time-varying correlations following an
           auto-regressive process. Add noise with ar process
        :param T: simulation length
        :param a0:
        :param a1:
        :return: random_corr: correlation process following specified dynamics."""
        eps = 1e-5
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
        :param a0: 0.02
        :param a1: 0.2
        :param b1: 0.78
        :return: random_corr_garch: correlation process following specified dynamics.
        :return: sigma: volatility process."""
        eps = 1e-5
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
            sigma[t] = np.sqrt(sigma_squared)
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

    def determinant_LU_factorization(self, corr_vec, n):
        """Method for determining the determinant of a given matrix. Determinants are computed using using
        LU factorization.
        :param corr_vec: time-varying correlation vector
        :param n: dimension correlation matrix
        :return: determinant."""
        cor_matrix = self.construct_correlation_matrix(corr_vec, n)
        det = np.linalg.det(cor_matrix)
        return det

    def generate_bivariate_dataset(self, ta, simulated_data_process, dt, proxy_type='pearson', T=500):
        """Method for generating a dataset with proxies (exponentially weighted) moving window correlation estimates
        for feature set and true correlation as the response variables.
        :param ta: technical analyzer object
        :param simulated_data_process: bivariate asset process with predefined correlation dynamics.
        :param dt: window length
        :param proxy_type: type definition of proxy for estimates of true correlation
        :param T: length test set
        :return: datasets with true correlation and proxy for output variable."""
        if proxy_type is 'pearson':
            mw_estimates = simulated_data_process[0].rolling(window=dt).corr(other=simulated_data_process[1])
            # Feature set consists of lagged asset price and mw correlation estimate, e.g. x_t = MW_t-1
            dataset = simulated_data_process.iloc[:, :2].shift(periods=1, axis='index')  # Dataframe
            dataset['MW_t-1'] = mw_estimates.shift(periods=1, axis='index')
            dataset_proxy = dataset.copy()       # copy feature matrix
            # Dataset with true correlations as target variable and proxies
            dataset['rho_true'] = simulated_data_process['rho']
            dataset_proxy['rho_proxy'] = mw_estimates
        else:  # Kendall as proxy
            kendall_estimates = ta.kendall_correlation_estimation(simulated_data_process.iloc[:, :2], dt)
            # Feature set consists of lagged asset price and kendall correlation estimate, e.g. x_t = kendall_t-1
            dataset = simulated_data_process.iloc[:, :2].shift(periods=1, axis='index')  # Dataframe
            dataset['Kendall_t-1'] = kendall_estimates.shift(periods=1, axis='index')
            dataset_proxy = dataset.copy()  # copy feature matrix
            # Dataset with true correlations as target variable and proxies
            dataset['rho_true'] = simulated_data_process['rho']
            dataset_proxy['rho_proxy'] = kendall_estimates
        return dataset, dataset_proxy

    def generate_multivariate_dataset(self, ta, data, dt, proxy_type='pearson'):
        """
        :param ta: technical analyzer object
        :param data: dataframe with log returns
        :param dt: window length
        :param proxy_type: type definition of proxy for estimates of true correlation
        :return: dataset with approximated covariates and output variable."""
        kendall_estimates = ta.kendall_correlation_estimation(data, dt)
        # Feature set consists of lagged kendall correlation estimate amd lagged min. and max. asset returns
        dataset = kendall_estimates.shift(periods=1, axis='index')
        dataset['r_min'] = np.min(data, axis=1).shift(periods=1, axis='index')
        dataset['r_max'] = np.max(data, axis=1).shift(periods=1, axis='index')
        # Dataset with proxies
        result = pd.concat([dataset, kendall_estimates], axis=1, join='inner')
        return result

    def bootstrap_moving_window_estimate(self, data, delta_t, T=500, reps=1000, ciw=99, proxy_type='pearson'):
        """Method for measuring the estimation uncertainty associated to the correlation coefficients when moving
        window estimates are used for approximating true correlations.
        :param data: dataset used for the task of bootstrap resampling
        :param T: length of test set
        :param delta_t: window length for moving window estimates of Pearson correlation coefficient
        :param reps: number of bootstrap samples
        :param ciw: confidence interval width
        :param proxy_type: type definition of proxy for estimates of true correlation (pearson, emw, kendall)
        :return: correlation estimates with associated estimation uncertainty."""
        assets_price = data.tail(T + delta_t - 1).iloc[:, :-1]
        assets_price.reset_index(drop=True, inplace=True)
        rho_true = data.tail(T).iloc[:, -1]; rho_true.reset_index(drop=True, inplace=True)
        rho_estimates = np.full(T, np.nan)
        sd_rho_estimates = np.full(T, np.nan)  # bootstrapped standard error of rho estimates
        lower_percentiles = np.full(T, np.nan)  # Initialisation array containing lower percentile values
        upper_percentiles = np.full(T, np.nan)  # Initialisation array containing upper percentile values
        p_low = (100 - ciw) / 2
        p_high = 100 - p_low

        for j, t in enumerate(range(delta_t, T + delta_t)):
            sampling_data = np.asarray(assets_price.iloc[t - delta_t:t, :])
            # Bootstrap resampling procedure:
            # draw sample of size delta_t by randomly extracting time units with uniform probability, with replacement.
            rho_bootstrapped = np.full(reps, np.nan)
            for rep in range(reps):
                indices = np.random.randint(low=0, high=sampling_data.shape[0], size=delta_t)
                sample = sampling_data[indices]
                if proxy_type is 'emw':
                    # Setup bootstrap procedure for weighted moving window estimates
                    w = self.ta.exponential_weights(delta_t, delta_t / 3)
                    weight_vec_raw = w[indices]
                    sum_w = np.sum(weight_vec_raw)
                    weight_vec_norm = [i / sum_w for i in weight_vec_raw]  # Re-normalize weights to one
                    rho_bootstrapped[rep] = \
                        self.ta.pearson_weighted_correlation_estimation(sample[:, 0], sample[:, 1], delta_t,
                                                                        weight_vec_norm)
                elif proxy_type is 'pearson':
                    rho_bootstrapped[rep] = pearsonr(sample[:, 0], sample[:, 1])[0]
                elif proxy_type is 'kendall':
                    rho_bootstrapped[rep] = kendalltau(sample[:, 0], sample[:, 1])[0]
                else:
                    print('Please, choose an option from the supported set of proxies for true correlations (Pearson '
                          'moving window, Pearson exponentially weighted moving window, Kendall moving window')
            lower, upper = np.nanpercentile(rho_bootstrapped, [p_low, p_high])
            lower_percentiles[j] = lower
            upper_percentiles[j] = upper
            rho_estimates[j] = np.nanmean(rho_bootstrapped)
            sd_rho_estimates[j] = np.nanstd(rho_bootstrapped)
        return rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates

    def bootstrap_learner_estimate(self,  data, T=500, reps=1000, ciw=99, model='knn', n_neighbors=5):
        """"Method for measuring the estimation uncertainty associated to the correlation coefficients when a learner
        model is used for approximating true correlations.
        :param data: dataset used for the task of bootstrap resampling
        :param T: length of test set
        :param reps: number of bootstrap samples
        :param ciw: confidence interval width
        :param model: learner model (e.g. nearest neighbour or random forest regressors)
        :param n_neighbors: number of multivariate neighbours
        :return: correlation estimates with associated estimation uncertainty."""
        rho_estimates = np.full(T, np.nan)
        sd_rho_estimates = np.full(T, np.nan)  # bootstrapped standard error of rho estimates
        lower_percentiles = np.full(T, np.nan)  # Initialisation array containing lower percentile values
        upper_percentiles = np.full(T, np.nan)  # Initialisation array containing upper percentile values
        p_low = (100 - ciw) / 2
        p_high = 100 - p_low
        data.drop(data.head(251).index, inplace=True)
        data.reset_index(drop=True, inplace=True)
        t_train_init = data.shape[0] - T  # 1000 for T = 500

        for j, t in enumerate(range(t_train_init, data.shape[0])):  # j = {0, 499}, t = {1000, 1499}
            sampling_data = np.asarray(data.iloc[:t, :])  # True rolling window is [j:t, :]
            x_test = np.asarray(data.iloc[t, 0:-1])  # This is in fact x_t+1
            y_test = np.asarray(data.iloc[t, -1])    # This is in fact y_t+1
            # Bootstrap resampling procedure:
            # draw sample of size train_set by randomly extracting time units with uniform probability, with replacement
            rho_bootstrapped = np.full(reps, np.nan)
            for rep in range(reps):
                indices = np.random.randint(low=0, high=t, size=t)
                sample = sampling_data[indices]  # Use sample to make a prediction with learner model
                # Separate data into feature and response components
                X = np.asarray(sample[:, 0:-1])  # feature matrix (vectorize data for speed up)
                y = np.asarray(sample[:, -1])    # response vector
                X_train = X[0:t, :]
                y_train = y[0:t]
                # Obtain estimation uncertainty in Pearson correlation estimation rho_t using bootstrap resampling:
                if model is 'knn':
                    knn = KNeighborsRegressor(n_neighbors=5)  # n_neighbors=len(X_train)
                    rho_bootstrapped[rep] = knn.fit(X_train, y_train).predict(x_test.reshape(1, -1))
                elif model is 'rf':
                    rf = RandomForestRegressor(n_jobs=1, n_estimators=10, max_features=1).fit(X_train, y_train)
                    rho_bootstrapped[rep] = rf.predict(x_test.reshape(1, -1))
                else:
                    print('Please, choose an option from the supported set of learner algorithms (nearest neighbour, '
                          'random forest)')
            lower, upper = np.nanpercentile(rho_bootstrapped, [p_low, p_high])
            lower_percentiles[j] = lower
            upper_percentiles[j] = upper
            rho_estimates[j] = np.nanmean(rho_bootstrapped)
            sd_rho_estimates[j] = np.nanstd(rho_bootstrapped)
        return rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates

    def mse_knn_sensitivity_analysis(self, proxy_type='pearson', output_type='true'):
        """Method for creation of a dataframe containing information on MSE decomposition as a function of different
        parameterizations for knn learner model.
        :param proxy_type: type of moving window estimator used as covariate.
        :param output_type: output variable true correlation or proxy.
        :return: dataframe."""
        rho_bias_squared = np.full(1001, np.nan)
        rho_var_vec = np.full(1001, np.nan)
        rho_mse_vec = np.full(1001, np.nan)
        # Load mse decomposition data
        mse_knn5 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/mse_knn5_%s_%s_cor.pkl'
                                     % (output_type, output_type, proxy_type, output_type))
        mse_knn10 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/mse_knn10_%s_%s_cor.pkl'
                                      % (output_type, output_type, proxy_type, output_type))
        mse_knn25 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/mse_knn25_%s_%s_cor.pkl'
                                      % (output_type, output_type, proxy_type, output_type))
        mse_knn50 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/mse_knn50_%s_%s_cor.pkl'
                                      % (output_type, output_type, proxy_type, output_type))
        mse_knn_100_to_1000 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/'
                            'mse_knn100_to_1000_%s_%s_cor.pkl' % (output_type, output_type, proxy_type, output_type))
        # Creation of dataframe
        rho_mse_vec[5], rho_bias_squared[5], rho_var_vec[5] = mse_knn5.iloc[10, :]
        rho_mse_vec[10], rho_bias_squared[10], rho_var_vec[10] = mse_knn10.iloc[10, :]
        rho_mse_vec[25], rho_bias_squared[25], rho_var_vec[25] = mse_knn25.iloc[10, :]
        rho_mse_vec[50], rho_bias_squared[50], rho_var_vec[50] = mse_knn50.iloc[10, :]
        rho_mse_vec[100], rho_bias_squared[100], rho_var_vec[100] = mse_knn_100_to_1000.iloc[1, :]
        rho_mse_vec[200], rho_bias_squared[200], rho_var_vec[200] = mse_knn_100_to_1000.iloc[2, :]
        rho_mse_vec[300], rho_bias_squared[300], rho_var_vec[300] = mse_knn_100_to_1000.iloc[3, :]
        rho_mse_vec[400], rho_bias_squared[400], rho_var_vec[400] = mse_knn_100_to_1000.iloc[4, :]
        rho_mse_vec[500], rho_bias_squared[500], rho_var_vec[500] = mse_knn_100_to_1000.iloc[5, :]
        rho_mse_vec[600], rho_bias_squared[600], rho_var_vec[600] = mse_knn_100_to_1000.iloc[6, :]
        rho_mse_vec[700], rho_bias_squared[700], rho_var_vec[700] = mse_knn_100_to_1000.iloc[7, :]
        rho_mse_vec[800], rho_bias_squared[800], rho_var_vec[800] = mse_knn_100_to_1000.iloc[8, :]
        rho_mse_vec[900], rho_bias_squared[900], rho_var_vec[900] = mse_knn_100_to_1000.iloc[9, :]
        rho_mse_vec[1000], rho_bias_squared[1000], rho_var_vec[1000] = mse_knn_100_to_1000.iloc[10, :]
        # Dataframe with information on MSE decomposition as a function of different learner parameterizations
        data_frame = pd.DataFrame({'bias_squared': rho_bias_squared, 'variance': rho_var_vec,
                                   'MSE': rho_mse_vec})
        return data_frame

    def mse_rf_sensitivity_analysis(self, proxy_type='pearson', output_type='true'):
        """Method for creation of a dataframe containing information on MSE decomposition as a function of different
        parameterizations for rf learner model.
        :param proxy_type: type of moving window estimator used as covariate.
        :param output_type: output variable true correlation or proxy.
        :return: dataframe."""
        rho_bias_squared = np.full(4, np.nan)
        rho_var_vec = np.full(4, np.nan)
        rho_mse_vec = np.full(4, np.nan)
        # Load mse decomposition data
        mse_rf300_1_to_3 = self.mm.load_data('bivariate_analysis/%s_cor/mse_results_%s_cor/'
                            'mse_rf300_1_to_3_%s_%s_cor.pkl' % (output_type, output_type, proxy_type, output_type))
        rho_mse_vec[1], rho_bias_squared[1], rho_var_vec[1] = mse_rf300_1_to_3.iloc[1, :]
        rho_mse_vec[2], rho_bias_squared[2], rho_var_vec[2] = mse_rf300_1_to_3.iloc[2, :]
        rho_mse_vec[3], rho_bias_squared[3], rho_var_vec[3] = mse_rf300_1_to_3.iloc[3, :]
        # Dataframe with information on MSE decomposition as a function of different learner parameterizations
        data_frame = pd.DataFrame({'bias_squared': rho_bias_squared, 'variance': rho_var_vec,
                                   'MSE': rho_mse_vec})
        return data_frame








































