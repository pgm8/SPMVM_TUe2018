import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats.stats import pearsonr

from TechnicalAnalyzer import TechnicalAnalyzer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)


class PreProcessor(object):
    """Preprocessor class. This class has the responsibility to preprocess the data. More specifically, the class
    has the task of simulating random correlated asset paths in the bivariate case. Additionally. the class has the
    responsibility for estimating the uncertainty in the response variable through a bootstrap resampling procedure."""

    def __init__(self):
        """Initializer PreProcessor object."""
        self.ta = TechnicalAnalyzer()

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

    def generate_bivariate_dataset(self, ta, simulated_data_process, dt, weighted=False):
        """Method for generating a dataset with proxies (exponentially weighted) moving window correlation estimates
        for feature set and true correlation as the response variables.
        :param ta: technical analyzer object
        :param simulated_data_process: bivariate asset process with predefined correlation dynamics.
        :param dt: window length
        :param weighted: boolean whether to use weighted mw estimates as proxies
        :return: dataset (datastructure: dataframe)."""
        if weighted:
            emw_estimates = ta.pearson_weighted_correlation_estimation(simulated_data_process[0],
                                                                       simulated_data_process[1], dt)
            emw_estimates = pd.Series(emw_estimates)
            # Feature set consists of lagged asset price and mw correlation estimate, e.g. x_t = EMW_t-1
            dataset = simulated_data_process.iloc[:, :2].shift(periods=1, axis='index')  # Dataframe
            dataset['EMW_t-1'] = emw_estimates.shift(periods=1, axis='index')
        else:
            mw_estimates = simulated_data_process[0].rolling(window=dt).corr(other=simulated_data_process[1])
            # Feature set consists of lagged asset price and mw correlation estimate, e.g. x_t = EMW_t-1
            dataset = simulated_data_process.iloc[:, :2].shift(periods=1, axis='index')  # Dataframe
            dataset['MW_t-1'] = mw_estimates.shift(periods=1, axis='index')
        # Dataset with true correlations as response variable
        dataset['rho_true'] = simulated_data_process['rho']
        return dataset

    def bootstrap_moving_window_estimate(self, data, delta_t, T=500, reps=1000, ciw=99, proxy_type='mw'):
        """Method for measuring the estimation uncertainty associated to the correlation coefficients when moving
        window estimates are used for approximating true correlations.
        :param data: dataset used for the task of bootstrap resampling
        :param T: length of test set
        :param delta_t: window length for moving window estimates of Pearson correlation coefficient
        :param reps: number of bootstrap samples
        :param ciw: confidence interval width
        :param proxy_type: type definition of proxy for estimates of true correlation
        :return: correlation estimates with associated confidence intervals."""
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
                elif 'mw':
                    rho_bootstrapped[rep] = pearsonr(sample[:, 0], sample[:, 1])[0]
                else:
                    print('Please, choose an option from the supported set proxies for true correlations (moving'
                          'window, exponentially weighted moving window')
            lower, upper = np.percentile(rho_bootstrapped, [p_low, p_high])
            lower_percentiles[j] = lower
            upper_percentiles[j] = upper
            rho_estimates[j] = np.mean(rho_bootstrapped)
            sd_rho_estimates[j] = np.std(rho_bootstrapped)
        return rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates

    def bootstrap_learner_estimate(self,  data, T=500, reps=1000, ciw=99, model='knn'):
        """"Method for measuring the estimation uncertainty associated to the correlation coefficients when a learner
        model is used for approximating true correlations.
        :param data: dataset used for the task of bootstrap resampling
        :param T: length of test set
        :param reps: number of bootstrap samples
        :param ciw: confidence interval width
        :param model: learner model (e.g. nearest neighbour or random forest regressors)
        :return: correlation estimates with associated confidence intervals."""
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
                    knn = KNeighborsRegressor(n_neighbors=5)
                    rho_bootstrapped[rep] = knn.fit(X_train, y_train).predict(x_test.reshape(1, -1))
                elif model is 'rf':
                    rf = RandomForestRegressor(n_estimators=10, max_features='sqrt')
                    rho_bootstrapped[rep] = rf.fit(X_train, y_train).predict(x_test.reshape(1, -1))
                else:
                    print('Please, choose an option from the supported set of learner algorithms (nearest neighbour, '
                          'random forest)')
            lower, upper = np.percentile(rho_bootstrapped, [p_low, p_high])
            lower_percentiles[j] = lower
            upper_percentiles[j] = upper
            rho_estimates[j] = np.mean(rho_bootstrapped)
            sd_rho_estimates[j] = np.std(rho_bootstrapped)
        return rho_estimates, lower_percentiles, upper_percentiles, sd_rho_estimates



































