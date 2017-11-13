import numpy as np
from math import sqrt, exp


class TechnicalAnalyzer2(object):
    """Technical analysis class. This class has the responsibility to engineer the feature space."""

    def __init__(self):
        """Initializer TechnicalAnalyzer object."""

    def _exponential_weights(self, dt, theta):
        """Private method for obtaining vector of weights where the functional form of the weights follow the
        exponential function. The definition of our weight function ensures the sum of weights is equal to one.
        As a recommendation from Pozzi et al. (2012): theta = dt / 3 for dt = {21, 251} [days]/
        :param dt: window length
        :param theta: weight's characteristic time, i.e. 1 / theta is the exponential decay factor
        :return: w: vector containing weights according to exponential function."""
        w = np.full(dt, np.nan)
        w0 = (1 - exp(-1 / theta)) / (1 - exp(-dt / theta))
        for t in range(1, dt + 1):
            w[t - 1] = w0 * exp((t - dt) / theta)
        return w

    def pearson_weighted_correlation_estimation(self, y_i, y_j, dt):
        """Method for estimation of Pearson weighted time-varying correlation coefficient between two assets. In this work,
         the functional form for the weights follows the exponential function.
        :param y_i: vector containing price path of asset i
        :param y_j: vector containing price path of asset j
        :param dt: window length, i.e. window containing dt consecutive observations
        :return: rho_weighted: vector containing pearson weighted correlation estimates."""
        rho_weighted = np.full(len(y_i),
                               np.nan)  # Initialise empty vector that will contain Pearson weighted correlations
        theta = dt / 3.0
        w_vec = self._exponential_weights(dt, theta)
        for t in range(dt - 1, len(y_i)):
            # Compute weighted means over dt consecutive observations
            y_i_weighted = sum(map(lambda (w, y): w * y, zip(w_vec, y_i[t - dt + 1:t])))  # t is current time
            y_j_weighted = sum(map(lambda (w, y): w * y, zip(w_vec, y_j[t - dt + 1:t])))
            # Compute weighted standard deviations over dt consecutive observations
            sd_i_weighted = sqrt(sum(map(lambda (w, y): w * (y - y_i_weighted) ** 2, zip(w_vec, y_i[t - dt + 1:t]))))
            sd_j_weighted = sqrt(sum(map(lambda (w, y): w * (y - y_j_weighted) ** 2, zip(w_vec, y_j[t - dt + 1:t]))))
            # Compute Pearson weighted correlation coefficient
            sd_ij_weighted = sum(map(lambda (w, x, y): w * (x - y_i_weighted) * (y - y_j_weighted),
                                     zip(w_vec, y_i[t - dt + 1:t], y_j[t - dt + 1:t])))
            rho_weighted[t] = sd_ij_weighted / (sd_i_weighted * sd_j_weighted)
        return rho_weighted

    def daily_log_return(self, data):
        """Method for computing daily asset return.
        :param data:
        :return:
        """
        data['log return'] = np.log(data['Close']).diff()
        return data

    def get_data(self, data):
        """Method to return the dataframe with return and response variable as the for last and last
         column, respectively. This reordering gives a good structure for subsequent processing."""
        cols = list(data.columns.values)
        cols.pop(cols.index('log return'))
        cols.pop(cols.index('response'))
        data = data[cols+['log return', 'response']]
        return data
