import numpy as np



class TechnicalAnalyzer(object):
    """Technical analysis class. This class has the responsibility to engineer the feature space."""

    def __init__(self):
        """Initializer TechnicalAnalyzer object."""

    def kendall_correlation_estimation(self):
        """Method for estimation of pair wise Kendall time-varying correlation coefficients.

        :return:
        """



    @staticmethod
    def exponential_weights(dt, theta):
        """Static method for obtaining vector of weights where the functional form of the weights follow the
        exponential function. The definition of our weight function ensures the sum of weights is equal to one.
        As a recommendation from Pozzi et al. (2012): theta = dt / 3 for dt = {21, 251} [days]
        :param dt: window length
        :param theta: weight's characteristic time, i.e. 1 / theta is the exponential decay factor
        :return: w: vector containing weights according to exponential function."""
        w = np.full(dt, np.nan)
        w0 = (1 - np.exp(-1 / theta)) / (1 - np.exp(-dt / theta))
        for t in range(1, dt + 1):
            w[t - 1] = w0 * np.exp((t - dt) / theta)
        return w

    def pearson_weighted_correlation_estimation(self, y_i, y_j, dt, weight_vec=None):
        """Method for estimation of Pearson weighted time-varying correlation coefficient between two assets.
        In this work, the functional form for the weights follows the exponential function.
        :param y_i: vector containing price path of asset i
        :param y_j: vector containing price path of asset j
        :param dt: window length, i.e. window containing dt consecutive observations
        :param weight_vec: vector containing weights used for smoothing
        :return: rho_weighted: vector containing pearson weighted correlation estimates."""
        if weight_vec is None:
            theta = dt / 3
            w_vec = self.exponential_weights(dt, theta=theta)
            # Body for non-bootstrap context
            rho_weighted = np.full(len(y_i), np.nan)  # Initialise empty vector for Pearson weighted correlations
            for t in range(dt - 1, len(y_i)):
                # Compute weighted means over dt consecutive observations
                y_i_weighted = sum(map(lambda w_y: w_y[0] * w_y[1], zip(w_vec, y_i[t - dt + 1:t])))  # t is current time
                y_j_weighted = sum(map(lambda w_y: w_y[0] * w_y[1], zip(w_vec, y_j[t - dt + 1:t])))
                # Compute weighted standard deviations over dt consecutive observations
                sd_i_weighted = np.sqrt(sum(map(lambda w_y: w_y[0] * np.power((w_y[1] - y_i_weighted), 2),
                                                zip(w_vec, y_i[t - dt + 1:t]))))
                sd_j_weighted = np.sqrt(sum(map(lambda w_y: w_y[0] * np.power((w_y[1] - y_j_weighted), 2),
                                                zip(w_vec, y_j[t - dt + 1:t]))))
                # Compute Pearson weighted correlation coefficient
                sd_ij_weighted = sum(map(lambda w_x_y: w_x_y[0] * (w_x_y[1] - y_i_weighted) * (w_x_y[2] - y_j_weighted),
                                        zip(w_vec, y_i[t - dt + 1:t], y_j[t - dt + 1:t])))
                rho_weighted[t] = sd_ij_weighted / (sd_i_weighted * sd_j_weighted)
        else:
            w_vec = weight_vec
            # Body for bootstrap context
            # Compute weighted means over dt consecutive observations
            y_i_weighted = sum(map(lambda w_y: w_y[0] * w_y[1], zip(w_vec, y_i)))
            y_j_weighted = sum(map(lambda w_y: w_y[0] * w_y[1], zip(w_vec, y_j)))
            # Compute weighted standard deviations over dt consecutive observations
            sd_i_weighted = np.sqrt(sum(map(lambda w_y: w_y[0] * np.power((w_y[1] - y_i_weighted), 2),
                                            zip(w_vec, y_i))))
            sd_j_weighted = np.sqrt(sum(map(lambda w_y: w_y[0] * np.power((w_y[1] - y_j_weighted), 2),
                                            zip(w_vec, y_j))))
            # Compute Pearson weighted correlation coefficient
            sd_ij_weighted = sum(map(lambda w_x_y: w_x_y[0] * (w_x_y[1] - y_i_weighted) * (w_x_y[2] - y_j_weighted),
                                     zip(w_vec, y_i, y_j)))
            rho_weighted = sd_ij_weighted / (sd_i_weighted * sd_j_weighted)
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
