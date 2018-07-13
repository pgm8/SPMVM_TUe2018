

import numpy as np
import pandas as pd
import time
from scipy.linalg import sqrtm
from scipy.stats._continuous_distns import chi2
from PreProcessor import PreProcessor
from ModuleManager import ModuleManager


#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)
np.set_printoptions(precision=4, linewidth=220)

def main():
    preprocesser = PreProcessor()
    mm = ModuleManager()

    def generate_random_points_on_hyperellipsoid(vol_data, cor_data,
                                                 alpha_vec=np.array([0.9, 0.95, 0.975, 0.99]),
                                                 n_sample=int(1e4), dim=30):
        header = alpha_vec
        result = pd.DataFrame(columns=header)
        for i in range(vol_data.shape[0]):
            start_time = time.time()
            var_estimates = []
            vol_mat = np.diag(vol_data.iloc[i, :])
            cor_mat = preprocesser.construct_correlation_matrix(corr_vec=cor_data.iloc[i, :], n=dim)
            H = preprocesser.construct_covariance_matrix(vol_matrix=vol_mat, corr_matrix=cor_mat)
            r = np.random.randn(H.shape[0], n_sample)
            # u contains random points on the unit hypersphere
            u = r / np.linalg.norm(r, axis=0)
            for alpha in alpha_vec:
                y = chi2.ppf(q=alpha, df=dim)         #-2*np.log(1-alpha)  #
                # Transform points on the unit hypersphere to the hyperellipsoid
                xrandom = sqrtm(H).dot(np.sqrt(y) * u)
                # Compute the lowest (equally) weighted average of random points on the hyperellipsoid.
                # This is the maximum loss with alpha percent probability, i.e. Value-at-Risk
                xrandom_min = np.min(np.array([np.mean(x) for x in xrandom.T]))
                var_estimates.append(xrandom_min)
            result = pd.merge(result, pd.DataFrame(np.asarray(var_estimates).reshape(1, -1),
                                                   columns=header), how='outer')
            print((i, time.time() - start_time))
        return result



    ##################################################################################################################
    ###                                      Multivariate Quantile Computation                                     ###
    ##################################################################################################################
    dim = 30
    vol_data = mm.load_data('multivariate_analysis/volatilities_garch_norm_DJI30_1994_1995.pkl')
    #cor_data = mm.load_data('multivariate_analysis/cor_DCC_mvnorm_DJI30_1994_1995.pkl')
    cor_data = mm.load_data('multivariate_analysis/pearson/pearson_cor_estimates/cor_rf100_pearson_10_DJI30_1994_1995.pkl')

    result = generate_random_points_on_hyperellipsoid(vol_data=vol_data, cor_data=cor_data)
    print(result)
    #mm.save_data('multivariate_analysis/VaR/var_dcc_mvnorm_1994_1995_nsample_1e6.pkl', result)
    #mm.transform_pickle_to_csv('multivariate_analysis/VaR/var_dcc_mvnorm_1994_1995_nsample_1e6.pkl')
    #mm.save_data('multivariate_analysis/VaR/var_knn_idw_garch_1994_1995_nsample_1e5_ch2_alpha.pkl', result)
    #mm.transform_pickle_to_csv('multivariate_analysis/VaR/var_knn_idw_garch_1994_1995_nsample_1e5_ch2_alpha.pkl')





###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()