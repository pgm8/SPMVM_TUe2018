

import numpy as np
from PreProcessor import PreProcessor
from ModuleManager import ModuleManager

def main():
    preprocesser = PreProcessor()
    mm = ModuleManager()

    ##################################################################################################################
    ###                                      Multivariate Quantile Computation                                     ###
    ##################################################################################################################
    cor_data = mm.load_data('multivariate_analysis/cor_DCC_mvnorm_DJI30_1994_1995.pkl')
    print(cor_data)



















###############################
####         MAIN          ####
###############################
if __name__ == '__main__':
    main()