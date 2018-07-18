#################################################################################################################
rm(list=ls())  # remove all variables in R
install.packages("rugarch")
install.packages("rmgarch")
install.packages("rootSolve")
install.packages("WeightedPortTest")
install.packages("zoo")
library(tseries)
library(rugarch)
library(rmgarch)
library(parallel)
library(zoo)
library(fGarch)
setwd("~/Documents/Python/PycharmProjects/ml_tue2017/source/main/resources/Data/multivariate_analysis")
source("R/fun_VaR_backtesting.R")
set.seed(42)  # 42:The answer to life, the universe and everything.


####################################################################################################
######                               Tranquil Market Conditions                              #######
####################################################################################################
# Data sample import 
data <- read.csv("DJI30_returns_1987_2001.csv", row.names=1, header=T)[1:224] # Data sample: 17/3/1987-29/12/1995 intial in sample period: 1720 observations
data$Date <- NULL

T <- 504  # Out-of-sample test sample: 3/1/1994-29/12/1995
N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
# Same first stage conditional mean filtration (unconditional mean)
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1)) 


## Dynamic Conditional Correlation model with various error distributions
dccGarch_mvnorm_tranquil <- dcc_garch_modeling(data=a_t, t=t, distribution.model="sstd", distribution="mvnorm")
dccGarch_gjr_tranquil <-  dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="mvnorm") 

# Write matrices containing time-varying volatilies and correlations to csv file
# Add column names to file containing conditional correlations
col_names <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv", row.names=1)
colnames(dccGarch_mvnorm_tranquil$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
colnames(dccGarch_gjr_tranquil$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
write.csv(uniGarch_gjr_tranquil$D_t_file, file="volatilities_norm_DJI30_1994_1995.csv")
write.csv(dccGarch_mvnorm_tranquil$R_t_file, file="cor_DCC_mvnorm_DJI30_1994_1995.csv")
write.csv(dccGarch_gjr_tranquil$D_t_file, file="volatilities_gjr_DJI30_1994_1995.csv")
write.csv(dccGarch_gjr_tranquil$R_t_file, file="cor_DCC_gjr_DJI30_1994_1995.csv")

####    Value-at-Risk Estimation   ###   
alpha <- c(0.99, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01)
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)

## Load conditional correlations and volatilities data
vol_data_tranquil_garch<- read.csv(file="volatilities_garch_norm_DJI30_1994_1995.csv", row.names=1)
vol_data_tranquil_gjr<- read.csv(file="volatilities_gjr_sstd_DJI30_1994_1995.csv", row.names=1)
cor_DCC_garch_tranquil <- read.csv(file="cor_DCC_garch_mvnorm_DJI30_1994_1995.csv", row.names=1)
cor_DCC_gjr_tranquil <- read.csv(file="cor_DCC_gjr_mvnorm_DJI30_1994_1995.csv", row.names=1)
# Nearest neighbor
cor_KNN5_pearson_tranquil <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv", row.names=1)
cor_KNN5_kendall_tranquil <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_1994_1995.csv", row.names=1)
cor_KNN_idw_pearson_tranquil <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_1994_1995.csv", row.names=1)
cor_KNN_idw_kendall_tranquil <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_1994_1995.csv", row.names=1)
# Random forest
cor_RF10_pearson_tranquil <- read.csv(file="pearson/pearson_cor_estimates/cor_rf10_pearson_10_DJI30_1994_1995.csv", row.names=1)
cor_RF10_kendall_tranquil <- read.csv(file="kendall/kendall_cor_estimates/cor_rf10_kendall_10_DJI30_1994_1995.csv", row.names=1)
cor_RF100_pearson_tranquil <- read.csv(file="pearson/pearson_cor_estimates/cor_rf100_pearson_10_DJI30_1994_1995.csv", row.names=1)
cor_RF100_kendall_tranquil <- read.csv(file="kendall/kendall_cor_estimates/cor_rf100_kendall_10_DJI30_1994_1995.csv", row.names=1)


## Compute portfolio Value-at-Risk
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns for a long position in an equally weighted portfolio
# Garch-Normal 
VaR_DCC_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_DCC_garch_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_DCC_garch_mvnorm_1994_1995.csv", T, w)
VaR_KNN5_pearson_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_KNN5_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN5_pearson_garch_1994_1995.csv", T, w)
VaR_KNN5_kendall_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_KNN5_kendall_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN5_kendall_garch_1994_1995.csv", T, w)
VaR_KNN_idw_pearson_tranquil_garch<- portfolio_VaR(vol_data_tranquil_garch, cor_KNN_idw_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN_idw_pearson_garch_1994_1995.csv", T, w)
VaR_KNN_idw_kendall_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_KNN_idw_kendall_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN_idw_kendall_garch_1994_1995.csv", T, w)
VaR_RF10_pearson_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_RF10_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF10_pearson_garch_1994_1995.csv", T, w)
VaR_RF10_kendall_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_RF10_kendall_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF10_kendall_garch_1994_1995.csv", T, w)
VaR_RF100_pearson_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_RF100_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF100_pearson_garch_1994_1995.csv", T, w)
VaR_RF100_kendall_tranquil_garch <- portfolio_VaR(vol_data_tranquil_garch, cor_RF100_kendall_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF100_kendall_garch_1994_1995.csv",  T, w)
# GJR-SSTD (GJR-Norm for DCC)
VaR_DCC_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_garch, cor_DCC_gjr_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_DCC_gjr_mvnorm_1994_1995.csv", T, w) 
VaR_KNN5_pearson_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_KNN5_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN5_pearson_gjr_1994_1995.csv", T, w)
VaR_KNN5_kendall_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_KNN5_kendall_tranquil,mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN5_kendall_gjr_1994_1995.csv", T, w)
VaR_KNN_idw_pearson_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_KNN_idw_pearson_tranquil,mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN_idw_pearson_gjr_1994_1995.csv", T, w)
VaR_KNN_idw_kendall_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_KNN_idw_kendall_tranquil,mu_portfolio_loss, alpha, file="VaR/tranquil/var_KNN_idw_kendall_gjr_1994_1995.csv", T, w)
VaR_RF10_pearson_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_RF10_pearson_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF10_pearson_gjr_1994_1995.csv", T, w)
VaR_RF10_kendall_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_RF10_kendall_tranquil, mu_portfolio_loss, alpha, file="VaR/tranquil/var_RF10_kendall_gjr_1994_1995.csv", T, w)
VaR_RF100_pearson_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_RF100_pearson_tranquil, mu_portfolio_loss, alpha, T, file="VaR/tranquil/var_RF100_pearson_gjr_1994_1995.csv", w)
VaR_RF100_kendall_tranquil_gjr <- portfolio_VaR(vol_data_tranquil_gjr, cor_RF100_kendall_tranquil, mu_portfolio_loss, alpha, T, file="VaR/tranquil/var_RF100_kendall_gjr_1994_1995.csv", w)

## Backtest portfolio Value-at-Risk
# Garch-Normal 
backtest_DCC_tranquil_garch <- uc_ind_test(VaR_est=VaR_DCC_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_DCC_garch_mvnorm_1994_1995.csv")
backtest_KNN5_pearson_tranquil_garch <- uc_ind_test(VaR_est=VaR_KNN5_pearson_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_KNN5_pearson_garch_1994_1995.csv")
backtest_KNN5_kendall_tranquil_garch <- uc_ind_test(VaR_est=VaR_KNN5_kendall_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_KNN5_kendall_garch_1994_1995.csv") 
backtest_KNN_idw_pearson_tranquil_garch <- uc_ind_test(VaR_est=VaR_KNN_idw_pearson_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_KNN_idw_pearson_garch_1994_1995.csv")
backtest_KNN_idw_kendall_tranquil_garch <- uc_ind_test(VaR_est=VaR_KNN_idw_kendall_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_KNN_idw_kendall_garch_1994_1995.csv") 
backtest_RF10_pearson_tranquil_garch <- uc_ind_test(VaR_est=VaR_RF10_pearson_tranquil_garch, cl=alpha,  file="backtest/tranquil/backtest_RF10_pearson_garch_1994_1995.csv")
backtest_RF10_kendall_tranquil_garch <- uc_ind_test(VaR_est=VaR_RF10_kendall_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_RF10_kendall_garch_1994_1995.csv") 
backtest_RF100_pearson_tranquil_garch <- uc_ind_test(VaR_est=VaR_RF100_pearson_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_RF100_pearson_garch_1994_1995.csv")
backtest_RF100_kendall_tranquil_garch <- uc_ind_test(VaR_est=VaR_RF100_kendall_tranquil_garch, cl=alpha, file="backtest/tranquil/backtest_RF100_kendall_garch_1994_1995.csv") 
# GJR-SSTD (GJR-Norm for DCC)
backtest_DCC_gjr_tranquil_mvt <- uc_ind_test(VaR_est=VaR_DCC_tranquil_gjr, cl=alpha,  file="backtest/tranquil/backtest_DCC_gjr_mvnorm_1994_1995.csv") 
backtest_KNN5_pearson_tranquil_gjr <- uc_ind_test(VaR_est=VaR_KNN5_pearson_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_KNN5_pearson_gjr_1994_1995.csv") 
backtest_KNN5_kendall_tranquil_gjr <- uc_ind_test(VaR_est=VaR_KNN5_kendall_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_KNN5_kendall_gjr_1994_1995.csv") 
backtest_KNN_idw_pearson_tranquil_gjr <- uc_ind_test(VaR_est=VaR_KNN_idw_pearson_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_KNN_idw_pearson_gjr_1994_1995.csv") 
backtest_KNN_idw_kendall_tranquil_gjr <- uc_ind_test(VaR_est=VaR_KNN_idw_kendall_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_KNN_idw_kendall_gjr_1994_1995.csv") 
backtest_RF10_pearson_tranquil_gjr <- uc_ind_test(VaR_est=VaR_RF10_pearson_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_RF10_pearson_gjr_1994_1995.csv") 
backtest_RF10_kendall_tranquil_gjr <- uc_ind_test(VaR_est=VaR_RF10_kendall_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_RF10_kendall_gjr_1994_1995.csv") 
backtest_RF100_pearson_tranquil_gjr <- uc_ind_test(VaR_est=VaR_RF100_pearson_tranquil_gjr, cl=alpha, file="backtest/tranquil/backtest_RF100_pearson_gjr_1994_1995.csv") 
backtest_RF100_kendall_tranquil_gjr <- uc_ind_test(VaR_est=VaR_RF100_kendall_tranquil_gjr, cl=alpha,  file="backtest/tranquil/backtest_RF100_kendall_gjr_1994_1995.csv")

# Non-rejection regions tranwuil market conditions
for (a in alpha){
  print(sprintf("CI %f: [%i,%i]",a, regions_uc_test(T=T, alpha=a)$lb, regions_uc_test(T=T, alpha=a)$ub))
}






####################################################################################################
######                               Volatile Market Conditions                              #######
####################################################################################################
data <- read.csv("DJI30_returns_1987_2001.csv", row.names=1, header=T) # Data sample: 17/3/1987-31/12/2001 
data$Date <- NULL

T <- 500  # Out-of-sample test sample: 3/1/2000-31/12/2001
N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1)) 

## Dynamic Conditional Correlation model with various error distributions
dccGarch_mvnorm_vol <- dcc_garch_modeling(data=a_t, t=t, distribution.model="sstd", distribution="mvnorm")
dccGarch_gjr_vol <-  dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="gjr") 
# Write matrices containing time-varying volatilies and correlations to csv file
# Add column names to file containing conditional correlations
col_names <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv", row.names=1)
colnames(dccGarch_mvnorm_vol$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
colnames(dccGarch_gjr_vol$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
write.csv(dccGarch_mvnorm_vol$D_t_file, file="volatilities_sstd_DJI30_2000_2001.csv")
write.csv(dccGarch_mvnorm_vol$R_t_file, file="cor_DCC_mvnorm_DJI30_2000_2001.csv")
write.csv(dccGarch_gjr_vol$D_t_file, file="volatilities_norm_gjr_DJI30_2000_2001.csv")
write.csv(dccGarch_gjr_vol$R_t_file, file="cor_DCC_norm_gjr_DJI30_2000_2001.csv")

####    Value-at-Risk Estimation   ###   
alpha <- c(0.99, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01)
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)

## Load conditional correlations and volatilities data
vol_data_volatile_garch<- read.csv(file="volatilities_garch_norm_DJI30_2000_2001.csv", row.names=1)
vol_data_volatile_gjr<- read.csv(file="volatilities_gjr_sstd_DJI30_2000_2001.csv", row.names=1)
cor_DCC_garch_volatile <- read.csv(file="cor_DCC_garch_mvnorm_DJI30_2000_2001.csv", row.names=1)
cor_DCCgarch_vol_gjr <- read.csv(file="cor_DCC_norm_gjr_DJI30_2000_2001.csv", row.names=1)
# Nearest neighbor
cor_KNN5_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN5_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN_idw_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN_idw_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_2000_2001.csv", row.names=1)

# Alt KNN_pearson_garch_gjr
cor_KNN300_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_knn300_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN500_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_knn500_pearson_10_DJI30_2000_2001.csv", row.names=1)

# Random forest
cor_RF10_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_rf10_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_RF10_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_rf10_kendall_10_DJI30_2000_2001.csv", row.names=1)
cor_RF100_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_rf100_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_RF100_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_rf100_kendall_10_DJI30_2000_2001.csv", row.names=1)


## Compute portfolio Value-at-Risk
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns for a long position in an equally weighted portfolio
# Garch-Normal 
VaR_DCC_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_DCC_garch_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_DCC_garch_mvnorm_2000_2001.csv", T, w)
VaR_KNN5_pearson_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_KNN5_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN5_pearson_garch_2000_2001.csv", T, w)
VaR_KNN5_kendall_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_KNN5_kendall_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN5_kendall_garch_2000_2001.csv", T, w)
VaR_KNN_idw_pearson_volatile_garch<- portfolio_VaR(vol_data_volatile_garch, cor_KNN_idw_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN_idw_pearson_garch_2000_2001.csv", T, w)
VaR_KNN_idw_kendall_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_KNN_idw_kendall_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN_idw_kendall_garch_2000_2001.csv", T, w)
# Alt VaR KNN_pearson_garch
VaR_KNN300_pearson_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_KNN300_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN300_pearson_garch_2000_2001.csv", T, w)
VaR_KNN500_pearson_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_KNN500_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN500_pearson_garch_2000_2001.csv", T, w)

VaR_RF10_pearson_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_RF10_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF10_pearson_garch_2000_2001.csv", T, w)
VaR_RF10_kendall_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_RF10_kendall_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF10_kendall_garch_2000_2001.csv", T, w)
VaR_RF100_pearson_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_RF100_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF100_pearson_garch_2000_2001.csv", T, w)
VaR_RF100_kendall_volatile_garch <- portfolio_VaR(vol_data_volatile_garch, cor_RF100_kendall_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF100_kendall_garch_2000_2001.csv",  T, w)
# GJR-SSTD (GJR-Norm for DCC)
VaR_DCC_volatile_gjr <- portfolio_VaR(vol_data_volatile_garch, cor_DCC_gjr_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_DCC_gjr_mvnorm_2000_2001.csv", T, w) 
VaR_KNN5_pearson_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN5_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN5_pearson_gjr_2000_2001.csv", T, w)
VaR_KNN5_kendall_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN5_kendall_volatile,mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN5_kendall_gjr_2000_2001.csv", T, w)
VaR_KNN_idw_pearson_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN_idw_pearson_volatile,mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN_idw_pearson_gjr_2000_2001.csv", T, w)
VaR_KNN_idw_kendall_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN_idw_kendall_volatile,mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN_idw_kendall_gjr_2000_2001.csv", T, w)
# Alt VaR KNN_pearson_gjr
VaR_KNN300_pearson_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN300_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN300_pearson_gjr_2000_2001.csv", T, w)
VaR_KNN500_kendall_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_KNN500_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_KNN500_pearson_gjr_2000_2001.csv", T, w)


VaR_RF10_pearson_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_RF10_pearson_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF10_pearson_gjr_2000_2001.csv", T, w)
VaR_RF10_kendall_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_RF10_kendall_volatile, mu_portfolio_loss, alpha, file="VaR/volatile/var_RF10_kendall_gjr_2000_2001.csv", T, w)
VaR_RF100_pearson_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_RF100_pearson_volatile, mu_portfolio_loss, alpha, T, file="VaR/volatile/var_RF100_pearson_gjr_2000_2001.csv", w)
VaR_RF100_kendall_volatile_gjr <- portfolio_VaR(vol_data_volatile_gjr, cor_RF100_kendall_volatile, mu_portfolio_loss, alpha, T, file="VaR/volatile/var_RF100_kendall_gjr_2000_2001.csv", w)

## Backtest portfolio Value-at-Risk
# Garch-Normal 
backtest_DCC_volatile_garch <- uc_ind_test(VaR_est=VaR_DCC_volatile_garch, cl=alpha, file="backtest/volatile/backtest_DCC_garch_mvnorm_2000_2001.csv")
backtest_KNN5_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN5_pearson_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN5_pearson_garch_2000_2001.csv")
backtest_KNN5_kendall_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN5_kendall_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN5_kendall_garch_2000_2001.csv") 
backtest_KNN_idw_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN_idw_pearson_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN_idw_pearson_garch_2000_2001.csv")
backtest_KNN_idw_kendall_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN_idw_kendall_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN_idw_kendall_garch_2000_2001.csv") 
# Alt VaR KNN_pearson_garch
backtest_KNN300_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN300_pearson_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN300_pearson_garch_2000_2001.csv")
backtest_KNN500_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_KNN500_pearson_volatile_garch, cl=alpha, file="backtest/volatile/backtest_KNN500_pearson_garch_2000_2001.csv")

backtest_RF10_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_RF10_pearson_volatile_garch, cl=alpha,  file="backtest/volatile/backtest_RF10_pearson_garch_2000_2001.csv")
backtest_RF10_kendall_volatile_garch <- uc_ind_test(VaR_est=VaR_RF10_kendall_volatile_garch, cl=alpha, file="backtest/volatile/backtest_RF10_kendall_garch_2000_2001.csv") 
backtest_RF100_pearson_volatile_garch <- uc_ind_test(VaR_est=VaR_RF100_pearson_volatile_garch, cl=alpha, file="backtest/volatile/backtest_RF100_pearson_garch_2000_2001.csv")
backtest_RF100_kendall_volatile_garch <- uc_ind_test(VaR_est=VaR_RF100_kendall_volatile_garch, cl=alpha, file="backtest/volatile/backtest_RF100_kendall_garch_2000_2001.csv") 
# GJR-SSTD (GJR-Norm for DCC)
backtest_DCC_gjr_volatile_mvt <- uc_ind_test(VaR_est=VaR_DCC_volatile_gjr, cl=alpha,  file="backtest/volatile/backtest_DCC_gjr_mvnorm_2000_2001.csv") 
backtest_KNN5_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN5_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN5_pearson_gjr_2000_2001.csv") 
backtest_KNN5_kendall_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN5_kendall_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN5_kendall_gjr_2000_2001.csv") 
backtest_KNN_idw_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN_idw_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN_idw_pearson_gjr_2000_2001.csv") 
backtest_KNN_idw_kendall_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN_idw_kendall_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN_idw_kendall_gjr_2000_2001.csv") 
# Alt VaR KNN_pearson_gjr
backtest_KNN300_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN300_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN300_pearson_gjr_2000_2001.csv") 
backtest_KNN500_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_KNN500_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_KNN500_pearson_gjr_2000_2001.csv") 

backtest_RF10_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_RF10_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_RF10_pearson_gjr_2000_2001.csv") 
backtest_RF10_kendall_volatile_gjr <- uc_ind_test(VaR_est=VaR_RF10_kendall_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_RF10_kendall_gjr_2000_2001.csv") 
backtest_RF100_pearson_volatile_gjr <- uc_ind_test(VaR_est=VaR_RF100_pearson_volatile_gjr, cl=alpha, file="backtest/volatile/backtest_RF100_pearson_gjr_2000_2001.csv") 
backtest_RF100_kendall_volatile_gjr <- uc_ind_test(VaR_est=VaR_RF100_kendall_volatile_gjr, cl=alpha,  file="backtest/volatile/backtest_RF100_kendall_gjr_2000_2001.csv")

# Non-rejection regions volatile market conditions
for (a in alpha){
  print(sprintf("CI %f: [%i,%i]",a, regions_uc_test(T=T, alpha=a)$lb, regions_uc_test(T=T, alpha=a)$ub))
}






####################################################################################################
######                        Conditional Variance Model Specification                       #######
####################################################################################################


####################################################################################################
# Ljung-Box tests for autocorrelation up to the tenth lag squared log-differences
Box.test(a_t[,1], lag = 10, type='Ljung-Box')
LB_test <- 0
for (i in 1:30) {
  result <- Box.test(a_t_dist[,i], lag = 40, type='Ljung-Box')$p.value
  if (result < 0.05) {
    LB_test <- LB_test+1
  }
}

####################################################################################################
# Results fitting GJR(1,1) with skewed student t marginal distributions for conditional variances
library(WeightedPortTest)
N = 30
#a_t_dist <- head(data, -504) # Tranquil market conditions
a_t_dist <- head(data,-500) # Volatile market conditions
a_t_dist$Date <- NULL
a_t_dist <- a_t_dist - rep(colMeans(a_t_dist), rep.int(nrow(a_t_dist), ncol(a_t_dist))) 

# Consider gjr-GARCH model
cl = makePSOCKcluster(10)
univ_garch_spec1 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1, 1)), mean.model=list(armaOrder=c(3,1), include.mean=FALSE), 
                              distribution.model="norm")
univ_garch_spec2 <- ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1, 1)), mean.model=list(armaOrder=c(3,1), include.mean=FALSE), 
                               distribution.model="sstd")
multi_univ_garch_spec1 <- multispec(replicate(N,univ_garch_spec1))
multi_univ_garch_spec2 <- multispec(replicate(N,univ_garch_spec2))
fit.multi_garch1 <- multifit(multi_univ_garch_spec1, a_t_dist, cluster=cl)
fit.multi_garch2 <- multifit(multi_univ_garch_spec2, a_t_dist, cluster=cl)
stopCluster(cl)

omega <- c(); alpha <- c(); beta <- c(); gamma <-c(); skew <- c(); shape <- c(); ar <- c() 
omega_reject <- 0; alpha_reject <- 0; beta_reject <- 0; gamma_reject <- 0; skew_reject <- 0; shape_reject <- 0; LM_stat <- 0; LB_stat <- 0; KS_stat <- 0; ar_reject <- 0
fit.multi_garch <- fit.multi_garch1
for (i in 1:30) {
  omega[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["omega"]]
  alpha[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["alpha1"]]
  beta[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["beta1"]]
  if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["omega", 4] < 0.05) { # if omega < 0.05, reject H0: The arch parameter is not different from zero
    omega_reject <- omega_reject+1
  }
  if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["alpha1", 4] < 0.05) { # if alpha1 < 0.05, reject H0: The arch parameter is not different from zero
    alpha_reject <- alpha_reject+1
  }
  if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["beta1", 4] < 0.05) { # if beta1 < 0.05, reject H0: The garch parameter is not different from zero
    beta_reject <- beta_reject+1
  }
  # gamma[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["gamma1"]]
  # if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["gamma1", 4] < 0.05) { # if gamma1 < 0.05, reject H0: The garch parameter is not different from zero
  #   gamma_reject <- gamma_reject+1
  # }
  # skew[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["skew"]]
  # if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["skew", 4] < 0.05) { # if skew < 0.05, reject H0: The skew parameter is not different from zero
  #   skew_reject <- skew_reject+1
  # }
  # shape[i] <- fit.multi_garch@fit[[i]]@fit[["coef"]][["shape"]]
  # if (fit.multi_garch@fit[[i]]@fit[["robust.matcoef"]]["shape", 4] < 0.05) { # if LM_result < 0.05, reject H0: The shape parameter is not different from zero
  #   shape_reject <- shape_reject+1
  # }
  LB_result <- Box.test(fit.multi_garch@fit[[i]]@fit[["residuals"]] / fit.multi_garch@fit[[i]]@fit[["sigma"]] , lag=10, type='Ljung-Box')$p.value
  if (LB_result < 0.05) {
    LB_stat <- LB_stat+1
  }
  LM_result <- Weighted.LM.test(fit.multi_garch@fit[[i]]@fit[["residuals"]], fit.multi_garch@fit[[i]]@fit[["var"]], lag=10)[["p.value"]]
  if (LM_result < 0.05) { # if LM_results < 0.05, reject H0: There is no autocorrelation in squared residuals
    LM_stat <- LM_stat+1
  }
  # standardized residuals for model checking
  x_t <- fit.multi_garch@fit[[i]]@fit[["residuals"]] / fit.multi_garch@fit[[i]]@fit[["sigma"]]
  distn="pnorm"
  #dist_result <- ks.test(x=x_t, y=distn, nu=shape[i], xi=skew[i])$p.value
  dist_result <- ks.test(x=x_t, y=distn)$p.value
  if (dist_result < 0.05) {
    KS_stat <- KS_stat+1
  }
}

# Cross-sectional distribution statistics
mean(omega); quantile(omega,  probs = c(5, 25, 50, 75, 95)/100)
mean(alpha); quantile(alpha,  probs = c(5, 25, 50, 75, 95)/100)
mean(beta); quantile(beta,  probs = c(5, 25, 50, 75, 95)/100)
mean(gamma); quantile(gamma,  probs = c(5, 25, 50, 75, 95)/100)
mean(shape); quantile(shape,  probs = c(5, 25, 50, 75, 95)/100)
mean(skew); quantile(skew,  probs = c(5, 25, 50, 75, 95)/100)






