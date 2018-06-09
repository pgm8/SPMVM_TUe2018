#################################################################################################################
rm(list=ls())  # remove all variables in R
install.packages("rugarch")
install.packages("rmgarch")
library(rugarch)
library(rmgarch)
library(parallel)
setwd("~/Documents/Python/PycharmProjects/ml_tue2017/source/main/resources/Data/multivariate_analysis")
source("R/fun_VaR_backtesting.R")
set.seed(42)  # 42:The answer to life, the universe and everything.

# Data sample import 
df <- read.csv("DJI30_returns_1987_2001.csv", row.names=1, header=T)

####################################################################################################
######                               Tranquil Market Conditions                              #######
####################################################################################################
data <- df[1:2224,]  # Data sample: 17/3/1987-29/12/1995
data$Date <- NULL

T <- 504  # Out-of-sample test sample: 3/1/1994-29/12/1995
N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
# Same first stage conditional mean filtration (sample mean)
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1)) 

## Dynamic Conditional Correlation model with various error distributions
dccGarch_mvnorm_tranquil <- dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="mvnorm")
dccGarch_mvt_tranquil <-  dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="mvt") 

# Write matrices containing time-varying volatilies and correlations to csv file
# Add column names to file containing conditional correlations
col_names <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv", row.names=1)
colnames(dccGarch_mvnorm_tranquil$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
colnames(dccGarch_mvt_tranquil$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
write.csv(dccGarch_mvnorm_tranquil$D_t_file, file="volatilities_mvnorm_DJI30_1994_1995.csv")
write.csv(dccGarch_mvnorm_tranquil$R_t_file, file="cor_DCC_mvnorm_DJI30_1994_1995.csv")
write.csv(dccGarch_mvt_tranquil$D_t_file, file="volatilities_mvt_DJI30_1994_1995.csv")
write.csv(dccGarch_mvt_tranquil$R_t_file, file="cor_DCC_mvt_DJI30_1994_1995.csv")

####    Value-at-Risk Estimation   ###   
alpha <- c(0.99, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01)
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)

## Load conditional correlations and volatilities data
vol_data_tranquil_mvt<- read.csv(file="volatilities_mvt_DJI30_1994_1995.csv", row.names=1)
vol_data_tranquil_mvnorm<- read.csv(file="volatilities_mvnorm_DJI30_1994_1995.csv", row.names=1)
cor_DCCgarch_tranquil_mvt <- read.csv(file="cor_DCC_norm_mvt_DJI30_1994_1995.csv", row.names=1)
cor_DCCgarch_tranquil_mvnorm <- read.csv(file="cor_DCC_mvnorm_DJI30_1994_1995.csv", row.names=1)
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



## Compute portfolio conditional covariances
# Multivariate Normal distributed errors
sigma_DCCgarch_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_DCCgarch_tranquil_mvnorm, T=T, w=w) 
sigma_KNN5_pearson_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_KNN5_pearson_tranquil, T=T, w=w)
sigma_KNN5_kendall_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_KNN5_kendall_tranquil, T=T, w=w)
sigma_KNN_idw_pearson_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_KNN_idw_pearson_tranquil, T=T, w=w)
sigma_KNN_idw_kendall_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_KNN_idw_kendall_tranquil, T=T, w=w)

sigma_RF10_pearson_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_RF10_pearson_tranquil, T=T, w=w)
sigma_RF10_kendall_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_RF10_kendall_tranquil, T=T, w=w)
sigma_RF100_pearson_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_RF100_pearson_tranquil, T=T, w=w)
sigma_RF100_kendall_tranquil_mvnorm <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_RF100_kendall_tranquil, T=T, w=w)

# Multivariate Student t-distributed errors
sigma_DCCgarch_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvnorm, cor_DCCgarch_tranquil_mvt, T=T, w=w) 
sigma_KNN5_pearson_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_KNN5_pearson_tranquil, T=T, w=w)
sigma_KNN5_kendall_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_KNN5_kendall_tranquil, T=T, w=w)
sigma_KNN_idw_pearson_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_KNN_idw_pearson_tranquil, T=T, w=w)
sigma_KNN_idw_kendall_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_KNN_idw_kendall_tranquil, T=T, w=w)

sigma_RF10_pearson_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_RF10_pearson_tranquil, T=T, w=w)
sigma_RF10_kendall_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_RF10_kendall_tranquil, T=T, w=w)
sigma_RF100_pearson_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_RF100_pearson_tranquil, T=T, w=w)
sigma_RF100_kendall_tranquil_mvt <- sigma_vec_portfolio(vol_data_tranquil_mvt, cor_RF100_kendall_tranquil, T=T, w=w)

## Value-at-Risk Computation 
# Multivariate Normal distributed errors
dcc_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_tranquil_mvnorm, mu= mu_portfolio_loss, cl=alpha)
knn5_pearson_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_tranquil_mvnorm,mu= mu_portfolio_loss, cl=alpha)
knn5_kendall_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)
knn_idw_pearson_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)
knn_idw_kendall_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)

rf10_pearson_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF10_pearson_tranquil_mvnorm,mu= mu_portfolio_loss, cl=alpha)
rf10_kendall_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF10_kendall_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)
rf100_pearson_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF100_pearson_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)
rf100_kendall_VaR_tranquil_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF100_kendall_tranquil_mvnorm, mu=mu_portfolio_loss, cl=alpha)

# Multivariate Student t-distributed errors
dcc_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_tranquil_mvt, mu= mu_portfolio_loss, cl=alpha)
knn5_pearson_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_tranquil_mvt,mu= mu_portfolio_loss, cl=alpha)
knn5_kendall_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)
knn_idw_pearson_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)
knn_idw_kendall_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)

rf10_pearson_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_RF10_pearson_tranquil_mvt,mu= mu_portfolio_loss, cl=alpha)
rf10_kendall_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_RF10_kendall_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)
rf100_pearson_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_RF100_pearson_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)
rf100_kendall_VaR_tranquil_mvt <- VaR_estimates(sigma_portfolio=sigma_RF100_kendall_tranquil_mvt, mu=mu_portfolio_loss, cl=alpha)


## Value-at-Risk Backtesting   
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns assuming equally weighted portfolio
# Multivariate Normal distributed errors
backtest_dccGarch_tranquil_mvnorm <- uc_ind_test(VaR_est=dcc_VaR_tranquil_mvnorm, cl=alpha)
backtest_KNN5_pearson_tranquil_mvnorm <- uc_ind_test(VaR_est=knn5_pearson_VaR_tranquil_mvnorm, cl=alpha)
backtest_KNN5_kendall_tranquil_mvnorm <- uc_ind_test(VaR_est=knn5_kendall_VaR_tranquil_mvnorm, cl=alpha)
backtest_KNN_idw_pearson_tranquil_mvnorm <- uc_ind_test(VaR_est=knn_idw_pearson_VaR_tranquil_mvnorm, cl=alpha)
backtest_KNN_idw_kendall_tranquil_mvnorm <- uc_ind_test(VaR_est=knn_idw_kendall_VaR_tranquil_mvnorm, cl=alpha)

backtest_RF10_pearson_tranquil_mvnorm <- uc_ind_test(VaR_est=rf10_pearson_VaR_tranquil_mvnorm, cl=alpha)
backtest_RF10_kendall_tranquil_mvnorm <- uc_ind_test(VaR_est=rf10_kendall_VaR_tranquil_mvnorm, cl=alpha)
backtest_RF100_pearson_tranquil_mvnorm <- uc_ind_test(VaR_est=rf100_pearson_VaR_tranquil_mvnorm, cl=alpha)
backtest_RF100_kendall_tranquil_mvnorm <- uc_ind_test(VaR_est=rf100_kendall_VaR_tranquil_mvnorm, cl=alpha)
# Multivariate Student t-distributed errors
backtest_dccGarch_tranquil_mvt <- uc_ind_test(VaR_est=dcc_VaR_tranquil_mvt, cl=alpha)
backtest_KNN5_pearson_tranquil_mvt <- uc_ind_test(VaR_est=knn5_pearson_VaR_tranquil_mvt, cl=alpha)
backtest_KNN5_kendall_tranquil_mvt <- uc_ind_test(VaR_est=knn5_kendall_VaR_tranquil_mvt, cl=alpha)
backtest_KNN_idw_pearson_tranquil_mvt <- uc_ind_test(VaR_est=knn_idw_pearson_VaR_tranquil_mvt, cl=alpha)
backtest_KNN_idw_kendall_tranquil_mvt <- uc_ind_test(VaR_est=knn_idw_kendall_VaR_tranquil_mvt, cl=alpha)

backtest_RF10_pearson_tranquil_mvt <- uc_ind_test(VaR_est=rf10_pearson_VaR_tranquil_mvt, cl=alpha)
backtest_RF10_kendall_tranquil_mvt <- uc_ind_test(VaR_est=rf10_kendall_VaR_tranquil_mvt, cl=alpha)
backtest_RF100_pearson_tranquil_mvt <- uc_ind_test(VaR_est=rf100_pearson_VaR_tranquil_mvt, cl=alpha)
backtest_RF100_kendall_tranquil_mvt <- uc_ind_test(VaR_est=rf100_kendall_VaR_tranquil_mvt, cl=alpha)
# Write backtest results to csv file
write.csv(backtest_dccGarch_tranquil_mvnorm, file="backtest/backtest_dccGarch_mvnorm_1994_1995.csv") 
write.csv(backtest_KNN5_pearson_tranquil_mvnorm, file="backtest/backtest_KNN5_pearson_mvnorm_1994_1995.csv") 
write.csv(backtest_KNN5_kendall_tranquil_mvnorm, file="backtest/backtest_KNN5_kendall_mvnorm_1994_1995.csv") 
write.csv(backtest_KNN_idw_pearson_tranquil_mvnorm, file="backtest/backtest_KNN_idw_pearson_mvnorm_1994_1995.csv") 
write.csv(backtest_KNN_idw_kendall_tranquil_mvnorm, file="backtest/backtest_KNN_idw_kendall_mvnorm_1994_1995.csv") 

write.csv(backtest_RF10_pearson_tranquil_mvnorm, file="backtest/backtest_RF10_pearson_mvnorm_1994_1995.csv") 
write.csv(backtest_RF10_kendall_tranquil_mvnorm, file="backtest/backtest_RF10_kendall_mvnorm_1994_1995.csv") 
write.csv(backtest_RF100_pearson_tranquil_mvnorm, file="backtest/backtest_RF100_pearson_mvnorm_1994_1995.csv") 
write.csv(backtest_RF100_kendall_tranquil_mvnorm, file="backtest/backtest_RF100_kendall_mvnorm_1994_1995.csv") 

write.csv(backtest_dccGarch_tranquil_mvt, file="backtest/backtest_dccGarch_mvt_1994_1995.csv") 
write.csv(backtest_KNN5_pearson_tranquil_mvt, file="backtest/backtest_KNN5_pearson_mvt_1994_1995.csv") 
write.csv(backtest_KNN5_kendall_tranquil_mvt, file="backtest/backtest_KNN5_kendall_mvt_1994_1995.csv") 
write.csv(backtest_KNN_idw_pearson_tranquil_mvt, file="backtest/backtest_KNN_idw_pearson_mvt_1994_1995.csv") 
write.csv(backtest_KNN_idw_kendall_tranquil_mvt, file="backtest/backtest_KNN_idw_kendall_mvt_1994_1995.csv") 

write.csv(backtest_RF10_pearson_tranquil_mvt, file="backtest/backtest_RF10_pearson_mvt_1994_1995.csv") 
write.csv(backtest_RF10_kendall_tranquil_mvt, file="backtest/backtest_RF10_kendall_mvt_1994_1995.csv") 
write.csv(backtest_RF100_pearson_tranquil_mvt, file="backtest/backtest_RF100_pearson_mvt_1994_1995.csv") 
write.csv(backtest_RF100_kendall_tranquil_mvt, file="backtest/backtest_RF100_kendall_mvt_1994_1995.csv")

# Non-rejection regions volatile market conditions
for (a in alpha){
  print(sprintf("CI %f: (%i,%i)",a, regions_uc_test(T=T, alpha=a)$lb, regions_uc_test(T=T, alpha=a)$ub))
}

####################################################################################################
######                               Volatile Market Conditions                              #######
####################################################################################################
data <- df  # Data sample: 17/3/1987-31/12/2001
data$Date <- NULL

T <- 500  # Out-of-sample test sample: 3/1/2000-31/12/2001
N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1)) 

## Dynamic Conditional Correlation model with various error distributions
dccGarch_mvnorm_vol <- dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="mvnorm")
dccGarch_mvt_vol <-  dcc_garch_modeling(data=a_t, t=t, distribution.model="norm", distribution="mvt") 
# Write matrices containing time-varying volatilies and correlations to csv file
# Add column names to file containing conditional correlations
col_names <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv", row.names=1)
colnames(dccGarch_mvnorm_vol$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
colnames(dccGarch_mvt_vol$R_t_file) <- c(colnames(col_names))[1:(N*(N-1)/2)]
write.csv(dccGarch_mvnorm_vol$D_t_file, file="volatilities_mvnorm_DJI30_2000_2001.csv")
write.csv(dccGarch_mvnorm_vol$R_t_file, file="cor_DCC_mvnorm_DJI30_2000_2001.csv")
write.csv(dccGarch_mvt_vol$D_t_file, file="volatilities_norm_mvt_DJI30_2000_2001.csv")
write.csv(dccGarch_mvt_vol$R_t_file, file="cor_DCC_norm_mvt_DJI30_2000_2001.csv")

####    Value-at-Risk Estimation   ###   
alpha <- c(0.99, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01)
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)

## Load conditional correlations and volatilities data
vol_data_vol_mvt<- read.csv(file="volatilities_mvt_DJI30_2000_2001.csv", row.names=1)
vol_data_vol_mvnorm<- read.csv(file="volatilities_mvnorm_DJI30_2000_2001.csv", row.names=1)
cor_DCCgarch_vol_mvnorm <- read.csv(file="cor_DCC_mvnorm_DJI30_2000_2001.csv", row.names=1)
cor_DCCgarch_vol_mvt <- read.csv(file="cor_DCC_mvt_DJI30_2000_2001.csv", row.names=1)
# Nearest neighbor
cor_KNN5_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN5_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN_idw_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_KNN_idw_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_2000_2001.csv", row.names=1)
# Random forest
cor_RF10_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_rf10_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_RF10_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_rf10_kendall_10_DJI30_2000_2001.csv", row.names=1)
cor_RF100_pearson_volatile <- read.csv(file="pearson/pearson_cor_estimates/cor_rf100_pearson_10_DJI30_2000_2001.csv", row.names=1)
cor_RF100_kendall_volatile <- read.csv(file="kendall/kendall_cor_estimates/cor_rf100_kendall_10_DJI30_2000_2001.csv", row.names=1)

# Compute portfolio conditional covariances
# Multivariate Normal distributed errors
sigma_DCCgarch_vol_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_DCCgarch_vol_mvnorm, T=T, w=w) 
sigma_KNN5_pearson_vol_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_KNN5_pearson_vol, T=T, w=w)
sigma_KNN5_kendall_vol_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_KNN5_kendall_vol, T=T, w=w)
sigma_KNN_idw_pearson_vol_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_KNN_idw_pearson_vol, T=T, w=w)
sigma_KNN_idw_kendall_vol_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_KNN_idw_kendall_vol, T=T, w=w)

sigma_RF10_pearson_volatile_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_RF10_pearson_volatile, T=T, w=w)
sigma_RF10_kendall_volatile_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_RF10_kendall_volatile, T=T, w=w)
sigma_RF100_pearson_volatile_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_RF100_pearson_volatile, T=T, w=w)
sigma_RF100_kendall_volatile_mvnorm <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_RF100_kendall_volatile, T=T, w=w)

# Multivariate Student t-distributed errors
sigma_DCCgarch_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvnorm, cor_DCCgarch_vol_mvt, T=T, w=w) 
sigma_KNN5_pearson_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_KNN5_pearson_vol, T=T, w=w)
sigma_KNN5_kendall_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_KNN5_kendall_vol, T=T, w=w)
sigma_KNN_idw_pearson_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_KNN_idw_pearson_vol, T=T, w=w)
sigma_KNN_idw_kendall_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_KNN_idw_kendall_vol, T=T, w=w)

sigma_RF10_pearson_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_RF10_pearson_volatile, T=T, w=w)
sigma_RF10_kendall_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_RF10_kendall_volatile, T=T, w=w)
sigma_RF100_pearson_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_RF100_pearson_volatile, T=T, w=w)
sigma_RF100_kendall_vol_mvt <- sigma_vec_portfolio(vol_data_vol_mvt, cor_RF100_kendall_volatile, T=T, w=w)

## Value-at-Risk Computation 
# Multivariate Normal distributed errors
dcc_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_vol_mvnorm, mu= mu_portfolio_loss, cl=alpha)
knn5_pearson_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_vol_mvnorm,mu= mu_portfolio_loss, cl=alpha)
knn5_kendall_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_vol_mvnorm, mu=mu_portfolio_loss, cl=alpha)
knn_idw_pearson_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_vol_mvnorm, mu=mu_portfolio_loss, cl=alpha)
knn_idw_kendall_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_vol_mvnorm, mu=mu_portfolio_loss, cl=alpha)


rf10_pearson_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF10_pearson_volatile_mvnorm,mu= mu_portfolio_loss, cl=alpha)
rf10_kendall_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF10_kendall_volatile_mvnorm, mu=mu_portfolio_loss, cl=alpha)
rf100_pearson_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF100_pearson_volatile_mvnorm, mu=mu_portfolio_loss, cl=alpha)
rf100_kendall_VaR_vol_mvnorm <- VaR_estimates(sigma_portfolio=sigma_RF100_kendall_volatile_mvnorm, mu=mu_portfolio_loss, cl=alpha)



# Multivariate Student t-distributed errors
dcc_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_vol_mvt, mu= mu_portfolio_loss, cl=alpha)
knn5_pearson_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_vol_mvt,mu= mu_portfolio_loss, cl=alpha)
knn5_kendall_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_vol_mvt, mu=mu_portfolio_loss, cl=alpha)
knn_idw_pearson_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_vol_mvt, mu=mu_portfolio_loss, cl=alpha)
knn_idw_kendall_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_vol_mvt, mu=mu_portfolio_loss, cl=alpha)

rf10_pearson_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_RF10_pearson_volatile_mvt,mu= mu_portfolio_loss, cl=alpha)
rf10_kendall_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_RF10_kendall_volatile_mvt, mu=mu_portfolio_loss, cl=alpha)
rf100_pearson_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_RF100_pearson_volatile_mvt, mu=mu_portfolio_loss, cl=alpha)
rf100_kendall_VaR_vol_mvt <- VaR_estimates(sigma_portfolio=sigma_RF100_kendall_volatile_mvt, mu=mu_portfolio_loss, cl=alpha)

## Value-at-Risk Backtesting   
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns assuming equally weighted portfolio
# Multivariate Normal distributed errors
backtest_dccGarch_vol_mvnorm <- uc_ind_test(VaR_est=dcc_VaR_vol_mvnorm, cl=alpha)
backtest_KNN5_pearson_vol_mvnorm <- uc_ind_test(VaR_est=knn5_pearson_VaR_vol_mvnorm, cl=alpha)
backtest_KNN5_kendall_vol_mvnorm <- uc_ind_test(VaR_est=knn5_kendall_VaR_vol_mvnorm, cl=alpha)
backtest_KNN_idw_pearson_vol_mvnorm <- uc_ind_test(VaR_est=knn_idw_pearson_VaR_vol_mvnorm, cl=alpha)
backtest_KNN_idw_kendall_vol_mvnorm <- uc_ind_test(VaR_est=knn_idw_kendall_VaR_vol_mvnorm, cl=alpha)

backtest_RF10_pearson_vol_mvnorm <- uc_ind_test(VaR_est=rf10_pearson_VaR_vol_mvnorm, cl=alpha)
backtest_RF10_kendall_vol_mvnorm <- uc_ind_test(VaR_est=rf10_kendall_VaR_vol_mvnorm, cl=alpha)
backtest_RF100_pearson_vol_mvnorm <- uc_ind_test(VaR_est=rf100_pearson_VaR_vol_mvnorm, cl=alpha)
backtest_RF100_kendall_vol_mvnorm <- uc_ind_test(VaR_est=rf100_kendall_VaR_vol_mvnorm, cl=alpha)



# Multivariate Student t-distributed errors
backtest_dccGarch_vol_mvt <- uc_ind_test(VaR_est=dcc_VaR_vol_mvt, cl=alpha)
backtest_KNN5_pearson_vol_mvt <- uc_ind_test(VaR_est=knn5_pearson_VaR_vol_mvt, cl=alpha)
backtest_KNN5_kendall_vol_mvt <- uc_ind_test(VaR_est=knn5_kendall_VaR_vol_mvt, cl=alpha)
backtest_KNN_idw_pearson_vol_mvt <- uc_ind_test(VaR_est=knn_idw_pearson_VaR_vol_mvt, cl=alpha)
backtest_KNN_idw_kendall_vol_mvt <- uc_ind_test(VaR_est=knn_idw_kendall_VaR_vol_mvt, cl=alpha)

backtest_RF10_pearson_vol_mvt <- uc_ind_test(VaR_est=rf10_pearson_VaR_vol_mvt, cl=alpha)
backtest_RF10_kendall_vol_mvt <- uc_ind_test(VaR_est=rf10_kendall_VaR_vol_mvt, cl=alpha)
backtest_RF100_pearson_vol_mvt <- uc_ind_test(VaR_est=rf100_pearson_VaR_vol_mvt, cl=alpha)
backtest_RF100_kendall_vol_mvt <- uc_ind_test(VaR_est=rf100_kendall_VaR_vol_mvt, cl=alpha)

# Write backtest results to csv file
write.csv(backtest_dccGarch_vol_mvnorm, file="backtest_dccGarch_mvnorm_2000_2001.csv") 
write.csv(backtest_KNN5_pearson_vol_mvnorm, file="backtest_KNN5_pearson_mvnorm_2000_2001.csv") 
write.csv(backtest_KNN5_kendall_vol_mvnorm, file="backtest_KNN5_kendall_mvnorm_2000_2001.csv") 
write.csv(backtest_KNN_idw_pearson_vol_mvnorm, file="backtest_KNN_idw_pearson_mvnorm_2000_2001.csv") 
write.csv(backtest_KNN_idw_kendall_vol_mvnorm, file="backtest_KNN_idw_kendall_mvnorm_2000_2001.csv")

write.csv(backtest_RF10_pearson_vol_mvnorm, file="backtest/backtest_RF10_pearson_mvnorm_2000_2001.csv") 
write.csv(backtest_RF10_kendall_vol_mvnorm, file="backtest/backtest_RF10_kendall_mvnorm_2000_2001.csv") 
write.csv(backtest_RF100_pearson_vol_mvnorm, file="backtest/backtest_RF100_pearson_mvnorm_2000_2001.csv") 
write.csv(backtest_RF100_kendall_vol_mvnorm, file="backtest/backtest_RF100_kendall_mvnorm_2000_2001.csv") 


write.csv(backtest_dccGarch_vol_mvt, file="backtest_dccGarch_mvt_2000_2001.csv") 
write.csv(backtest_KNN5_pearson_vol_mvt, file="backtest_KNN5_pearson_mvt_2000_2001.csv") 
write.csv(backtest_KNN5_kendall_vol_mvt, file="backtest_KNN5_kendall_mvt_2000_2001.csv") 
write.csv(backtest_KNN_idw_pearson_vol_mvt, file="backtest_KNN_idw_pearson_mvt_2000_2001.csv") 
write.csv(backtest_KNN_idw_kendall_vol_mvt, file="backtest_KNN_idw_kendall_mvt_2000_2001.csv") 


write.csv(backtest_RF10_pearson_vol_mvt, file="backtest/backtest_RF10_pearson_mvt_2000_2001.csv") 
write.csv(backtest_RF10_kendall_vol_mvt, file="backtest/backtest_RF10_kendall_mvt_2000_2001.csv") 
write.csv(backtest_RF100_pearson_vol_mvt, file="backtest/backtest_RF100_pearson_mvt_2000_2001.csv") 
write.csv(backtest_RF100_kendall_vol_mvt, file="backtest/backtest_RF100_kendall_mvt_2000_2001.csv") 




# Non-rejection regions volatile market conditions
for (a in alpha){
  print(sprintf("CI %f: (%i,%i)",a, regions_uc_test(T=T, alpha=a)$lb, regions_uc_test(T=T, alpha=a)$ub))
}





############################################################################################################
## Plot Daily Log Returns and Value-at-Risk Exceedances
x <- seq(1, T)
plot(VaR_true, type="p", pch = 20, col="black", main="Daily Returns and Value-at-Risk Exceedances",
     xlab="time", ylab="Log Return")
plot(VaR_true, type="l", col="black", main="Daily Returns and Value-at-Risk Exceedances",
     xlab="time", ylab="Log Return")
lines(rf10_pearson_VaR_tranquil_mvnorm[,toString(0.99)], col="green")
lines(rf100_pearson_VaR_tranquil_mvnorm[,toString(0.99)], col="red")
lines(dcc_VaR_vol[,toString(0.99)], col="red")
lines(rf10_kendall_VaR_tranquil_vnorm[,toString(0.99)], col="blue")
lines(knn_idw_pearson_VaR_vol[,toString(0.99)], col="brown")
lines(knn_idw_kendall_VaR_vol[,toString(0.99)], col="orange")
legend(x=3.5, y=14, legend=c("dcc", "knn(5)"), col=c("red", "green"), lty=1, bty="n")




