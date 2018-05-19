#################################################################################################################
rm(list=ls())  # remove all variables in R
set.seed(42)
setwd("~/Documents/Python/PycharmProjects/thesisOML/ml_tue2017/source/main/resources/Data/multivariate_analysis")
df <- read.csv("DJI30_returns_stable.csv")

N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
data <- df[-c(1:10), ]  # Data sample 3/1/1995-31/12/1999
rownames(data) <- seq(length=nrow(data))  # reset index
data$X <- NULL  # Drop first column with indices 
data$Date <- NULL

T <- 252  # Length out-of-sample test set
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1))   # c(1011:1262)

####################################################################################################
######                      DCC-GARCH Student-t distributed errors (rugarch)                 #######
####################################################################################################
install.packages("rugarch")
install.packages("rmgarch")
library(rugarch)
library(rmgarch)
library(parallel)

# Two-stage quasi-likelihood function to find parameters under multivariate Student-t distributed errors
dcc_garch_modeling <- function(data=a_t, distribution.model="norm", distribution="mvnorm") {
  D_t_file <- matrix(NaN, T, N)
  colnames(D_t_file) <- c(colnames(data))[1:N]
  R_t_file <- matrix(NaN, T, N*(N-1)/2)
  cl = makePSOCKcluster(10)
  for (i in seq_along(t)) {  
    tic <- Sys.time()
    data_train <- data[i:(t[i]+1),1:N]  # Rolling forward: data[i:(t[i]+1),1:N]  
    # Specify univariate GARCH(1,1) with marginal Student-t dist. errors for each component series
    univ_garch_spec <- ugarchspec(variance.model=list(model="fGARCH", submodel="GARCH", garchOrder=c(1, 1)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model=distribution.model)
    multi_univ_garch_spec <- multispec(replicate(N,univ_garch_spec))
    # Specify DCC-GARCH(1,1) with multivariate Student-t errors for stand. residual series
    dcc_spec <- dccspec(multi_univ_garch_spec, dccOrder=c(1,1), model="DCC", distribution=distribution)
    # Fit component series
    fit.multi_garch <- multifit(multi_univ_garch_spec, data_train, cluster=cl)
    # Estimate DCC-GARCH(1,1) mvt model
    fit.dcc = dccfit(dcc_spec, data=data_train, solver='solnp', cluster=cl, fit.control=list(eval.se = FALSE), fit=fit.multi_garch)
    # Save conditional volatilities and correlations
    D_t_file[i, ] <- tail(sigma(fit.multi_garch),1)
    R_t_file[i, ] <- t(rcor(fit.dcc)[,,dim(rcor(fit.dcc))[3]])[lower.tri(t(rcor(fit.dcc)[,,dim(rcor(fit.dcc))[3]]),diag=FALSE)]
    print(i)
    print(t[i])
    print(Sys.time()-tic)
  }
  stopCluster(cl)
  return(list("D_t_file"=D_t_file, "R_t_file"=R_t_file))
}
dccGarch_mvnorm <- dcc_garch_modeling(data=a_t, distribution.model="norm", distribution="mvnorm")
dccGarch_mvt <-  dcc_garch_modeling(data=a_t, distribution.model="std", distribution="mvt") 

# Add column names to file containing conditional correlations
correlation_KNNpearson <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_stable.csv")
correlation_KNNpearson$X <- NULL
colnames(R_t_file) <- c(colnames(correlation_KNNpearson))[1:(N*(N-1)/2)]
# Write matrices containing time-varying volatilies and correlations to csv file
write.csv(D_t_file, file="volatilities_norm_DJI30_stable.csv")
write.csv(R_t_file, file="cor_DCC_mvn_DJI30_stable.csv")

####################################################################################################
######                               Nearest Neighbors Algorithm                             #######
####################################################################################################
# Load conditional correlations and volatilities data
volatility_data <- read.csv(file="volatilities_norm_DJI30_stable.csv")
cor_DCCgarch <- read.csv(file="cor_DCC_mvn_DJI30_stable.csv")

cor_KNN5_pearson <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_stable.csv")
cor_KNN5_kendall <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_stable.csv")
cor_KNN_idw_pearson <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_stable.csv")
cor_KNN_idw_kendall <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_stable.csv")

# Data pre-processing
volatility_data$X <- NULL  # Drop first column with indices 
cor_DCCgarch$X <- NULL
cor_KNN5_pearson$X <- NULL
cor_KNN5_kendall$X <- NULL
cor_KNN_idw_pearson$X <- NULL
cor_KNN_idw_kendall$X <- NULL

# Initialise variables
N <- ncol(volatility_data)  # Number of assets under consideration

cor_mat <- function(cor_vec, dim=N) { # function assumes row matrix indexing
  # cor_vec := vector with uppertriangular correlation values index by row
  R_t <- matrix(0,N,N)
  R_t[lower.tri(R_t, diag=FALSE)] <- as.numeric(cor_vec)
  R_t <- t(R_t)
  R_t[lower.tri(R_t, diag=FALSE)] <- t(R_t)[lower.tri(R_t, diag=FALSE)]
  diag(R_t) <- rep(1, times=N)
  if (isSymmetric(R_t) == FALSE) {
    print("Houston we have a problem, our correlation matrix is not symmetric!")
  }
  return(R_t)
}

cov_mat_portfolio <- function(vol_vec, cor_vec, dim=N) {
  D_t <- diag(as.numeric(vol_vec))
  R_t <- cor_mat(cor_vec, dim=N)   # Symmetric matrix from upper triangular matrix
  H_t <- D_t %*% R_t %*% D_t  # Conditional covariance matrix 
  return(H_t)
}

sigma_vec_portfolio <- function(volatility_matrix, cor_matrix) {
  # volatility_matrix := matrix with conditional volatilities
  # cor_matrix := matrix with conditional correlations
  sigma_t <- rep(NaN,T)  
  for (i in 1:T) {
    H_t <- cov_mat_portfolio(volatility_matrix[i,], cor_matrix[i,])  
    sigma_t[i] <- sqrt(t(w)%*%H_t%*%w)  # Portfolio sdv 
  }
  return(sigma_t)
}
# Compute portfolio conditional covariances
sigma_DCCgarch <- sigma_vec_portfolio(volatility_data, cor_DCCgarch) 
sigma_KNN5_pearson <- sigma_vec_portfolio(volatility_data, cor_KNN5_pearson)
sigma_KNN5_kendall <- sigma_vec_portfolio(volatility_data, cor_KNN5_kendall)
sigma_KNN_idw_pearson <- sigma_vec_portfolio(volatility_data, cor_KNN_idw_pearson)
sigma_KNN_idw_kendall <- sigma_vec_portfolio(volatility_data, cor_KNN_idw_kendall)


####################################################################################################
######                               Value-at-Risk Computation                               #######
####################################################################################################
alpha <- c(0.99, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01)
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)

VaR_estimates <- function(sigma_portfolio, mu=mu_portfolio_loss, cl=alpha) {
  VaR_mat <- matrix(data=NaN, nrow=T, ncol=length(cl))
  colnames(VaR_mat) <- 1-cl  #  Set column names to corresponding conf. level VaR estimates
  for (i in 1:T) {
    for (a in cl) {
      VaR_mat[i, toString(a)] <- mu+sigma_portfolio[i]*qnorm(1-a)
    }
  }
  return(VaR_mat)
}
dcc_VaR <- VaR_estimates(sigma_portfolio=sigma_DCCgarch)
knn5_pearson_VaR <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson)
knn5_kendall_VaR <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall)
knn_idw_pearson_VaR <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson)
knn_idw_kendall_VaR <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall)

####################################################################################################
######                               Value-at-Risk Backtesting                               #######
####################################################################################################
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns
# BACKTESTING function: Unconditional coverage test (proportion of failures Kupiec test) and 
# Independence test (Christoffersen Markov test)
pof_ind_test <- function(VaR_est, cl=alpha) {
  # Initialise datastructure to save test results
  row_names <- c("exceedances", "LR_pof", "LR_crit_pof", "decision_pof",
                 "pi01", "pi11", "LR_ind", "LR_crit_ind", "decision_ind")
  Kupiec_Christoffersen_mat <- matrix(data=NaN, nrow=length(row_names), ncol=length(cl))
  colnames(Kupiec_Christoffersen_mat) <- 1-cl  #  Set column names to corresponding conf. level VaR estimates
  rownames(Kupiec_Christoffersen_mat) <- row_names
  Kupiec_Christoffersen_mat
  for (a in cl) {
    I_sum <- sum(VaR_true<VaR_est[,toString(1-a)])   # Sum of exceedances
    LR_pof <- 2*log((((1-(I_sum/T))/(1-a))^(T-I_sum))*(((I_sum/T)/a)^I_sum))
    LR_crit_pof <- qchisq(p=0.95, df=1)  # Coverage test with confidence level 0.95 
    decision_pof <- ifelse(LR_pof > LR_crit_pof, "Reject H0", "Fail to reject H0")
    # Compute Markov test numbers
    I <- ifelse(VaR_true<VaR_est[,toString(1-a)], 1, 0)  #  Hit function series
    T_00=T_10=T_01=T_11 <- 0  # Markov test numbers
    for (i in 2:252) {  2:T
      T_00 <- ifelse(I[i-1]==0 & I[i]==0, T_00+1, T_00)
      T_10 <- ifelse(I[i-1]==1 & I[i]==0, T_10+1, T_10)
      T_01 <- ifelse(I[i-1]==0 & I[i]==1, T_01+1, T_01)
      T_11 <- ifelse(I[i-1]==1 & I[i]==1, T_11+1, T_11)
    }
    # Compute Markov test probabilities
    pi_01=pi_11 <- 0          
    pi_01 <- T_01 / (T_00+T_01)
    pi_11 <- T_11 / (T_10+T_11)
    pi_2 <- (T_01+T_11) / (T_00+T_10+T_01+T_11)
    LR_ind <- -2*log(((1-pi_2)^(T_00+T_10)*pi_2^(T_01+T_11)) / 
                       ((1-pi_01)^T_00*pi_01^T_01*(1-pi_11)^T_10*pi_11^T_11))
    LR_crit_ind <- qchisq(p=0.95, df=1)  # Coverage test with confidence level 0.95 
    decision_ind <- ifelse(LR_ind > LR_crit_ind, "Reject H0", "Fail to reject H0")
    Kupiec_Christoffersen_mat[, toString(1-a)] <- c(I_sum, LR_pof, LR_crit_pof,decision_pof,
                                                     pi_01, pi_11, LR_ind, LR_crit_ind, decision_ind)
  }
  return(Kupiec_Christoffersen_mat)
}
backtest_dccGarch <- pof_ind_test(VaR_est=dcc_VaR)
backtest_KNN5_pearson <- pof_ind_test(VaR_est=knn5_pearson_VaR)
backtest_KNN5_kendall <- pof_ind_test(VaR_est=knn5_kendall_VaR)
backtest_KNN_idw_pearson <- pof_ind_test(VaR_est=knn_idw_pearson_VaR)
backtest_KNN_idw_kendall <- pof_ind_test(VaR_est=knn_idw_kendall_VaR)

View(backtest_dccGarch)
View(backtest_KNN5_pearson)
View(backtest_KNN5_kendall)
View(backtest_KNN_idw_pearson)
View(backtest_KNN_idw_kendall)

# Verify correct computation VaR backtest measures
install.packages("GAS")
library(GAS)
for (a in alpha) {
  print(a)
  print(BacktestVaR(VaR_true, knn5_pearson_VaR[,toString(a)], alpha=1-a)$LRuc)
  #print(BacktestVaR(VaR_true, knn5_pearson_VaR[,toString(a)], alpha=1-a)$LRcc)
  
}

# DCC model seems to overestimate risk measure and is therefore rejected in right tail of the loss distribution
# Overestimation of risk is not necessarily a problem for regulatory bodies but comes at a cost of profitability for
# e.g. banks/ hedge funds etc.
# KNN5 not rejected in right tail of loss distribution. Seems to accurately estimate risk measure. Model performance does 
# degrade considerably for quantiles < 0.9. This implies model performance is not consistent across the entire loss distribution.

# Write backtest results to csv file
write.csv(backtest_dccGarch, file="backtest_dccGarch_mvn_stable.csv") 
write.csv(backtest_KNN5_pearson, file="backtest_KNN5_pearson_norm_stable.csv") 
write.csv(backtest_KNN5_kendall, file="backtest_KNN5_kendall_norm_stable.csv") 
write.csv(backtest_KNN_idw_pearson, file="backtest_KNN_idw_pearson_norm_stable.csv") 
write.csv(backtest_KNN_idw_kendall, file="backtest_KNN_idw_kendall_norm_stable.csv") 

############################################################################################################
## Plot Daily Log Returns and Value-at-Risk Exceedances
x <- seq(1, T)
plot(VaR_true, type="p", pch = 20, col="black", main="Daily Returns and Value-at-Risk Exceedances",
     xlab="time", ylab="Log Return")
plot(VaR_true, type="l", col="black", main="Daily Returns and Value-at-Risk Exceedances",
     xlab="time", ylab="Log Return")
lines(knn5_pearson_VaR_vol[,toString(0.99)], col="green")
lines(dcc_VaR_vol[,toString(0.99)], col="red")
lines(knn5_kendall_VaR_vol[,toString(0.99)], col="blue")
lines(knn_idw_pearson_VaR_vol[,toString(0.99)], col="brown")
lines(knn_idw_kendall_VaR_vol[,toString(0.99)], col="orange")
legend(x=3.5, y=14, legend=c("dcc", "knn(5)"), col=c("red", "green"), lty=1, bty="n")



####################################################################################################
######                               Volatile Market Conditions                              #######
####################################################################################################
df <- read.csv("DJI30_returns_1987_2001.csv")

N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
data <- df[-c(1:10), ]  # Data sample 2/1/2004-31/12/2008
rownames(data) <- seq(length=nrow(data))  # reset index
data$X <- NULL  # Drop first column with indices 
data$Date <- NULL

#T <- 252  # Length out-of-sample test set
T <- 500
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1))   # c(1006:1258)

#dccGarch_mvnorm <- dcc_garch_modeling(data=a_t, distribution.model="norm", distribution="mvnorm")
#dccGarch_mvt <-  dcc_garch_modeling(data=a_t, distribution.model="std", distribution="mvt") 
#dccGarch_sstd_mvt <-  dcc_garch_modeling(data=a_t, distribution.model="sstd", distribution="mvt") 

# Add column names to file containing conditional correlations
correlation_KNNpearson <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_stable.csv")
correlation_KNNpearson$X <- NULL
colnames(dccGarch_mvt$R_t_file) <- c(colnames(correlation_KNNpearson))[1:(N*(N-1)/2)]
# Write matrices containing time-varying volatilies and correlations to csv file
write.csv(dccGarch_mvt$D_t_file, file="volatilities_sstd_DJI30_volatile.csv")
write.csv(dccGarch_mvt$R_t_file, file="cor_DCC_sstd_mvt_DJI30_volatile.csv")

####    Nearest Neighbors Algorithm   
# Load conditional correlations and volatilities data
volatility_data_vol <- read.csv(file="volatilities_mvnorm_DJI30_2000_2001_ext.csv")
cor_DCCgarch_vol <- read.csv(file="cor_DCC_mvnorm_DJI30_2000_2001_ext.csv")
cor_KNN5_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_2000_2001.csv")
cor_KNN5_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_2000_2001.csv")
cor_KNN_idw_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_2000_2001.csv")
cor_KNN_idw_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_2000_2001.csv")

# Data pre-processing
volatility_data_vol$X <- NULL  # Drop first column with indices 
cor_DCCgarch_vol$X <- NULL  
cor_KNN5_pearson_vol$X <- NULL
cor_KNN5_kendall_vol$X <- NULL
cor_KNN_idw_pearson_vol$X <- NULL
cor_KNN_idw_kendall_vol$X <- NULL

# Initialise variables
N <- ncol(volatility_data_vol)  # Number of assets under consideration

# Compute portfolio conditional covariances
sigma_DCCgarch_vol <- sigma_vec_portfolio(volatility_data_vol, cor_DCCgarch_vol) 
sigma_KNN5_pearson_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN5_pearson_vol)
sigma_KNN5_kendall_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN5_kendall_vol)
sigma_KNN_idw_pearson_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN_idw_pearson_vol)
sigma_KNN_idw_kendall_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN_idw_kendall_vol)

# Value-at-Risk Computation 
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)
dcc_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_vol)
knn5_pearson_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_vol)
knn5_kendall_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_vol)
knn_idw_pearson_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_vol)
knn_idw_kendall_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_vol)

# Value-at-Risk Backtesting   
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns


backtest_dccGarch_vol <- pof_ind_test(VaR_est=dcc_VaR_vol)
backtest_KNN5_pearson_vol <- pof_ind_test(VaR_est=knn5_pearson_VaR_vol)
backtest_KNN5_kendall_vol <- pof_ind_test(VaR_est=knn5_kendall_VaR_vol)
backtest_KNN_idw_pearson_vol <- pof_ind_test(VaR_est=knn_idw_pearson_VaR_vol)
backtest_KNN_idw_kendall_vol <- pof_ind_test(VaR_est=knn_idw_kendall_VaR_vol)

View(backtest_dccGarch_vol)
View(backtest_KNN5_pearson_vol)
View(backtest_KNN5_kendall_vol)
View(backtest_KNN_idw_pearson_vol)
View(backtest_KNN_idw_kendall_vol)

####################################################################################################
######                               Alternative data sample                                 #######
####################################################################################################
# Period: 1987-2001
df <- read.csv("DJI30_returns_1987_2001.csv")

N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
#data <- df[-c(1:10), ]  # Data sample 2/1/2004-31/12/2008
data <- df[-c(1:9), ]  # Data sample 30/3/1987-31/12/2001
rownames(data) <- seq(length=nrow(data))  # reset index
data$X <- NULL  # Drop first column with indices 
data$Date <- NULL
View(data)

T <- 500  # Length out-of-sample test set
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1))  

dccGarch_mvnorm <- dcc_garch_modeling(data=a_t, distribution.model="norm", distribution="mvnorm")
#dccGarch_mvt <-  dcc_garch_modeling(data=a_t, distribution.model="std", distribution="mvt") 

# Add column names to file containing conditional correlations
correlation_KNNpearson <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_stable.csv")
correlation_KNNpearson$X <- NULL
colnames(dccGarch_mvt$R_t_file) <- c(colnames(correlation_KNNpearson))[1:(N*(N-1)/2)]
# Write matrices containing time-varying volatilies and correlations to csv file
write.csv(dccGarch_mvnorm$D_t_file, file="volatilities_mvnorm_DJI30_1994_1995_rol.csv")
write.csv(dccGarch_mvnorm$R_t_file, file="cor_DCC_mvnorm_DJI30_1994_1995_rol.csv")


####################################################################################################
######                               Tranquil Market Conditions                              #######
####################################################################################################
df <- read.csv("DJI30_returns_1987_2001.csv")
df <- df[1:2224,]
N <- 30  # number of assets under consideration
w <- c(rep(1/N, N))  # asset weight vector (assume equal weights)
data <- df[-c(1:10), ]  # Data sample 2/1/2004-31/12/2008
rownames(data) <- seq(length=nrow(data))  # reset index
data$X <- NULL  # Drop first column with indices 
data$Date <- NULL

#T <- 252  # Length out-of-sample test set
T <- 504
a_t <- data - rep(colMeans(data), rep.int(nrow(data), ncol(data)))  # r_t - mu_t = a_t = epsilon_t
t <- c((nrow(data)-T):(nrow(data)-1))   # c(1710:2213)

dccGarch_mvt <-  dcc_garch_modeling(data=a_t, distribution.model="std", distribution="mvt") 
dccGarch_mvnorm <- dcc_garch_modeling(data=a_t, distribution.model="norm", distribution="mvnorm")

####    Nearest Neighbors Algorithm   
# Load conditional correlations and volatilities data
volatility_data_vol <- read.csv(file="volatilities_mvt_DJI30_1994_1995_ext.csv")
cor_DCCgarch_vol <- read.csv(file="cor_DCC_mvt_DJI30_1994_1995_ext.csv")
cor_KNN5_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn5_pearson_10_DJI30_1994_1995.csv")
cor_KNN5_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn5_kendall_10_DJI30_1994_1995.csv")
cor_KNN_idw_pearson_vol <- read.csv(file="pearson/pearson_cor_estimates/cor_knn_idw_pearson_10_DJI30_1994_1995.csv")
cor_KNN_idw_kendall_vol <- read.csv(file="kendall/kendall_cor_estimates/cor_knn_idw_kendall_10_DJI30_1994_1995.csv")

# Data pre-processing
volatility_data_vol$X <- NULL  # Drop first column with indices 
cor_DCCgarch_vol$X <- NULL  
cor_KNN5_pearson_vol$X <- NULL
cor_KNN5_kendall_vol$X <- NULL
cor_KNN_idw_pearson_vol$X <- NULL
cor_KNN_idw_kendall_vol$X <- NULL

# Initialise variables
N <- ncol(volatility_data_vol)  # Number of assets under consideration

# Compute portfolio conditional covariances
sigma_DCCgarch_vol <- sigma_vec_portfolio(volatility_data_vol, cor_DCCgarch_vol) 
sigma_KNN5_pearson_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN5_pearson_vol)
sigma_KNN5_kendall_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN5_kendall_vol)
sigma_KNN_idw_pearson_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN_idw_pearson_vol)
sigma_KNN_idw_kendall_vol <- sigma_vec_portfolio(volatility_data_vol, cor_KNN_idw_kendall_vol)

# Value-at-Risk Computation 
mu_portfolio_loss <- w%*%colMeans(data)  # Expected portfolio return (assumed constant through sample mean)
dcc_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_DCCgarch_vol)
knn5_pearson_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN5_pearson_vol)
knn5_kendall_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN5_kendall_vol)
knn_idw_pearson_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_pearson_vol)
knn_idw_kendall_VaR_vol <- VaR_estimates(sigma_portfolio=sigma_KNN_idw_kendall_vol)

# Value-at-Risk Backtesting   
VaR_true <- as.matrix(tail(data, T))%*%w  #  Out-of-sample realized returns


backtest_dccGarch_vol <- pof_ind_test(VaR_est=dcc_VaR_vol)
backtest_KNN5_pearson_vol <- pof_ind_test(VaR_est=knn5_pearson_VaR_vol)
backtest_KNN5_kendall_vol <- pof_ind_test(VaR_est=knn5_kendall_VaR_vol)
backtest_KNN_idw_pearson_vol <- pof_ind_test(VaR_est=knn_idw_pearson_VaR_vol)
backtest_KNN_idw_kendall_vol <- pof_ind_test(VaR_est=knn_idw_kendall_VaR_vol)

View(backtest_dccGarch_vol)
View(backtest_KNN5_pearson_vol)
View(backtest_KNN5_kendall_vol)
View(backtest_KNN_idw_pearson_vol)
View(backtest_KNN_idw_kendall_vol)


