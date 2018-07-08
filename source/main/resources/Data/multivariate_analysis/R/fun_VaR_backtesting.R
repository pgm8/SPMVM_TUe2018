####################################################################################################
######                      DCC-GARCH Student-t distributed errors (rugarch)                 #######
####################################################################################################
# Two-stage quasi-likelihood function to find parameters under multivariate Student-t distributed errors
dcc_garch_modeling <- function(data=a_t, t=t, distribution.model="norm", distribution="mvnorm") {
  D_t_file <- matrix(NaN, T, N)
  colnames(D_t_file) <- c(colnames(data))[1:N]
  R_t_file <- matrix(NaN, T, N*(N-1)/2)
  cl = makePSOCKcluster(10)
  for (i in seq_along(t)) {  # for (i in seq_along(t))  # tranquil: {1720:2223}, voaltile: {3225:3734}
    tic <- Sys.time()
    data_train <- data[1:t[i],1:N]  # Rolling forward: data[i:t[i],1:N]  
    # Specify univariate GARCH(1,1) with marginal Student-t dist. errors for each component series
    if (distribution.model=="norm"){
    univ_garch_spec <- ugarchspec(variance.model=list(model="fGARCH", submodel="GARCH", garchOrder=c(1, 1)),
                                  mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model=distribution.model)
    } else{
    univ_garch_spec <- ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1, 1)),
                                  mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model=distribution.model)
    }
    multi_univ_garch_spec <- multispec(replicate(N,univ_garch_spec))
    # Specify DCC-GARCH(1,1) with multivariate Student-t errors for stand. residual series
    #dcc_spec <- dccspec(multi_univ_garch_spec, dccOrder=c(1,1), model="DCC", distribution=distribution)
    # Fit component series
    fit.multi_garch <- multifit(multi_univ_garch_spec, data_train, cluster=cl)
    # Estimate DCC-GARCH(1,1) mvt model
    #fit.dcc = dccfit(dcc_spec, data=data_train, solver='solnp', cluster=cl, fit.control=list(eval.se = FALSE), fit=fit.multi_garch)
    # Save conditional volatilities and correlations
    uniGarch.forc <- multiforecast(fit.multi_garch, n.ahead=1, n.roll=0, cluster=cl) # T+1 forecasts conditional volatilities
    #dcc.forc <- dccforecast(fit.dcc, n.ahead=1, n.roll=0, cluster=cl)  # T+1 forecasts
    D_t_file[i, ] <- as.numeric(sigma(uniGarch.forc))
    #D_t_file[i, ] <- as.numeric(sigma(dcc.forc))
    #R_t_file[i, ] <- t(dcc.forc@mforecast[["R"]][[1]][,,1])[lower.tri(t(dcc.forc@mforecast[["R"]][[1]][,,1]),diag=FALSE)]
    print(i)
    print(t[i])
    print(Sys.time()-tic)
  }
  stopCluster(cl)
  #return(list("D_t_file"=D_t_file, "R_t_file"=R_t_file))
  return(list("D_t_file"=D_t_file))
}
####################################################################################################
######                                Value-at-Risk Estimation                               #######
####################################################################################################
cor_mat <- function(cor_vec, dim) { # function assumes row matrix indexing
  # cor_vec := vector with uppertriangular correlation values index by row
  R_t <- matrix(0,dim,dim)
  R_t[lower.tri(R_t, diag=FALSE)] <- as.numeric(cor_vec)
  R_t <- t(R_t)
  R_t[lower.tri(R_t, diag=FALSE)] <- t(R_t)[lower.tri(R_t, diag=FALSE)]
  diag(R_t) <- rep(1, times=dim)
  if (isSymmetric(R_t) == FALSE) {
    print("Houston we have a problem, our correlation matrix is not symmetric!")
  }
  return(R_t)
}

cov_mat_portfolio <- function(vol_vec, cor_vec) {
  D_t <- diag(as.numeric(vol_vec))
  R_t <- cor_mat(cor_vec, dim=length(vol_vec))   # Symmetric matrix from upper triangular matrix
  H_t <- D_t %*% R_t %*% D_t  # Conditional covariance matrix 
  return(H_t)
}

sigma_vec_portfolio <- function(volatility_matrix, cor_matrix, T=T, w=w) {
  # volatility_matrix := matrix with conditional volatilities
  # cor_matrix := matrix with conditional correlations
  sigma_t <- rep(NaN,T)
  for (i in 1:T) {
    H_t <- cov_mat_portfolio(volatility_matrix[i,], cor_matrix[i,])
    sigma_t[i] <- sqrt(t(w)%*%H_t%*%w)  # Portfolio sdv
  }
  return(sigma_t)
}

VaR_estimates_alt <- function(volatility_matrix, cor_matrix, mu=mu_portfolio_loss, cl=alpha) {
  VaR_mat <- matrix(data=NaN, nrow=T, ncol=length(cl))
  colnames(VaR_mat) <- 1-cl  #  Set column names to corresponding conf. level VaR estimates
  for (i in 1:3) {  # dim(vol_data_tranquil_mvnorm)[1]
    print(i)
    H_t <- cov_mat_portfolio(as.numeric(volatility_matrix[i,]), as.numeric(cor_matrix[i,]))
    for (a in cl) {
      VaR_mat[i, toString(a)] <- mu_portfolio_loss + qmvnorm(p=a, interval=c(-5, 5), tail="lower", mu=c(rep(0, 30)), sigma=H_t)$quantile
    }
  }
  return(VaR_mat)
}

VaR_estimates <- function(sigma_portfolio, mu=mu_portfolio_loss, cl=alpha) {
  VaR_mat <- matrix(data=NaN, nrow=T, ncol=length(cl))
  colnames(VaR_mat) <- 1-cl  #  Set column names to corresponding conf. level VaR estimates
  for (i in 1:length(sigma_portfolio)) {
    for (a in cl) {
      VaR_mat[i, toString(a)] <- mu+sigma_portfolio[i]*qnorm(1-a)
    }
  }
  return(VaR_mat)
}

####################################################################################################
######                                Value-at-Risk Backtesting                              #######
####################################################################################################
# Kupiec's Unconditional Coverage Test & Christoffersen Markov Test for independence 
uc_ind_test <- function(VaR_est, cl=alpha) {
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
    for (i in 2:nrow(VaR_est)) {  
      T_00 <- ifelse(I[i-1]==0 & I[i]==0, T_00+1, T_00)
      T_10 <- ifelse(I[i-1]==1 & I[i]==0, T_10+1, T_10)
      T_01 <- ifelse(I[i-1]==0 & I[i]==1, T_01+1, T_01)
      T_11 <- ifelse(I[i-1]==1 & I[i]==1, T_11+1, T_11)
    }
    # Compute Markov transition probabilities
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


regions_uc_test_approx <- function(T, alpha) {
  "Kupiec test uses Binomial distribution to construct a likelihood ratio, which is APPROXIMATELY
  centrally chi-squared with one degree of freedom, assuming H0. By considering the -2Log() of the
  likelihood ratio and setting this equal to epsilon quantile of chi^2(1,0), denoted by y, one can 
  find the roots of the function. Finding the roots equals finding the non-rejection interval values,
  approximately."
  # T: length backtest period
  # alpha: confidence level VaR
  # y: significance level test = 95th percentile of the Chi-Squared distribution with 1 degree of freedom
  y <- qchisq(.95, df=1)
  result <- uniroot.all(function(x, y, t=T, a=alpha) {2*log(((t-x)/(a*T))^(t-x)*((x/((1-a)*t))^x))-y}, 
          y=y, lower=0, upper=T)
  return(list("lb"=floor(result[1]), "ub"=ceiling(result[2])))
}
  

# X ~ Binomial(500, 0.05) 
regions_uc_test <- function(T=T, alpha=alpha, eps=0.05) {
  " Use binomial distribution (hit sequence) to construct non-rejection regions."
  # T: length backtest period
  # alpha: confidence level VaR
  # eps: significance level test 
  q <- 1-alpha
  lb_init <- qbinom(p=(eps/2), size=T, prob=q)  # 1: P(X < lb_init) <= eps/2
  # Find maximum x1 for which (1) holds 
  if(pbinom(q=lb_init, size=T, prob=q) > (eps/2)) {
    lb_init <- lb_init-1
  }
  ub_init <- qbinom(p=(eps/2), size=T, prob=q, lower.tail=FALSE)  # 35
  if(pbinom(q=ub_init, size=T, prob=q, lower.tail=FALSE) > (eps/2)) {
    ub_init <- ub_init+1
  }
  # [a, b]
  op_1 <- pbinom(q=lb_init, size=T, prob=q) + pbinom(q=ub_init, size=T, prob=q, lower.tail=FALSE)
  lb_opt <- lb_init
  ub_opt <- ub_init
  op_opt <- op_1
  # [a+1, b]
  op_2 <- pbinom(q=lb_init+1, size=T, prob=q) + pbinom(q=ub_init, size=T, prob=q, lower.tail=FALSE)
  if ((op_2 <= eps) & (eps-op_2 <= eps-op_opt)) {
    lb_opt <- lb_init+1
    ub_opt <- ub_init
    op_opt <- op_2
  }
  # [a, b-1]
  op_3 <- pbinom(q=lb_init, size=T, prob=q) + pbinom(q=ub_init-1, size=T, prob=q, lower.tail=FALSE)
  if ((op_3 <= eps) & (eps-op_3 <= eps-op_opt)) {  
    lb_opt <- lb_init
    ub_opt <- ub_init-1
    op_opt <- op_3
  }
  return(list("lb"=lb_opt+1, "ub"=ub_opt))
}
