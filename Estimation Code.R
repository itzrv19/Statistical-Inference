############################################################
# CSPALT - UGR MODEL
# Classical and Bayesian Estimation
# Progressive Type-II Censoring
############################################################

############################################################
# SECTION 1 : LIBRARIES
############################################################

rm(list = ls())
cat("\014")

library(stats)
library(numDeriv)
library(MASS)
library(coda)
library(ggplot2)

set.seed(1234)

############################################################
# SECTION 2 : GLOBAL SETTINGS
############################################################

NSIM   <- 100
N_MCMC <- 10000
BURNIN <- 2500
THIN   <- 5

ALPHA_LEVEL <- 0.05
EPS <- 1e-10

############################################################
# SECTION 3 : UGR DISTRIBUTION FUNCTIONS
############################################################

# Normal condition PDF
f1_UGR <- function(x, alpha, beta){
  
  z <- -log(x)
  
  val <- 2 * alpha * beta^2 * (z/x) *
    exp(-(beta*z)^2) *
    (1 - exp(-(beta*z)^2))^(alpha - 1)
  
  val[val < EPS] <- EPS
  
  return(val)
}

# Normal condition survival
S1_UGR <- function(x, alpha, beta){
  
  z <- -log(x)
  
  val <- (1 - exp(-(beta*z)^2))^alpha
  
  val[val < EPS] <- EPS
  
  return(val)
}

# Normal condition CDF
F1_UGR <- function(x, alpha, beta){
  
  1 - S1_UGR(x, alpha, beta)
}

# Accelerated PDF
f2_UGR <- function(x, alpha, beta, lambda){
  
  z <- -log(x)
  
  val <- 2 * alpha * beta^2 * lambda * (z/x) *
    exp(-(beta*z)^2) *
    (1 - exp(-(beta*z)^2))^(alpha*lambda - 1)
  
  val[val < EPS] <- EPS
  
  return(val)
}

# Accelerated survival
S2_UGR <- function(x, alpha, beta, lambda){
  
  z <- -log(x)
  
  val <- (1 - exp(-(beta*z)^2))^(alpha*lambda)
  
  val[val < EPS] <- EPS
  
  return(val)
}

# Accelerated CDF
F2_UGR <- function(x, alpha, beta, lambda){
  
  1 - S2_UGR(x, alpha, beta, lambda)
}

############################################################
# SECTION 4 : INVERSE CDF FUNCTIONS
############################################################

inv_F1 <- function(u, alpha, beta){
  
  exp(-sqrt(-log(1 - (1-u)^(1/alpha)))/beta)
}

inv_F2 <- function(u, alpha, beta, lambda){
  
  exp(-sqrt(-log(1 - (1-u)^(1/(alpha*lambda))))/beta)
}

############################################################
# SECTION 5 : CENSORING SCHEMES
############################################################

CS1 <- c(rep(0,29),10)
CS2 <- c(10,rep(0,29))
CS3 <- c(rep(1,10),rep(0,20))
CS4 <- c(rep(0,10),10,rep(0,19))
CS5 <- c(rep(2,5),rep(0,25))
CS6 <- c(rep(0,15),5,rep(0,14))
CS7 <- c(rep(0,5),3,rep(0,10),2,rep(0,13))

CS_LIST <- list(CS1,CS2,CS3,CS4,CS5,CS6,CS7)
CS_NAMES <- c("CS1","CS2","CS3","CS4","CS5","CS6","CS7")

############################################################
# SECTION 6 : PARAMETER SETTINGS
############################################################

PARAMETERS <- list(
  c(1.6,0.5,2),
  c(2.5,1.2,3)
)

############################################################
# SECTION 7 : PROGRESSIVE SAMPLE GENERATION
############################################################

Generate_Progressive <- function(n,m,R,
                                 alpha,beta,
                                 lambda=NULL,
                                 accelerated=FALSE){
  
  W <- runif(m)
  V <- rep(0,m)
  
  for(i in 1:m){
    
    denom <- i + sum(rev(R)[1:i])
    
    V[i] <- W[i]^(1/denom)
  }
  
  U <- rep(0,m)
  
  for(i in 1:m){
    
    U[i] <- 1 - prod(V[(m-i+1):m])
  }
  
  if(accelerated == FALSE){
    
    X <- inv_F1(U,alpha,beta)
    
  } else {
    
    X <- inv_F2(U,alpha,beta,lambda)
  }
  
  sort(X)
}

############################################################
# SECTION 8 : LOG LIKELIHOOD FUNCTION
############################################################

logLik_CSPALT <- function(par,x1,x2,S1,S2){
  
  alpha  <- par[1]
  beta   <- par[2]
  lambda <- par[3]
  
  if(alpha <= 0 || beta <= 0 || lambda <= 1){
    return(1e10)
  }
  
  d1 <- f1_UGR(x1,alpha,beta)
  d2 <- f2_UGR(x2,alpha,beta,lambda)
  
  s1 <- S1_UGR(x1,alpha,beta)
  s2 <- S2_UGR(x2,alpha,beta,lambda)
  
  if(any(!is.finite(d1)) || any(!is.finite(d2)) ||
     any(!is.finite(s1)) || any(!is.finite(s2))){
    return(1e10)
  }
  
  ll1 <- sum(log(d1)) + sum(S1*log(s1))
  ll2 <- sum(log(d2)) + sum(S2*log(s2))
  
  val <- -(ll1 + ll2)
  
  if(!is.finite(val)){
    return(1e10)
  }
  
  return(val)
}

############################################################
# SECTION 9 : MLE ESTIMATION
############################################################

MLE_CSPALT <- function(x1,x2,S1,S2){
  
  fit <- optim(
    par=c(1,0.5,2),
    fn=logLik_CSPALT,
    x1=x1,
    x2=x2,
    S1=S1,
    S2=S2,
    method="L-BFGS-B",
    lower=c(0.001,0.001,1.001)
  )
  
  est <- fit$par
  
  H <- tryCatch(
    hessian(logLik_CSPALT,
            est,
            x1=x1,
            x2=x2,
            S1=S1,
            S2=S2),
    error=function(e) diag(1,3)
  )
  
  Fisher <- tryCatch(solve(H),error=function(e) diag(NA,3))
  
  SE <- sqrt(abs(diag(Fisher)))
  
  z <- qnorm(1 - ALPHA_LEVEL/2)
  
  ACI <- cbind(
    est - z*SE,
    est + z*SE
  )
  
  colnames(ACI) <- c("Lower","Upper")
  
  list(
    estimate=est,
    Fisher=Fisher,
    SE=SE,
    ACI=ACI
  )
}

############################################################
# SECTION 10 : GAMMA PRIOR
############################################################

log_prior_gamma <- function(alpha,beta,lambda,
                            a1=2,b1=1,
                            a2=2,b2=1,
                            a3=2,b3=1){
  
  dgamma(alpha,a1,b1,log=TRUE) +
    dgamma(beta,a2,b2,log=TRUE) +
    dgamma(lambda,a3,b3,log=TRUE)
}

############################################################
# SECTION 11 : GAMMA-DIRICHLET PRIOR
############################################################

log_prior_GD <- function(alpha,beta,lambda,
                         a0=3,b0=1,
                         a1=2,a2=2,
                         a3=2,b3=1){
  
  if(alpha <=0 || beta <=0 || lambda <=1){
    return(-Inf)
  }
  
  val <-
    lgamma(a1+a2) - lgamma(a1) - lgamma(a2) +
    (a0-a1-a2)*log(alpha+beta) +
    (a1-1)*log(alpha) +
    (a2-1)*log(beta) -
    b0*(alpha+beta) +
    dgamma(lambda,a3,b3,log=TRUE)
  
  return(val)
}

############################################################
# SECTION 12 : POSTERIOR FUNCTIONS
############################################################

log_post_gamma <- function(par,x1,x2,S1,S2){
  
  alpha <- par[1]
  beta <- par[2]
  lambda <- par[3]
  
  ll <- -logLik_CSPALT(par,x1,x2,S1,S2)
  
  lp <- log_prior_gamma(alpha,beta,lambda)
  
  ll + lp
}

log_post_GD <- function(par,x1,x2,S1,S2){
  
  alpha <- par[1]
  beta <- par[2]
  lambda <- par[3]
  
  ll <- -logLik_CSPALT(par,x1,x2,S1,S2)
  
  lp <- log_prior_GD(alpha,beta,lambda)
  
  ll + lp
}


############################################################
# SECTION 13 :MH_Gibbs
############################################################

MH_Gibbs <- function(x1,x2,S1,S2,
                     prior="gamma"){
  
  chain <- matrix(0,nrow=N_MCMC,ncol=3)
  
  colnames(chain) <- c("alpha","beta","lambda")
  
  chain[1,] <- c(1,0.5,2)
  
  acc <- c(0,0,0)
  
  for(i in 2:N_MCMC){
    
    current <- chain[i-1,]
    
    ########################################################
    # ALPHA
    ########################################################
    
    alpha_prop <- abs(rnorm(1,current[1],0.05))
    
    prop1 <- c(alpha_prop,current[2],current[3])
    
    if(prior=="gamma"){
      
      logR <- log_post_gamma(prop1,x1,x2,S1,S2) -
        log_post_gamma(current,x1,x2,S1,S2)
      
    } else {
      
      logR <- log_post_GD(prop1,x1,x2,S1,S2) -
        log_post_GD(current,x1,x2,S1,S2)
    }
    
    if(log(runif(1)) < logR){
      
      current[1] <- alpha_prop
      acc[1] <- acc[1] + 1
    }
    
    ########################################################
    # BETA
    ########################################################
    
    beta_prop <- abs(rnorm(1,current[2],0.02))
    
    prop2 <- c(current[1],beta_prop,current[3])
    
    if(prior=="gamma"){
      
      logR <- log_post_gamma(prop2,x1,x2,S1,S2) -
        log_post_gamma(current,x1,x2,S1,S2)
      
    } else {
      
      logR <- log_post_GD(prop2,x1,x2,S1,S2) -
        log_post_GD(current,x1,x2,S1,S2)
    }
    
    if(log(runif(1)) < logR){
      
      current[2] <- beta_prop
      acc[2] <- acc[2] + 1
    }
    
    ########################################################
    # LAMBDA
    ########################################################
    
    lambda_prop <- rnorm(1,current[3],0.02)
    
    if(lambda_prop <= 1){
      
      lambda_prop <- current[3]
    }
    
    prop3 <- c(current[1],current[2],lambda_prop)
    
    if(prior=="gamma"){
      
      logR <- log_post_gamma(prop3,x1,x2,S1,S2) -
        log_post_gamma(current,x1,x2,S1,S2)
      
    } else {
      
      logR <- log_post_GD(prop3,x1,x2,S1,S2) -
        log_post_GD(current,x1,x2,S1,S2)
    }
    
    if(log(runif(1)) < logR){
      
      current[3] <- lambda_prop
      acc[3] <- acc[3] + 1
    }
    
    chain[i,] <- current
  }
  
  ##########################################################
  # BURN-IN REMOVAL + THINNING
  ##########################################################
  
  chain_post <- chain[(BURNIN+1):N_MCMC,]
  
  chain_post <- chain_post[
    seq(1,nrow(chain_post),THIN),
  ]
  
  ##########################################################
  # ACCEPTANCE RATES
  ##########################################################
  
  cat("Acceptance Rates :\n")
  print(acc/N_MCMC)
  
  list(
    chain=chain_post,
    acceptance=acc/N_MCMC
  )
}


############################################################
# SECTION 14 : BAYES ESTIMATION
############################################################

Bayes_SELF <- function(chain){
  
  apply(chain,2,mean)
}

Bayes_LINEX <- function(chain,s=2){
  
  c(
    -(1/s)*log(mean(exp(-s*chain[,1]))),
    -(1/s)*log(mean(exp(-s*chain[,2]))),
    -(1/s)*log(mean(exp(-s*chain[,3])))
  )
}

############################################################
# SECTION 15 : BAYESIAN CREDIBLE INTERVALS
############################################################

Bayes_CI <- function(chain){
  
  apply(chain,2,quantile,c(0.025,0.975))
}

HPD_CI <- function(chain){
  
  HPDinterval(as.mcmc(chain))
}


############################################################
# SECTION 16 : PERFORMANCE MEASURES
############################################################

Compute_MSE <- function(est,true){
  
  mean((est-true)^2)
}

Compute_ABS <- function(est,true){
  
  mean(abs(est-true))
}

Compute_CP <- function(lower,upper,true){
  
  mean((lower <= true) & (true <= upper))
}

Compute_AL <- function(lower,upper){
  
  mean(upper-lower)
}

############################################################
# SECTION 17 : RESULT TABLES
############################################################
TABLE_MLE <- data.frame()

TABLE_ACI <- data.frame()

TABLE_SELF_GAMMA <- data.frame()
TABLE_LINEX2_GAMMA <- data.frame()
TABLE_LINEXN2_GAMMA <- data.frame()

TABLE_SELF_GD <- data.frame()
TABLE_LINEX2_GD <- data.frame()
TABLE_LINEXN2_GD <- data.frame()

TABLE_BCI <- data.frame()

TABLE_COMPARE <- data.frame()

############################################################
# SECTION 18 : SIMULATION
############################################################

for(setting in 1:length(PARAMETERS)){
  
  alpha_true <- PARAMETERS[[setting]][1]
  beta_true <- PARAMETERS[[setting]][2]
  lambda_true <- PARAMETERS[[setting]][3]
  
  for(cs in 1:length(CS_LIST)){
    
    cat("RUNNING : Setting",setting,
        CS_NAMES[cs],"\n")
    
    R <- CS_LIST[[cs]]
    
    ########################################################
    #MATRICES
    ########################################################
    
    MLE_STORE <- matrix(0,NSIM,3)
    
    SELF_G_STORE <- matrix(0,NSIM,3)
    LINEX2_G_STORE <- matrix(0,NSIM,3)
    LINEXN2_G_STORE <- matrix(0,NSIM,3)
    
    SELF_GD_STORE <- matrix(0,NSIM,3)
    LINEX2_GD_STORE <- matrix(0,NSIM,3)
    LINEXN2_GD_STORE <- matrix(0,NSIM,3)
    
    ########################################################
    # INTERVAL
    ########################################################
    
    ACI_LOW <- matrix(0,NSIM,3)
    ACI_UP  <- matrix(0,NSIM,3)
    
    BCI_LOW <- matrix(0,NSIM,3)
    BCI_UP  <- matrix(0,NSIM,3)
    
    ########################################################
    # MAIN SIMULATION LOOP
    ########################################################
    
    for(sim in 1:NSIM){
      
      ######################################################
      # DATA GENERATION
      ######################################################
      
      x1 <- Generate_Progressive(
        n=40,m=30,R=R,
        alpha_true,beta_true,
        accelerated=FALSE
      )
      
      x2 <- Generate_Progressive(
        n=40,m=30,R=R,
        alpha_true,beta_true,
        lambda_true,
        accelerated=TRUE
      )
      
      ######################################################
      # MLE
      ######################################################
      
      mle <- MLE_CSPALT(x1,x2,R,R)
      
      MLE_STORE[sim,] <- mle$estimate
      
      ######################################################
      #ACI
      ######################################################
      
      ACI_LOW[sim,] <- mle$ACI[,1]
      ACI_UP[sim,]  <- mle$ACI[,2]
      
      ######################################################
      # GAMMA PRIOR
      ######################################################
      
      MH_G <- MH_Gibbs(x1,x2,R,R,"gamma")
      
      SELF_G_STORE[sim,] <- Bayes_SELF(MH_G$chain)
      
      LINEX2_G_STORE[sim,] <- Bayes_LINEX(MH_G$chain,2)
      
      LINEXN2_G_STORE[sim,] <- Bayes_LINEX(MH_G$chain,-2)
      
      ######################################################
      # BCI
      ######################################################
      
      BCI_TEMP <- Bayes_CI(MH_G$chain)
      
      BCI_LOW[sim,] <- BCI_TEMP[1,]
      BCI_UP[sim,]  <- BCI_TEMP[2,]
      
      ######################################################
      # GAMMA-DIRICHLET PRIOR
      ######################################################
      
      MH_GD <- MH_Gibbs(x1,x2,R,R,"GD")
      
      SELF_GD_STORE[sim,] <- Bayes_SELF(MH_GD$chain)
      
      LINEX2_GD_STORE[sim,] <- Bayes_LINEX(MH_GD$chain,2)
      
      LINEXN2_GD_STORE[sim,] <- Bayes_LINEX(MH_GD$chain,-2)
      
      ######################################################
      # SIM COUNT
      ######################################################
      
      if(sim %% 5 == 0){
        
        cat("Simulation :",sim,"\n")
      }
    }
    
    ########################################################
    # MLE TABLE
    ########################################################
    
    TABLE_MLE <- rbind(
      TABLE_MLE,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(MLE_STORE),
        
        MSE=c(
          Compute_MSE(MLE_STORE[,1],alpha_true),
          Compute_MSE(MLE_STORE[,2],beta_true),
          Compute_MSE(MLE_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # ACI TABLE
    ########################################################
    
    TABLE_ACI <- rbind(
      TABLE_ACI,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        CP=c(
          Compute_CP(ACI_LOW[,1],ACI_UP[,1],alpha_true),
          Compute_CP(ACI_LOW[,2],ACI_UP[,2],beta_true),
          Compute_CP(ACI_LOW[,3],ACI_UP[,3],lambda_true)
        ),
        
        AL=c(
          Compute_AL(ACI_LOW[,1],ACI_UP[,1]),
          Compute_AL(ACI_LOW[,2],ACI_UP[,2]),
          Compute_AL(ACI_LOW[,3],ACI_UP[,3])
        )
      )
    )
    
    ########################################################
    # BCI TABLE
    ########################################################
    
    TABLE_BCI <- rbind(
      TABLE_BCI,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        CP=c(
          Compute_CP(BCI_LOW[,1],BCI_UP[,1],alpha_true),
          Compute_CP(BCI_LOW[,2],BCI_UP[,2],beta_true),
          Compute_CP(BCI_LOW[,3],BCI_UP[,3],lambda_true)
        ),
        
        AL=c(
          Compute_AL(BCI_LOW[,1],BCI_UP[,1]),
          Compute_AL(BCI_LOW[,2],BCI_UP[,2]),
          Compute_AL(BCI_LOW[,3],BCI_UP[,3])
        )
      )
    )
    
    ########################################################
    # SELF GAMMA TABLE
    ########################################################
    
    TABLE_SELF_GAMMA <- rbind(
      TABLE_SELF_GAMMA,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(SELF_G_STORE),
        
        MSE=c(
          Compute_MSE(SELF_G_STORE[,1],alpha_true),
          Compute_MSE(SELF_G_STORE[,2],beta_true),
          Compute_MSE(SELF_G_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # LINEX2 GAMMA TABLE
    ########################################################
    
    TABLE_LINEX2_GAMMA <- rbind(
      TABLE_LINEX2_GAMMA,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(LINEX2_G_STORE),
        
        MSE=c(
          Compute_MSE(LINEX2_G_STORE[,1],alpha_true),
          Compute_MSE(LINEX2_G_STORE[,2],beta_true),
          Compute_MSE(LINEX2_G_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # LINEX-2 GAMMA TABLE
    ########################################################
    
    TABLE_LINEXN2_GAMMA <- rbind(
      TABLE_LINEXN2_GAMMA,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(LINEXN2_G_STORE),
        
        MSE=c(
          Compute_MSE(LINEXN2_G_STORE[,1],alpha_true),
          Compute_MSE(LINEXN2_G_STORE[,2],beta_true),
          Compute_MSE(LINEXN2_G_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # SELF GD TABLE
    ########################################################
    
    TABLE_SELF_GD <- rbind(
      TABLE_SELF_GD,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(SELF_GD_STORE),
        
        MSE=c(
          Compute_MSE(SELF_GD_STORE[,1],alpha_true),
          Compute_MSE(SELF_GD_STORE[,2],beta_true),
          Compute_MSE(SELF_GD_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # LINEX2 GD TABLE
    ########################################################
    
    TABLE_LINEX2_GD <- rbind(
      TABLE_LINEX2_GD,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(LINEX2_GD_STORE),
        
        MSE=c(
          Compute_MSE(LINEX2_GD_STORE[,1],alpha_true),
          Compute_MSE(LINEX2_GD_STORE[,2],beta_true),
          Compute_MSE(LINEX2_GD_STORE[,3],lambda_true)
        )
      )
    )
    
    ########################################################
    # LINEX-2 GD TABLE
    ########################################################
    
    TABLE_LINEXN2_GD <- rbind(
      TABLE_LINEXN2_GD,
      data.frame(
        Setting=setting,
        CS=CS_NAMES[cs],
        Parameter=c("alpha","beta","lambda"),
        
        AE=colMeans(LINEXN2_GD_STORE),
        
        MSE=c(
          Compute_MSE(LINEXN2_GD_STORE[,1],alpha_true),
          Compute_MSE(LINEXN2_GD_STORE[,2],beta_true),
          Compute_MSE(LINEXN2_GD_STORE[,3],lambda_true)
        )
      )
    )
  }
}

############################################################
# SECTION 19 : EXPORT TABLES TO CSV
############################################################

write.csv(TABLE_MLE,
          "Table1_MLE.csv",
          row.names=FALSE)

write.csv(TABLE_ACI,
          "Table2_ACI.csv",
          row.names=FALSE)

write.csv(TABLE_SELF_GAMMA,
          "Table3_SELF_Gamma.csv",
          row.names=FALSE)

write.csv(TABLE_LINEX2_GAMMA,
          "Table4_LINEX2_Gamma.csv",
          row.names=FALSE)

write.csv(TABLE_LINEXN2_GAMMA,
          "Table5_LINEXN2_Gamma.csv",
          row.names=FALSE)

write.csv(TABLE_BCI,
          "Table6_BCI.csv",
          row.names=FALSE)

write.csv(TABLE_SELF_GD,
          "Table7_SELF_GD.csv",
          row.names=FALSE)

write.csv(TABLE_LINEX2_GD,
          "Table8_LINEX2_GD.csv",
          row.names=FALSE)

write.csv(TABLE_LINEXN2_GD,
          "Table9_LINEXN2_GD.csv",
          row.names=FALSE)
############################################################
# SECTION 20 : REAL DATA ANALYSIS
############################################################

x_normal <- exp(-c(7.74,17.05,20.46,21.02,22.66,
                   43.40,47.30,139.07))

x_acc <- exp(-c(0.27,0.40,0.69,0.79,2.75,
                3.91,9.88,13.95,15.93,27.80))

S_real1 <- rep(0,length(x_normal))
S_real2 <- rep(0,length(x_acc))

REAL_MLE <- MLE_CSPALT(x_normal,x_acc,S_real1,S_real2)

REAL_GAMMA <- MH_Gibbs(x_normal,x_acc,S_real1,S_real2,"gamma")

REAL_SELF <- Bayes_SELF(REAL_GAMMA$chain)
REAL_LINEX <- Bayes_LINEX(REAL_GAMMA$chain,2)
REAL_BCI <- Bayes_CI(REAL_GAMMA$chain)

REAL_TABLE <- data.frame(
  Parameter=c("alpha","beta","lambda"),
  MLE=REAL_MLE$estimate,
  SELF=REAL_SELF,
  LINEX=REAL_LINEX,
  BCI_Lower=REAL_BCI[1,],
  BCI_Upper=REAL_BCI[2,]
)

write.csv(REAL_TABLE,
          "RealData_Results.csv",
          row.names=FALSE)

############################################################
# SECTION 21 : TRACE PLOTS
############################################################

png("TracePlot_Alpha.png")
plot(REAL_GAMMA$chain[,1],type='l',main='Trace Plot Alpha')
dev.off()

png("TracePlot_Beta.png")
plot(REAL_GAMMA$chain[,2],type='l',main='Trace Plot Beta')
dev.off()

png("TracePlot_Lambda.png")
plot(REAL_GAMMA$chain[,3],type='l',main='Trace Plot Lambda')
dev.off()

############################################################
# CODE KHATAM
############################################################

