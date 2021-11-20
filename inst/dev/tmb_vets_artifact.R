library(TMB)
library(mvtnorm)
library(zoo)
library(xts)
library(tsvets)
library(tsvetsad)
wdir <- "~/github/tsvetsad/inst/dev/"
source(paste0("~/github/tsvetsad/R/estimate.R"))
compile(paste0(wdir,"tmbvets.cpp"))
dyn.load(dynlib(paste0(wdir,"tmbvets")))

# sigma <- matrix(c(4,2,2,3), ncol=2)
# x <- rmvnorm(n=500, mean=c(1,2), sigma=sigma)
# xx <- cov(x)

data("dji30ret", package = "rugarch")
# xx <- cov(dji30ret)

xx <- scale(coredata(dji30ret), scale = FALSE)
xx <- as.matrix(xx)
obj <- MakeADFun(data = list(E = xx), parameters = list(rho = 1), DLL="testing")
obj$report()$llh


R <- matrix(0.6, ncol(x), ncol(x))
diag(R) <- 1
S <- matrix(0, ncol(x), ncol(x))
diag(S) <- apply(x, 2, sd)

sum(abs((S %*% R %*% t(S)) - obj$report()$S))

#####################################################
x <- tsdatasets::austretail
x <- x[,c(1,2,5,6)]
XX <- xts(matrix(0, ncol = 2, nrow = nrow(x)), index(x))
XX[50,1] <- 1
XX[60,2] <- 1
V <- matrix(0, ncol = 2, nrow = 4)
V[] <- 1
# last columns of Amat are the inital states which for now we keep
spec <- vets_modelspec(y = x, level = "diagonal", slope = "diagonal", seasonal = "diagonal",
                       damped = "full", frequency = 12, lambda = NULL, xreg = XX, xreg_include = V,
                       dependence = "shrink")
mod <- estimate(spec, solver = "nlminb")
L <- prepare_inputs_vets(spec)
pars <- L$par_list
pars$pars <- pars$pars + 1e-3
names(pars$pars) <- L$parnames_estimate
obj <- MakeADFun(data = L$data, parameters = pars, DLL = "tmbvets")

obj$report()$A
obj$report()$F
obj$report()$beta
obj$report()$E
obj$report()$loglik
obj$report()$Yhat
obj$fn()
obj$gr()
obj$report()

env <- L
L$loglik <- 1
lb <- env$lower
ub <- env$upper
pars <- env$par_list$pars
pars <- pars + 1e-2

vetsenv <- new.env()
vetsenv$loglik <- 1
vetsenv$tmb_names <- L$tmb_names



sol <- nlminb(start = pars, objective = L$llh_fun, gradient = L$grad_fun, hessian = L$hess_fun,
              control = list(trace = 1), lower = lb, upper = ub, vetsenv = vetsenv, fun = obj)



L$llh_fun()
pars <- 1:length(L$par_list$pars)
L$data$pindex[L$data$mindex] <- pars[L$data$pindex]
A <- matrix(L$data$amat, L$data$adim[1], L$data$adim[2])
A[,1:L$data$adim[2]]


env <- L$data
pars <- pars$pars
env$amat[env$mindex + 1] <- pars[env$pindex + 1]

Amat <- matrix(env$amat, env$adim[1], env$adim[2])[,1:env$adim[3]]
Fmat <- env$fmat
States <- env$States
if (env$phiindex[1] > 0) {
    Fmat[env$findex+1] <- env$amat[(env$phiindex[1]+1):((env$phiindex[1]) + env$phiindex[2])]
}
Fmat <- matrix(Fmat, env$fdim[1], env$fdim[2])
if (env$xindex[1] >= 0) {
    beta <- env$amat[(env$xindex[1]+1):((env$xindex[1]) + env$xindex[2])]
    beta <- matrix(beta, env$betadim)
    xreg <- env$xreg
} else {
    beta <- matrix(1, ncol = env$model[2], nrow = 1)
    xreg <- env$xreg
}
