prepare_inputs_vets <- function(spec)
{
    d <- list()
    env <- spec$vets_env
    d$pindex <- env$parameter_index - 1
    d$mindex <- env$matrix_index - 1
    amat <- env$Amat
    d$adim <- c(dim(amat), env$Amat_index[2])
    d$amat <- as.vector(amat)
    if (any(is.na(d$amat))) {
        d$amat[which(is.na(d$amat))] <- 0.0
    }
    if (any(is.na(env$ymat))) {
        d$Y <- na.fill(env$ymat, fill = 0)
    } else {
        d$Y <- env$ymat
    }
    fmat <- env$Fmat
    d$fdim <- dim(fmat)
    d$fmat <- as.vector(fmat)
    if (any(is.na(d$fmat))) {
        # c++ indexing
        d$findex <- which(is.na(d$fmat)) - 1
        d$fmat[which(is.na(d$fmat))] <- 0
        d$phiindex <- c((env$Phi_index[1]-1)*d$adim[1] + 1,(d$adim[1]*env$Phi_index[2])) - 1
        d$phiindex <- c(d$phiindex[1], length(d$phiindex[1]:d$phiindex[2]))
    } else {
        d$findex <- c(0,0)
        d$phiindex <- env$Phi_index
    }
    d$states <- env$States
    d$xindex <- env$X_index
    if (d$xindex[1] > 0) {
        d$xindex <- c((env$X_index[1] - 1) * d$adim[1] + 1,(d$adim[1]*env$X_index[2])) - 1
        d$xindex <- c(d$xindex[1], length(d$xindex[1]:d$xindex[2]))
        d$betadim <- c(env$model[2], nrow(env$xreg))
    } else {
        d$betadim <- c(env$model[2],1)
    }
    # d$good
    d$X <- coredata(env$xreg)
    d$G <- as.matrix(env$Gmat)
    d$H <- as.matrix(env$Hmat)
    d$vmodel <- env$model
    d$good <- t(spec$target$good_matrix)
    d$select <- t(spec$target$good_index)
    # switch models
    llh_fun <- function(pars, fun, vetsenv) {
        names(pars) <- vetsenv$tmb_names
        lik <- fun$fn(pars)
        flag <- fun$report()$stability_test
        if (flag == 0 | is.na(lik)) {
            lik <- vetsenv$lik + 0.25 * abs(vetsenv$lik)
            vetsenv$lik <- lik
        } else {
            vetsenv$lik <- lik
        }
        return(lik)
    }

    # map
    grad_fun <- function(pars, fun, vetsenv)
    {
        names(pars) <- vetsenv$tmb_names
        grad <- fun$gr(pars)
        if (any(is.na(grad))) grad[which(is.na(grad))] <- 1e12
        return(matrix(grad, nrow = 1))
    }

    hess_fun <- function(pars, fun, vetsenv)
    {
        names(pars) <- vetsenv$tmb_names
        fun$he(pars)
    }
    par_list <- list(pars = env$pars)

    d$model <- spec$dependence$type
    L <- list(data = d, par_list = par_list, map = list(), lower = env$lower_index, upper = env$upper_index,
              llh_fun = llh_fun, grad_fun = grad_fun, hess_fun = hess_fun,
              parnames_estimate = env$parnames, tmb_names = env$parnames, parnames_all = env$parnames)
    return(L)
}

#' Estimates a VETS model given a specification object using maximum likelihood and autodiff
#'
#' @param object An object of class tsvets.spec.
#' @param solver Only \dQuote{nlminb} currently supported.
#' @param use_hessian Whether to include the hessian in the calculation.
#' @param control Solver control parameters.
#' @param ... additional parameters passed to the estimation function
#' @return An list of coefficients and other information.
#' @details This function is not expected to be used by itself but rather as a plugin
#' to be called from the estimate method of the tsvets package.
#' @export estimate_ad.tsvets.spec
#' @aliases estimate_ad
#' @export
#'
estimate_ad.tsvets.spec <- function(object, solver = "nlminb", control = list(trace = 0, eval.max = 300, iter.max = 500), use_hessian = FALSE, ...)
{
    spec_list <- prepare_inputs_vets(object)
    other_opts <- list(...)
    if (!is.null(other_opts$silent)) {
        silent <- other_opts$silent
    } else {
        silent <- TRUE
    }
    names(spec_list$par_list$pars) <- spec_list$parnames_estimate
    fun <- try(MakeADFun(data = spec_list$data, hessian = use_hessian, parameters = spec_list$par_list, DLL = "tsvetsad_TMBExports",
                         trace = FALSE, silent = silent), silent = FALSE)
    fun$env$tracemgc <- FALSE
    if (inherits(fun, 'try-error')) {
        stop("\nestimate_ad found an error. Please use non ad version of estimator and contact developer with reproducible example.")
    }
    vetsenv <- new.env()
    vetsenv$lik <- 1
    vetsenv$grad <- NULL
    vetsenv$parameter_names <- spec_list$parnames_all
    vetsenv$parnames_estimate <- spec_list$parnames_estimate
    vetsenv$tmb_names <- spec_list$tmb_names
    ## if (solver != "nlminb") warning("\nonly nlminb solver currently supported for issm with autodiff. Using nlminb.")
    if (use_hessian) hessian <- spec_list$hess_fun else hessian <- NULL

    if (solver == "nlminb") {
        sol <- nlminb(start = fun$par, objective = spec_list$llh_fun,
                      gradient = spec_list$grad_fun, hessian = hessian,
                      lower = spec_list$lower, upper = spec_list$upper,  control = control,
                      fun = fun, vetsenv = vetsenv)
    } else {
        sol <- optim(par = fun$par, fn = spec_list$llh_fun, gr = spec_list$grad_fun,
                     lower = spec_list$lower, upper = spec_list$upper,  control = control,
                     method = "L-BFGS-B", fun = fun, vetsenv = vetsenv)
    }
    pars <- sol$par
    names(pars) <- vetsenv$tmb_names
    llh <- spec_list$llh_fun(pars, fun, vetsenv)
    gradient <- spec_list$grad_fun(pars, fun, vetsenv)
    hessian <- spec_list$hess_fun(pars, fun, vetsenv)
    names(pars) <- vetsenv$parnames_estimate
    colnames(gradient) <- vetsenv$estimation_names
    colnames(hessian) <- rownames(hessian) <- vetsenv$estimation_names
    out <- list(pars = pars, llh = llh, gradient = gradient, hessian = hessian, solver_out = sol)
    return(out)
}
