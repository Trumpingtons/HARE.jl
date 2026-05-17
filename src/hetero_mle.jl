"""
Maximum likelihood estimators for heteroskedastic regression.

Three link functions are supported, corresponding to the SAS PROC AUTOREG
HETERO statement conventions:

  - Exponential  (LINK=EXP):    σ²_i = exp(γ₀ + z_i'γ)       Harvey (1976)
  - Quadratic    (LINK=SQUARE): σ_i  = γ₀ + z_i'γ            Glejser (1969)
  - Linear       (LINK=LINEAR): σ²_i = γ₀ + z_i'γ

The log-likelihood is maximised jointly over (β, γ) using L-BFGS with
ForwardDiff automatic differentiation.  Two-step FWLS estimates serve as
warm starts.  Standard errors come from the inverse observed information matrix.
"""

function _hetero_negloglik(theta, X_full, Z_full, y, link)
    k     = size(X_full, 2)
    beta  = theta[1:k]
    gamma = theta[k+1:end]
    resid = y .- X_full * beta
    if link === :exponential
        log_sigma2 = Z_full * gamma
        return 0.5 * (sum(log_sigma2) + sum(resid.^2 .* exp.(-log_sigma2)))
    elseif link === :quadratic
        sigma   = Z_full * gamma
        sigma_c = max.(sigma, 1e-10)
        return sum(log.(sigma_c)) + 0.5 * sum(resid.^2 ./ sigma_c.^2)
    else  # :linear
        sigma2   = Z_full * gamma
        sigma2_c = max.(sigma2, 1e-10)
        return 0.5 * (sum(log.(sigma2_c)) + sum(resid.^2 ./ sigma2_c))
    end
end

function _hetero_warm_start(X_full, Z_full, y, link)
    ols   = lm(X_full, y)
    u     = residuals(ols)
    beta0 = coef(ols)
    if link === :exponential
        _, gamma0, _ = harvey_weights(Z_full, u)
    elseif link === :quadratic
        _, gamma0, _ = glejser_weights(Z_full, u)
    else  # :linear — constant variance start (always feasible)
        gamma0 = vcat(mean(u.^2), zeros(size(Z_full, 2) - 1))
    end
    return vcat(beta0, gamma0)
end

function _hetero_mle_core(X_full, Z_full, y, link, theta0; tol, maxiter)
    n   = length(y)
    nll = theta -> _hetero_negloglik(theta, X_full, Z_full, y, link)
    res = optimize(nll, theta0, LBFGS(), Optim.Options(iterations=maxiter, g_tol=tol);
                   autodiff=:forward)
    theta_hat = Optim.minimizer(res)
    H         = ForwardDiff.hessian(nll, theta_hat)
    vcov_full = inv(H)
    loglik    = -Optim.minimum(res) - n/2 * log(2π)
    return theta_hat, vcov_full, loglik, Optim.converged(res), Optim.iterations(res)
end

function _hetero_mle(X_full, Z_full, y, link, coefnames, mf; tol, maxiter)
    k         = size(X_full, 2)
    theta0    = _hetero_warm_start(X_full, Z_full, y, link)
    theta_hat, vcov_full, loglik, conv, iters =
        _hetero_mle_core(X_full, Z_full, y, link, theta0; tol=tol, maxiter=maxiter)
    beta       = theta_hat[1:k]
    gamma      = theta_hat[k+1:end]
    vcov_beta  = vcov_full[1:k, 1:k]
    vcov_gamma = vcov_full[k+1:end, k+1:end]
    f          = X_full * beta
    return HeteroMLEResult(beta, coefnames, mf, vcov_beta, y .- f, f,
                           gamma, vcov_gamma, loglik, link, conv, iters)
end

"""
    exponential_mle(X, y; Z=nothing, intercept=true, tol=1e-10, maxiter=1000) -> HeteroMLEResult
    exponential_mle(formula, data; Z=nothing, tol=1e-10, maxiter=1000) -> HeteroMLEResult

Maximum likelihood estimation of the heteroskedastic regression model with an
exponential variance function (Harvey 1976):

```math
\\begin{aligned}
y_i &= \\mathbf{x}_i^\\top\\boldsymbol{\\beta} + \\varepsilon_i, \\qquad
       \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma_i^2) \\\\[4pt]
\\sigma_i^2 &= \\exp(\\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma})
\\end{aligned}
```

The log-likelihood is maximised jointly over ``(\\boldsymbol{\\beta}, \\boldsymbol{\\gamma})``
using L-BFGS with automatic differentiation.  Two-step Harvey FWLS estimates
serve as warm starts.  Standard errors come from the inverse observed information matrix.

Corresponds to SAS `PROC AUTOREG HETERO LINK=EXP`.

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `Z`       : n × p auxiliary regressor matrix for the variance equation,
              **without** a constant column (default: `X`). A constant is
              prepended internally.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).
- `tol`     : gradient norm tolerance for L-BFGS convergence (default `1e-10`).
- `maxiter` : maximum number of L-BFGS iterations (default `1000`).

# Returns
[`HeteroMLEResult`](@ref) with `link = :exponential`.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function exponential_mle(X, y; intercept::Bool = true, Z=nothing, tol=1e-10, maxiter=1000,
                          coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                          mf=nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    return _hetero_mle(X_full, Z_full, y, :exponential, coefnames, mf; tol=tol, maxiter=maxiter)
end

function exponential_mle(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return exponential_mle(X, y; intercept=false, Z=Z, coefnames=cn, mf=mf, kwargs...)
end

"""
    quadratic_mle(X, y; Z=nothing, intercept=true, tol=1e-10, maxiter=1000) -> HeteroMLEResult
    quadratic_mle(formula, data; Z=nothing, tol=1e-10, maxiter=1000) -> HeteroMLEResult

Maximum likelihood estimation of the heteroskedastic regression model with a
quadratic (linear standard deviation) variance function (Glejser 1969):

```math
\\begin{aligned}
y_i &= \\mathbf{x}_i^\\top\\boldsymbol{\\beta} + \\varepsilon_i, \\qquad
       \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma_i^2) \\\\[4pt]
\\sigma_i &= \\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma}
\\end{aligned}
```

The log-likelihood is maximised jointly over ``(\\boldsymbol{\\beta}, \\boldsymbol{\\gamma})``
using L-BFGS with automatic differentiation.  Two-step Glejser FWLS estimates
serve as warm starts.  Standard errors come from the inverse observed information matrix.

Corresponds to SAS `PROC AUTOREG HETERO LINK=SQUARE`.

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `Z`       : n × p auxiliary regressor matrix for the variance equation,
              **without** a constant column (default: `X`). A constant is
              prepended internally.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).
- `tol`     : gradient norm tolerance for L-BFGS convergence (default `1e-10`).
- `maxiter` : maximum number of L-BFGS iterations (default `1000`).

# Returns
[`HeteroMLEResult`](@ref) with `link = :quadratic`.

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function quadratic_mle(X, y; intercept::Bool = true, Z=nothing, tol=1e-10, maxiter=1000,
                        coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                        mf=nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    return _hetero_mle(X_full, Z_full, y, :quadratic, coefnames, mf; tol=tol, maxiter=maxiter)
end

function quadratic_mle(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return quadratic_mle(X, y; intercept=false, Z=Z, coefnames=cn, mf=mf, kwargs...)
end

"""
    linear_mle(X, y; Z=nothing, intercept=true, tol=1e-10, maxiter=1000) -> HeteroMLEResult
    linear_mle(formula, data; Z=nothing, tol=1e-10, maxiter=1000) -> HeteroMLEResult

Maximum likelihood estimation of the heteroskedastic regression model with a
linear variance function:

```math
\\begin{aligned}
y_i &= \\mathbf{x}_i^\\top\\boldsymbol{\\beta} + \\varepsilon_i, \\qquad
       \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma_i^2) \\\\[4pt]
\\sigma_i^2 &= \\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma}
\\end{aligned}
```

The log-likelihood is maximised jointly over ``(\\boldsymbol{\\beta}, \\boldsymbol{\\gamma})``
using L-BFGS with automatic differentiation.  OLS of squared OLS residuals on Z
serves as a warm start.  Standard errors come from the inverse observed information matrix.

Corresponds to SAS `PROC AUTOREG HETERO LINK=LINEAR`.

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `Z`       : n × p auxiliary regressor matrix for the variance equation,
              **without** a constant column (default: `X`). A constant is
              prepended internally.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).
- `tol`     : gradient norm tolerance for L-BFGS convergence (default `1e-10`).
- `maxiter` : maximum number of L-BFGS iterations (default `1000`).

# Returns
[`HeteroMLEResult`](@ref) with `link = :linear`.
"""
function linear_mle(X, y; intercept::Bool = true, Z=nothing, tol=1e-10, maxiter=1000,
                    coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                    mf=nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    return _hetero_mle(X_full, Z_full, y, :linear, coefnames, mf; tol=tol, maxiter=maxiter)
end

function linear_mle(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return linear_mle(X, y; intercept=false, Z=Z, coefnames=cn, mf=mf, kwargs...)
end
