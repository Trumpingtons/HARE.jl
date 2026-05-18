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

For groupwise heteroskedasticity the MLE is computed via coordinate descent
(the same iteration as iterated_groupwise) with standard errors from the
exact observed information matrix (X′WX)⁻¹, not GLM's dispersion-scaled vcov.
"""

function _hetero_negloglik(theta, X_full, Z_full, y, link)
    k     = size(X_full, 2)
    beta  = theta[1:k]
    gamma = theta[k+1:end]
    resid = y .- X_full * beta
    if link === :exponential || link === :power
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
        _, gamma0, _ = exponential_weights(Z_full, u)
    elseif link === :power
        aux    = lm(Z_full, log.(u.^2) .- _HARVEY_C)
        gamma0 = coef(aux)
    elseif link === :quadratic
        _, gamma0, _ = quadratic_weights(Z_full, u)
    else  # :linear — constant variance start (always feasible)
        gamma0 = vcat(mean(u.^2), zeros(size(Z_full, 2) - 1))
    end
    return vcat(beta0, gamma0)
end

function _hetero_mle_core(X_full, Z_full, y, link, theta0; tol, maxiter)
    n   = length(y)
    nll = theta -> _hetero_negloglik(theta, X_full, Z_full, y, link)
    if link === :linear
        k  = size(X_full, 2)
        np = length(theta0)
        lb = fill(-Inf, np); lb[k+1] = 1e-8
        ub = fill(Inf, np)
        g! = (G, theta) -> ForwardDiff.gradient!(G, nll, theta)
        res = optimize(nll, g!, lb, ub, theta0, Fminbox(LBFGS()),
                       Optim.Options(iterations=maxiter, g_tol=tol))
    else
        res = optimize(nll, theta0, LBFGS(), Optim.Options(iterations=maxiter, g_tol=tol);
                       autodiff=:forward)
    end
    theta_hat = Optim.minimizer(res)
    H         = Symmetric(ForwardDiff.hessian(nll, theta_hat))
    vcov_full = inv(H)
    if link === :linear && any(diag(vcov_full) .<= 0)
        vcov_full = pinv(H)
    end
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
    return MLEResult(beta, coefnames, mf, vcov_beta, y .- f, f,
                           gamma, vcov_gamma, loglik, link, conv, iters)
end

"""
    exponential_mle(X, y; Z=nothing, intercept=true, tol=1e-10, maxiter=1000) -> MLEResult
    exponential_mle(formula, data; Z=nothing, tol=1e-10, maxiter=1000) -> MLEResult

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
[`MLEResult`](@ref) with `link = :exponential`.

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
    return exponential_mle(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    quadratic_mle(X, y; Z=nothing, intercept=true, tol=1e-10, maxiter=1000) -> MLEResult
    quadratic_mle(formula, data; Z=nothing, tol=1e-10, maxiter=1000) -> MLEResult

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
[`MLEResult`](@ref) with `link = :quadratic`.

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
    return quadratic_mle(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    linear_mle(X, y; Z=nothing, intercept=true, tol=1e-8, maxiter=1000) -> MLEResult
    linear_mle(formula, data; Z=nothing, tol=1e-8, maxiter=1000) -> MLEResult

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
[`MLEResult`](@ref) with `link = :linear`.
"""
function linear_mle(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=1000,
                    coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                    mf=nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    return _hetero_mle(X_full, Z_full, y, :linear, coefnames, mf; tol=tol, maxiter=maxiter)
end

function linear_mle(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return linear_mle(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    power_mle(X, y; Z, intercept=true, tol=1e-10, maxiter=1000) -> MLEResult
    power_mle(formula, data; Z, tol=1e-10, maxiter=1000) -> MLEResult

Maximum likelihood estimation of the heteroskedastic regression model with a
power variance function:

```math
\\begin{aligned}
y_i &= \\mathbf{x}_i^\\top\\boldsymbol{\\beta} + \\varepsilon_i, \\qquad
       \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma_i^2) \\\\[4pt]
\\sigma_i^2 &= \\sigma^2 \\prod_j |z_{ij}|^{2\\delta_j}
\\end{aligned}
```

Equivalent to Harvey's exponential model on ``\\log|Z|``: the variance equation
is ``\\log\\sigma_i^2 = \\gamma_0 + \\gamma_1\\log|z_{i1}| + \\cdots`` so the
reported ``\\boldsymbol{\\gamma} = (\\gamma_0, \\gamma_1, \\ldots)`` satisfies
``\\gamma_0 = \\log\\sigma^2`` and ``\\gamma_j = 2\\delta_j``. Standard errors are
exact (from the observed information matrix), with no delta-method approximation.

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `Z`       : n × p auxiliary regressor matrix **without** a constant column.
              A constant is prepended internally. **Required** for the power model.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).
- `tol`     : gradient norm tolerance for L-BFGS convergence (default `1e-10`).
- `maxiter` : maximum number of L-BFGS iterations (default `1000`).

# Returns
[`MLEResult`](@ref) with `link = :power`. The `gamma` field contains
``(\\log\\sigma^2, 2\\delta_1, \\ldots)``.
"""
function power_mle(X, y; intercept::Bool = true, Z=nothing, tol=1e-10, maxiter=1000,
                   coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                   mf=nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_orig = isnothing(Z) ? X_full[:, 2:end] : Z
    Z_log  = hcat(ones(n), log.(max.(abs.(Z_orig), 1e-10)))
    return _hetero_mle(X_full, Z_log, y, :power, coefnames, mf; tol=tol, maxiter=maxiter)
end

function power_mle(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return power_mle(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    groupwise_mle(X, y, groups; intercept=true, tol=1e-8, maxiter=100) -> GroupwiseResult
    groupwise_mle(formula, data, groups; tol=1e-8, maxiter=100) -> GroupwiseResult

Maximum likelihood estimation for groupwise heteroskedasticity.

The model is:

```math
y_i = \\mathbf{x}_i^\\top\\boldsymbol{\\beta} + \\varepsilon_i, \\qquad
\\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma_g^2), \\quad i \\in g
```

The MLE first-order conditions are:

- ``\\hat{\\boldsymbol{\\beta}}``: WLS with weights ``1/\\sigma_g^2``
- ``\\hat{\\sigma}_g^2 = \\frac{1}{n_g}\\sum_{i \\in g}\\hat{u}_i^2``

These are solved by coordinate descent (alternating WLS and variance updates)
until ``\\max|\\hat{\\boldsymbol{\\beta}}^{(t)} - \\hat{\\boldsymbol{\\beta}}^{(t-1)}| < \\texttt{tol}``.

Standard errors come from the exact observed information matrix
``(\\mathbf{X}^\\top\\mathbf{W}\\mathbf{X})^{-1}``, **not** from GLM's
dispersion-scaled covariance. This is the key difference from
[`iterated_groupwise`](@ref), which gives identical point estimates but
slightly different standard errors in finite samples.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `groups`   : length-n vector of group labels (any type: integers, strings, …).
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum number of iterations (default `100`).

# Returns
[`GroupwiseResult`](@ref).

# References
Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, §9.7.2.
"""
function groupwise_mle(X, y, groups; intercept::Bool = true, tol = 1e-8, maxiter = 100,
                       coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                       mf = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    glabels = _gw_labels(groups)
    gsizes  = [count(==(g), groups) for g in glabels]

    beta   = coef(lm(X_full, y))
    sigma2 = _gw_sigma2(y .- X_full * beta, groups, glabels)

    local w
    for i in 1:maxiter
        beta_old = copy(beta)
        w        = _gw_weights(groups, glabels, sigma2)
        beta     = (X_full' * (w .* X_full)) \ (X_full' * (w .* y))
        sigma2   = _gw_sigma2(y .- X_full * beta, groups, glabels)

        if maximum(abs.(beta .- beta_old)) < tol
            f        = X_full * beta
            vcov_beta = inv(X_full' * (w .* X_full))
            ll       = _gw_loglik(y .- f, groups, glabels, sigma2)
            return GroupwiseResult(beta, coefnames, mf, vcov_beta, y .- f, f,
                                   glabels, gsizes, sigma2, ll, i, true)
        end
    end

    f        = X_full * beta
    vcov_beta = inv(X_full' * (w .* X_full))
    ll       = _gw_loglik(y .- f, groups, glabels, sigma2)
    return GroupwiseResult(beta, coefnames, mf, vcov_beta, y .- f, f,
                           glabels, gsizes, sigma2, ll, maxiter, false)
end

function groupwise_mle(formula::FormulaTerm, data, groups; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return groupwise_mle(X, y, groups; intercept = false, coefnames = cn, mf = mf, kwargs...)
end
