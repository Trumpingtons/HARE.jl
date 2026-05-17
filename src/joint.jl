"""
Joint HARE estimators for simultaneous maximum likelihood estimation of
AR(1) autocorrelation and multiplicative heteroskedasticity.

The model is:
    u_t = rho * u_{t-1} + sigma_t * e_t,    log(sigma_t^2) = z_t' * gamma

Two-step variant uses the concentrated likelihood (beta profiled out analytically).
Iterated variant uses coordinate descent over beta and (rho, gamma) until convergence.
"""

# Apply the doubly-transformed system: Prais-Winsten transformation scaled by sigma_t.
# The resulting ystar2, Xstar2 have iid N(0,1) errors under the joint model.
function _joint_transform(X, y, rho, sigma)
    n, k = size(X)
    s    = sqrt(max(1 - rho^2, 0.0))
    ystar2  = similar(sigma, n)
    Xstar2  = similar(sigma, n, k)
    ystar2[1]    = s * y[1] / sigma[1]
    Xstar2[1, :] = s .* X[1, :] ./ sigma[1]
    for t in 2:n
        ystar2[t]    = (y[t] - rho * y[t-1]) / sigma[t]
        Xstar2[t, :] = (X[t, :] .- rho .* X[t-1, :]) ./ sigma[t]
    end
    return ystar2, Xstar2
end

# Exact log-likelihood (excluding constant -n/2 * log(2*pi)) given all parameters.
function _joint_loglik(rho, gamma, X, y, Z, beta)
    n  = length(y)
    u  = y .- X * beta
    log_var = Z * gamma
    sigma   = sqrt.(exp.(log_var))
    s   = sqrt(max(1 - rho^2, 0.0))
    ll  = 0.5 * log(max(1 - rho^2, 1e-15))
    ll -= 0.5 * sum(log_var)
    ll -= 0.5 * (s * u[1] / sigma[1])^2
    for t in 2:n
        ll -= 0.5 * ((u[t] - rho * u[t-1]) / sigma[t])^2
    end
    return ll
end

# Return unconstrained starting values (atanh(rho0), gamma0) from OLS residuals.
function _joint_init(X, y, Z)
    u     = residuals(lm(X, y))
    rho0  = clamp(estimate_rho(u), -0.99, 0.99)
    gamma0 = coef(lm(Z, log.(u.^2) .- _HARVEY_C))
    return atanh(rho0), gamma0
end

# Hessian-based vcov for the gamma block, given a negll function and its minimiser.
# gamma lives at indices 2:end of p (index 1 is atanh(rho)), so no delta-method needed.
function _gamma_vcov_from_hessian(negll, p_opt)
    H = ForwardDiff.hessian(negll, p_opt)
    return inv(H)[2:end, 2:end]
end

# Minimize the negative concentrated log-likelihood (beta profiled out analytically).
function _joint_opt_concentrated(X, y, Z)
    rho_unc0, gamma0 = _joint_init(X, y, Z)
    p0 = vcat(rho_unc0, gamma0)
    function negll(p)
        rho    = tanh(p[1])
        gamma  = p[2:end]
        sigma  = sqrt.(exp.(Z * gamma))
        ystar2, Xstar2 = _joint_transform(X, y, rho, sigma)
        beta_hat = Xstar2 \ ystar2
        rss = sum(abs2, ystar2 - Xstar2 * beta_hat)
        return -(0.5 * log(max(1 - rho^2, 1e-15)) - 0.5 * sum(Z * gamma) - 0.5 * rss)
    end
    res        = optimize(negll, p0, LBFGS(), Optim.Options(iterations=2000, g_tol=1e-8))
    p_opt      = res.minimizer
    rho_hat    = tanh(p_opt[1])
    gamma_hat  = p_opt[2:end]
    gamma_vcov = _gamma_vcov_from_hessian(negll, p_opt)
    return rho_hat, gamma_hat, gamma_vcov, -Optim.minimum(res), Optim.converged(res)
end

# Minimize the negative log-likelihood given fixed beta (over rho, gamma only).
function _joint_opt_given_beta(beta, X, y, Z, rho_unc0, gamma0)
    p0    = vcat(rho_unc0, gamma0)
    negll(p) = -_joint_loglik(tanh(p[1]), p[2:end], X, y, Z, beta)
    res   = optimize(negll, p0, LBFGS(), Optim.Options(iterations=500, g_tol=1e-8))
    p_opt = res.minimizer
    gamma_vcov = _gamma_vcov_from_hessian(negll, p_opt)
    return tanh(p_opt[1]), p_opt[2:end], gamma_vcov, Optim.converged(res)
end

"""
    two_step_joint(X, y; intercept=true, Z=nothing) -> JointResult
    two_step_joint(formula, data; Z=nothing) -> JointResult

Two-step Joint HARE estimator. Jointly estimates AR(1) autocorrelation and
multiplicative heteroskedasticity via the concentrated maximum likelihood:

**Step 1.** Maximize the concentrated log-likelihood over (rho, gamma), profiling
  out beta analytically at each evaluation.
**Step 2.** One GLS step on the doubly-transformed system to recover beta_hat.

The variance model is `log(sigma_t^2) = z_t' * gamma` with auxiliary regressors `Z`
(default: augmented `X`). Standard errors are conditional on (rho_hat, gamma_hat).

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : n x p auxiliary regressor matrix for the variance equation,
               **without** a constant column (default: `X`). A constant is
               prepended internally.

# Returns
[`JointResult`](@ref).

# Examples
```jldoctest
julia> using HARE, Random, StatsBase

julia> Random.seed!(42); n = 100; X = randn(n, 1); y = hcat(ones(n), X) * [1.0, 2.0] .+ 0.5 .* randn(n);

julia> m = two_step_joint(X, y);

julia> length(coef(m))
2

julia> m.converged
true
```

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461-465.

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*,
42(3), 579-590.
"""
function two_step_joint(X, y; intercept::Bool = true, Z=nothing,
                        coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                        mf=nothing)
    n                                    = length(y)
    X                                    = intercept ? hcat(ones(eltype(X), n), X) : X
    Z                                    = isnothing(Z) ? X : hcat(ones(n), Z)
    rho_hat, gamma_hat, gamma_vcov, ll, conv = _joint_opt_concentrated(X, y, Z)
    sigma_hat                            = sqrt.(exp.(Z * gamma_hat))
    ystar2, Xstar2                       = _joint_transform(X, y, rho_hat, sigma_hat)
    model                                = lm(Xstar2, ystar2)
    beta_hat                             = coef(model)
    f                                    = X * beta_hat
    ll                                   = _joint_loglik(rho_hat, gamma_hat, X, y, Z, beta_hat)
    return JointResult(beta_hat, coefnames, mf, vcov(model), y .- f, f, rho_hat, gamma_hat, gamma_vcov, ll, 1, conv)
end

function two_step_joint(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_joint(X, y; intercept=false, Z=Z, coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_joint(X, y; intercept=true, Z=nothing, tol=1e-8, maxiter=100) -> JointResult
    iterated_joint(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> JointResult

Iterated Joint HARE estimator. Uses coordinate descent to jointly estimate
AR(1) autocorrelation and multiplicative heteroskedasticity:

Each iteration:
1. Optimize (rho, gamma) via maximum likelihood with beta fixed.
2. Update beta via one GLS step on the doubly-transformed system.

Convergence criterion:

    max(max|beta_hat(i) - beta_hat(i-1)|, |rho(i) - rho(i-1)|) < tol

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : n x p auxiliary regressor matrix for the variance equation,
               **without** a constant column (default: `X`). A constant is
               prepended internally.
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum number of iterations (default `100`).

# Returns
[`JointResult`](@ref).

# References
Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*,
42(3), 579-590.
"""
function iterated_joint(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                        coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                        mf=nothing)
    n               = length(y)
    X               = intercept ? hcat(ones(eltype(X), n), X) : X
    Z               = isnothing(Z) ? X : hcat(ones(n), Z)
    beta            = coef(lm(X, y))
    rho_unc, gamma  = _joint_init(X, y, Z)
    rho             = tanh(rho_unc)
    gamma_vcov      = zeros(size(Z, 2), size(Z, 2))

    local model
    for i in 1:maxiter
        beta_old = copy(beta)
        rho_old  = rho

        rho_unc_cur            = atanh(clamp(rho, -0.99, 0.99))
        rho, gamma, gamma_vcov, _ = _joint_opt_given_beta(beta, X, y, Z, rho_unc_cur, gamma)

        sigma          = sqrt.(exp.(Z * gamma))
        ystar2, Xstar2 = _joint_transform(X, y, rho, sigma)
        model          = lm(Xstar2, ystar2)
        beta           = coef(model)

        if max(maximum(abs.(beta .- beta_old)), abs(rho - rho_old)) < tol
            f  = X * beta
            ll = _joint_loglik(rho, gamma, X, y, Z, beta)
            return JointResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, gamma, gamma_vcov, ll, i, true)
        end
    end

    f  = X * beta
    ll = _joint_loglik(rho, gamma, X, y, Z, beta)
    return JointResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, gamma, gamma_vcov, ll, maxiter, false)
end

function iterated_joint(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_joint(X, y; intercept=false, Z=Z, coefnames=cn, mf=mf, kwargs...)
end
