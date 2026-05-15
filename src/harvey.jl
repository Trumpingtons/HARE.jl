"""
Feasible Weighted Least Squares (FWLS) estimators for multiplicative
heteroskedasticity via the Harvey model.

Includes two-step and iterated versions.
"""

"""
    two_step_harvey(X, y; intercept=true) -> HarveyResult
    two_step_harvey(formula, data) -> HarveyResult

Two-step Feasible Weighted Least Squares (FWLS) correcting for
multiplicative heteroskedasticity.

**Step 1.** Fit OLS and obtain residuals u_hat.
**Step 2.** Model the variance via Harvey's log-linear auxiliary regression
`log(u_hat_i^2) = X_i * gamma + v_i` and set `w_i = 1 / exp(X_i * gamma_hat)`.
**Step 3.** Re-fit by WLS with weights w.

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).

# Returns
[`HarveyResult`](@ref) with `coef`, `vcov`, `residuals`, `fitted`, `iterations`,
`converged`. Use `stderror(result)` to obtain standard errors from the vcov.

# Examples
```jldoctest
julia> using HARE, Random, StatsBase

julia> Random.seed!(42); n = 50; X = randn(n, 2); y = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ randn(n);

julia> m = two_step_harvey(X, y);

julia> length(coef(m))
3

julia> m.converged
true
```

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461-465.
"""
function two_step_harvey(X, y; intercept::Bool = true,
                         coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                         mf=nothing)
    X                    = intercept ? hcat(ones(eltype(X), size(X,1)), X) : X
    ols                  = lm(X, y)
    w, gamma, gamma_vcov = harvey_weights(X, residuals(ols))
    model                = lm(X, y, weights = w)
    beta                 = coef(model)
    f                    = X * beta
    return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, 1, true)
end

function two_step_harvey(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_harvey(X, y; intercept=false, coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_harvey(X, y; intercept=true, tol=1e-8, maxiter=100) -> HarveyResult
    iterated_harvey(formula, data; tol=1e-8, maxiter=100) -> HarveyResult

Iterated Feasible Weighted Least Squares (IFWLS). Repeats the Harvey-weight
FWLS step until the coefficient vector converges:

    max|beta_hat(i) - beta_hat(i-1)| < tol

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `tol`      : convergence tolerance on the sup-norm of coefficient change (default `1e-8`).
- `maxiter`  : maximum number of iterations (default `100`).

# Returns
[`HarveyResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461-465.
"""
function iterated_harvey(X, y; intercept::Bool = true, tol=1e-8, maxiter=100,
                         coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                         mf=nothing)
    X          = intercept ? hcat(ones(eltype(X), size(X,1)), X) : X
    model      = lm(X, y)
    gamma      = zeros(size(X, 2))
    gamma_vcov = zeros(size(X, 2), size(X, 2))
    for i in 1:maxiter
        beta_old             = coef(model)
        w, gamma, gamma_vcov = harvey_weights(X, residuals(model))
        model                = lm(X, y, weights = w)
        if maximum(abs.(coef(model) .- beta_old)) < tol
            beta = coef(model)
            f    = X * beta
            return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, maxiter, false)
end

function iterated_harvey(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_harvey(X, y; intercept=false, coefnames=cn, mf=mf, kwargs...)
end
