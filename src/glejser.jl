"""
Feasible Weighted Least Squares (FWLS) estimators for heteroskedasticity
via the Glejser auxiliary regression model.

Includes two-step and iterated versions, with optional separate auxiliary
regressors Z for the variance equation.
"""

"""
    two_step_glejser(X, y; intercept=true) -> GlejserResult
    two_step_glejser(formula, data; Z=nothing) -> GlejserResult

Two-step Feasible Weighted Least Squares (FWLS) correcting for
heteroskedasticity via Glejser's auxiliary regression.

**Step 1.** Fit OLS and obtain residuals u_hat.
**Step 2.** Model the standard deviation via `|u_hat_i| = Z_i * gamma + v_i` and set
  `w_i = 1 / (Z_i * gamma_hat)^2`.
**Step 3.** Re-fit by WLS with weights w.

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : n x p auxiliary regressor matrix for the variance equation
               (default: augmented `X`). Must include a constant column.

# Returns
[`GlejserResult`](@ref) with `coef`, `vcov`, `residuals`, `fitted`,
`iterations`, `converged`.

# Examples
```jldoctest
julia> using HARE, Random, StatsBase

julia> Random.seed!(42); n = 50; X = randn(n, 2); y = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ randn(n);

julia> m = two_step_glejser(X, y);

julia> length(coef(m))
3

julia> m.converged
true
```

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316-323.
"""
function two_step_glejser(X, y; intercept::Bool = true, Z=nothing,
                          coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                          mf=nothing)
    X                    = intercept ? hcat(ones(eltype(X), size(X,1)), X) : X
    Z                    = isnothing(Z) ? X : Z
    ols                  = lm(X, y)
    w, gamma, gamma_vcov = glejser_weights(Z, residuals(ols))
    model                = lm(X, y, weights = w)
    beta                 = coef(model)
    f                    = X * beta
    return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, 1, true)
end

function two_step_glejser(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_glejser(X, y; intercept=false, Z = isnothing(Z) ? X : Z, coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_glejser(X, y; intercept=true, Z=nothing, tol=1e-8, maxiter=100) -> GlejserResult
    iterated_glejser(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> GlejserResult

Iterated Feasible Weighted Least Squares (IFWLS) via Glejser's auxiliary
regression. Repeats the weight-update step until the coefficient vector
converges:

    max|beta_hat(i) - beta_hat(i-1)| < tol

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : n x p auxiliary regressor matrix for the variance equation
               (default: augmented `X`). Must include a constant column.
- `tol`      : convergence tolerance on the sup-norm of coefficient change (default `1e-8`).
- `maxiter`  : maximum number of iterations (default `100`).

# Returns
[`GlejserResult`](@ref).

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316-323.
"""
function iterated_glejser(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                          coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                          mf=nothing)
    X          = intercept ? hcat(ones(eltype(X), size(X,1)), X) : X
    Z          = isnothing(Z) ? X : Z
    model      = lm(X, y)
    gamma      = zeros(size(Z, 2))
    gamma_vcov = zeros(size(Z, 2), size(Z, 2))
    for i in 1:maxiter
        beta_old             = coef(model)
        w, gamma, gamma_vcov = glejser_weights(Z, residuals(model))
        model                = lm(X, y, weights = w)
        if maximum(abs.(coef(model) .- beta_old)) < tol
            beta = coef(model)
            f    = X * beta
            return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, maxiter, false)
end

function iterated_glejser(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_glejser(X, y; intercept=false, Z = isnothing(Z) ? X : Z, coefnames=cn, mf=mf, kwargs...)
end
