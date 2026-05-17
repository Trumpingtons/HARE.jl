"""
Cochrane-Orcutt GLS estimator for AR(1) autocorrelation.

Cochrane-Orcutt quasi-differences the data starting from t=2, discarding the
first observation.  Prais-Winsten strictly dominates it by retaining the first
observation via scaling by sqrt(1-rho^2) at zero additional cost.
"""

# Quasi-difference transformation dropping the first observation (CO convention).
function co_transform(X, y, rho)
    n = length(y)
    ystar = y[2:n] .- rho .* y[1:n-1]
    Xstar = X[2:n, :] .- rho .* X[1:n-1, :]
    return ystar, Xstar
end

"""
    two_step_cochrane_orcutt(X, y; intercept=true) -> CochranOrcuttResult
    two_step_cochrane_orcutt(formula, data) -> CochranOrcuttResult

Two-step Cochrane-Orcutt GLS estimator for AR(1) autocorrelation.

**Step 1.** Fit OLS, estimate rho via the Cochrane-Orcutt moment estimator.
**Step 2.** Apply the Cochrane-Orcutt quasi-differencing transformation (dropping
  the first observation) and fit OLS on the (n−1) × k transformed system.

!!! warning "Use Prais-Winsten instead"
    Prais-Winsten strictly dominates Cochrane-Orcutt: it retains the first
    observation via the scaling `sqrt(1 − ρ²)` at zero additional cost, yielding
    coefficient estimates with smaller variance.  `two_step_cochrane_orcutt` is
    provided for compatibility with software that uses the CO convention
    (e.g. Python `statsmodels.GLSAR`, Stata `prais, corc`).

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).

# Returns
[`CochranOrcuttResult`](@ref).

# Examples
```jldoctest
julia> using HARE, Random, StatsBase

julia> Random.seed!(42); n = 50; u = zeros(n); for t in 2:n; u[t] = 0.7*u[t-1] + randn(); end;

julia> X = randn(n, 1); y = hcat(ones(n), X) * [1.0, 2.0] .+ u;

julia> m = two_step_cochrane_orcutt(X, y);

julia> length(coef(m))
2

julia> m.converged
true
```

# References
Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32–61.
"""
function two_step_cochrane_orcutt(X, y; intercept::Bool = true,
                                  coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                                  mf=nothing)
    X              = intercept ? hcat(ones(eltype(X), length(y)), X) : X
    rho            = estimate_rho(residuals(lm(X, y)))
    ystar, Xstar   = co_transform(X, y, rho)
    model          = lm(Xstar, ystar)
    beta           = coef(model)
    f              = X * beta
    return CochranOrcuttResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, 1, true)
end

function two_step_cochrane_orcutt(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_cochrane_orcutt(X, y; intercept=false, coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_cochrane_orcutt(X, y; intercept=true, tol=1e-8, maxiter=100) -> CochranOrcuttResult
    iterated_cochrane_orcutt(formula, data; tol=1e-8, maxiter=100) -> CochranOrcuttResult

Iterated Cochrane-Orcutt GLS estimator for AR(1) errors.  Alternates between
estimating rho and re-fitting by CO quasi-differencing until
`|rho(i) - rho(i-1)| < tol`.

!!! warning "Use Prais-Winsten instead"
    Prais-Winsten strictly dominates Cochrane-Orcutt: it retains the first
    observation via the scaling `sqrt(1 − ρ²)` at zero additional cost, yielding
    coefficient estimates with smaller variance.  `iterated_cochrane_orcutt` is
    provided for compatibility with software that uses the CO convention
    (e.g. Python `statsmodels.GLSAR`, Stata `prais, corc`).

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `tol`      : convergence tolerance on |rho(i) - rho(i-1)| (default `1e-8`).
- `maxiter`  : maximum number of iterations (default `100`).

# Returns
[`CochranOrcuttResult`](@ref).

# References
Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32–61.
"""
function iterated_cochrane_orcutt(X, y; intercept::Bool = true, tol=1e-8, maxiter=100,
                                   coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                                   mf=nothing)
    X     = intercept ? hcat(ones(eltype(X), length(y)), X) : X
    rho   = estimate_rho(residuals(lm(X, y)))
    model = lm(X, y)
    for i in 1:maxiter
        rho_old      = rho
        ystar, Xstar = co_transform(X, y, rho)
        model        = lm(Xstar, ystar)
        rho          = estimate_rho(y .- X * coef(model))
        if abs(rho - rho_old) < tol
            beta = coef(model)
            f    = X * beta
            return CochranOrcuttResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return CochranOrcuttResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, maxiter, false)
end

function iterated_cochrane_orcutt(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_cochrane_orcutt(X, y; intercept=false, coefnames=cn, mf=mf, kwargs...)
end
