"""
Prais–Winsten (Cochrane–Orcutt) GLS estimators for AR(1) autocorrelation.

Includes two-step and iterated versions. The Prais–Winsten approach retains
the first observation via scaling, unlike Cochrane–Orcutt.
"""

"""
    two_step_prais_winsten(X, y) -> PraisWinstenResult
    two_step_prais_winsten(formula, data) -> PraisWinstenResult

Two-step Prais–Winsten GLS estimator correcting for AR(1) autocorrelation.

**Step 1.** Fit OLS, estimate rho via the Cochrane–Orcutt moment estimator.
**Step 2.** Apply the Prais–Winsten transformation (retaining the first
observation) and fit OLS on the transformed system.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).

# Returns
[`PraisWinstenResult`](@ref). Use `stderror(result)` to obtain standard errors.

# Examples
```jldoctest
julia> using HARE, Random

julia> Random.seed!(42); n = 50; u = zeros(n); for t in 2:n; u[t] = 0.7*u[t-1] + randn(); end;

julia> X = hcat(ones(n), randn(n)); y = X * [1.0, 2.0] .+ u;

julia> m = two_step_prais_winsten(X, y);

julia> length(coef(m))
2

julia> m.converged
true
```

# References
Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial
correlation. *Cowles Commission Discussion Paper*, No. 383.
"""
function two_step_prais_winsten(X, y; coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    rho            = estimate_rho(residuals(lm(X, y)))
    ystar, Xstar = pw_transform(X, y, rho)
    model        = lm(Xstar, ystar)
    beta            = coef(model)
    f            = X * beta
    return PraisWinstenResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, 1, true)
end

function two_step_prais_winsten(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_prais_winsten(X, y; coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_prais_winsten(X, y; tol=1e-8, maxiter=100) -> PraisWinstenResult
    iterated_prais_winsten(formula, data; tol=1e-8, maxiter=100) -> PraisWinstenResult

Iterated Prais–Winsten (Cochrane–Orcutt) GLS estimator for AR(1) errors.
Alternates between estimating rho and re-fitting by Prais–Winsten GLS until
|rho⁽ⁱ⁾ − rho⁽ⁱ⁻¹⁾| < `tol`.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `tol`    : convergence tolerance on |Δrho| (default `1e-8`).
- `maxiter`: maximum number of iterations (default `100`).

# Returns
[`PraisWinstenResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial
correlation. *Cowles Commission Discussion Paper*, No. 383.

Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32–61.
"""
function iterated_prais_winsten(X, y; tol=1e-8, maxiter=100, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    rho     = estimate_rho(residuals(lm(X, y)))
    model = lm(X, y)
    for i in 1:maxiter
        rho_old        = rho
        ystar, Xstar = pw_transform(X, y, rho)
        model        = lm(Xstar, ystar)
        rho            = estimate_rho(y .- X * coef(model))
        if abs(rho - rho_old) < tol
            beta = coef(model)
            f = X * beta
            return PraisWinstenResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, i, true)
        end
    end
    beta = coef(model)
    f = X * beta
    return PraisWinstenResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, maxiter, false)
end

function iterated_prais_winsten(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_prais_winsten(X, y; coefnames=cn, mf=mf, kwargs...)
end
