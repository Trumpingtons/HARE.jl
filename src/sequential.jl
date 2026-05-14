"""
Sequential HARE estimators for correction of AR(1) autocorrelation
and multiplicative heteroskedasticity.

Includes two-step and iterated versions.
"""

"""
    two_step_sequential(X, y) -> SequentialResult
    two_step_sequential(formula, data) -> SequentialResult

Two-step Feasible GLS correcting sequentially for AR(1) autocorrelation **and**
multiplicative heteroskedasticity (Sequential HARE).

**Step 1.** OLS → estimate rho via Cochrane–Orcutt.
**Step 2.** Prais–Winsten transformation → (X★, y★).
**Step 3.** Harvey log-variance auxiliary regression on (X★, y★) residuals → weights w.
**Step 4.** WLS on (X★, y★) with weights w.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).

# Returns
[`SequentialResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*,
42(3), 579–590.
"""
function two_step_sequential(X, y; coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    rho          = estimate_rho(residuals(lm(X, y)))
    ystar, Xstar = pw_transform(X, y, rho)
    u_pw         = residuals(lm(Xstar, ystar))
    w, gamma     = harvey_weights(Xstar, u_pw)
    model        = lm(Xstar, ystar, wts = w)
    beta         = coef(model)
    f            = X * beta
    return SequentialResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, gamma, 1, true)
end

function two_step_sequential(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_sequential(X, y; coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_sequential(X, y; tol=1e-8, maxiter=100) -> SequentialResult
    iterated_sequential(formula, data; tol=1e-8, maxiter=100) -> SequentialResult

Iterated Feasible GLS correcting sequentially for AR(1) autocorrelation **and**
multiplicative heteroskedasticity. Each iteration estimates rho, applies the
Prais–Winsten transformation, computes Harvey weights, and fits WLS.
Convergence criterion:

    max(‖betâ⁽ⁱ⁾ − betâ⁽ⁱ⁻¹⁾‖∞, |rho⁽ⁱ⁾ − rho⁽ⁱ⁻¹⁾|) < tol

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `tol`    : convergence tolerance (default `1e-8`).
- `maxiter`: maximum number of iterations (default `100`).

# Returns
[`SequentialResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*,
42(3), 579–590.
"""
function iterated_sequential(X, y; tol=1e-8, maxiter=100, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    model = lm(X, y)
    rho   = estimate_rho(residuals(model))
    gamma = zeros(size(X, 2))
    for i in 1:maxiter
        beta_old     = coef(model)
        rho_old      = rho
        ystar, Xstar = pw_transform(X, y, rho)
        u_pw         = residuals(lm(Xstar, ystar))
        w, gamma     = harvey_weights(Xstar, u_pw)
        model        = lm(Xstar, ystar, wts = w)
        rho          = estimate_rho(y .- X * coef(model))
        if max(maximum(abs.(coef(model) .- beta_old)), abs(rho - rho_old)) < tol
            beta = coef(model)
            f    = X * beta
            return SequentialResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, gamma, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return SequentialResult(beta, coefnames, mf, vcov(model), y .- f, f, rho, gamma, maxiter, false)
end

function iterated_sequential(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_sequential(X, y; coefnames=cn, mf=mf, kwargs...)
end
