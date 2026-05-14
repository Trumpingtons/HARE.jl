"""
Feasible Weighted Least Squares (FWLS) estimators for multiplicative
heteroskedasticity via the Harvey model.

Includes two-step and iterated versions.
"""

"""
    two_step_harvey(X, y) -> HarveyResult
    two_step_harvey(formula, data) -> HarveyResult

Two-step Feasible Weighted Least Squares (FWLS) correcting for
multiplicative heteroskedasticity.

**Step 1.** Fit OLS and obtain residuals û.
**Step 2.** Model the variance via Harvey's log-linear auxiliary regression
`log(ûᵢ²) = Xᵢ γ + vᵢ` and set `wᵢ = 1 / exp(Xᵢ γ̂)`.
**Step 3.** Re-fit by WLS with weights w.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).

# Returns
[`HarveyResult`](@ref) with `coef`, `vcov`, `residuals`, `fitted`, `iterations`,
`converged`. Use `stderror(result)` to obtain standard errors from the vcov.

# Examples
```jldoctest
julia> using HARE, Random

julia> Random.seed!(42); n = 50; X = hcat(ones(n), randn(n), randn(n)); y = X * [1.0, 2.0, -1.0] .+ randn(n);

julia> m = two_step_harvey(X, y);

julia> length(coef(m))
3

julia> m.converged
true
```

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function two_step_harvey(X, y; coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    ols      = lm(X, y)
    w, gamma = harvey_weights(X, residuals(ols))
    model    = lm(X, y, wts = w)
    beta     = coef(model)
    f        = X * beta
    return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, 1, true)
end

function two_step_harvey(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_harvey(X, y; coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_harvey(X, y; tol=1e-8, maxiter=100) -> HarveyResult
    iterated_harvey(formula, data; tol=1e-8, maxiter=100) -> HarveyResult

Iterated Feasible Weighted Least Squares (IFWLS). Repeats the Harvey-weight
FWLS step until the coefficient vector converges:

    ‖betâ⁽ⁱ⁾ − betâ⁽ⁱ⁻¹⁾‖∞ < tol

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `tol`    : convergence tolerance on the sup-norm of coefficient change (default `1e-8`).
- `maxiter`: maximum number of iterations (default `100`).

# Returns
[`HarveyResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function iterated_harvey(X, y; tol=1e-8, maxiter=100, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    model = lm(X, y)
    gamma = zeros(size(X, 2))
    for i in 1:maxiter
        beta_old    = coef(model)
        w, gamma    = harvey_weights(X, residuals(model))
        model       = lm(X, y, wts = w)
        if maximum(abs.(coef(model) .- beta_old)) < tol
            beta = coef(model)
            f    = X * beta
            return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return HarveyResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, maxiter, false)
end

function iterated_harvey(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_harvey(X, y; coefnames=cn, mf=mf, kwargs...)
end
