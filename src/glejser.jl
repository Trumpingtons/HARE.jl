"""
Feasible Weighted Least Squares (FWLS) estimators for heteroskedasticity
via the Glejser auxiliary regression model.

Includes two-step and iterated versions, with optional separate auxiliary
regressors Z for the variance equation.
"""

"""
    two_step_glejser(X, y; Z=X) -> GlejserResult
    two_step_glejser(formula, data; Z=nothing) -> GlejserResult

Two-step Feasible Weighted Least Squares (FWLS) correcting for
heteroskedasticity via Glejser's auxiliary regression.

**Step 1.** Fit OLS and obtain residuals û.
**Step 2.** Model the standard deviation via `|ûᵢ| = Zᵢ γ + vᵢ` and set
  `wᵢ = 1 / (Zᵢ γ̂)²`.
**Step 3.** Re-fit by WLS with weights w.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `Z`      : n × p auxiliary regressor matrix for the variance equation
             (default: `X`). Must include a constant column.

# Returns
[`GlejserResult`](@ref) with `coef`, `vcov`, `residuals`, `fitted`,
`iterations`, `converged`.

# Examples
```jldoctest
julia> using HARE, Random

julia> Random.seed!(42); n = 50; X = hcat(ones(n), randn(n), randn(n)); y = X * [1.0, 2.0, -1.0] .+ randn(n);

julia> m = two_step_glejser(X, y);

julia> length(coef(m))
3

julia> m.converged
true
```

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function two_step_glejser(X, y; Z=X, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    ols      = lm(X, y)
    w, gamma = glejser_weights(Z, residuals(ols))
    model    = lm(X, y, wts = w)
    beta     = coef(model)
    f        = X * beta
    return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, 1, true)
end

function two_step_glejser(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_glejser(X, y; Z = isnothing(Z) ? X : Z, coefnames=cn, mf=mf, kwargs...)
end

"""
    iterated_glejser(X, y; Z=X, tol=1e-8, maxiter=100) -> GlejserResult
    iterated_glejser(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> GlejserResult

Iterated Feasible Weighted Least Squares (IFWLS) via Glejser's auxiliary
regression. Repeats the weight-update step until the coefficient vector
converges:

    ‖betâ⁽ⁱ⁾ − betâ⁽ⁱ⁻¹⁾‖∞ < tol

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `Z`      : n × p auxiliary regressor matrix for the variance equation
             (default: `X`). Must include a constant column.
- `tol`    : convergence tolerance on the sup-norm of coefficient change (default `1e-8`).
- `maxiter`: maximum number of iterations (default `100`).

# Returns
[`GlejserResult`](@ref).

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function iterated_glejser(X, y; Z=X, tol=1e-8, maxiter=100, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    model = lm(X, y)
    gamma = zeros(size(Z, 2))
    for i in 1:maxiter
        beta_old    = coef(model)
        w, gamma    = glejser_weights(Z, residuals(model))
        model       = lm(X, y, wts = w)
        if maximum(abs.(coef(model) .- beta_old)) < tol
            beta = coef(model)
            f    = X * beta
            return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, i, true)
        end
    end
    beta = coef(model)
    f    = X * beta
    return GlejserResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, maxiter, false)
end

function iterated_glejser(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_glejser(X, y; Z = isnothing(Z) ? X : Z, coefnames=cn, mf=mf, kwargs...)
end
