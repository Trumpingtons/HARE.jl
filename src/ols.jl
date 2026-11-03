"""
Ordinary least squares with the classical covariance matrix, the benchmark
against which the feasible GLS estimators are compared.
"""

"""
    ols(X, y; intercept=true) -> OLSResult
    ols(formula, data) -> OLSResult

Ordinary least squares estimator with the classical covariance matrix
sigma2_hat * inv(X'X), where sigma2_hat = RSS / (n - k). The coefficients come
from a Householder QR decomposition of X, and a rank-deficient X is an error.
The estimates, standard errors, residuals and log-likelihood agree with
GLM.jl's `lm` to machine precision, which the test suite checks.

# Arguments
- `X`        : n x k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`
               automatically.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).

# Returns
[`OLSResult`](@ref). Use `stderror(result)` to obtain standard errors.

# Examples
```jldoctest
julia> using HARE, Random

julia> Random.seed!(42); n = 50; X = randn(n, 1); y = hcat(ones(n), X) * [1.0, 2.0] .+ randn(n);

julia> m = ols(X, y);

julia> length(coef(m))
2

julia> coefnames(m)
2-element Vector{String}:
 "(Intercept)"
 "x1"
```
"""
function ols(X, y; intercept::Bool = true,
             coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
             mf=nothing)
    X      = intercept ? hcat(ones(eltype(X), size(X,1)), X) : X
    fit    = _ls(X, y)
    n      = length(fit.residuals)
    loglik = -n / 2 * (log(2pi * sum(abs2, fit.residuals) / n) + 1)  # Gaussian, at the ML variance RSS / n
    return OLSResult(fit.coef, coefnames, mf, fit.vcov, fit.residuals, fit.fitted, loglik)
end

function ols(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return ols(X, y; intercept=false, coefnames=cn, mf=mf, kwargs...)
end
