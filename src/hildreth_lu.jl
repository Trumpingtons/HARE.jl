"""
Hildreth–Lu grid-search estimator for AR(1) autocorrelation.

Performs a grid search over candidate rho values to minimize RSS. Guaranteed
to find the global minimum over the grid (unlike iterative methods).
"""

"""
    hildreth_lu(X, y; nrho=200) -> HildrethLuResult
    hildreth_lu(formula, data; nrho=200) -> HildrethLuResult

Hildreth–Lu grid-search estimator for AR(1) errors. Evaluates the GLS
transformation over a uniform grid of `nrho` candidate values of rho on
(−0.99, 0.99) and selects the value that minimises the residual sum of squares.

Unlike iterative methods, Hildreth–Lu is guaranteed to find the global
minimum over the search grid and is particularly robust when the likelihood
surface is multimodal. Note: the first observation is dropped for each
candidate rho (Cochrane–Orcutt convention), so the effective sample size is n−1.

# Arguments
- `X`    : n × k regressor matrix (including constant).
- `y`    : response vector of length n.
- `nrho` : number of grid points for rho ∈ (−0.99, 0.99) (default `200`).

# Returns
[`HildrethLuResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Hildreth, C., & Lu, J. Y. (1960). Demand relations with autocorrelated
disturbances. *Michigan State University Agricultural Experiment Station
Technical Bulletin*, No. 276.
"""
function hildreth_lu(X, y; nrho=200, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    best_rss   = Inf
    best_model = nothing
    best_rho   = 0.0
    for rho in range(-0.99, 0.99, length=nrho)
        ystar = y[2:end] .- rho .* y[1:end-1]
        Xstar = X[2:end, :] .- rho .* X[1:end-1, :]
        model = lm(Xstar, ystar)
        rss   = sum(residuals(model).^2)
        if rss < best_rss
            best_rss   = rss
            best_model = model
            best_rho   = rho
        end
    end
    beta = coef(best_model)
    f = X * beta
    return HildrethLuResult(beta, coefnames, mf, vcov(best_model), y .- f, f, best_rho, best_rss, nrho, true)
end

function hildreth_lu(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return hildreth_lu(X, y; coefnames=cn, mf=mf, kwargs...)
end
