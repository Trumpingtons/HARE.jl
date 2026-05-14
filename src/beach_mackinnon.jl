"""
Beach–MacKinnon exact maximum likelihood estimator for AR(1) autocorrelation.

Uses Brent's method to maximize the exact concentrated log-likelihood,
including the Jacobian term from the transformation.
"""

"""
    beach_mackinnon(X, y; tol=1e-10) -> BeachMacKinnonResult
    beach_mackinnon(formula, data; tol=1e-10) -> BeachMacKinnonResult

Beach–MacKinnon exact maximum likelihood estimator for the linear regression
model with AR(1) errors. Maximises the exact concentrated log-likelihood

    ℓ(rho) = −n/2 · log(S(rho)/n) + ½ · log(1 − rho²)

where the Jacobian term `½ log(1−rho²)` distinguishes this from the
Cochrane–Orcutt/Prais–Winsten approximation. rho is found by Brent's method;
beta_hat and σ̂² are then recovered analytically. Standard errors are conditional
on rhô.

# Arguments
- `X`      : n × k regressor matrix (including constant).
- `y`      : response vector of length n.
- `formula`: `@formula` expression (formula method).
- `data`   : Tables.jl-compatible data source (formula method).
- `tol`    : absolute tolerance for Brent's method (default `1e-10`).

# Returns
[`BeachMacKinnonResult`](@ref). Use `stderror(result)` to obtain standard errors.

# References
Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for
regression with autocorrelated errors. *Econometrica*, 46(1), 51–58.
"""
function beach_mackinnon(X, y; tol=1e-10, coefnames::Vector{String} = ["x$i" for i in 1:size(X,2)], mf=nothing)
    n = length(y)
    function neg_loglik(rho)
        ystar, Xstar = pw_transform(X, y, rho)
        beta            = (Xstar'Xstar) \ (Xstar'ystar)
        u            = ystar .- Xstar * beta
        S            = dot(u, u)
        return n/2 * log(S/n) - 1/2 * log(1 - rho^2)
    end
    result       = optimize(neg_loglik, -0.999, 0.999, Brent(); abs_tol = tol)
    rho_hat        = Optim.minimizer(result)
    ystar, Xstar = pw_transform(X, y, rho_hat)
    model        = lm(Xstar, ystar)
    beta            = coef(model)
    f            = X * beta
    return BeachMacKinnonResult(beta, coefnames, mf, vcov(model), y .- f, f, rho_hat, -Optim.minimum(result), Optim.converged(result))
end

function beach_mackinnon(formula::FormulaTerm, data; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return beach_mackinnon(X, y; coefnames=cn, mf=mf, kwargs...)
end
