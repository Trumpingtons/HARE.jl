"""
Auxiliary functions shared across all estimators.

These utilities handle:
- AR(1) coefficient estimation (Cochrane-Orcutt)
- Heteroskedasticity weighting (Harvey model)
- Prais-Winsten GLS transformation
- Formula parsing and data extraction
"""

"""
    estimate_rho(u) -> Float64

Estimate the AR(1) autocorrelation coefficient from OLS residuals using the
Cochrane-Orcutt moment estimator:

    rho_hat = sum(u_t * u_{t-1}) / sum(u_{t-1}^2)

# Arguments
- `u`: vector of OLS residuals of length n.

# References
Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32-61.
"""
estimate_rho(u) = sum(u[2:end] .* u[1:end-1]) / sum(u[1:end-1].^2)

"""
    harvey_weights(X, u) -> (Vector{Float64}, Vector{Float64})

Compute inverse-variance weights for FWLS using Harvey's multiplicative
heteroskedasticity model. The log-variance is modelled as a linear function
of the regressors:

    log(sigma_i^2) = X_i * gamma

Weights are `w_i = 1 / exp(X_i * gamma_hat)`, consistent estimates of
`1/sigma_i^2` under the Harvey model.

## Bias correction

The naive auxiliary regression of `log(u_hat_i^2)` on `X` yields a biased
estimator of `gamma` because `log(u_hat_i^2) = X_i*gamma + log(eps_i^2)` and

    E[log(eps_i^2)] = E[log(chi^2(1))] = digamma(1/2) + log(2) ≈ -1.2703.

This non-zero mean shifts the OLS estimate by `(X'X)^{-1} X' * c * 1`, which
in general contaminates all coefficients, not just the intercept, whenever the
regressors have non-zero means.

The correction used here subtracts the constant `c` from the dependent variable
before fitting:

    log(u_hat_i^2) - c = X_i * gamma + v_i,   E[v_i] = 0.

Because `c * 1` lies in the column space of `X` (the intercept column is always
present), OLS absorbs it exactly into the intercept, yielding consistent
estimates of all elements of `gamma` regardless of regressor means. The fitted
values `X * gamma_hat` then directly estimate `log(sigma_i^2)`, so no
post-hoc adjustment is needed for the returned weights.

# Arguments
- `X`: n x k regressor matrix (including a constant column).
- `u`: vector of residuals from a preliminary OLS fit.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461-465.

Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.

Judge, G. G., Griffiths, W. E., Hill, R. C., Lütkepohl, H., & Lee, T. C.
(1985). *The Theory and Practice of Econometrics* (2nd ed.). Wiley.

Amemiya, T. (1985). *Advanced Econometrics*. Harvard University Press.
"""
function harvey_weights(X, u)
    c     = -1.2703264837432   # E[log(chi^2(1))] = digamma(1/2) + log(2)
    aux   = lm(X, log.(u.^2) .- c)
    gamma = coef(aux)
    return 1.0 ./ exp.(fitted(aux)), gamma, vcov(aux)
end

"""
    glejser_weights(Z, u) -> Vector{Float64}

Compute inverse-variance weights for FWLS using Glejser's auxiliary regression.
The standard deviation is modelled as a linear function of auxiliary regressors:

    |u_hat_i| ~ Z_i * gamma

Weights are `w_i = 1 / sigma_hat_i^2` where `sigma_hat_i = max(Z_i * gamma_hat, 1e-8)`.
Negative or near-zero fitted values are clipped to `1e-8` to guarantee positive weights.

# Arguments
- `Z`: n x p auxiliary regressor matrix (including a constant column).
- `u`: vector of residuals from a preliminary OLS fit.

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316-323.
"""
function glejser_weights(Z, u)
    aux       = lm(Z, abs.(u))
    gamma     = coef(aux)
    sigma_hat = max.(fitted(aux), 1e-8)
    return 1.0 ./ sigma_hat.^2, gamma, vcov(aux)
end

"""
    pw_transform(X, y, rho) -> (ystar, Xstar)

Apply the Prais-Winsten GLS transformation for AR(1) errors. The first
observation is scaled by `sqrt(1 - rho^2)` (retaining it, unlike Cochrane-Orcutt),
and subsequent observations are quasi-differenced:

    ystar_1 = sqrt(1 - rho^2) * y_1,    ystar_t = y_t - rho * y_{t-1}  (t >= 2)

# Arguments
- `X`  : n x k regressor matrix.
- `y`  : response vector of length n.
- `rho`: AR(1) coefficient in (-1, 1).

# References
Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial
correlation. *Cowles Commission Discussion Paper*, No. 383.
"""
function pw_transform(X, y, rho)
    n, k  = size(X)
    ystar = zeros(n)
    Xstar = zeros(n, k)
    s = sqrt(1 - rho^2)
    ystar[1]    = s * y[1]
    Xstar[1, :] = s .* X[1, :]
    for t in 2:n
        ystar[t]    = y[t]    - rho * y[t-1]
        Xstar[t, :] = X[t, :] .- rho .* X[t-1, :]
    end
    return ystar, Xstar
end

"""
    _extract_Xy(formula, data) -> (X, y)

Internal helper. Parse a StatsModels formula and a Tables.jl-compatible data
source into a design matrix `X` and response vector `y`.
"""
function _extract_Xy(formula::FormulaTerm, data)
    mf = ModelFrame(formula, data)
    X  = Matrix{Float64}(ModelMatrix(mf).m)
    y  = vec(response(mf))
    cn = StatsModels.coefnames(mf)
    return X, y, cn, mf
end
