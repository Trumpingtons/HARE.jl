"""
Auxiliary functions shared across all estimators.

These utilities handle:
- AR(1) coefficient estimation (Cochrane–Orcutt)
- Heteroskedasticity weighting (Harvey model)
- Prais–Winsten GLS transformation
- Formula parsing and data extraction
"""

"""
    estimate_rho(u) -> Float64

Estimate the AR(1) autocorrelation coefficient from OLS residuals using the
Cochrane–Orcutt moment estimator:

    ρ̂ = Σ(uₜ uₜ₋₁) / Σ(uₜ₋₁²)

# Arguments
- `u`: vector of OLS residuals of length n.

# References
Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32–61.
"""
estimate_rho(u) = sum(u[2:end] .* u[1:end-1]) / sum(u[1:end-1].^2)

"""
    harvey_weights(X, u) -> Vector{Float64}

Compute inverse-variance weights for FWLS using Harvey's multiplicative
heteroskedasticity model. The log-variance is modelled as a linear function
of the regressors:

    log(σᵢ²) ≈ Xᵢ γ

Weights are `wᵢ = 1 / exp(Xᵢ γ̂)`, consistent estimates of `1/σᵢ²` under
the Harvey model.

# Arguments
- `X`: n × k regressor matrix (including a constant column).
- `u`: vector of residuals from a preliminary OLS fit.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function harvey_weights(X, u)
    aux   = lm(X, log.(u.^2))
    gamma = coef(aux)
    return 1.0 ./ exp.(fitted(aux)), gamma
end

"""
    glejser_weights(Z, u) -> Vector{Float64}

Compute inverse-variance weights for FWLS using Glejser's auxiliary regression.
The standard deviation is modelled as a linear function of auxiliary regressors:

    |ûᵢ| ≈ Zᵢ γ

Weights are `wᵢ = 1 / sigma_hatᵢ²` where `sigma_hatᵢ = max(Zᵢ γ̂, ε)`. Negative or near-zero
fitted values are clipped to `1e-8` to guarantee positive weights.

# Arguments
- `Z`: n × p auxiliary regressor matrix (including a constant column).
- `u`: vector of residuals from a preliminary OLS fit.

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function glejser_weights(Z, u)
    aux   = lm(Z, abs.(u))
    gamma = coef(aux)
    sigma_hat     = max.(fitted(aux), 1e-8)
    return 1.0 ./ sigma_hat.^2, gamma
end

"""
    pw_transform(X, y, rho) -> (ystar, Xstar)

Apply the Prais–Winsten GLS transformation for AR(1) errors. The first
observation is scaled by `√(1−ρ²)` (retaining it, unlike Cochrane–Orcutt),
and subsequent observations are quasi-differenced:

    y★₁ = √(1−ρ²) y₁,    y★ₜ = yₜ − ρ yₜ₋₁  (t ≥ 2)

# Arguments
- `X`  : n × k regressor matrix.
- `y`  : response vector of length n.
- `rho`: AR(1) coefficient ρ ∈ (−1, 1).

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
