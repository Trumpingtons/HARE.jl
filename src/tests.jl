"""
Result of [`harvey_test`](@ref). Fields: `n` (sample size), `lm` (LM statistic), `dof` (degrees of freedom).
"""
struct HarveyTestResult
    n::Int
    lm::Float64
    dof::Int
end

"""
    harvey_test(X, y; intercept=true, Z=nothing) -> HarveyTestResult
    harvey_test(formula, data; Z=nothing) -> HarveyTestResult

Lagrange multiplier test for multiplicative heteroskedasticity (Harvey 1976).

Tests H₀: γ = 0 in the exponential variance model log(σᵢ²) = γ₀ + zᵢ'γ.

The test statistic is

    LM = ESS / trigamma(1/2)

where ESS is the explained sum of squares from the auxiliary OLS regression of
log(ûᵢ²) − c on Z (c = E[log(χ²(1))] = −1.2703628454614782), and
trigamma(1/2) = π²/2 is the variance of log(χ²(1)).  The statistic is
distributed as χ²(p) under H₀, where p = dim(γ) (columns of Z excluding the
intercept).

This is Harvey's (1976) exact score-test formula.  It differs from the
generic LM form n·R² but is asymptotically equivalent.

# Arguments
- `X`        : n × k regressor matrix without a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, without a
               constant column (default: `X`). A constant is prepended internally.

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function harvey_test(X::AbstractMatrix, y::AbstractVector;
                     intercept::Bool = true, Z = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    e      = residuals(lm(X_full, y))
    y_aux  = log.(e.^2) .- _HARVEY_C
    aux    = lm(Z_full, y_aux)
    ess    = sum((y_aux .- mean(y_aux)).^2) - sum(residuals(aux).^2)
    stat   = max(ess, 0.0) / (π^2 / 2)
    return HarveyTestResult(n, stat, size(Z_full, 2) - 1)
end

function harvey_test(formula::FormulaTerm, data; Z = nothing, kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return harvey_test(X, y; intercept = false, Z = _extract_Z_maybe(Z, data), kwargs...)
end

StatsAPI.dof(t::HarveyTestResult)    = t.dof
StatsAPI.pvalue(t::HarveyTestResult) = ccdf(Chisq(t.dof), t.lm)

function Base.show(io::IO, t::HarveyTestResult)
    println(io, "Harvey test for multiplicative heteroskedasticity (exponential link)")
    println(io, "  H₀: γ = 0  in  log(σᵢ²) = γ₀ + zᵢ'γ")
    println(io, "  LM = $(round(t.lm, digits=4))   df = $(t.dof)   p-value = $(round(StatsAPI.pvalue(t), digits=4))")
end

"""
Result of [`glejser_test`](@ref). Fields: `n` (sample size), `lm` (LM statistic), `dof` (degrees of freedom).
"""
struct GlejserTestResult
    n::Int
    lm::Float64
    dof::Int
end

"""
    glejser_test(X, y; intercept=true, Z=nothing) -> GlejserTestResult
    glejser_test(formula, data; Z=nothing) -> GlejserTestResult

Lagrange multiplier test for heteroskedasticity with a linear standard deviation
(Glejser 1969).

Tests H₀: γ = 0 in the quadratic variance model σᵢ = γ₀ + zᵢ'γ.

The test statistic is

    LM = ESS_|û| / (σ̂² · (1 − 2/π))

where ESS_|û| is the explained sum of squares from the auxiliary OLS regression
of |ûᵢ| on Z, σ̂² = (1/n)Σûᵢ² is the OLS variance estimate from the main
regression, and (1 − 2/π) = Var(|N(0,1)|) is the variance of the half-normal.
The statistic is distributed as χ²(p) under H₀, where p = dim(γ) (columns of
Z excluding the intercept).

This is Glejser's (1969) exact score-test formula.  It differs from the
generic LM form n·R² but is asymptotically equivalent.

# Arguments
- `X`        : n × k regressor matrix without a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, without a
               constant column (default: `X`). A constant is prepended internally.

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American
Statistical Association*, 64(325), 316–323.
"""
function glejser_test(X::AbstractMatrix, y::AbstractVector;
                      intercept::Bool = true, Z = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    e      = residuals(lm(X_full, y))
    y_abs  = abs.(e)
    aux    = lm(Z_full, y_abs)
    σ̂²    = sum(e.^2) / n
    ess    = sum((y_abs .- mean(y_abs)).^2) - sum(residuals(aux).^2)
    stat   = max(ess, 0.0) / (σ̂² * (1 - 2/π))
    return GlejserTestResult(n, stat, size(Z_full, 2) - 1)
end

function glejser_test(formula::FormulaTerm, data; Z = nothing, kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return glejser_test(X, y; intercept = false, Z = _extract_Z_maybe(Z, data), kwargs...)
end

StatsAPI.dof(t::GlejserTestResult)    = t.dof
StatsAPI.pvalue(t::GlejserTestResult) = ccdf(Chisq(t.dof), t.lm)

function Base.show(io::IO, t::GlejserTestResult)
    println(io, "Glejser test for heteroskedasticity (quadratic link / linear SD)")
    println(io, "  H₀: γ = 0  in  σᵢ = γ₀ + zᵢ'γ")
    println(io, "  LM = $(round(t.lm, digits=4))   df = $(t.dof)   p-value = $(round(StatsAPI.pvalue(t), digits=4))")
end

# ---------------------------------------------------------------------------
# Thin wrappers around HypothesisTests — HARE-convention constructors
# (X without intercept, y as response) that run OLS internally.
# ---------------------------------------------------------------------------

"""
    breusch_pagan_test(X, y; intercept=true) -> WhiteTest
    breusch_pagan_test(formula, data) -> WhiteTest

Breusch-Pagan / Koenker LM test for heteroskedasticity (linear variance link).

Tests H₀: σᵢ² is constant against σᵢ² = f(Xβ). The LM statistic n·R²
from the regression of ê² on X is distributed as χ²(k−1) under H₀,
where k is the number of columns in X (including the intercept).

Returns a `HypothesisTests.WhiteTest` with `type = :linear`; call `pvalue`
on the result as usual.

# References
Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity
and random coefficient variation. *Econometrica*, 47(5), 1287–1294.

Koenker, R. (1981). A note on studentizing a test for heteroscedasticity.
*Journal of Econometrics*, 17(1), 107–112.
"""
function breusch_pagan_test(X::AbstractMatrix, y::AbstractVector;
                             intercept::Bool = true)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    e      = residuals(lm(X_full, y))
    return WhiteTest(float.(X_full), float.(e); type = :linear)
end

function breusch_pagan_test(formula::FormulaTerm, data; kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return breusch_pagan_test(X, y; intercept = false, kwargs...)
end

"""
    white_test(X, y; intercept=true, type=:White) -> WhiteTest
    white_test(formula, data; type=:White) -> WhiteTest

White's general test for heteroskedasticity.

The `type` keyword selects which auxiliary regressors are used:
- `:White` (default) — linear terms, squares, and cross-products.
- `:linear_and_squares` — linear terms and squares only (no cross-products).
- `:linear` — linear terms only (equivalent to Breusch-Pagan/Koenker).

The LM statistic n·R² from the regression of ê² on the selected terms
is distributed as χ²(dof) under H₀ of homoskedasticity.

Returns a `HypothesisTests.WhiteTest`; call `pvalue` on the result as usual.

# References
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
"""
function white_test(X::AbstractMatrix, y::AbstractVector;
                    intercept::Bool = true, type::Symbol = :White)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    e      = residuals(lm(X_full, y))
    return WhiteTest(float.(X_full), float.(e); type = type)
end

function white_test(formula::FormulaTerm, data; kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return white_test(X, y; intercept = false, kwargs...)
end

"""
    durbin_watson_test(X, y; intercept=true) -> DurbinWatsonTest
    durbin_watson_test(formula, data) -> DurbinWatsonTest

Durbin-Watson test for first-order serial correlation in OLS residuals.

The test statistic DW = Σ(eₜ − eₜ₋₁)² / Σeₜ² lies in [0, 4]; values
near 2 indicate no autocorrelation, near 0 positive, near 4 negative.

P-values use Pan's exact algorithm for n < 100 and a normal approximation
otherwise. One-sided p-values can be obtained with
`pvalue(t; tail = :right)` (positive autocorrelation) or
`pvalue(t; tail = :left)` (negative autocorrelation).

Returns a `HypothesisTests.DurbinWatsonTest`; call `pvalue` on the result
as usual.

Note: the test is invalid if X contains a lagged dependent variable.

# References
Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least
squares regression, II. *Biometrika*, 38(1–2), 159–177.
"""
function durbin_watson_test(X::AbstractMatrix, y::AbstractVector;
                             intercept::Bool = true)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    e      = residuals(lm(X_full, y))
    return DurbinWatsonTest(float.(X_full), float.(e))
end

function durbin_watson_test(formula::FormulaTerm, data; kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return durbin_watson_test(X, y; intercept = false, kwargs...)
end

"""
    breusch_godfrey_test(X, y, lag; intercept=true) -> BreuschGodfreyTest
    breusch_godfrey_test(formula, data, lag) -> BreuschGodfreyTest

Breusch-Godfrey LM test for serial correlation up to order `lag`.

Unlike the Durbin-Watson test, this test remains valid when X contains
lagged dependent variables. The LM statistic is distributed as χ²(lag)
under H₀ of no serial correlation.

Returns a `HypothesisTests.BreuschGodfreyTest`; call `pvalue` on the
result as usual.

# References
Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models.
*Australian Economic Papers*, 17(31), 334–355.

Godfrey, L. G. (1978). Testing against general autoregressive and moving
average error models when the regressors include lagged dependent variables.
*Econometrica*, 46(6), 1293–1301.
"""
function breusch_godfrey_test(X::AbstractMatrix, y::AbstractVector, lag::Int;
                               intercept::Bool = true)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    e      = residuals(lm(X_full, y))
    return BreuschGodfreyTest(float.(X_full), float.(e), lag)
end

function breusch_godfrey_test(formula::FormulaTerm, data, lag::Int; kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return breusch_godfrey_test(X, y, lag; intercept = false, kwargs...)
end

# ---------------------------------------------------------------------------

"""
Result of a Wald test for linear restrictions R*beta = r.

Fields: `stat` (F statistic), `df` (numerator), `df_residual` (denominator), `pvalue`.
"""
struct WaldTestResult
    stat::Float64
    df::Int
    df_residual::Int
    pvalue::Float64
end

function Base.show(io::IO, w::WaldTestResult)
    println(io, "Wald test (F):")
    println(io, "F($(w.df), $(w.df_residual)) = $(round(w.stat, digits=4))   p-value = $(round(w.pvalue, digits=4))")
end

"""
Result of a likelihood ratio test comparing two nested Beach-MacKinnon models.

Fields: `stat` (LR statistic), `df`, `pvalue`.
"""
struct LRTestResult
    stat::Float64
    df::Int
    pvalue::Float64
end

function Base.show(io::IO, lr::LRTestResult)
    println(io, "Likelihood ratio test:")
    println(io, "LR = $(round(lr.stat, digits=4))   df = $(lr.df)   p-value = $(round(lr.pvalue, digits=4))")
end

"""
    wald_test(m, R, r = zeros(size(R, 1))) -> WaldTestResult

Wald test for the linear restriction H0: R*beta = r.

`R` is a q x k restriction matrix; `r` is a q-vector (default: zero vector).
The test statistic F = (R*beta_hat - r)' * (R*V*R')^(-1) * (R*beta_hat - r) / q
is compared to F(q, n-k).
"""
function wald_test(m::HAREModel, R::AbstractMatrix,
                   r::AbstractVector = zeros(size(R, 1)))
    size(R, 2) == length(coef(m)) ||
        throw(DimensionMismatch("R has $(size(R,2)) columns but model has $(length(coef(m))) coefficients"))
    size(R, 1) == length(r) ||
        throw(DimensionMismatch("R has $(size(R,1)) rows but r has $(length(r)) elements"))

    q    = size(R, 1)
    diff = R * coef(m) .- r
    F    = dot(diff, (R * vcov(m) * R') \ diff) / q
    p    = ccdf(FDist(q, dof_residual(m)), F)
    return WaldTestResult(F, q, dof_residual(m), p)
end

"""
    lrtest(m_r, m_u) -> LRTestResult

Likelihood ratio test comparing two nested Beach-MacKinnon models.

`m_r` is the restricted (fewer parameters) model; `m_u` is the unrestricted model.
LR = 2*(L_u - L_r) ~ chi^2(q) where q = dof(m_u) - dof(m_r).
"""
function StatsModels.lrtest(m_r::AR1Result, m_u::AR1Result)
    df = dof(m_u) - dof(m_r)
    df > 0 || throw(ArgumentError(
        "unrestricted model must have more parameters than restricted model"))
    stat = 2 * (loglikelihood(m_u) - loglikelihood(m_r))
    return LRTestResult(stat, df, ccdf(Chisq(df), stat))
end
