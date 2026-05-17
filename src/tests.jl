"""
    HarveyTest(X, y; intercept=true, Z=nothing) -> HarveyTest
    HarveyTest(formula, data; Z=nothing) -> HarveyTest

Lagrange multiplier test for multiplicative heteroskedasticity (Harvey 1976).

Tests H₀: γ = 0 in the exponential variance model log(σᵢ²) = γ₀ + zᵢ'γ.

The test statistic is LM = n·R² from the auxiliary regression of
log(ûᵢ²) − c on Z (where c = E[log(χ²(1))] = −1.2703628454614782),
distributed as χ²(p) under H₀, where p = dim(γ) (columns of Z excluding
the intercept).

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
struct HarveyTest
    n::Int
    lm::Float64
    dof::Int
end

function HarveyTest(X::AbstractMatrix, y::AbstractVector;
                    intercept::Bool = true, Z = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    e      = residuals(lm(X_full, y))
    aux    = lm(Z_full, log.(e.^2) .- _HARVEY_C)
    stat   = n * max(r2(aux), 0.0)
    return HarveyTest(n, stat, size(Z_full, 2) - 1)
end

function HarveyTest(formula::FormulaTerm, data; Z = nothing, kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return HarveyTest(X, y; intercept = false, Z = Z, kwargs...)
end

StatsAPI.dof(t::HarveyTest)    = t.dof
StatsAPI.pvalue(t::HarveyTest) = ccdf(Chisq(t.dof), t.lm)

function Base.show(io::IO, t::HarveyTest)
    println(io, "Harvey test for multiplicative heteroskedasticity (exponential link)")
    println(io, "  H₀: γ = 0  in  log(σᵢ²) = γ₀ + zᵢ'γ")
    println(io, "  LM = $(round(t.lm, digits=4))   df = $(t.dof)   p-value = $(round(StatsAPI.pvalue(t), digits=4))")
end

"""
    GlejserTest(X, y; intercept=true, Z=nothing) -> GlejserTest
    GlejserTest(formula, data; Z=nothing) -> GlejserTest

Lagrange multiplier test for heteroskedasticity with a linear standard deviation
(Glejser 1969).

Tests H₀: γ = 0 in the quadratic variance model σᵢ = γ₀ + zᵢ'γ.

The test statistic is LM = n·R² from the auxiliary regression of |ûᵢ| on Z,
distributed as χ²(p) under H₀, where p = dim(γ) (columns of Z excluding the
intercept).  The sqrt(π/2) scaling used in estimation cancels in R² and does
not affect the test statistic.

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
struct GlejserTest
    n::Int
    lm::Float64
    dof::Int
end

function GlejserTest(X::AbstractMatrix, y::AbstractVector;
                     intercept::Bool = true, Z = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    Z_full = isnothing(Z) ? X_full : hcat(ones(n), Z)
    e      = residuals(lm(X_full, y))
    aux    = lm(Z_full, abs.(e))
    stat   = n * max(r2(aux), 0.0)
    return GlejserTest(n, stat, size(Z_full, 2) - 1)
end

function GlejserTest(formula::FormulaTerm, data; Z = nothing, kwargs...)
    X, y, _, _ = _extract_Xy(formula, data)
    return GlejserTest(X, y; intercept = false, Z = Z, kwargs...)
end

StatsAPI.dof(t::GlejserTest)    = t.dof
StatsAPI.pvalue(t::GlejserTest) = ccdf(Chisq(t.dof), t.lm)

function Base.show(io::IO, t::GlejserTest)
    println(io, "Glejser test for heteroskedasticity (quadratic link / linear SD)")
    println(io, "  H₀: γ = 0  in  σᵢ = γ₀ + zᵢ'γ")
    println(io, "  LM = $(round(t.lm, digits=4))   df = $(t.dof)   p-value = $(round(StatsAPI.pvalue(t), digits=4))")
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
function StatsModels.lrtest(m_r::BeachMacKinnonResult, m_u::BeachMacKinnonResult)
    df = dof(m_u) - dof(m_r)
    df > 0 || throw(ArgumentError(
        "unrestricted model must have more parameters than restricted model"))
    stat = 2 * (loglikelihood(m_u) - loglikelihood(m_r))
    return LRTestResult(stat, df, ccdf(Chisq(df), stat))
end
