"""
Result of a Wald test for linear restrictions Rβ = r.

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
Result of a likelihood ratio test comparing two nested Beach–MacKinnon models.

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

Wald test for the linear restriction H₀: Rβ = r.

`R` is a q × k restriction matrix; `r` is a q-vector (default: zero vector).
The test statistic F = (Rβ̂ − r)'(RVR')⁻¹(Rβ̂ − r) / q is compared to F(q, n−k).
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

Likelihood ratio test comparing two nested Beach–MacKinnon models.

`m_r` is the restricted (fewer parameters) model; `m_u` is the unrestricted model.
LR = 2(ℓᵤ − ℓᵣ) ~ χ²(q) where q = dof(m_u) − dof(m_r).
"""
function StatsModels.lrtest(m_r::BeachMacKinnonResult, m_u::BeachMacKinnonResult)
    df = dof(m_u) - dof(m_r)
    df > 0 || throw(ArgumentError(
        "unrestricted model must have more parameters than restricted model"))
    stat = 2 * (loglikelihood(m_u) - loglikelihood(m_r))
    return LRTestResult(stat, df, ccdf(Chisq(df), stat))
end
