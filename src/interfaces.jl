"""
StatsBase.jl interface implementations for HAREModel result objects.

Implements the standard statistical interface for all result types:
`coef`, `vcov`, `stderror`, `residuals`, `fitted`, `predict`.
"""

StatsBase.islinear(m::HAREModel)                    = true
StatsBase.loglikelihood(m::BeachMacKinnonResult)   = m.loglik
StatsBase.loglikelihood(m::JointResult)            = m.loglik
StatsBase.responsename(m::HAREModel)      = isnothing(m.mf) ? nothing : string(m.mf.f.lhs)
StatsBase.coef(m::HAREModel)              = m.coef
StatsBase.coefnames(m::HAREModel)         = m.coefnames
StatsBase.vcov(m::HAREModel)              = m.vcov
StatsBase.stderror(m::HAREModel)          = sqrt.(diag(m.vcov))
StatsBase.residuals(m::HAREModel)         = m.residuals
StatsBase.response(m::HAREModel)          = fitted(m) .+ residuals(m)
StatsBase.fitted(m::HAREModel)            = m.fitted
StatsBase.predict(m::HAREModel)           = fitted(m)
StatsBase.predict(m::HAREModel, Xnew::AbstractMatrix) = Xnew * coef(m)

function StatsBase.predict(m::HAREModel, newdata)
    isnothing(m.mf) && throw(ArgumentError(
        "Model was fitted from a matrix; use predict(m, Xnew::AbstractMatrix) instead"))
    return Matrix{Float64}(modelcols(m.mf.f.rhs, newdata)) * coef(m)
end

StatsModels.formula(m::HAREModel)         = isnothing(m.mf) ? nothing : m.mf.f
StatsModels.termnames(m::HAREModel)       = isnothing(m.mf) ? nothing : StatsModels.termnames(m.mf.f.rhs)
StatsBase.nobs(m::HAREModel)              = length(m.residuals)
StatsBase.dof(m::HAREModel)               = length(m.coef)
StatsBase.dof_residual(m::HAREModel)      = nobs(m) - dof(m)

tstat(m::HAREModel)                       = coef(m) ./ stderror(m)

function pvalues(m::HAREModel; dist=:t)
    d = dist === :t ? TDist(dof_residual(m)) : Normal()
    return 2 .* ccdf.(d, abs.(tstat(m)))
end

StatsBase.rss(m::HAREModel)               = sum(abs2, residuals(m))
sigma2(m::HAREModel)                      = StatsBase.rss(m) / dof_residual(m)

function StatsBase.r2(m::HAREModel)
    y = response(m)
    return 1 - StatsBase.rss(m) / sum(abs2, y .- mean(y))
end

StatsBase.adjr2(m::HAREModel)             = 1 - (1 - r2(m)) * (nobs(m) - 1) / dof_residual(m)

function StatsBase.confint(m::HAREModel; level::Real=0.95, dist=:t)
    d = dist === :t ? TDist(dof_residual(m)) : Normal()
    q = quantile(d, (1 + level) / 2)
    return hcat(coef(m) .- q .* stderror(m), coef(m) .+ q .* stderror(m))
end
