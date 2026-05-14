function StatsBase.coeftable(m::HAREModel; level::Real=0.95)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    ci = confint(m; level=level)
    StatsBase.CoefTable(
        hcat(coef(m), stderror(m), tstat(m), pvalues(m), ci[:,1], ci[:,2]),
        ["Coef.", "Std. Error", "t", "Pr(>|t|)", "Lower $(levstr)%", "Upper $(levstr)%"],
        coefnames(m), 4, 3
    )
end

_show_stats(io::IO, m::HAREModel) =
    println(io, "N: $(nobs(m))   R²: $(round(r2(m), digits=4))   Adj. R²: $(round(adjr2(m), digits=4))")

_show_extra(io::IO, m::HarveyResult) =
    println(io, "gamma: [$(join(round.(m.gamma, digits=4), ", "))]   Converged: $(m.converged)   Iterations: $(m.iterations)")

_show_extra(io::IO, m::GlejserResult) =
    println(io, "gamma: [$(join(round.(m.gamma, digits=4), ", "))]   Converged: $(m.converged)   Iterations: $(m.iterations)")

_show_extra(io::IO, m::PraisWinstenResult) =
    println(io, "rho: $(round(m.rho, digits=4))   Converged: $(m.converged)   Iterations: $(m.iterations)")

_show_extra(io::IO, m::HildrethLuResult) =
    println(io, "rho: $(round(m.rho, digits=4))   Grid points: $(m.iterations)")

_show_extra(io::IO, m::SequentialResult) =
    println(io, "rho: $(round(m.rho, digits=4))   gamma: [$(join(round.(m.gamma, digits=4), ", "))]   Converged: $(m.converged)   Iterations: $(m.iterations)")

_show_extra(io::IO, m::JointResult) =
    println(io, "rho: $(round(m.rho, digits=4))   gamma: [$(join(round.(m.gamma, digits=4), ", "))]   Log-likelihood: $(round(m.loglik, digits=4))   Converged: $(m.converged)   Iterations: $(m.iterations)")

_show_extra(io::IO, m::BeachMacKinnonResult) =
    println(io, "rho: $(round(m.rho, digits=4))   Log-likelihood: $(round(m.loglik, digits=4))   Converged: $(m.converged)")

function Base.show(io::IO, m::HAREModel)
    println(io, "$(typeof(m)):\n\nCoefficients:")
    show(io, MIME("text/plain"), coeftable(m))
    println(io)
    _show_extra(io, m)
    _show_stats(io, m)
    return nothing
end
