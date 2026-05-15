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
    println(io, "N: $(nobs(m))   R^2: $(round(r2(m), digits=4))   Adj. R^2: $(round(adjr2(m), digits=4))")

function _gamma_coeftable(gamma, gamma_vcov)
    p     = length(gamma)
    names = ["(Intercept)"; ["z$i" for i in 1:(p-1)]]
    se    = sqrt.(diag(gamma_vcov))
    t     = gamma ./ se
    pv    = 2 .* ccdf.(Normal(), abs.(t))
    q     = quantile(Normal(), 0.975)
    StatsBase.CoefTable(
        hcat(gamma, se, t, pv, gamma .- q .* se, gamma .+ q .* se),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        names, 4, 3
    )
end

function _show_extra(io::IO, m::HarveyResult)
    println(io, "Variance equation (gamma):")
    show(io, MIME("text/plain"), _gamma_coeftable(m.gamma, m.gamma_vcov))
    println(io)
    println(io, "Converged: $(m.converged)   Iterations: $(m.iterations)")
end

function _show_extra(io::IO, m::GlejserResult)
    println(io, "Std. deviation equation (gamma):")
    show(io, MIME("text/plain"), _gamma_coeftable(m.gamma, m.gamma_vcov))
    println(io)
    println(io, "Converged: $(m.converged)   Iterations: $(m.iterations)")
end

function _rho_coeftable(rho, n)
    se = sqrt((1 - rho^2) / n)
    z  = rho / se
    pv = 2 * ccdf(Normal(), abs(z))
    q  = quantile(Normal(), 0.975)
    StatsBase.CoefTable(
        hcat([rho], [se], [z], [pv], [rho - q*se], [rho + q*se]),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        ["ρ"], 4, 3
    )
end

function _show_extra(io::IO, m::PraisWinstenResult)
    println(io, "AR(1) coefficient:")
    show(io, MIME("text/plain"), _rho_coeftable(m.rho, nobs(m)))
    println(io)
    println(io, "Converged: $(m.converged)   Iterations: $(m.iterations)")
end

function _show_extra(io::IO, m::HildrethLuResult)
    println(io, "AR(1) coefficient:")
    show(io, MIME("text/plain"), _rho_coeftable(m.rho, nobs(m)))
    println(io)
    println(io, "Grid points: $(m.iterations)")
end

function _show_extra(io::IO, m::SequentialResult)
    println(io, "AR(1) coefficient:")
    show(io, MIME("text/plain"), _rho_coeftable(m.rho, nobs(m)))
    println(io)
    println(io, "Variance equation (gamma):")
    show(io, MIME("text/plain"), _gamma_coeftable(m.gamma, m.gamma_vcov))
    println(io)
    println(io, "Converged: $(m.converged)   Iterations: $(m.iterations)")
end

function _show_extra(io::IO, m::JointResult)
    println(io, "AR(1) coefficient:")
    show(io, MIME("text/plain"), _rho_coeftable(m.rho, nobs(m)))
    println(io)
    println(io, "Variance equation (gamma):")
    show(io, MIME("text/plain"), _gamma_coeftable(m.gamma, m.gamma_vcov))
    println(io)
    println(io, "Log-likelihood: $(round(m.loglik, digits=4))   Converged: $(m.converged)   Iterations: $(m.iterations)")
end

function _show_extra(io::IO, m::BeachMacKinnonResult)
    println(io, "AR(1) coefficient:")
    show(io, MIME("text/plain"), _rho_coeftable(m.rho, nobs(m)))
    println(io)
    println(io, "Log-likelihood: $(round(m.loglik, digits=4))   Converged: $(m.converged)")
end

function Base.show(io::IO, m::HAREModel)
    println(io, "Coefficients:")
    show(io, MIME("text/plain"), coeftable(m))
    println(io)
    _show_extra(io, m)
    _show_stats(io, m)
    return nothing
end
