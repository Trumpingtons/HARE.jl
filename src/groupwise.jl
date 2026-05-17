"""
Groupwise heteroskedasticity estimators.

The model is:
    y_i = x_i' beta + u_i,    Var(u_i) = sigma_g^2   for all i in group g.

The variance is constant within each group and unrestricted across groups.
The MLE for sigma_g^2 is the within-group sample variance of residuals; the
iterated FWLS therefore converges to the MLE at convergence (Greene §9.7.2).
"""

function _gw_labels(groups)
    seen = Dict{eltype(groups), Nothing}()
    labels = eltype(groups)[]
    for g in groups
        if !haskey(seen, g)
            seen[g] = nothing
            push!(labels, g)
        end
    end
    return labels
end

function _gw_sigma2(u, groups, glabels)
    [mean(u[groups .== g].^2) for g in glabels]
end

function _gw_weights(groups, glabels, sigma2)
    g_to_s2 = Dict(zip(glabels, sigma2))
    [1.0 / g_to_s2[g] for g in groups]
end

function _gw_loglik(u, groups, glabels, sigma2)
    n  = length(u)
    ll = -n / 2 * log(2π)
    g_to_s2 = Dict(zip(glabels, sigma2))
    for g in glabels
        idx = groups .== g
        s2  = g_to_s2[g]
        ll -= sum(idx) / 2 * log(s2) + sum(u[idx].^2) / (2 * s2)
    end
    return ll
end

"""
    two_step_groupwise(X, y, groups; intercept=true) -> GroupwiseResult
    two_step_groupwise(formula, data, groups) -> GroupwiseResult

Two-step Feasible GLS for groupwise heteroskedasticity (Greene §9.7.2).

**Step 1.** OLS → estimate σ̂²_g = (1/n_g) Σ_{i∈g} û_i² for each group g.
**Step 2.** WLS with weights w_i = 1/σ̂²_g.

Standard errors are conditional on σ̂²_g (WLS sandwich).

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `groups`  : length-n vector of group labels (any type: integers, strings, …).
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).

# Returns
[`GroupwiseResult`](@ref).

# References
Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, §9.7.2.
"""
function two_step_groupwise(X, y, groups; intercept::Bool = true,
                            coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                            mf = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    glabels = _gw_labels(groups)
    gsizes  = [count(==(g), groups) for g in glabels]

    u      = residuals(lm(X_full, y))
    sigma2 = _gw_sigma2(u, groups, glabels)
    w      = _gw_weights(groups, glabels, sigma2)
    model  = lm(X_full, y, weights = w)
    beta   = coef(model)
    f      = X_full * beta
    ll     = _gw_loglik(y .- f, groups, glabels, sigma2)
    return GroupwiseResult(beta, coefnames, mf, vcov(model), y .- f, f,
                           glabels, gsizes, sigma2, ll, 1, true)
end

function two_step_groupwise(formula::FormulaTerm, data, groups; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return two_step_groupwise(X, y, groups; intercept = false, coefnames = cn, mf = mf, kwargs...)
end

"""
    iterated_groupwise(X, y, groups; intercept=true, tol=1e-8, maxiter=100) -> GroupwiseResult
    iterated_groupwise(formula, data, groups; tol=1e-8, maxiter=100) -> GroupwiseResult

Iterated Feasible GLS for groupwise heteroskedasticity (Greene §9.7.2).

Each iteration:
1. Update σ̂²_g = (1/n_g) Σ_{i∈g} û_i² from current residuals.
2. Update β by WLS with weights w_i = 1/σ̂²_g.

Convergence criterion: max|β(i) − β(i−1)| < tol.

At convergence this is equivalent to the joint MLE of (β, σ²_1, …, σ²_G),
since the MLE first-order conditions for σ²_g reduce to the within-group
sample variance.

# Arguments
- `X`       : n × k regressor matrix **without** a constant column.
- `y`       : response vector of length n.
- `groups`  : length-n vector of group labels.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula` : `@formula` expression (formula method).
- `data`    : Tables.jl-compatible data source (formula method).
- `tol`     : convergence tolerance (default `1e-8`).
- `maxiter` : maximum number of iterations (default `100`).

# Returns
[`GroupwiseResult`](@ref).

# References
Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, §9.7.2.
"""
function iterated_groupwise(X, y, groups; intercept::Bool = true, tol = 1e-8, maxiter = 100,
                            coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                            mf = nothing)
    n      = length(y)
    X_full = intercept ? hcat(ones(eltype(X), n), X) : X
    glabels = _gw_labels(groups)
    gsizes  = [count(==(g), groups) for g in glabels]

    model  = lm(X_full, y)
    beta   = coef(model)
    sigma2 = _gw_sigma2(residuals(model), groups, glabels)

    for i in 1:maxiter
        beta_old = copy(beta)
        w        = _gw_weights(groups, glabels, sigma2)
        model    = lm(X_full, y, weights = w)
        beta     = coef(model)
        sigma2   = _gw_sigma2(y .- X_full * beta, groups, glabels)

        if maximum(abs.(beta .- beta_old)) < tol
            f  = X_full * beta
            ll = _gw_loglik(y .- f, groups, glabels, sigma2)
            return GroupwiseResult(beta, coefnames, mf, vcov(model), y .- f, f,
                                   glabels, gsizes, sigma2, ll, i, true)
        end
    end

    f  = X_full * beta
    ll = _gw_loglik(y .- f, groups, glabels, sigma2)
    return GroupwiseResult(beta, coefnames, mf, vcov(model), y .- f, f,
                           glabels, gsizes, sigma2, ll, maxiter, false)
end

function iterated_groupwise(formula::FormulaTerm, data, groups; kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return iterated_groupwise(X, y, groups; intercept = false, coefnames = cn, mf = mf, kwargs...)
end
