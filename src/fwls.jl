"""
Feasible Weighted Least Squares (FWLS) estimators for heteroskedastic regression.

Four variance models are supported:

  - Exponential: σ²_i = exp(γ₀ + z_i'γ)   Harvey (1976)
  - Quadratic:   σ_i  = γ₀ + z_i'γ         Glejser (1969)
  - Linear:      σ²_i = γ₀ + z_i'γ
  - Groupwise:   σ²_i = σ²_g  for all i in group g

Each model has a two-step variant and an iterated variant.
"""

# --- Shared FWLS internals ---

function _fwls_two_step(X, y, Z, weight_fn, link, coefnames, mf)
    w, gamma, gamma_vcov = weight_fn(Z, residuals(lm(X, y)))
    model = lm(X, y, weights = w)
    beta  = coef(model)
    f     = X * beta
    return FWLSResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, 1, true, link)
end

function _fwls_iterated(X, y, Z, weight_fn, link, coefnames, mf; tol, maxiter)
    model      = lm(X, y)
    gamma      = zeros(size(Z, 2))
    gamma_vcov = zeros(size(Z, 2), size(Z, 2))
    for i in 1:maxiter
        beta_old             = coef(model)
        w, gamma, gamma_vcov = weight_fn(Z, residuals(model))
        model                = lm(X, y, weights = w)
        if maximum(abs.(coef(model) .- beta_old)) < tol
            beta = coef(model)
            f    = X * beta
            return FWLSResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, i, true, link)
        end
    end
    beta = coef(model)
    f    = X * beta
    return FWLSResult(beta, coefnames, mf, vcov(model), y .- f, f, gamma, gamma_vcov, maxiter, false, link)
end

# --- Exponential (Harvey 1976) ---

"""
    exponential_fwls(X, y; Z=nothing, intercept=true) -> FWLSResult
    exponential_fwls(formula, data; Z=nothing) -> FWLSResult

Two-step Feasible Weighted Least Squares for multiplicative heteroskedasticity
(Harvey 1976). Models the log-variance as a linear function of auxiliary regressors:

```math
\\begin{aligned}
\\sigma_i^2 &= \\exp(\\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma})
\\end{aligned}
```

**Step 1.** OLS → residuals û.
**Step 2.** Auxiliary regression `log(û²ᵢ) - c = Zᵢγ + vᵢ` → weights `wᵢ = 1/exp(Zᵢγ̂)`.
**Step 3.** WLS with weights w.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.

# Returns
[`FWLSResult`](@ref).

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function exponential_fwls(X, y; intercept::Bool = true, Z=nothing,
                           coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                           mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_two_step(X, y, Z, exponential_weights, :exponential, coefnames, mf)
end

function exponential_fwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return exponential_fwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    exponential_ifwls(X, y; Z=nothing, intercept=true, tol=1e-8, maxiter=100) -> FWLSResult
    exponential_ifwls(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> FWLSResult

Iterated FWLS for multiplicative heteroskedasticity (Harvey 1976). Repeats the
weight-update step until `max|β̂(i) - β̂(i-1)| < tol`.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum iterations (default `100`).

# Returns
[`FWLSResult`](@ref).

# References
Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.
"""
function exponential_ifwls(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                            coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                            mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_iterated(X, y, Z, exponential_weights, :exponential, coefnames, mf; tol=tol, maxiter=maxiter)
end

function exponential_ifwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return exponential_ifwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

# --- Quadratic (Glejser 1969) ---

"""
    quadratic_fwls(X, y; Z=nothing, intercept=true) -> FWLSResult
    quadratic_fwls(formula, data; Z=nothing) -> FWLSResult

Two-step Feasible Weighted Least Squares for heteroskedasticity with a linear
standard deviation (Glejser 1969):

```math
\\begin{aligned}
\\sigma_i &= \\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma}
\\end{aligned}
```

**Step 1.** OLS → residuals û.
**Step 2.** Auxiliary regression `|ûᵢ| = Zᵢγ + vᵢ` → weights `wᵢ = 1/σ̂²ᵢ`.
**Step 3.** WLS with weights w.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.

# Returns
[`FWLSResult`](@ref).

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function quadratic_fwls(X, y; intercept::Bool = true, Z=nothing,
                         coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                         mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_two_step(X, y, Z, quadratic_weights, :quadratic, coefnames, mf)
end

function quadratic_fwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return quadratic_fwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    quadratic_ifwls(X, y; Z=nothing, intercept=true, tol=1e-8, maxiter=100) -> FWLSResult
    quadratic_ifwls(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> FWLSResult

Iterated FWLS for heteroskedasticity with a linear standard deviation (Glejser 1969).
Repeats the weight-update step until `max|β̂(i) - β̂(i-1)| < tol`.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum iterations (default `100`).

# Returns
[`FWLSResult`](@ref).

# References
Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the
American Statistical Association*, 64(325), 316–323.
"""
function quadratic_ifwls(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                          coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                          mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_iterated(X, y, Z, quadratic_weights, :quadratic, coefnames, mf; tol=tol, maxiter=maxiter)
end

function quadratic_ifwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return quadratic_ifwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

# --- Linear ---

"""
    linear_fwls(X, y; Z=nothing, intercept=true) -> FWLSResult
    linear_fwls(formula, data; Z=nothing) -> FWLSResult

Two-step Feasible Weighted Least Squares for heteroskedasticity with a linear
variance function:

```math
\\begin{aligned}
\\sigma_i^2 &= \\gamma_0 + \\mathbf{z}_i^\\top\\boldsymbol{\\gamma}
\\end{aligned}
```

**Step 1.** OLS → residuals û.
**Step 2.** Auxiliary OLS of `û²ᵢ` on `Zᵢ` → weights `wᵢ = 1/σ̂²ᵢ`.
**Step 3.** WLS with weights w.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.

# Returns
[`FWLSResult`](@ref).
"""
function linear_fwls(X, y; intercept::Bool = true, Z=nothing,
                     coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                     mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_two_step(X, y, Z, linear_weights, :linear, coefnames, mf)
end

function linear_fwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return linear_fwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    linear_ifwls(X, y; Z=nothing, intercept=true, tol=1e-8, maxiter=100) -> FWLSResult
    linear_ifwls(formula, data; Z=nothing, tol=1e-8, maxiter=100) -> FWLSResult

Iterated FWLS for heteroskedasticity with a linear variance function. Repeats
the weight-update step until `max|β̂(i) - β̂(i-1)| < tol`.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix for the variance equation, **without**
               a constant column (default: `X`). A constant is prepended internally.
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum iterations (default `100`).

# Returns
[`FWLSResult`](@ref).
"""
function linear_ifwls(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                      coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                      mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_iterated(X, y, Z, linear_weights, :linear, coefnames, mf; tol=tol, maxiter=maxiter)
end

function linear_ifwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return linear_ifwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    power_fwls(X, y; Z, intercept=true) -> FWLSResult
    power_fwls(formula, data; Z) -> FWLSResult

Two-step FWLS for heteroskedasticity with a power variance function.
The variance is modelled as:

```math
\\sigma_i^2 = \\sigma^2 \\prod_j |z_{ij}|^{2\\delta_j}
```

Equivalently, ``\\log\\sigma_i^2 = \\log\\sigma^2 + 2\\delta_1\\log|z_{i1}| + \\cdots``,
which is the exponential (Harvey) model on ``\\log|Z|``. The reported ``\\boldsymbol{\\gamma} = (\\gamma_0, \\gamma_1, \\ldots)`` satisfies
``\\gamma_0 = \\log\\sigma^2`` and ``\\gamma_j = 2\\delta_j``. Standard errors are
exact — no delta-method approximation.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix **without** a constant column. A
               constant is prepended internally. **Required** for the power model.

# Returns
[`FWLSResult`](@ref) with `link = :power`.
"""
function power_fwls(X, y; intercept::Bool = true, Z=nothing,
                    coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                    mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_two_step(X, y, Z, power_weights, :power, coefnames, mf)
end

function power_fwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return power_fwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

"""
    power_ifwls(X, y; Z, intercept=true, tol=1e-8, maxiter=100) -> FWLSResult
    power_ifwls(formula, data; Z, tol=1e-8, maxiter=100) -> FWLSResult

Iterated FWLS for heteroskedasticity with a power variance function. Repeats
the weight-update step until `max|β̂(i) - β̂(i-1)| < tol`.

See [`power_fwls`](@ref) for the variance model definition.

# Arguments
- `X`        : n × k regressor matrix **without** a constant column.
- `y`        : response vector of length n.
- `intercept`: if `true` (default), a constant column is prepended to `X`.
- `formula`  : `@formula` expression (formula method).
- `data`     : Tables.jl-compatible data source (formula method).
- `Z`        : auxiliary regressor matrix **without** a constant column.
- `tol`      : convergence tolerance (default `1e-8`).
- `maxiter`  : maximum iterations (default `100`).

# Returns
[`FWLSResult`](@ref) with `link = :power`.
"""
function power_ifwls(X, y; intercept::Bool = true, Z=nothing, tol=1e-8, maxiter=100,
                     coefnames::Vector{String} = intercept ? ["(Intercept)"; ["x$i" for i in 1:size(X,2)]] : ["x$i" for i in 1:size(X,2)],
                     mf=nothing)
    n = size(X, 1)
    X = intercept ? hcat(ones(eltype(X), n), X) : X
    Z = isnothing(Z) ? X : hcat(ones(n), Z)
    return _fwls_iterated(X, y, Z, power_weights, :power, coefnames, mf; tol=tol, maxiter=maxiter)
end

function power_ifwls(formula::FormulaTerm, data; Z=nothing, kwargs...)
    X, y, cn, mf = _extract_Xy(formula, data)
    return power_ifwls(X, y; intercept=false, Z=_extract_Z_maybe(Z, data), coefnames=cn, mf=mf, kwargs...)
end

# --- Groupwise ---

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
