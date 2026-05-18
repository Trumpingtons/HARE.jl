"""
    HAREModel

Abstract supertype for all HARE estimator result objects.
Subtypes support the StatsBase interface: `coef`, `stderror`, `vcov`,
`residuals`, `fitted`, and `predict`.
"""
abstract type HAREModel <: StatsBase.RegressionModel end

"""
    FWLSResult <: HAREModel

Result of a Feasible Weighted Least Squares (FWLS) estimator for
heteroskedastic regression. Returned by [`exponential_fwls`](@ref),
[`exponential_ifwls`](@ref), [`quadratic_fwls`](@ref),
[`quadratic_ifwls`](@ref), [`linear_fwls`](@ref), and [`linear_ifwls`](@ref).

# Fields
- `coef`       : coefficient vector beta_hat.
- `coefnames`  : coefficient names.
- `mf`         : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`       : estimated covariance matrix of beta_hat.
- `residuals`  : residuals in the original space, y - X*beta_hat.
- `fitted`     : fitted values X*beta_hat.
- `gamma`      : estimated variance-equation coefficients gamma_hat.
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from the auxiliary OLS regression).
- `iterations` : number of FWLS iterations performed.
- `converged`  : `true` if the convergence criterion was satisfied.
- `link`       : variance model — `:exponential`, `:quadratic`, or `:linear`.
"""
struct FWLSResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    gamma::Vector{Float64}
    gamma_vcov::Matrix{Float64}
    iterations::Int
    converged::Bool
    link::Symbol
end

"""
    AR1Result <: HAREModel

Result of an AR(1) serial-correlation estimator. Returned by
[`two_step_prais_winsten`](@ref), [`iterated_prais_winsten`](@ref),
[`two_step_cochrane_orcutt`](@ref), [`iterated_cochrane_orcutt`](@ref),
[`hildreth_lu`](@ref), and [`beach_mackinnon`](@ref).

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : estimated AR(1) coefficient rho_hat.
- `loglik`    : maximised log-likelihood (`:beach_mackinnon` only; `NaN` otherwise).
- `rss`       : minimised transformed RSS (`:hildreth_lu` only; `NaN` otherwise).
- `iterations`: iterations performed (grid points for `:hildreth_lu`; `0` for `:beach_mackinnon`).
- `converged` : `true` if the convergence criterion was satisfied.
- `method`    : estimator used — `:prais_winsten`, `:cochrane_orcutt`, `:hildreth_lu`, or `:beach_mackinnon`.
"""
struct AR1Result <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    loglik::Float64
    rss::Float64
    iterations::Int
    converged::Bool
    method::Symbol
end

"""
    SequentialResult <: HAREModel

Result of the Sequential HARE estimator ([`two_step_sequential`](@ref) or
[`iterated_sequential`](@ref)). Corrects for AR(1) autocorrelation first
(Prais-Winsten), then multiplicative heteroskedasticity (Harvey), sequentially.

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : estimated AR(1) coefficient rho_hat.
- `gamma`      : estimated log-variance coefficients gamma_hat (from the Harvey step).
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from the auxiliary OLS regression).
- `iterations` : number of sequential iterations performed.
- `converged`  : `true` if the joint convergence criterion was satisfied.
"""
struct SequentialResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    gamma::Vector{Float64}
    gamma_vcov::Matrix{Float64}
    iterations::Int
    converged::Bool
end


"""
    JointResult <: HAREModel

Result of the Joint HARE estimator ([`two_step_joint`](@ref) or
[`iterated_joint`](@ref)). Estimates AR(1) autocorrelation and multiplicative
heteroskedasticity simultaneously via maximum likelihood.

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat (conditional on rho_hat, gamma_hat).
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : estimated AR(1) coefficient rho_hat.
- `gamma`      : estimated log-variance coefficients gamma_hat (for `log(sigma_t^2) = z_t' * gamma`).
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from the observed information matrix).
- `loglik`     : log-likelihood at the final estimates.
- `iterations` : number of iterations performed.
- `converged`  : `true` if the convergence criterion was satisfied.
"""
struct JointResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    gamma::Vector{Float64}
    gamma_vcov::Matrix{Float64}
    loglik::Float64
    iterations::Int
    converged::Bool
end

"""
    MLEResult <: HAREModel

Result of a heteroskedastic regression MLE estimator ([`exponential_mle`](@ref),
[`quadratic_mle`](@ref), or [`linear_mle`](@ref)).

# Fields
- `coef`       : coefficient vector beta_hat.
- `coefnames`  : coefficient names.
- `mf`         : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`       : estimated covariance matrix of beta_hat (from observed information).
- `residuals`  : residuals in the original space, y - X*beta_hat.
- `fitted`     : fitted values X*beta_hat.
- `gamma`      : estimated variance equation coefficients gamma_hat (including intercept).
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from observed information).
- `loglik`     : maximised log-likelihood value.
- `link`       : link function — `:exponential`, `:quadratic`, or `:linear`.
- `converged`  : `true` if the L-BFGS convergence criterion was satisfied.
- `iterations` : number of L-BFGS iterations performed.
"""
struct MLEResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    gamma::Vector{Float64}
    gamma_vcov::Matrix{Float64}
    loglik::Float64
    link::Symbol
    converged::Bool
    iterations::Int
end

"""
    GroupwiseResult <: HAREModel

Result of a groupwise heteroskedasticity estimator ([`two_step_groupwise`](@ref),
[`iterated_groupwise`](@ref), or [`groupwise_mle`](@ref)).

# Fields
- `coef`        : coefficient vector beta_hat.
- `coefnames`   : coefficient names.
- `mf`          : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`        : estimated covariance matrix of beta_hat. For `groupwise_mle` this is (X′WX)⁻¹; for FWLS estimators it is GLM's dispersion-scaled WLS covariance.
- `residuals`   : residuals in the original space, y - X*beta_hat.
- `fitted`      : fitted values X*beta_hat.
- `group_labels`: group identifiers (one per group, in order of first occurrence).
- `group_sizes` : number of observations per group.
- `sigma2`      : estimated group variances σ̂²_g, one per group.
- `loglik`      : log-likelihood at the final estimates.
- `iterations`  : number of FWLS iterations performed.
- `converged`   : `true` if the convergence criterion was satisfied.
"""
struct GroupwiseResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    group_labels::Vector
    group_sizes::Vector{Int}
    sigma2::Vector{Float64}
    loglik::Float64
    iterations::Int
    converged::Bool
end

