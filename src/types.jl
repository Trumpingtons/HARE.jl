"""
    HAREModel

Abstract supertype for all HARE estimator result objects.
Subtypes support the StatsBase interface: `coef`, `stderror`, `vcov`,
`residuals`, `fitted`, and `predict`.
"""
abstract type HAREModel <: StatsBase.RegressionModel end

"""
    HarveyResult <: HAREModel

Result of a Feasible WLS estimator ([`two_step_harvey`](@ref) or
[`iterated_harvey`](@ref)).

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : OLS-style residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `gamma`      : estimated log-variance coefficients gamma_hat (from `log(u_hat_i^2) = X_i * gamma`).
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from the auxiliary OLS regression).
- `iterations` : number of FWLS iterations performed.
- `converged`  : `true` if the convergence criterion was satisfied.
"""
struct HarveyResult <: HAREModel
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
end

"""
    PraisWinstenResult <: HAREModel

Result of a Prais-Winsten estimator ([`two_step_prais_winsten`](@ref) or
[`iterated_prais_winsten`](@ref)).

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : estimated AR(1) coefficient rho_hat.
- `iterations`: number of Cochrane-Orcutt iterations performed.
- `converged` : `true` if |rho(i) - rho(i-1)| < tolerance.
"""
struct PraisWinstenResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    iterations::Int
    converged::Bool
end

"""
    HildrethLuResult <: HAREModel

Result of the Hildreth-Lu grid-search estimator ([`hildreth_lu`](@ref)).

# Fields
- `coef`      : coefficient vector beta_hat at the optimal rho.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : grid value of rho_hat that minimised RSS.
- `rss`       : minimised residual sum of squares.
- `iterations`: number of grid points evaluated.
- `converged` : always `true` (grid search always completes).
"""
struct HildrethLuResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    rss::Float64
    iterations::Int
    converged::Bool
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
    GlejserResult <: HAREModel

Result of a Glejser FWLS estimator ([`two_step_glejser`](@ref) or
[`iterated_glejser`](@ref)).

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat.
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `gamma`      : estimated std-dev coefficients gamma_hat (from `|u_hat_i| = Z_i * gamma`).
- `gamma_vcov` : estimated covariance matrix of gamma_hat (from the auxiliary OLS regression).
- `iterations` : number of FWLS iterations performed.
- `converged`  : `true` if the convergence criterion was satisfied.
"""
struct GlejserResult <: HAREModel
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
    BeachMacKinnonResult <: HAREModel

Result of the Beach-MacKinnon exact MLE ([`beach_mackinnon`](@ref)).

# Fields
- `coef`      : coefficient vector beta_hat.
- `coefnames` : coefficient names.
- `mf`        : `ModelFrame` from formula fit; `nothing` for matrix fit.
- `vcov`      : estimated covariance matrix of beta_hat (conditional on rho_hat).
- `residuals` : residuals in the original space, y - X*beta_hat.
- `fitted`    : fitted values X*beta_hat.
- `rho`       : MLE of the AR(1) coefficient rho_hat.
- `loglik`    : maximised concentrated log-likelihood value.
- `converged` : `true` if Brent's method converged.
"""
struct BeachMacKinnonResult <: HAREModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    mf::Any
    vcov::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    rho::Float64
    loglik::Float64
    converged::Bool
end
