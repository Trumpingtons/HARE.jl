# # HARE.jl — Tutorial
#
# This tutorial walks through all eight estimators in HARE.jl using simulated
# regression data. It serves three purposes:
#
# 1. **Documentation** — rendered as a page in the Documenter.jl site via Literate.jl.
# 2. **Executable demo** — run directly with `julia --project examples/demo.jl`.
# 3. **CI smoke test** — executed during the Docs CI job; a failure here breaks
#    the docs build, keeping the README examples honest.

# ## Setup

using HARE
using Random, LinearAlgebra, GLM, StatsBase

# ## Simulating Regression Data
#
# We simulate three datasets from the model yᵢ = 1 + 2x₁ᵢ − x₂ᵢ + uᵢ,
# each with a different error structure.

Random.seed!(1234)

n      = 500
x1     = randn(n)
x2     = randn(n)
X      = hcat(ones(n), x1, x2)
β_true = [1.0, 2.0, -1.0]

# **Heteroskedastic errors** — variance grows exponentially with x₁ (Harvey model):
σ       = exp.(0.5 .* x1)
y_het   = X * β_true .+ σ .* randn(n)
data_het = (y = y_het, x1 = x1, x2 = x2)

# **AR(1) errors** — ρ = 0.7:
ρ_true = 0.7
ϵ      = randn(n)
u_ar   = zeros(n)
for t in 2:n
    u_ar[t] = ρ_true * u_ar[t-1] + ϵ[t]
end
y_ar   = X * β_true .+ u_ar
data_ar = (y = y_ar, x1 = x1, x2 = x2)

# **Combined AR(1) + heteroskedastic errors**:
u_hae = zeros(n)
for t in 2:n
    u_hae[t] = ρ_true * u_hae[t-1] + σ[t] * ϵ[t]
end
y_hae    = X * β_true .+ u_hae
data_hae = (y = y_hae, x1 = x1, x2 = x2)

# ## Heteroskedasticity Correction
#
# Both estimators use Harvey's (1976) log-variance model.

# **Two-step FWLS** — single Harvey-weighted WLS pass:
fwls2 = two_step_harvey(X, y_het)
println("Two-Step FWLS  β̂ = ", round.(coef(fwls2), digits=3))
println("               SE = ", round.(stderror(fwls2), digits=3))

# **Iterated FWLS** — repeats until coefficient convergence:
fwlsi = iterated_harvey(X, y_het)
println("Iterated FWLS  β̂ = ", round.(coef(fwlsi), digits=3))
println("               SE = ", round.(stderror(fwlsi), digits=3))
println("   iterations = ", fwlsi.iterations, "  converged = ", fwlsi.converged)

# Both estimators accept a `@formula` interface identical to GLM.jl:
fwls2_f = two_step_harvey(@formula(y ~ x1 + x2), data_het)
@assert coef(fwls2) ≈ coef(fwls2_f)

# The result structs expose the full StatsBase interface:
@assert length(residuals(fwls2)) == n
@assert length(fitted(fwls2))    == n
@assert size(vcov(fwls2))        == (3, 3)

# ## AR(1) Correction
#
# Three estimators address first-order autocorrelation.

# **Two-step Prais–Winsten** — retains the first observation (unlike Cochrane–Orcutt):
pw2 = two_step_prais_winsten(X, y_ar)
println("\nTwo-Step Prais–Winsten  β̂ = ", round.(coef(pw2), digits=3))
println("                        ρ̂ = ", round(pw2.rho, digits=4))

# **Iterated Prais–Winsten** — alternates ρ estimation and GLS until |Δρ| < tol:
pwi = iterated_prais_winsten(X, y_ar)
println("Iterated Prais–Winsten  β̂ = ", round.(coef(pwi), digits=3))
println("                        ρ̂ = ", round(pwi.rho, digits=4),
        "  iterations = ", pwi.iterations)

# **Hildreth–Lu** — grid search over ρ ∈ (−0.99, 0.99); robust to multimodal likelihood:
hl = hildreth_lu(X, y_ar)
println("Hildreth–Lu             β̂ = ", round.(coef(hl), digits=3))
println("                        ρ̂ = ", round(hl.rho, digits=4),
        "  RSS = ", round(hl.rss, digits=2))

# ## Joint AR(1) + Heteroskedasticity Correction
#
# HARE-GLS (Oberhofer–Kmenta 1974) corrects for both error structures simultaneously.

# **Two-step HARE-GLS**:
hae2 = two_step_haegls(X, y_hae)
println("\nTwo-Step HARE-GLS   β̂ = ", round.(coef(hae2), digits=3))
println("                   ρ̂ = ", round(hae2.rho, digits=4))

# **Iterated HARE-GLS**:
haei = iterated_haegls(X, y_hae)
println("Iterated HARE-GLS   β̂ = ", round.(coef(haei), digits=3))
println("                   ρ̂ = ", round(haei.rho, digits=4),
        "  iterations = ", haei.iterations,
        "  converged = ", haei.converged)

# ## Beach–MacKinnon Exact MLE
#
# Maximises the exact concentrated log-likelihood including the Jacobian term
# `½ log(1−ρ²)`. More efficient than iterated Prais–Winsten in small samples.

bm = beach_mackinnon(X, y_ar)
println("\nBeach–MacKinnon MLE  β̂ = ", round.(coef(bm), digits=3))
println("                     ρ̂ = ", round(bm.rho, digits=4))
println("                log ℓ = ", round(bm.loglik, digits=2),
        "  converged = ", bm.converged)

# ## Summary
#
# | Estimator                | Error structure       | Struct                  |
# |:-------------------------|:----------------------|:------------------------|
# | `two_step_harvey`          | Heteroskedasticity    | `HarveyResult`            |
# | `iterated_harvey`          | Heteroskedasticity    | `HarveyResult`            |
# | `two_step_prais_winsten` | AR(1)                 | `PraisWinstenResult`    |
# | `iterated_prais_winsten` | AR(1)                 | `PraisWinstenResult`    |
# | `hildreth_lu`            | AR(1)                 | `HildrethLuResult`      |
# | `two_step_haegls`        | AR(1) + Het.          | `HAREGLSResult`          |
# | `iterated_haegls`        | AR(1) + Het.          | `HAREGLSResult`          |
# | `beach_mackinnon`        | AR(1) exact MLE       | `BeachMacKinnonResult`  |
#
# All result structs implement the StatsBase interface:
# `coef`, `stderror`, `vcov`, `residuals`, `fitted`, `predict`.

println("\nTrue β: ", β_true, "  True ρ: ", ρ_true)
