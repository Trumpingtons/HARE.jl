# # HARE.jl -- Tutorial
#
# This tutorial demonstrates all estimators in HARE.jl.
# Each section simulates data appropriate for that estimator.

# ## Setup

using HARE
using Random, LinearAlgebra, GLM, StatsBase

Random.seed!(1234)

n      = 2500
x1     = randn(n)
x2     = randn(n)
X      = hcat(x1, x2)
X_full = hcat(ones(n), X)
b_true = [1.0, 2.0, -1.0];

# ## Harvey Estimator
#
# The Harvey model assumes the log-variance is linear in the regressors:
# log(sigma_i^2) = X_full' * gamma.
# We simulate data with log variance proportional to x1.
#
# Note: the auxiliary OLS regresses log(u_hat^2) on X. Because E[log(chi^2(1))] ≈ -1.2703,
# the raw gamma_0 estimate is biased downward by that constant (slope coefficients are
# consistent when regressors are mean-zero).  HARE corrects gamma_0 by adding 1.2703 back,
# so all reported gamma values are consistent.  This correction is purely cosmetic for gamma:
# WLS is invariant to a constant weight scaling, so beta estimates are unaffected either way.

gamma_h  = [0.0, 1.0, 0.0]
sigma_h  = exp.(0.5 .* (X_full * gamma_h))
y_harvey = X_full * b_true .+ sigma_h .* randn(n);

# **Matrix interface** -- pass regressors without a constant column:
two_step_harvey(X, y_harvey)

# **Formula interface** (all estimators support this):
data_harvey = (y = y_harvey, x1 = x1, x2 = x2)
two_step_harvey(@formula(y ~ x1 + x2), data_harvey)

# ## Glejser Estimator
#
# The Glejser model assumes the standard deviation is linear in auxiliary regressors Z:
# sigma_i = Z_i' * gamma.
# We simulate data with sigma linear in |x1|.
#
# Because the variance depends on |x1|, not x1 itself, we supply a custom auxiliary
# regressor matrix Z = [ones(n), |x1|, x2] so the Glejser auxiliary regression is
# correctly specified.  Using the raw regressors X would yield a badly biased gamma.

sigma_g   = 0.5 .+ 0.3 .* abs.(x1)
y_glejser = X_full * b_true .+ sigma_g .* randn(n);

Z_glejser = hcat(ones(n), abs.(x1), x2);   # correctly specified auxiliary regressors

# **Matrix interface** -- pass regressors without a constant column:
two_step_glejser(X, y_glejser; Z = Z_glejser)

# **Formula interface:**
data_glejser = (y = y_glejser, x1 = x1, x2 = x2);
two_step_glejser(@formula(y ~ x1 + x2), data_glejser; Z = Z_glejser)

# ## AR(1) Estimators
#
# We simulate data with AR(1) errors: u_t = 0.7 * u_{t-1} + e_t.

eps  = randn(n)
u_ar = zeros(n)
for t in 2:n; u_ar[t] = 0.7 * u_ar[t-1] + eps[t]; end
y_ar = X_full * b_true .+ u_ar;

# **Prais-Winsten** -- retains the first observation via GLS transformation:
two_step_prais_winsten(X, y_ar)

# **Hildreth-Lu** -- grid search over rho in (-0.99, 0.99):
hildreth_lu(X, y_ar)

# **Beach-MacKinnon** -- exact maximum likelihood:
beach_mackinnon(X, y_ar)

# ## Combined AR(1) + Heteroskedasticity
#
# The DGP is u_t = rho * u_{t-1} + sigma_t * eps_t with Harvey-type heteroskedasticity
# in the innovations.  True parameters: rho = 0.7, gamma = [0.0, 1.0, 0.0] (reusing
# sigma_h from the Harvey section above).
#
# Note: the Sequential estimator runs the Harvey auxiliary regression on the
# Prais-Winsten transformed regressors Xstar, so its reported gamma refers to that
# transformed system and will only approximately match the true values.  The Joint
# estimator uses the original regressors and exact MLE, so its gamma should be close
# to [0.0, 1.0, 0.0].

u_hae = zeros(n)
for t in 2:n; u_hae[t] = 0.7 * u_hae[t-1] + sigma_h[t] * eps[t]; end
y_hae = X_full * b_true .+ u_hae;

# **Sequential HARE** -- corrects for AR(1) first, then heteroskedasticity:
two_step_sequential(X, y_hae)

# **Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously:
two_step_joint(X, y_hae)
