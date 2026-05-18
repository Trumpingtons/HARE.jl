# # HARE.jl -- Tutorial
#
# This tutorial demonstrates estimators in HARE.jl using the formula interface.
# Every function also accepts a matrix interface `f(X, y; ...)` where `X` is the
# regressor matrix **without** a constant column; both signatures are shown for the
# first estimator and listed for all estimators in the [API Reference](api.md).
# Each section simulates data appropriate for that estimator.

# ## Setup
#
# ```math
# \begin{aligned}
# y_i &= \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\,\sigma_i^2)
# \end{aligned}
# ```
# For this tutorial, as exogenous variables we generate two sets of random samples
# ($\mathbf{x}_{1}$) and ($\mathbf{x}_{2}$), normally distributed. The response variable
# ($y$) is given by
# ```math
# \begin{aligned}
# y_i &= \beta_0+\beta_{1}\, \mathbf{x}_{1i} + \beta_{2}\, \mathbf{x}_{2i} + \varepsilon_i
# \end{aligned}
# ```

using HARE
using Random, LinearAlgebra, GLM, StatsBase

Random.seed!(1234)

nobs   = 500
x1     = randn(nobs)
x2     = randn(nobs)
b = [1.0, 2.0, -1.0];              # beta true parameters

# ## Exponential Variance Model:
#
# The Harvey (1976) model specifies multiplicative heteroskedasticity via an exponential
# variance function:
#
# ```math
# \begin{aligned}
# \sigma_i^2 &= \sigma^2_0 \exp(\mathbf{z}_i^\top \boldsymbol{\gamma})
# \end{aligned}
# ```
#
# We simulate data with log variance proportional to $|\mathbf{x}_1|$ and $\sigma_0=1$.

sigma_0 = 1.0;
gamma   = [1.0, 0.0];
sigma_exp = sigma_0 * exp.((gamma[1] .* abs.(x1) .+ gamma[2] .* x2) ./ 2);

# The simulated dataset ($data$) is

y    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_exp .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);

# **MLE** (`exponential_mle`): jointly maximises the log-likelihood over
# $(\boldsymbol{\beta}, \boldsymbol{\gamma})$ via L-BFGS, warm-started from the Harvey
# auxiliary regression. Equivalent to R's `nlme::gls(weights = varComb(varExp(...)))`.
#
# `exponential_mle(formula, data; Z=nothing) -> HeteroMLEResult`
exponential_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))

# **Two-step FWLS** (`exponential_fwls`): a closed-form alternative that requires no
# numerical optimisation. Preferable when MLE fails to converge or when the normality
# assumption is in doubt, since FWLS is consistent under any error distribution with
# the correct variance model.
#
# `exponential_fwls(formula, data; Z=nothing) -> ExponentialFWLSResult`
exponential_fwls(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))

# **Matrix interface** (available for every estimator; `X` and `Z` without a constant column):
X   = hcat(x1, x2);
Z_h = hcat(abs.(x1), x2);
exponential_fwls(X, y; Z = Z_h)

# ## Power Variance Model
#
# The power variance model (Harvey 1976, multiplicative form) specifies:
#
# ```math
# \begin{aligned}
# \sigma_i^2 &= \sigma^2 \prod_j |z_{ij}|^{2\delta_j}
# \end{aligned}
# ```
#
# Equivalently, $\log\sigma_i^2 = \log\sigma^2 + 2\delta_1\log|z_{i1}| + \cdots$,
# which is the Harvey exponential model applied to $\log|Z|$.
# The reported $\boldsymbol{\gamma} = (\sigma^2, \delta_1, \ldots)$ is obtained via the
# delta-method reparametrisation.
#
# We simulate data with $\sigma_i^2 = |x_1|$ (i.e.\ $\sigma^2 = 1$, $\delta_1 = 0.5$).

sigma_pow = abs.(x1) .^ 0.5;
y_pow     = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_pow .* randn(nobs);
data_pow  = (y = y_pow, x1 = x1, x2 = x2);

# **MLE** (`power_mle`): joint MLE warm-started from the Harvey auxiliary regression
# on $\log|Z|$. Reports $(\sigma^2, \delta_1, \ldots)$ with delta-method standard errors.

power_mle(@formula(y ~ x1 + x2), data_pow; Z = @formula(0 ~ abs(x1)))

# **Two-step FWLS** (`power_fwls`):

power_fwls(@formula(y ~ x1 + x2), data_pow; Z = @formula(0 ~ abs(x1)))

# ## Quadratic Variance Model
#
# The Glejser (1969) model assumes the standard deviation is linear in auxiliary
# regressors $\mathbf{z}_i$:
#
# ```math
# \begin{aligned}
# \sigma_i &= \sigma_0 + \mathbf{z}_i^\top \boldsymbol{\gamma}
# \end{aligned}
# ```
#
# We simulate data with $\sigma_i$ linear in $|\mathbf{x}_1|$.
#
# Because the variance depends on $|\mathbf{x}_1|$, not $\mathbf{x}_1$ itself, we supply a custom auxiliary
# regressor matrix `Z = [|x1|, x2]` so the Glejser auxiliary regression is
# correctly specified.  Using the raw regressors would yield a badly biased gamma.

sigma = 0.5 .+ 0.3 .* abs.(x1);
y = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);
quadratic_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))

# ## Linear Variance Model
#
# The linear variance model specifies the variance as a linear combination of
# auxiliary regressors $\mathbf{z}_i$:
#
# ```math
# \begin{aligned}
# \sigma_i^2 &= \gamma_0 + \mathbf{z}_i^\top \boldsymbol{\gamma}
# \end{aligned}
# ```
#
# We simulate data with $\sigma_i^2$ linear in $|\mathbf{x}_1|$.

sigma2 = 0.5 .+ 0.3 .* abs.(x1);
y = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sqrt.(sigma2) .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);
linear_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))

# ## Groupwise Heteroskedasticity
#
# The groupwise model assumes a constant variance within each group $g$:
#
# ```math
# \begin{aligned}
# \sigma_i^2 &= \sigma_g^2, \quad i \in g
# \end{aligned}
# ```
#
# We simulate four groups with increasing variance.

groups   = repeat(["A", "B", "C", "D"], inner = nobs ÷ 4);
sigma_gw = [0.5, 1.0, 1.5, 2.0];
sigma_g  = [sigma_gw[findfirst(==( g), ["A","B","C","D"])] for g in groups];
y_gw     = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_g .* randn(nobs);
data_gw  = (y = y_gw, x1 = x1, x2 = x2);

# **MLE** (`groupwise_mle`): exact MLE standard errors from $(X^\top W X)^{-1}$:
groupwise_mle(@formula(y ~ x1 + x2), data_gw, groups)

# **Iterated FWLS** (`iterated_groupwise`): identical point estimates, GLM-scaled standard errors:
iterated_groupwise(@formula(y ~ x1 + x2), data_gw, groups)

# ## AR(1) Estimators
#
# We simulate data with AR(1) errors: $u_t = 0.7\, u_{t-1} + e_t$.

eps  = randn(nobs);
u_ar = zeros(nobs);
for t in 2:nobs; u_ar[t] = 0.7 * u_ar[t-1] + eps[t]; end
y_ar    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ u_ar;
data_ar = (y = y_ar, x1 = x1, x2 = x2);

# **Prais-Winsten** -- retains the first observation via GLS transformation:
two_step_prais_winsten(@formula(y ~ x1 + x2), data_ar)

# **Hildreth-Lu** -- grid search over $\rho \in (-0.99,\, 0.99)$:
hildreth_lu(@formula(y ~ x1 + x2), data_ar)

# **Beach-MacKinnon** -- exact maximum likelihood:
beach_mackinnon(@formula(y ~ x1 + x2), data_ar)

# ## R Cross-Validation
#
# Each MLE estimator has an exact R counterpart in `nlme::gls`.  Below we run
# both on the same dataset — generated once in Julia with `StableRNG` and
# saved to CSV so R can read it back — so the output tables can be compared
# directly.  The data uses `StableRNG(42)` for the predictors and
# `StableRNG(99)` for the noise.

using StableRNGs
rng_r  = StableRNG(42);
rng_r2 = StableRNG(99);
n_r    = 500;
x1_r   = randn(rng_r, n_r);
x2_r   = randn(rng_r, n_r);
Xr     = hcat(x1_r, x2_r);
br     = [1.0, 2.0, -1.0];

# ### Exponential MLE
#
# R code:
# ```r
# library(nlme)
# d <- read.csv("hare_exp.csv")   # x1, x2, y — written from Julia
# m <- gls(y ~ x1 + x2, data = d, weights = varExp(form = ~x1), method = "ML")
# summary(m)
# ```
#
# R output:
# ```
# Generalized least squares fit by maximum likelihood
#   Model: y ~ x1 + x2
#   Data: d
#        AIC      BIC    logLik
#   1399.339 1420.412 -694.6695
#
# Variance function:
#  Structure: Exponential of variance covariate
#  Formula: ~x1
#  Parameter estimates:
#     expon
# 0.5400336
#
# Coefficients:
#                  Value Std.Error   t-value p-value
# (Intercept)  0.9762948 0.0494178  19.75594       0
# x1           1.9933986 0.0311266  64.04173       0
# x2          -1.0114907 0.0298082 -33.93332       0
#
# Residual standard error: 1.001954
# Degrees of freedom: 500 total; 497 residual
# ```
#
# HARE: `gamma[1]` = $\log\sigma^2$ = `log(sigma^2)` from R;
# `gamma[2]` = $2\delta$ = `2 * 0.5400336`:

y_r_exp = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ exp.(0.5 .* x1_r) .* randn(rng_r2, n_r);
exponential_mle(Xr, y_r_exp; Z = hcat(x1_r))

# ### Power MLE
#
# R code:
# ```r
# d <- read.csv("hare_pow.csv")
# m <- gls(y ~ x1 + x2, data = d, weights = varPower(form = ~abs(x1)), method = "ML")
# summary(m)
# ```
#
# R output:
# ```
# Generalized least squares fit by maximum likelihood
#   Model: y ~ x1 + x2
#   Data: d
#        AIC      BIC    logLik
#   1141.608 1162.681 -565.8042
#
# Variance function:
#  Structure: Power of variance covariate
#  Formula: ~abs(x1)
#  Parameter estimates:
#     power
# 0.4773703
#
# Coefficients:
#                  Value Std.Error   t-value p-value
# (Intercept)  0.9955270 0.0228240  43.61754       0
# x1           2.1150261 0.0483900  43.70791       0
# x2          -0.9639498 0.0242141 -39.80943       0
#
# Residual standard error: 0.98068
# Degrees of freedom: 500 total; 497 residual
# ```
#
# HARE: `gamma[1]` = $\log\sigma^2$; `gamma[2]` = $2\delta$ = `2 * 0.4773703`:

y_r_pow = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ abs.(x1_r).^0.5 .* randn(rng_r2, n_r);
power_mle(Xr, y_r_pow; Z = hcat(abs.(x1_r)))

# ### Groupwise MLE
#
# R code:
# ```r
# d <- read.csv("hare_gw.csv")
# d$group <- factor(d$group)
# m <- gls(y ~ x1 + x2, data = d, weights = varIdent(form = ~1 | group), method = "ML")
# summary(m)
# ```
#
# R output:
# ```
# Generalized least squares fit by maximum likelihood
#   Model: y ~ x1 + x2
#   Data: d
#        AIC      BIC    logLik
#   1464.294 1493.797 -725.1472
#
# Variance function:
#  Structure: Different standard deviations per stratum
#  Formula: ~1 | group
#  Parameter estimates:
#        1        2        3        4
# 1.000000 1.661873 2.666309 3.399754
#
# Coefficients:
#                  Value Std.Error   t-value p-value
# (Intercept)  0.9999977 0.0373464  26.77628       0
# x1           1.9740476 0.0362961  54.38735       0
# x2          -0.9412746 0.0376907 -24.97368       0
#
# Residual standard error: 0.5237556
# Degrees of freedom: 500 total; 497 residual
# ```
#
# R's `varIdent` reports relative SDs per group (group 1 = reference = 1.0).
# HARE reports $\sigma_g^2$ directly for each group:

gw_r_groups = repeat(["A","B","C","D"], inner = n_r ÷ 4);
gw_r_sig    = Dict("A"=>0.5,"B"=>1.0,"C"=>1.5,"D"=>2.0);
y_r_gw      = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ [gw_r_sig[g] for g in gw_r_groups] .* randn(rng_r2, n_r);
groupwise_mle(Xr, y_r_gw, gw_r_groups)

# ## Consistency: Exponential MLE
#
# With a large sample the MLE should recover the true parameters
# $\boldsymbol{\beta} = [1, 2, -1]$ and $\boldsymbol{\gamma} = [0, 1, 0]$
# (intercept $\log\sigma_0^2 = 0$, slope on $x_1$ equal to 1, slope on $x_2$ equal to 0).

rng_c   = StableRNG(42);
N       = 1_000_000;
x1_c    = randn(rng_c, N);
x2_c    = randn(rng_c, N);
sigma_c = exp.(0.5 .* x1_c);
y_c     = 1.0 .+ 2.0 .* x1_c .- 1.0 .* x2_c .+ sigma_c .* randn(rng_c, N);
data_c  = (y = y_c, x1 = x1_c, x2 = x2_c);

m_c = exponential_mle(@formula(y ~ x1 + x2), data_c)

# ## Combined AR(1) + Heteroskedasticity
#
# The DGP is $u_t = \rho\, u_{t-1} + \sigma_t\, \varepsilon_t$ with Harvey-type
# heteroskedasticity in the innovations.
# True parameters: $\rho = 0.7$, $\boldsymbol{\gamma} = [1,\, 0]$
# (reusing `sigma_exp` from the Exponential Variance section above).
#
# The Sequential estimator runs the Harvey auxiliary regression on the
# Prais-Winsten transformed regressors $\mathbf{X}^*$, so its reported
# $\boldsymbol{\gamma}$ refers to that transformed system and will only
# approximately match the true values.  The Joint estimator uses the original
# regressors and exact MLE, so its $\boldsymbol{\gamma}$ should be close to
# $[1, 0]$.

u_hae = zeros(nobs);
for t in 2:nobs; u_hae[t] = 0.7 * u_hae[t-1] + sigma_exp[t] * eps[t]; end
y_hae    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ u_hae;
data_hae = (y = y_hae, x1 = x1, x2 = x2);

# **Sequential HARE** -- corrects for AR(1) first, then heteroskedasticity:
two_step_sequential(@formula(y ~ x1 + x2), data_hae)

# **Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously:
two_step_joint(@formula(y ~ x1 + x2), data_hae)

# ## Heteroskedasticity Tests
#
# HARE provides two score (LM) tests for heteroskedasticity that are unique to
# the package.  Both test $H_0\colon \gamma = 0$ (homoskedasticity) and are
# distributed as $\chi^2(p)$ under $H_0$, where $p$ is the number of variance
# regressors (columns of $Z$ excluding the intercept).
#
# We reuse the exponential-variance dataset from the first section ($n = 500$,
# $\log\sigma_i^2 = x_{1i}$), so both tests should reject strongly.
# The R equivalents are `skedastic::harvey()` and `skedastic::glejser()`.
#
# **Harvey test** -- LM statistic $\text{ESS}/\text{trigamma}(1/2)$ from the
# auxiliary regression of $\log\hat{u}_i^2$ on $Z$:

y_test   = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_exp .* randn(nobs);
data_test = (y = y_test, x1 = x1, x2 = x2);

harvey_test(@formula(y ~ x1 + x2), data_test)

# **Glejser test** -- LM statistic from the auxiliary regression of
# $|\hat{u}_i|$ on $Z$:

glejser_test(@formula(y ~ x1 + x2), data_test)
