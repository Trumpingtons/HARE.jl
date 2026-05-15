# HARE.jl

[![CI](https://github.com/Trumpingtons/HARE.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Trumpingtons/HARE.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/Trumpingtons/HARE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Trumpingtons/HARE.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://Trumpingtons.github.io/HARE.jl/stable)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://Trumpingtons.github.io/HARE.jl/dev)

**Heteroskedasticity and Autocorrelation Estimators** for Julia.

Feasible GLS correction methods for linear regression models covering
heteroskedasticity, AR(1) serial correlation, and their combination --
with both sequential and joint estimation strategies.

All estimators accept both matrix and `@formula` interfaces and return typed
result structs that subtype `StatsBase.RegressionModel`, making them
compatible with the broader Julia statistics ecosystem.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Trumpingtons/HARE.jl")
```

## Estimators

### Heteroskedasticity only

Two variance models are available. **Harvey** models log variance as a linear function of the regressors (`log(sigma_i^2) = x_i' * gamma`), estimated via WLS on log-squared OLS residuals. **Glejser** models the standard deviation directly (`sigma_i = z_i' * gamma`) using a separate auxiliary regressor matrix `Z` (defaults to `X`). Both offer a two-step variant (one WLS pass) and an iterated variant that alternates until convergence.

| Function | Variance model | Result type |
|:---------|:---------------|:------------|
| `two_step_harvey` | Harvey: `log(sigma_i^2) = x_i' * gamma` | `HarveyResult` |
| `iterated_harvey` | Harvey: `log(sigma_i^2) = x_i' * gamma` | `HarveyResult` |
| `two_step_glejser` | Glejser: `sigma_i = z_i' * gamma` | `GlejserResult` |
| `iterated_glejser` | Glejser: `sigma_i = z_i' * gamma` | `GlejserResult` |

### AR(1) autocorrelation only

Four estimators handle serial correlation under the AR(1) model `u_t = rho * u_{t-1} + e_t`. **Prais-Winsten** FGLS preserves the first observation via a GLS transformation, available in two-step and iterated forms. **Hildreth-Lu** is also GLS-based, but searches a grid of rho values and picks the one minimising RSS. **Beach-MacKinnon** maximises the exact concentrated likelihood over rho using Brent's method, with regression coefficients recovered analytically.

| Function | Method | Result type |
|:---------|:-------|:------------|
| `two_step_prais_winsten` | Prais-Winsten FGLS | `PraisWinstenResult` |
| `iterated_prais_winsten` | Prais-Winsten FGLS | `PraisWinstenResult` |
| `hildreth_lu` | Grid search over rho | `HildrethLuResult` |
| `beach_mackinnon` | Exact MLE (Brent) | `BeachMacKinnonResult` |

### AR(1) + Heteroskedasticity combined

Two strategies are available for handling both effects simultaneously.

**Sequential HARE** -- corrects for autocorrelation first (Prais-Winsten), then
heteroskedasticity (Harvey), alternating until convergence. The two effects
are estimated separately in each pass.

| Function | Result type |
|:---------|:------------|
| `two_step_sequential` | `SequentialResult` |
| `iterated_sequential` | `SequentialResult` |

**Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously via
maximum likelihood. The model is `u_t = rho * u_{t-1} + sigma_t * e_t` with
`log(sigma_t^2) = z_t' * gamma`. The two-step variant uses the concentrated
likelihood (beta profiled out analytically); the iterated variant uses
coordinate descent.

| Function | Result type |
|:---------|:------------|
| `two_step_joint` | `JointResult` |
| `iterated_joint` | `JointResult` |

## Quick Start

Pass regressors **without** a constant column -- the intercept is added automatically:

```julia
using HARE, Random

Random.seed!(1234)
n   = 500
x1  = randn(n); x2 = randn(n)
X = hcat(x1, x2)

# Heteroskedastic errors (Harvey DGP)
y_het = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ exp.(0.5 .* x1) .* randn(n)

m = iterated_harvey(X, y_het)
```

The `@formula` interface works identically to GLM.jl:

```julia
data = (y = y_het, x1 = x1, x2 = x2)
m = iterated_harvey(@formula(y ~ x1 + x2), data)
```

AR(1) correction with the Prais-Winsten estimator:

```julia
u = zeros(n)
for t in 2:n; u[t] = 0.7 * u[t-1] + randn(); end
y_ar = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ u

m = iterated_prais_winsten(X, y_ar)
m.rho     # estimated AR(1) coefficient, approx 0.70
```

Combined AR(1) + heteroskedasticity -- sequential approach:

```julia
sigma = exp.(0.3 .* x1)
u = zeros(n)
for t in 2:n; u[t] = 0.7 * u[t-1] + sigma[t] * randn(); end
y_hae = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ u

m = iterated_sequential(X, y_hae)
m.rho     # AR(1) estimate
```

Combined AR(1) + heteroskedasticity -- joint MLE:

```julia
m = iterated_joint(X, y_hae)
m.rho     # AR(1) estimate
m.gamma   # log-variance coefficients gamma_hat
m.loglik  # maximised log-likelihood
```

The Glejser estimator models `sigma_i` (not `sigma_i^2`) linearly in auxiliary
regressors. An alternative `Z` matrix can be supplied for the variance equation:

```julia
Z = hcat(ones(n), abs.(x1))   # variance depends on |x1| only
m = iterated_glejser(X, y_het; Z=Z)
```

The same `Z` keyword is available for `two_step_joint` and `iterated_joint`.

To fit a model **without** an intercept, pass `intercept=false`:

```julia
m = two_step_harvey(X, y_het; intercept=false)
```

## StatsBase Interface

All result types implement the full `StatsBase.RegressionModel` interface:

```julia
m = iterated_prais_winsten(@formula(y ~ x1 + x2), data)

# Coefficients
coef(m)           # coefficient vector beta_hat
stderror(m)       # standard errors
vcov(m)           # covariance matrix
coefnames(m)      # ["(Intercept)", "x1", "x2"]

# Sample information
nobs(m)           # number of observations
dof(m)            # number of parameters k
dof_residual(m)   # n - k

# Fitted values and residuals
fitted(m)         # X * beta_hat
residuals(m)      # y - X * beta_hat
response(m)       # y (recovered from fitted + residuals)

# Inference
tstat(m)          # t-statistics
pvalues(m)        # two-sided p-values (Student-t by default)
pvalues(m; dist=:normal)   # asymptotic normal p-values
confint(m)                 # 95% confidence intervals
confint(m; level=0.90)     # 90% confidence intervals

# Fit diagnostics
rss(m)            # residual sum of squares
sigma2(m)         # sigma_hat^2 = RSS / (n - k)
r2(m)             # R^2
adjr2(m)          # adjusted R^2

# Formula metadata (formula-fitted models only)
formula(m)        # fitted formula
termnames(m)      # ["(Intercept)", "x1", "x2"]
responsename(m)   # "y"
```

`BeachMacKinnonResult` and `JointResult` additionally expose `loglikelihood(m)`.

Typing `m` in the REPL displays a formatted coefficient table:

```
Coefficients:
---------------------------------------------------------------------------
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
---------------------------------------------------------------------------
(Intercept)    1.023       0.089   11.49    <1e-23      0.848      1.198
x1             2.015       0.092   21.90    <1e-64      1.834      2.196
x2            -0.998       0.091  -10.97    <1e-22     -1.177     -0.819
---------------------------------------------------------------------------
rho: 0.6934   Converged: true   Iterations: 8
N: 500   R^2: 0.9421   Adj. R^2: 0.9419
```

## Prediction

Matrix-fitted models accept a new design matrix:

```julia
m = iterated_harvey(X, y_het)
predict(m, Xnew)
```

Formula-fitted models accept a new data table and apply the original
encoding automatically:

```julia
m   = iterated_harvey(@formula(y ~ x1 + x2), data)
predict(m, newdata)   # Tables.jl-compatible: DataFrame, NamedTuple, etc.
```

## Hypothesis Tests

### Wald test (all estimators)

Tests the linear restriction H0: R*beta = r using the GLS covariance matrix.
The test statistic F = (R*beta_hat - r)' * (R*V*R')^(-1) * (R*beta_hat - r) / q
is compared to F(q, n-k).

```julia
m = iterated_prais_winsten(X, y_ar)

# Joint significance of x1 and x2
R = [0.0 1.0 0.0; 0.0 0.0 1.0]
wald_test(m, R)
# Wald test (F):
# F(2, 497) = 312.45   p-value = 0.0000

# Test beta_2 = 1.0
wald_test(m, [0.0 1.0 0.0], [1.0])
```

### Likelihood ratio test (Beach-MacKinnon and Joint HARE)

Compares two nested models with a `loglik` field. Restricted model is passed first.

```julia
X1 = X[:, 1:1]   # x1 only
m_r = beach_mackinnon(X1, y_ar)   # intercept + x1
m_u = beach_mackinnon(X, y_ar)  # intercept + x1 + x2

lrtest(m_r, m_u)
# Likelihood ratio test:
# LR = 95.43   df = 1   p-value = 0.0000
```

## Ecosystem Integration

`HAREModel <: StatsBase.RegressionModel` means HARE results work directly
with packages that support the standard interface.

## References

- Harvey, A. C. (1976). Estimating regression models with multiplicative heteroscedasticity. *Econometrica*, 44(3), 461-465.
- Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American Statistical Association*, 64(325), 316-323.
- Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial correlation. *Cowles Commission Discussion Paper*, No. 383.
- Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression to relationships containing auto-correlated error terms. *JASA*, 44(245), 32-61.
- Hildreth, C., & Lu, J. Y. (1960). Demand relations with autocorrelated disturbances. *MSU Agricultural Experiment Station Technical Bulletin*, No. 276.
- Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum likelihood estimates in generalized regression models. *Econometrica*, 42(3), 579-590.
- Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for regression with autocorrelated errors. *Econometrica*, 46(1), 51-58.
