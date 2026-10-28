```@meta
EditURL = "../../examples/demo.jl"
```

# HARE.jl -- Tutorial

This tutorial demonstrates estimators in HARE.jl using the formula interface.
Every function also accepts a matrix interface `f(X, y; ...)` where `X` is the
regressor matrix **without** a constant column; both signatures are shown for the
first estimator and listed for all estimators in the [API Reference](api.md).
Each section simulates data appropriate for that estimator.

## Setup

```math
\begin{aligned}
y_i &= \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\,\sigma_i^2)
\end{aligned}
```
For this tutorial, as exogenous variables we generate two sets of random samples
($\mathbf{x}_{1}$) and ($\mathbf{x}_{2}$), normally distributed. The response variable
($y$) is given by
```math
\begin{aligned}
y_i &= \beta_0+\beta_{1}\, \mathbf{x}_{1i} + \beta_{2}\, \mathbf{x}_{2i} + \varepsilon_i
\end{aligned}
```

````julia
using HARE
using Random, LinearAlgebra, GLM, StatsBase

Random.seed!(1234)

nobs   = 500
x1     = randn(nobs)
x2     = randn(nobs)
b = [1.0, 2.0, -1.0];              # beta true parameters
````

## Exponential Variance Model:

The Harvey (1976) model specifies multiplicative heteroskedasticity via an exponential
variance function:

```math
\begin{aligned}
\sigma_i^2 &= \sigma^2_0 \exp(\mathbf{z}_i^\top \boldsymbol{\gamma})
\end{aligned}
```

We simulate data with log variance proportional to $|\mathbf{x}_1|$ and $\sigma_0=1$.

````julia
sigma_0 = 1.0;
gamma   = [1.0, 0.0];
sigma_exp = sigma_0 * exp.((gamma[1] .* abs.(x1) .+ gamma[2] .* x2) ./ 2);
````

The simulated dataset ($data$) is

````julia
y    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_exp .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);
````

**MLE** (`exponential_mle`): jointly maximises the log-likelihood over
$(\boldsymbol{\beta}, \boldsymbol{\gamma})$ via L-BFGS, warm-started from the Harvey
auxiliary regression. Equivalent to R's `nlme::gls(weights = varComb(varExp(...)))`.

`exponential_mle(formula, data; Z=nothing) -> MLEResult`

````julia
exponential_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.00997   0.0632213   15.98    <1e-45   0.885755   1.13418
x1            1.90611   0.0881646   21.62    <1e-72   1.73289    2.07933
x2           -1.02156   0.061346   -16.65    <1e-49  -1.14209   -0.901027
─────────────────────────────────────────────────────────────────────────
Link: Exponential — σ²ᵢ = exp(γ₀ + zᵢ′γ)    [Harvey 1976]
Log-variance equation (gamma):
──────────────────────────────────────────────────────────────────────
              Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────
log σ₀²  -0.0344842   0.106269   -0.32    0.7456  -0.242768  0.1738
z1        1.05254     0.0988821  10.64    <1e-25   0.85873   1.24634
z2       -0.095787    0.0637538  -1.50    0.1330  -0.220742  0.0291681
──────────────────────────────────────────────────────────────────────
Log-likelihood: -927.5787   Converged: true   Iterations: 10
N: 500   R^2: 0.617   Adj. R^2: 0.6154

````

**Two-step FWLS** (`exponential_fwls`): a closed-form alternative that requires no
numerical optimisation. Preferable when MLE fails to converge or when the normality
assumption is in doubt, since FWLS is consistent under any error distribution with
the correct variance model.

`exponential_fwls(formula, data; Z=nothing) -> FWLSResult`

````julia
exponential_fwls(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.01124   0.0837127   12.08    <1e-28   0.846768   1.17572
x1            1.90355   0.12354     15.41    <1e-43   1.66082    2.14627
x2           -1.02852   0.08088    -12.72    <1e-31  -1.18743   -0.869612
─────────────────────────────────────────────────────────────────────────
Log-variance equation (gamma):
─────────────────────────────────────────────────────────────────────
             Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────
log σ₀²  -0.196615    0.169205  -1.16    0.2452  -0.528251  0.135021
z1        1.21344     0.15656    7.75    <1e-14   0.90659   1.5203
z2       -0.100246    0.101157  -0.99    0.3217  -0.29851   0.0980186
─────────────────────────────────────────────────────────────────────
N: 500   R^2: 0.6169   Adj. R^2: 0.6154

````

**Matrix interface** (available for every estimator; `X` and `Z` without a constant column):

````julia
X   = hcat(x1, x2);
Z_h = hcat(abs.(x1), x2);
exponential_fwls(X, y; Z = Z_h)
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.01124   0.0837127   12.08    <1e-28   0.846768   1.17572
x1            1.90355   0.12354     15.41    <1e-43   1.66082    2.14627
x2           -1.02852   0.08088    -12.72    <1e-31  -1.18743   -0.869612
─────────────────────────────────────────────────────────────────────────
Log-variance equation (gamma):
─────────────────────────────────────────────────────────────────────
             Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────
log σ₀²  -0.196615    0.169205  -1.16    0.2452  -0.528251  0.135021
z1        1.21344     0.15656    7.75    <1e-14   0.90659   1.5203
z2       -0.100246    0.101157  -0.99    0.3217  -0.29851   0.0980186
─────────────────────────────────────────────────────────────────────
N: 500   R^2: 0.6169   Adj. R^2: 0.6154

````

## Power Variance Model

The power variance model (Harvey 1976, multiplicative form) specifies:

```math
\begin{aligned}
\sigma_i^2 &= \sigma^2 \prod_j |z_{ij}|^{2\delta_j}
\end{aligned}
```

Equivalently, $\log\sigma_i^2 = \log\sigma^2 + 2\delta_1\log|z_{i1}| + \cdots$,
which is the Harvey exponential model applied to $\log|Z|$.
The reported $\boldsymbol{\gamma} = (\sigma^2, \delta_1, \ldots)$ is obtained via the
delta-method reparametrisation.

We simulate data with $\sigma_i^2 = |x_1|$ (i.e.\ $\sigma^2 = 1$, $\delta_1 = 0.5$).

````julia
sigma_pow = abs.(x1) .^ 0.5;
y_pow     = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_pow .* randn(nobs);
data_pow  = (y = y_pow, x1 = x1, x2 = x2);
````

**MLE** (`power_mle`): joint MLE warm-started from the Harvey auxiliary regression
on $\log|Z|$. Reports $(\sigma^2, \delta_1, \ldots)$ with delta-method standard errors.

````julia
power_mle(@formula(y ~ x1 + x2), data_pow; Z = @formula(0 ~ abs(x1)))
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.00317    0.0195705   51.26    <1e-99   0.964722   1.04162
x1            1.98875    0.0474852   41.88    <1e-99   1.89546    2.08205
x2           -0.999924   0.017665   -56.60    <1e-99  -1.03463   -0.965217
──────────────────────────────────────────────────────────────────────────
Link: Power       — log σ²ᵢ = γ₀ + Σγⱼlog|zᵢⱼ|
Power variance equation (log σ², 2δ):
─────────────────────────────────────────────────────────────────────
             Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────
log σ²  -0.0306619   0.0709218  -0.43    0.6655  -0.169666   0.108342
z1       1.01287     0.0564796  17.93    <1e-71   0.902172   1.12357
─────────────────────────────────────────────────────────────────────
Log-likelihood: -557.9243   Converged: true   Iterations: 11
N: 500   R^2: 0.8743   Adj. R^2: 0.8738

````

**Two-step FWLS** (`power_fwls`):

````julia
power_fwls(@formula(y ~ x1 + x2), data_pow; Z = @formula(0 ~ abs(x1)))
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.00132   0.011958    83.74    <1e-99   0.977826   1.02481
x1            1.99449   0.022658    88.03    <1e-99   1.94997    2.03901
x2           -1.00398   0.0109698  -91.52    <1e-99  -1.02553   -0.982426
─────────────────────────────────────────────────────────────────────────
Power variance equation (log σ², 2δ):
────────────────────────────────────────────────────────────────────
            Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────
log σ²  -0.210162    0.117995  -1.78    0.0749  -0.441428  0.0211047
z1       0.841352    0.093567   8.99    <1e-18   0.657964  1.02474
────────────────────────────────────────────────────────────────────
N: 500   R^2: 0.8744   Adj. R^2: 0.8739

````

## Quadratic Variance Model

The Glejser (1969) model assumes the standard deviation is linear in auxiliary
regressors $\mathbf{z}_i$:

```math
\begin{aligned}
\sigma_i &= \sigma_0 + \mathbf{z}_i^\top \boldsymbol{\gamma}
\end{aligned}
```

We simulate data with $\sigma_i$ linear in $|\mathbf{x}_1|$.

Because the variance depends on $|\mathbf{x}_1|$, not $\mathbf{x}_1$ itself, we supply a custom auxiliary
regressor matrix `Z = [|x1|, x2]` so the Glejser auxiliary regression is
correctly specified.  Using the raw regressors would yield a badly biased gamma.

````julia
sigma = 0.5 .+ 0.3 .* abs.(x1);
y = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);
quadratic_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.01448   0.0301162   33.69    <1e-99   0.955305   1.07365
x1            2.03922   0.0399564   51.04    <1e-99   1.96072    2.11772
x2           -1.04103   0.02877    -36.18    <1e-99  -1.09755   -0.984501
─────────────────────────────────────────────────────────────────────────
Link: Quadratic   — σᵢ  = γ₀ + zᵢ′γ          [Glejser 1969]
Std. deviation equation (gamma):
─────────────────────────────────────────────────────────────────
        Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────
σ₀  0.456078    0.0292284  15.60    <1e-54   0.398792   0.513365
z1  0.343226    0.0380666   9.02    <1e-18   0.268617   0.417835
z2  0.0137392   0.021326    0.64    0.5194  -0.0280591  0.0555374
─────────────────────────────────────────────────────────────────
Log-likelihood: -546.9842   Converged: true   Iterations: 10
N: 500   R^2: 0.9003   Adj. R^2: 0.8999

````

## Linear Variance Model

The linear variance model specifies the variance as a linear combination of
auxiliary regressors $\mathbf{z}_i$:

```math
\begin{aligned}
\sigma_i^2 &= \gamma_0 + \mathbf{z}_i^\top \boldsymbol{\gamma}
\end{aligned}
```

We simulate data with $\sigma_i^2$ linear in $|\mathbf{x}_1|$.

````julia
sigma2 = 0.5 .+ 0.3 .* abs.(x1);
y = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sqrt.(sigma2) .* randn(nobs);
data = (y = y, x1 = x1, x2 = x2);
linear_mle(@formula(y ~ x1 + x2), data; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.06012   0.0388198   27.31    <1e-99   0.983849   1.13639
x1            2.05141   0.0409823   50.06    <1e-99   1.97089    2.13193
x2           -1.00749   0.0375162  -26.85    <1e-98  -1.0812    -0.933782
─────────────────────────────────────────────────────────────────────────
Link: Linear      — σ²ᵢ = γ₀ + zᵢ′γ
Variance equation (gamma):
─────────────────────────────────────────────────────────────────
         Coef.  Std. Error     z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────
σ₀²  0.55191     0.0783311  7.05    <1e-11   0.398384    0.705436
z1   0.266642    0.0968623  2.75    0.0059   0.0767957   0.456489
z2   0.0289895   0.0432038  0.67    0.5022  -0.0556885   0.113667
─────────────────────────────────────────────────────────────────
Log-likelihood: -642.3411   Converged: true   Iterations: 4
N: 500   R^2: 0.8856   Adj. R^2: 0.8851

````

## Groupwise Heteroskedasticity

The groupwise model assumes a constant variance within each group $g$:

```math
\begin{aligned}
\sigma_i^2 &= \sigma_g^2, \quad i \in g
\end{aligned}
```

We simulate four groups with increasing variance.

````julia
groups   = repeat(["A", "B", "C", "D"], inner = nobs ÷ 4);
sigma_gw = [0.5, 1.0, 1.5, 2.0];
sigma_g  = [sigma_gw[findfirst(==( g), ["A","B","C","D"])] for g in groups];
y_gw     = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_g .* randn(nobs);
data_gw  = (y = y_gw, x1 = x1, x2 = x2);
````

**MLE** (`groupwise_mle`): exact MLE standard errors from $(X^\top W X)^{-1}$:

````julia
groupwise_mle(@formula(y ~ x1 + x2), data_gw, groups)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.997978   0.0354523   28.15    <1e-99   0.928323   1.06763
x1            2.0166     0.0326343   61.79    <1e-99   1.95249    2.08072
x2           -0.944209   0.0342432  -27.57    <1e-99  -1.01149   -0.876929
──────────────────────────────────────────────────────────────────────────
Group variances (G = 4):
  Group              n_g            σ²           σ
  ────────────────────────────────────────────────
  A                  125      0.218385    0.467317
  B                  125       0.90287    0.950195
  C                  125      2.254301    1.501433
  D                  125      4.065849    2.016395

Log-likelihood: -746.4561   Converged: true   Iterations: 6
N: 500   R^2: 0.7613   Adj. R^2: 0.7603

````

**Iterated FWLS** (`iterated_groupwise`): identical point estimates, GLM-scaled standard errors:

````julia
iterated_groupwise(@formula(y ~ x1 + x2), data_gw, groups)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.997978   0.0281327   35.47    <1e-99   0.942704    1.05325
x1            2.0166     0.0258966   77.87    <1e-99   1.96572     2.06748
x2           -0.944209   0.0271733  -34.75    <1e-99  -0.997597   -0.89082
──────────────────────────────────────────────────────────────────────────
Group variances (G = 4):
  Group              n_g            σ²           σ
  ────────────────────────────────────────────────
  A                  125      0.218385    0.467317
  B                  125       0.90287    0.950195
  C                  125      2.254301    1.501433
  D                  125      4.065849    2.016395

Log-likelihood: -746.4561   Converged: true   Iterations: 6
N: 500   R^2: 0.7613   Adj. R^2: 0.7603

````

## AR(1) Estimators

We simulate data with AR(1) errors: $u_t = 0.7\, u_{t-1} + e_t$.

````julia
eps  = randn(nobs);
u_ar = zeros(nobs);
for t in 2:nobs; u_ar[t] = 0.7 * u_ar[t-1] + eps[t]; end
y_ar    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ u_ar;
data_ar = (y = y_ar, x1 = x1, x2 = x2);
````

**Prais-Winsten** -- retains the first observation via GLS transformation:

````julia
two_step_prais_winsten(@formula(y ~ x1 + x2), data_ar)
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   1.04781   0.156339     6.70    <1e-10   0.740645   1.35498
x1            1.98397   0.0321344   61.74    <1e-99   1.92083    2.04711
x2           -0.97822   0.0344398  -28.40    <1e-99  -1.04589   -0.910554
─────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.727404   0.0306882  23.70    <1e-99   0.667256   0.787552
──────────────────────────────────────────────────────────────
N: 500   R^2: 0.7477   Adj. R^2: 0.7467

````

**Hildreth-Lu** -- grid search over $\rho \in (-0.99,\, 0.99)$:

````julia
hildreth_lu(@formula(y ~ x1 + x2), data_ar)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.04824    0.159783     6.56    <1e-09    0.73431   1.36218
x1            1.98394    0.0321171   61.77    <1e-99    1.92084   2.04705
x2           -0.978189   0.0344114  -28.43    <1e-99   -1.0458   -0.910579
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.731307   0.0305022  23.98    <1e-99   0.671523    0.79109
──────────────────────────────────────────────────────────────
Grid points: 200
N: 500   R^2: 0.7477   Adj. R^2: 0.7467

````

**Beach-MacKinnon** -- exact maximum likelihood:

````julia
beach_mackinnon(@formula(y ~ x1 + x2), data_ar)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.04781    0.155994     6.72    <1e-10   0.741318    1.35429
x1            1.98398    0.0321437   61.72    <1e-99   1.92082     2.04713
x2           -0.978225   0.0344497  -28.40    <1e-99  -1.04591    -0.91054
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.726796    0.030717  23.66    <1e-99   0.666591      0.787
──────────────────────────────────────────────────────────────
Log-likelihood: 22.6585   Converged: true
N: 500   R^2: 0.7477   Adj. R^2: 0.7467

````

## R Cross-Validation

Each MLE estimator has an exact R counterpart in `nlme::gls`.  Below we run
both on the same dataset — generated once in Julia with `StableRNG` and
saved to CSV so R can read it back — so the output tables can be compared
directly.  The data uses `StableRNG(42)` for the predictors and
`StableRNG(99)` for the noise.

````julia
using StableRNGs
rng_r  = StableRNG(42);
rng_r2 = StableRNG(99);
n_r    = 500;
x1_r   = randn(rng_r, n_r);
x2_r   = randn(rng_r, n_r);
Xr     = hcat(x1_r, x2_r);
br     = [1.0, 2.0, -1.0];
````

### Exponential MLE

R code:
```r
library(nlme)
d <- read.csv("hare_exp.csv")   # x1, x2, y — written from Julia
m <- gls(y ~ x1 + x2, data = d, weights = varExp(form = ~x1), method = "ML")
summary(m)
```

R output:
```
Generalized least squares fit by maximum likelihood
  Model: y ~ x1 + x2
  Data: d
       AIC      BIC    logLik
  1399.339 1420.412 -694.6695

Variance function:
 Structure: Exponential of variance covariate
 Formula: ~x1
 Parameter estimates:
    expon
0.5400336

Coefficients:
                 Value Std.Error   t-value p-value
(Intercept)  0.9762948 0.0494178  19.75594       0
x1           1.9933986 0.0311266  64.04173       0
x2          -1.0114907 0.0298082 -33.93332       0

Residual standard error: 1.001954
Degrees of freedom: 500 total; 497 residual
```

HARE: `gamma[1]` = $\log\sigma^2$ = `log(sigma^2)` from R;
`gamma[2]` = $2\delta$ = `2 * 0.5400336`:

````julia
y_r_exp = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ exp.(0.5 .* x1_r) .* randn(rng_r2, n_r);
exponential_mle(Xr, y_r_exp; Z = hcat(x1_r))
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.976295   0.0492883   19.81    <1e-64   0.879456   1.07313
x1            1.9934     0.0310571   64.19    <1e-99   1.93238    2.05442
x2           -1.01149    0.029809   -33.93    <1e-99  -1.07006   -0.952924
──────────────────────────────────────────────────────────────────────────
Link: Exponential — σ²ᵢ = exp(γ₀ + zᵢ′γ)    [Harvey 1976]
Log-variance equation (gamma):
──────────────────────────────────────────────────────────────────────
              Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────
log σ₀²  0.00390501   0.0633408   0.06    0.9508  -0.120241   0.128051
z1       1.08007      0.0594226  18.18    <1e-73   0.963601   1.19653
──────────────────────────────────────────────────────────────────────
Log-likelihood: -694.6695   Converged: true   Iterations: 9
N: 500   R^2: 0.738   Adj. R^2: 0.7369

````

### Power MLE

R code:
```r
d <- read.csv("hare_pow.csv")
m <- gls(y ~ x1 + x2, data = d, weights = varPower(form = ~abs(x1)), method = "ML")
summary(m)
```

R output:
```
Generalized least squares fit by maximum likelihood
  Model: y ~ x1 + x2
  Data: d
       AIC      BIC    logLik
  1141.608 1162.681 -565.8042

Variance function:
 Structure: Power of variance covariate
 Formula: ~abs(x1)
 Parameter estimates:
    power
0.4773703

Coefficients:
                 Value Std.Error   t-value p-value
(Intercept)  0.9955270 0.0228240  43.61754       0
x1           2.1150261 0.0483900  43.70791       0
x2          -0.9639498 0.0242141 -39.80943       0

Residual standard error: 0.98068
Degrees of freedom: 500 total; 497 residual
```

HARE: `gamma[1]` = $\log\sigma^2$; `gamma[2]` = $2\delta$ = `2 * 0.4773703`:

````julia
y_r_pow = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ abs.(x1_r).^0.5 .* randn(rng_r2, n_r);
power_mle(Xr, y_r_pow; Z = hcat(abs.(x1_r)))
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.995527   0.0227699   43.72    <1e-99    0.95079   1.04026
x1            2.11503    0.0483153   43.78    <1e-99    2.0201    2.20995
x2           -0.96395    0.0241777  -39.87    <1e-99   -1.01145  -0.916447
──────────────────────────────────────────────────────────────────────────
Link: Power       — log σ²ᵢ = γ₀ + Σγⱼlog|zᵢⱼ|
Power variance equation (log σ², 2δ):
─────────────────────────────────────────────────────────────────────
             Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────
log σ²  -0.0390182   0.0750069  -0.52    0.6029  -0.186029   0.107993
z1       0.954741    0.0718746  13.28    <1e-39   0.813869   1.09561
─────────────────────────────────────────────────────────────────────
Log-likelihood: -565.8042   Converged: true   Iterations: 10
N: 500   R^2: 0.8838   Adj. R^2: 0.8833

````

### Groupwise MLE

R code:
```r
d <- read.csv("hare_gw.csv")
d$group <- factor(d$group)
m <- gls(y ~ x1 + x2, data = d, weights = varIdent(form = ~1 | group), method = "ML")
summary(m)
```

R output:
```
Generalized least squares fit by maximum likelihood
  Model: y ~ x1 + x2
  Data: d
       AIC      BIC    logLik
  1464.294 1493.797 -725.1472

Variance function:
 Structure: Different standard deviations per stratum
 Formula: ~1 | group
 Parameter estimates:
       1        2        3        4
1.000000 1.661873 2.666309 3.399754

Coefficients:
                 Value Std.Error   t-value p-value
(Intercept)  0.9999977 0.0373464  26.77628       0
x1           1.9740476 0.0362961  54.38735       0
x2          -0.9412746 0.0376907 -24.97368       0

Residual standard error: 0.5237556
Degrees of freedom: 500 total; 497 residual
```

R's `varIdent` reports relative SDs per group (group 1 = reference = 1.0).
HARE reports $\sigma_g^2$ directly for each group:

````julia
gw_r_groups = repeat(["A","B","C","D"], inner = n_r ÷ 4);
gw_r_sig    = Dict("A"=>0.5,"B"=>1.0,"C"=>1.5,"D"=>2.0);
y_r_gw      = br[1] .+ br[2] .* x1_r .+ br[3] .* x2_r .+ [gw_r_sig[g] for g in gw_r_groups] .* randn(rng_r2, n_r);
groupwise_mle(Xr, y_r_gw, gw_r_groups)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.999998   0.0372344   26.86    <1e-98   0.926842   1.07315
x1            1.97405    0.0361872   54.55    <1e-99   1.90295    2.04515
x2           -0.941275   0.0375776  -25.05    <1e-89  -1.01511   -0.867444
──────────────────────────────────────────────────────────────────────────
Group variances (G = 4):
  Group              n_g            σ²           σ
  ────────────────────────────────────────────────
  A                  125      0.274323    0.523759
  B                  125      0.757638    0.870424
  C                  125      1.950139    1.396474
  D                  125       3.17067    1.780638

Log-likelihood: -725.1472   Converged: true   Iterations: 5
N: 500   R^2: 0.7556   Adj. R^2: 0.7547

````

## Consistency: Exponential MLE

With a large sample the MLE should recover the true parameters
$\boldsymbol{\beta} = [1, 2, -1]$ and $\boldsymbol{\gamma} = [0, 1, 0]$
(intercept $\log\sigma_0^2 = 0$, slope on $x_1$ equal to 1, slope on $x_2$ equal to 0).

````julia
rng_c   = StableRNG(42);
N       = 1_000_000;
x1_c    = randn(rng_c, N);
x2_c    = randn(rng_c, N);
sigma_c = exp.(0.5 .* x1_c);
y_c     = 1.0 .+ 2.0 .* x1_c .- 1.0 .* x2_c .+ sigma_c .* randn(rng_c, N);
data_c  = (y = y_c, x1 = x1_c, x2 = x2_c);

m_c = exponential_mle(@formula(y ~ x1 + x2), data_c)
````

````
Coefficients:
─────────────────────────────────────────────────────────────────────────────
                 Coef.   Std. Error         t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)   1.00085   0.00109866     910.97    <1e-99   0.998701   1.00301
x1            2.00132   0.00077413    2585.26    <1e-99   1.99981    2.00284
x2           -0.999302  0.000777795  -1284.79    <1e-99  -1.00083   -0.997778
─────────────────────────────────────────────────────────────────────────────
Link: Exponential — σ²ᵢ = exp(γ₀ + zᵢ′γ)    [Harvey 1976]
Log-variance equation (gamma):
─────────────────────────────────────────────────────────────────────────────
               Coef.  Std. Error       z  Pr(>|z|)    Lower 95%     Upper 95%
─────────────────────────────────────────────────────────────────────────────
log σ₀²  -0.00301577  0.00141422   -2.13    0.0330  -0.00578758  -0.000243958
z1        0.998305    0.0014144   705.82    <1e-99   0.995533     1.00108
z2        7.49778e-5  0.00141193    0.05    0.9576  -0.00269235   0.0028423
─────────────────────────────────────────────────────────────────────────────
Log-likelihood: -1.4174502441e6   Converged: true   Iterations: 8
N: 1000000   R^2: 0.7534   Adj. R^2: 0.7534

````

## Combined AR(1) + Heteroskedasticity

The DGP is $u_t = \rho\, u_{t-1} + \sigma_t\, \varepsilon_t$ with Harvey-type
heteroskedasticity in the innovations.
True parameters: $\rho = 0.7$, $\boldsymbol{\gamma} = [1,\, 0]$ on $(|x_1|, x_2)$
(reusing `sigma_exp` from the Exponential Variance section above).

Both estimators run the log-variance regression in the original space: the
Sequential estimator regresses the log squared Prais-Winsten residuals on
$\mathbf{Z}$, since the innovations $\sigma_t \varepsilon_t$ have variance
$\sigma_0^2 \exp(\mathbf{z}_t^\top \boldsymbol{\gamma})$; the Joint estimator maximises
the exact likelihood. As in the sections above, `Z = [|x1|, x2]` is passed, so
both report $\boldsymbol{\gamma}$ for the variables the variance depends on.

````julia
u_hae = zeros(nobs);
for t in 2:nobs; u_hae[t] = 0.7 * u_hae[t-1] + sigma_exp[t] * eps[t]; end
y_hae    = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ u_hae;
data_hae = (y = y_hae, x1 = x1, x2 = x2);
````

**Sequential HARE** -- corrects for AR(1) first, then heteroskedasticity:

````julia
two_step_sequential(@formula(y ~ x1 + x2), data_hae; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.0166     0.331721     3.06    0.0023   0.364854   1.66835
x1            1.9606     0.0757041   25.90    <1e-93   1.81186    2.10933
x2           -0.955632   0.0655662  -14.58    <1e-39  -1.08445   -0.826811
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.745123   0.0298259  24.98    <1e-99   0.686666   0.803581
──────────────────────────────────────────────────────────────
Log-variance equation (gamma):
─────────────────────────────────────────────────────────────────────
             Coef.  Std. Error     z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────
log σ₀²  0.0337159    0.173542  0.19    0.8460  -0.30642     0.373852
z1       0.857706     0.160573  5.34    <1e-07   0.542989    1.17242
z2       0.143196     0.10375   1.38    0.1675  -0.0601497   0.346542
─────────────────────────────────────────────────────────────────────
N: 500   R^2: 0.5228   Adj. R^2: 0.5209

````

**Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously:

````julia
two_step_joint(@formula(y ~ x1 + x2), data_hae; Z = @formula(0 ~ abs(x1) + x2))
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.04166     0.234634    4.44    <1e-04   0.580665   1.50266
x1            1.96357     0.055024   35.69    <1e-99   1.85547    2.07168
x2           -0.964916    0.049563  -19.47    <1e-62  -1.06229   -0.867537
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
─────────────────────────────────────────────────────────────
     Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────
ρ  0.73498    0.030325  24.24    <1e-99   0.675544   0.794416
─────────────────────────────────────────────────────────────
Log-variance equation (gamma):
──────────────────────────────────────────────────────────────────────
              Coef.  Std. Error     z  Pr(>|z|)   Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────
log σ₀²  0.1395       0.108134   1.29    0.1970  -0.0724395   0.351439
z1       0.712469     0.101468   7.02    <1e-11   0.513596    0.911342
z2       0.00611855   0.0678506  0.09    0.9281  -0.126866    0.139103
──────────────────────────────────────────────────────────────────────
Log-likelihood: -439.3689
N: 500   R^2: 0.5234   Adj. R^2: 0.5214

````

## Heteroskedasticity Tests

HARE provides two score (LM) tests for heteroskedasticity that are unique to
the package.  Both test $H_0\colon \gamma = 0$ (homoskedasticity) and are
distributed as $\chi^2(p)$ under $H_0$, where $p$ is the number of variance
regressors (columns of $Z$ excluding the intercept).

We reuse the exponential-variance dataset from the first section ($n = 500$,
$\log\sigma_i^2 = x_{1i}$), so both tests should reject strongly.
The R equivalents are `skedastic::harvey()` and `skedastic::glejser()`.

**Harvey test** -- LM statistic $\text{ESS}/\text{trigamma}(1/2)$ from the
auxiliary regression of $\log\hat{u}_i^2$ on $Z$:

````julia
y_test   = b[1] .+ b[2] .* x1 .+ b[3] .* x2 .+ sigma_exp .* randn(nobs);
data_test = (y = y_test, x1 = x1, x2 = x2);

harvey_test(@formula(y ~ x1 + x2), data_test)
````

````
Harvey test for multiplicative heteroskedasticity (exponential link)
  H₀: γ = 0  in  log(σᵢ²) = γ₀ + zᵢ'γ
  LM = 0.4511   df = 2   p-value = 0.7981

````

**Glejser test** -- LM statistic from the auxiliary regression of
$|\hat{u}_i|$ on $Z$:

````julia
glejser_test(@formula(y ~ x1 + x2), data_test)
````

````
Glejser test for heteroskedasticity (quadratic link / linear SD)
  H₀: γ = 0  in  σᵢ = γ₀ + zᵢ'γ
  LM = 2.3974   df = 2   p-value = 0.3016

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

