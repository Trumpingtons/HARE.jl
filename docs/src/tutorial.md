```@meta
EditURL = "../../examples/demo.jl"
```

# HARE.jl -- Tutorial

This tutorial demonstrates all estimators in HARE.jl.
Each section simulates data appropriate for that estimator.

## Setup

````julia
using HARE
using Random, LinearAlgebra, GLM, StatsBase

Random.seed!(1234)

n      = 100_000
x1     = randn(n)
x2     = randn(n)
X      = hcat(x1, x2)
X_full = hcat(ones(n), X)
b_true = [1.0, 2.0, -1.0];
````

## Harvey Estimator

The Harvey (1976) model specifies multiplicative heteroskedasticity via an exponential
variance function:

```math
\begin{aligned}
y_i &= \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\,\sigma_i^2) \\[4pt]
\sigma_i^2 &= \exp(\mathbf{x}_i^\top \boldsymbol{\gamma})
\end{aligned}
```

We simulate data with log variance proportional to x1.

Note: the auxiliary OLS regresses log(u_hat^2) - c on X, where
c = E[log(chi^2(1))] = digamma(1/2) + log(2) = -1.2703628454614782.  Subtracting c centres
the error term so that E[v_i] = 0.  Because c·1 lies in the column space of X
(an intercept is always present), OLS absorbs the shift exactly into the intercept,
yielding consistent estimates of all gamma coefficients regardless of whether the
regressors have non-zero means.  The WLS beta estimates are unaffected in any case
because a constant shift to the intercept of log(sigma^2) rescales all weights by
the same factor, which cancels in the WLS normal equations.

````julia
gamma_h  = [0.0, 1.0, 0.0]
sigma_h  = exp.(0.5 .* (X_full * gamma_h))
y_harvey = X_full * b_true .+ sigma_h .* randn(n);
````

**Matrix interface** -- pass regressors without a constant column; `Z` defaults to the
augmented mean regressors but can be any matrix with a constant column:

````julia
two_step_harvey(X, y_harvey)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.00788  0.00270571   372.50    <1e-99    1.00257   1.01318
x1            2.00605  0.00192616  1041.48    <1e-99    2.00228   2.00983
x2           -1.00136  0.00193257  -518.15    <1e-99   -1.00515  -0.997573
──────────────────────────────────────────────────────────────────────────
Variance equation (gamma):
─────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error       z  Pr(>|z|)   Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.0116627  0.00706096   -1.65    0.0986  -0.025502   0.00217647
z1            0.999699   0.00704527  141.90    <1e-99   0.98589    1.01351
z2            0.0026104  0.00710384    0.37    0.7133  -0.0113129  0.0165337
─────────────────────────────────────────────────────────────────────────────
N: 100000   R^2: 0.7529   Adj. R^2: 0.7529

````

**Formula interface** (all estimators support this):

````julia
data_harvey = (y = y_harvey, x1 = x1, x2 = x2)
two_step_harvey(@formula(y ~ x1 + x2), data_harvey)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.00788  0.00270571   372.50    <1e-99    1.00257   1.01318
x1            2.00605  0.00192616  1041.48    <1e-99    2.00228   2.00983
x2           -1.00136  0.00193257  -518.15    <1e-99   -1.00515  -0.997573
──────────────────────────────────────────────────────────────────────────
Variance equation (gamma):
─────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error       z  Pr(>|z|)   Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.0116627  0.00706096   -1.65    0.0986  -0.025502   0.00217647
z1            0.999699   0.00704527  141.90    <1e-99   0.98589    1.01351
z2            0.0026104  0.00710384    0.37    0.7133  -0.0113129  0.0165337
─────────────────────────────────────────────────────────────────────────────
N: 100000   R^2: 0.7529   Adj. R^2: 0.7529

````

## Glejser Estimator

The Glejser (1969) model assumes the standard deviation is linear in auxiliary
regressors $\mathbf{z}_i$:

```math
\begin{aligned}
y_i &= \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\,\sigma_i^2) \\[4pt]
\sigma_i &= \mathbf{z}_i^\top \boldsymbol{\gamma}
\end{aligned}
```

We simulate data with sigma linear in |x1|.

Because the variance depends on |x1|, not x1 itself, we supply a custom auxiliary
regressor matrix Z = [ones(n), |x1|, x2] so the Glejser auxiliary regression is
correctly specified.  Using the raw regressors X would yield a badly biased gamma.

````julia
sigma_g   = 0.5 .+ 0.3 .* abs.(x1)
y_glejser = X_full * b_true .+ sigma_g .* randn(n);

Z_glejser = hcat(ones(n), abs.(x1), x2);   # correctly specified auxiliary regressors
````

**Matrix interface** -- pass regressors without a constant column:

````julia
two_step_glejser(X, y_glejser; Z = Z_glejser)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.00399  0.00147948   678.61    <1e-99    1.00109    1.00689
x1            1.99645  0.00198644  1005.04    <1e-99    1.99256    2.00035
x2           -1.00321  0.00148741  -674.47    <1e-99   -1.00613   -1.0003
──────────────────────────────────────────────────────────────────────────
Std. deviation equation (gamma):
────────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)    Lower 95%    Upper 95%
────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.496552    0.00302554  164.12    <1e-99   0.490622    0.502482
z1            0.304044    0.00301881  100.72    <1e-99   0.298127    0.30996
z2           -0.00282386  0.00183203   -1.54    0.1232  -0.00641456  0.000766848
────────────────────────────────────────────────────────────────────────────────
N: 100000   R^2: 0.8958   Adj. R^2: 0.8958

````

**Formula interface:**

````julia
data_glejser = (y = y_glejser, x1 = x1, x2 = x2);
two_step_glejser(@formula(y ~ x1 + x2), data_glejser; Z = Z_glejser)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.00399  0.00147948   678.61    <1e-99    1.00109    1.00689
x1            1.99645  0.00198644  1005.04    <1e-99    1.99256    2.00035
x2           -1.00321  0.00148741  -674.47    <1e-99   -1.00613   -1.0003
──────────────────────────────────────────────────────────────────────────
Std. deviation equation (gamma):
────────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)    Lower 95%    Upper 95%
────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.496552    0.00302554  164.12    <1e-99   0.490622    0.502482
z1            0.304044    0.00301881  100.72    <1e-99   0.298127    0.30996
z2           -0.00282386  0.00183203   -1.54    0.1232  -0.00641456  0.000766848
────────────────────────────────────────────────────────────────────────────────
N: 100000   R^2: 0.8958   Adj. R^2: 0.8958

````

## AR(1) Estimators

We simulate data with AR(1) errors: u_t = 0.7 * u_{t-1} + e_t.

````julia
eps  = randn(n)
u_ar = zeros(n)
for t in 2:n; u_ar[t] = 0.7 * u_ar[t-1] + eps[t]; end
y_ar = X_full * b_true .+ u_ar;
````

**Prais-Winsten** -- retains the first observation via GLS transformation:

````julia
two_step_prais_winsten(X, y_ar)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.991258  0.010647      93.10    <1e-99    0.97039   1.01213
x1            1.99996   0.00257471   776.77    <1e-99    1.99491   2.005
x2           -0.998589  0.00259584  -384.69    <1e-99   -1.00368  -0.993501
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
───────────────────────────────────────────────────────────────
      Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────
ρ  0.703354  0.00224787  312.90    <1e-99   0.698948    0.70776
───────────────────────────────────────────────────────────────
N: 100000   R^2: 0.7164   Adj. R^2: 0.7164

````

**Hildreth-Lu** -- grid search over rho in (-0.99, 0.99):

````julia
hildreth_lu(X, y_ar)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.991258  0.0105797     93.69    <1e-99   0.970522   1.01199
x1            1.99995   0.00257704   776.07    <1e-99   1.9949     2.005
x2           -0.998589  0.00259818  -384.34    <1e-99  -1.00368   -0.993497
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
───────────────────────────────────────────────────────────────
      Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────
ρ  0.701457  0.00225379  311.23    <1e-99    0.69704   0.705875
───────────────────────────────────────────────────────────────
Grid points: 200
N: 100000   R^2: 0.7164   Adj. R^2: 0.7164

````

**Beach-MacKinnon** -- exact maximum likelihood:

````julia
beach_mackinnon(X, y_ar)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.991258  0.0106469     93.10    <1e-99    0.97039   1.01213
x1            1.99996   0.00257471   776.77    <1e-99    1.99491   2.005
x2           -0.998589  0.00259584  -384.69    <1e-99   -1.00368  -0.993501
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
───────────────────────────────────────────────────────────────
      Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────
ρ  0.703351  0.00224788  312.90    <1e-99   0.698946   0.707757
───────────────────────────────────────────────────────────────
Log-likelihood: 121.9405   Converged: true
N: 100000   R^2: 0.7164   Adj. R^2: 0.7164

````

## Combined AR(1) + Heteroskedasticity

The DGP is u_t = rho * u_{t-1} + sigma_t * eps_t with Harvey-type heteroskedasticity
in the innovations.  True parameters: rho = 0.7, gamma = [0.0, 1.0, 0.0] (reusing
sigma_h from the Harvey section above).

Note: the Sequential estimator runs the Harvey auxiliary regression on the
Prais-Winsten transformed regressors Xstar, so its reported gamma refers to that
transformed system and will only approximately match the true values.  The Joint
estimator uses the original regressors and exact MLE, so its gamma should be close
to [0.0, 1.0, 0.0].

````julia
u_hae = zeros(n)
for t in 2:n; u_hae[t] = 0.7 * u_hae[t-1] + sigma_h[t] * eps[t]; end
y_hae = X_full * b_true .+ u_hae;
````

**Sequential HARE** -- corrects for AR(1) first, then heteroskedasticity:

````julia
two_step_sequential(X, y_hae)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.993548  0.00829009   119.85    <1e-99   0.977299   1.0098
x1            1.99851   0.00155863  1282.22    <1e-99   1.99546    2.00157
x2           -0.998394  0.00156198  -639.18    <1e-99  -1.00146   -0.995332
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
───────────────────────────────────────────────────────────────
      Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────
ρ  0.703059   0.0022488  312.64    <1e-99   0.698651   0.707466
───────────────────────────────────────────────────────────────
Variance equation (gamma):
─────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.00363408  0.00702222   -0.52    0.6048  -0.0173974  0.0101292
z1            1.01024     0.00700662  144.18    <1e-99   0.996505   1.02397
z2           -0.0068742   0.00706487   -0.97    0.3305  -0.0207211  0.0069727
─────────────────────────────────────────────────────────────────────────────
N: 100000   R^2: 0.6048   Adj. R^2: 0.6048

````

**Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously:

````julia
two_step_joint(X, y_hae)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.993475  0.0106618     93.18    <1e-99   0.972578   1.01437
x1            1.99852   0.00202304   987.88    <1e-99   1.99456    2.00249
x2           -0.998417  0.00202758  -492.42    <1e-99  -1.00239   -0.994443
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
───────────────────────────────────────────────────────────────
      Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────
ρ  0.701447  0.00225382  311.23    <1e-99   0.697029   0.705864
───────────────────────────────────────────────────────────────
Variance equation (gamma):
──────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)   Lower 95%   Upper 95%
──────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.00244232  0.00447215   -0.55    0.5850  -0.0112076  0.00632293
z1            1.00172     0.00446364  224.42    <1e-99   0.99297    1.01047
z2           -0.00283999  0.0045213    -0.63    0.5299  -0.0117016  0.00602161
──────────────────────────────────────────────────────────────────────────────
Log-likelihood: -49976.5514
N: 100000   R^2: 0.6048   Adj. R^2: 0.6048

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

