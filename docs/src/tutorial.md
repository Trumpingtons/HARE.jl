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

n      = 2500
x1     = randn(n)
x2     = randn(n)
X      = hcat(x1, x2)
X_full = hcat(ones(n), X)
b_true = [1.0, 2.0, -1.0];
````

## Harvey Estimator

The Harvey model assumes the log-variance is linear in the regressors:
log(sigma_i^2) = X_full' * gamma.
We simulate data with log variance proportional to x1.

Note: the auxiliary OLS regresses log(u_hat^2) on X. Because E[log(chi^2(1))] ≈ -1.2703,
the raw gamma_0 estimate is biased downward by that constant (slope coefficients are
consistent when regressors are mean-zero).  HARE corrects gamma_0 by adding 1.2703 back,
so all reported gamma values are consistent.  This correction is purely cosmetic for gamma:
WLS is invariant to a constant weight scaling, so beta estimates are unaffected either way.

````julia
gamma_h  = [0.0, 1.0, 0.0]
sigma_h  = exp.(0.5 .* (X_full * gamma_h))
y_harvey = X_full * b_true .+ sigma_h .* randn(n);
````

**Matrix interface** -- pass regressors without a constant column:

````julia
two_step_harvey(X, y_harvey)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.993486   0.015939    62.33    <1e-99   0.962231   1.02474
x1            2.0109     0.0105925  189.84    <1e-99   1.99013    2.03167
x2           -0.992086   0.0106388  -93.25    <1e-99  -1.01295   -0.971224
──────────────────────────────────────────────────────────────────────────
Variance equation (gamma):
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)  -0.0677727   0.0466003  -1.45    0.1459  -0.159108   0.0235622
z1            1.02532     0.0453294  22.62    <1e-99   0.936473   1.11416
z2            0.0701096   0.0469752   1.49    0.1356  -0.0219601  0.162179
───────────────────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.7677   Adj. R^2: 0.7676

````

**Formula interface** (all estimators support this):

````julia
data_harvey = (y = y_harvey, x1 = x1, x2 = x2)
two_step_harvey(@formula(y ~ x1 + x2), data_harvey)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.993486   0.015939    62.33    <1e-99   0.962231   1.02474
x1            2.0109     0.0105925  189.84    <1e-99   1.99013    2.03167
x2           -0.992086   0.0106388  -93.25    <1e-99  -1.01295   -0.971224
──────────────────────────────────────────────────────────────────────────
Variance equation (gamma):
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)  -0.0677727   0.0466003  -1.45    0.1459  -0.159108   0.0235622
z1            1.02532     0.0453294  22.62    <1e-99   0.936473   1.11416
z2            0.0701096   0.0469752   1.49    0.1356  -0.0219601  0.162179
───────────────────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.7677   Adj. R^2: 0.7676

````

## Glejser Estimator

The Glejser model assumes the standard deviation is linear in auxiliary regressors Z:
sigma_i = Z_i' * gamma.
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
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.962089  0.0071769    134.05    <1e-99   0.948016   0.976163
x1            1.99499   0.00969822   205.71    <1e-99   1.97598    2.01401
x2           -1.00744   0.00724889  -138.98    <1e-99  -1.02165   -0.993221
───────────────────────────────────────────────────────────────────────────
Std. deviation equation (gamma):
────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.37481     0.0150662   24.88    <1e-99   0.345281   0.404339
z1            0.25976     0.0146483   17.73    <1e-69   0.231049   0.28847
z2           -0.00184489  0.00915904  -0.20    0.8404  -0.0197963  0.0161065
────────────────────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.9022   Adj. R^2: 0.9021

````

**Formula interface:**

````julia
data_glejser = (y = y_glejser, x1 = x1, x2 = x2);
two_step_glejser(@formula(y ~ x1 + x2), data_glejser; Z = Z_glejser)
````

````
Coefficients:
───────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.962089  0.0071769    134.05    <1e-99   0.948016   0.976163
x1            1.99499   0.00969822   205.71    <1e-99   1.97598    2.01401
x2           -1.00744   0.00724889  -138.98    <1e-99  -1.02165   -0.993221
───────────────────────────────────────────────────────────────────────────
Std. deviation equation (gamma):
────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.37481     0.0150662   24.88    <1e-99   0.345281   0.404339
z1            0.25976     0.0146483   17.73    <1e-69   0.231049   0.28847
z2           -0.00184489  0.00915904  -0.20    0.8404  -0.0197963  0.0161065
────────────────────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.9022   Adj. R^2: 0.9021

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
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.968808   0.0670879   14.44    <1e-44   0.837255   1.10036
x1            2.00084    0.0161649  123.78    <1e-99   1.96914    2.03254
x2           -0.98818    0.0166643  -59.30    <1e-99  -1.02086   -0.955502
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.700197    0.014279  49.04    <1e-99    0.67221   0.728183
──────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.7326   Adj. R^2: 0.7324

````

**Hildreth-Lu** -- grid search over rho in (-0.99, 0.99):

````julia
hildreth_lu(X, y_ar)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.968751   0.0674613   14.36    <1e-44   0.836465   1.10104
x1            2.00082    0.0161598  123.81    <1e-99   1.96913    2.0325
x2           -0.988186   0.016658   -59.32    <1e-99  -1.02085   -0.955522
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.701457   0.0142542  49.21    <1e-99    0.67352   0.729395
──────────────────────────────────────────────────────────────
Grid points: 200
N: 2500   R^2: 0.7326   Adj. R^2: 0.7324

````

**Beach-MacKinnon** -- exact maximum likelihood:

````julia
beach_mackinnon(X, y_ar)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.968809   0.0672612   14.40    <1e-44   0.836915   1.1007
x1            2.00083    0.0161591  123.82    <1e-99   1.96914    2.03251
x2           -0.988183   0.0166582  -59.32    <1e-99  -1.02085   -0.955517
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
─────────────────────────────────────────────────────────────
     Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────
ρ  0.70097   0.0142638  49.14    <1e-99   0.673014   0.728927
─────────────────────────────────────────────────────────────
Log-likelihood: -15.1905   Converged: true
N: 2500   R^2: 0.7326   Adj. R^2: 0.7324

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
(Intercept)   0.970088  0.0489969     19.80    <1e-80   0.874009   1.06617
x1            2.00261   0.00921368   217.35    <1e-99   1.98454    2.02068
x2           -0.998028  0.00936041  -106.62    <1e-99  -1.01638   -0.979673
───────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.693669   0.0144059  48.15    <1e-99   0.665434   0.721904
──────────────────────────────────────────────────────────────
Variance equation (gamma):
────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  0.0359084     0.0432769   0.83    0.4067  -0.0489127    0.12073
z1           1.03104       0.0420966  24.49    <1e-99   0.948535     1.11355
z2           0.000716541   0.043625    0.02    0.9869  -0.0847869    0.08622
────────────────────────────────────────────────────────────────────────────
Converged: true   Iterations: 1
N: 2500   R^2: 0.6226   Adj. R^2: 0.6223

````

**Joint HARE** -- estimates AR(1) and heteroskedasticity simultaneously:

````julia
two_step_joint(X, y_hae)
````

````
Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.97119    0.0661814   14.67    <1e-46   0.841414   1.10097
x1            2.00269    0.0123642  161.97    <1e-99   1.97845    2.02694
x2           -0.997066   0.0125651  -79.35    <1e-99  -1.02171   -0.972427
──────────────────────────────────────────────────────────────────────────
AR(1) coefficient:
──────────────────────────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
ρ  0.699849   0.0142858  48.99    <1e-99   0.671849   0.727849
──────────────────────────────────────────────────────────────
Variance equation (gamma):
────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.0115895    0.0283017   0.41    0.6822  -0.0438808  0.0670598
z1            0.993879     0.0268608  37.00    <1e-99   0.941233   1.04653
z2           -0.00853688   0.0289637  -0.29    0.7682  -0.0653047  0.0482309
────────────────────────────────────────────────────────────────────────────
Log-likelihood: -1220.2229   Converged: true   Iterations: 1
N: 2500   R^2: 0.6226   Adj. R^2: 0.6223

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

