# HARE.jl — Implementation Notes

This document describes, for each estimator in HARE.jl, the exact computational
procedure, the choices made at each step, and the references supporting those
choices.  It is intended as an internal reference for contributors and for users
who want to understand what the code is actually doing.

---

## 1. Shared infrastructure

### 1.1 Rho estimation (Cochrane-Orcutt moment estimator)

$$\hat{\rho} = \frac{\sum_{t=2}^{n} u_t\, u_{t-1}}{\sum_{t=2}^{n} u_{t-1}^2}$$

This is the standard first-order autoregressive moment estimator proposed by
Cochrane & Orcutt (1949).  It is consistent under mild regularity conditions and
cheap to compute.  It is used as the initial estimate of rho in all two-step
AR(1) procedures.

### 1.2 Prais-Winsten transformation

For $\rho \in (-1, 1)$, the AR(1) error structure is removed by quasi-differencing:

$$\begin{aligned}
y^*_1 &= \sqrt{1-\rho^2}\; y_1 \\
y^*_t &= y_t - \rho\, y_{t-1}, \quad t = 2, \ldots, n \\
\mathbf{X}^*_{1,:} &= \sqrt{1-\rho^2}\; \mathbf{X}_{1,:} \\
\mathbf{X}^*_{t,:} &= \mathbf{X}_{t,:} - \rho\, \mathbf{X}_{t-1,:}, \quad t = 2, \ldots, n
\end{aligned}$$

The first-observation scaling by $\sqrt{1-\rho^2}$ is the Prais-Winsten
correction (Prais & Winsten, 1954).  The alternative — simply dropping the first
observation — is the Cochrane-Orcutt convention.  Prais-Winsten is preferred
because it retains all $n$ observations and is asymptotically more efficient,
particularly when $\rho$ is large and $n$ is small.

---

## 2. Harvey (1976) FWLS — multiplicative heteroskedasticity

### Model

$$y_i = \mathbf{x}_i^\top\boldsymbol{\beta} + u_i, \quad u_i \sim \mathcal{N}(0, \sigma_i^2), \qquad \log\sigma_i^2 = \mathbf{z}_i^\top\boldsymbol{\gamma}$$

where $\mathbf{Z}$ is an auxiliary regressor matrix (by default equal to $\mathbf{X}$
augmented with an intercept; user-supplied $\mathbf{Z}$ must not include a constant —
one is prepended internally, consistent with the $\mathbf{X}$ convention).

### Procedure

**Step 1.** OLS to get residuals $\hat{u}$.

**Step 2.** Auxiliary log-linear regression for $\boldsymbol{\gamma}$.  Because
$u_i = \sigma_i \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0,1)$:

$$\log u_i^2 = \log\sigma_i^2 + \log\varepsilon_i^2 = \mathbf{z}_i^\top\boldsymbol{\gamma} + \log\varepsilon_i^2$$

$\log\varepsilon_i^2$ is distributed as $\log\chi^2(1)$, which has mean

$$\mathrm{E}[\log\chi^2(1)] = \psi(1/2) + \log 2 = -1.2703628454614782 \eqqcolon c$$

The naive OLS of $\log\hat{u}^2$ on $\mathbf{Z}$ therefore estimates
$\boldsymbol{\gamma}$ with all coefficients biased by
$(\mathbf{Z}^\top\mathbf{Z})^{-1}\mathbf{Z}^\top c\mathbf{1}$, where
$c = \mathrm{E}[\log\chi^2(1)]$.  When regressors have non-zero means this
contaminates every element of $\boldsymbol{\gamma}$, not just the intercept.

The correction subtracts $c$ from the dependent variable **before** fitting:

$$\log\hat{u}_i^2 - c = \mathbf{z}_i^\top\boldsymbol{\gamma} + v_i, \qquad \mathrm{E}[v_i] = 0$$

Because $c\mathbf{1}$ lies in the column space of $\mathbf{Z}$ (an intercept
column is always present), OLS absorbs the shift exactly into the intercept,
yielding consistent estimates of **all** elements of $\boldsymbol{\gamma}$
regardless of regressor means.  The fitted values
$\mathbf{z}_i^\top\hat{\boldsymbol{\gamma}}$ then directly estimate $\log\sigma_i^2$.

**Step 3.** WLS with weights $w_i = 1/\exp(\mathbf{z}_i^\top\hat{\boldsymbol{\gamma}})$.

**Step 4.** Recover $\boldsymbol{\beta}$ by WLS of $\mathbf{y}$ on $\mathbf{X}$ with weights $\mathbf{w}$.

### Iterated version

Repeat Steps 2–4 until $\max|\beta^{(i)} - \beta^{(i-1)}| < \text{tol}$.  Convergence of the
iterated FWLS to the MLE is established under regularity conditions by
Oberhofer & Kmenta (1974).

`iterated_harvey` is the preferred FWLS choice when the iterative reweighting
framework is explicitly desired.  For asymptotically efficient estimation,
`exponential_mle` (Section 4) achieves the MLE directly via joint L-BFGS.

### Standard errors

Reported vcov is the WLS sandwich $\hat{\sigma}^2(\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}$,
conditional on the estimated weights.  This is the standard feasible GLS
standard error; it does not account for the estimation uncertainty in
$\boldsymbol{\gamma}$.

### References

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*, 42(3),
579–590.

Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, §9.7.

---

## 3. Glejser (1969) FWLS — linear standard deviation model

### Model

$$\begin{aligned}
y_i &= \mathbf{x}_i^\top\boldsymbol{\beta} + u_i, \quad u_i \sim \mathcal{N}(0, \sigma_i^2) \\
\sigma_i &= \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}
\end{aligned}$$

where $\mathbf{Z}$ is an auxiliary regressor matrix (by default equal to $\mathbf{X}$
without the intercept column; a constant is prepended internally, consistent with
the $\mathbf{X}$ convention).

### Procedure

**Step 1.** OLS to get residuals $\hat{u}$.

**Step 2.** Auxiliary regression for $\boldsymbol{\gamma}$.  Under the model,
$|u_i| = \sigma_i|\varepsilon_i|$ with $\varepsilon_i \sim \mathcal{N}(0,1)$, so

$$\mathrm{E}[|u_i|] = \sigma_i\,\mathrm{E}[|\varepsilon_i|] = \sigma_i\sqrt{2/\pi}$$

OLS of $|\hat{u}|$ on $\mathbf{Z}$ therefore consistently estimates
$\boldsymbol{\gamma}\sqrt{2/\pi}$, not $\boldsymbol{\gamma}$ itself.  The
correction multiplies the raw OLS coefficients by $\sqrt{\pi/2}$:

$$\hat{\boldsymbol{\gamma}} = \sqrt{\pi/2}\;(\mathbf{Z}^\top\mathbf{Z})^{-1}\mathbf{Z}^\top|\hat{\mathbf{u}}|$$

The vcov is scaled accordingly by $\pi/2$.  For the WLS weights,
$\hat{\sigma}_i = \max(\mathbf{z}_i^\top\hat{\boldsymbol{\gamma}},\, 10^{-8})$
(clipped to avoid division by zero), and $w_i = 1/\hat{\sigma}_i^2$.

Without this correction the reported $\boldsymbol{\gamma}$ would be biased toward
zero by the factor $\sqrt{2/\pi} \approx 0.798$ for all $n$, making the estimator
appear inconsistent.  The WLS $\boldsymbol{\beta}$ itself is unaffected because a
constant weight scaling cancels in the normal equations, but the displayed
$\boldsymbol{\gamma}$ values and their standard errors would be wrong.

**Step 3.** WLS of $\mathbf{y}$ on $\mathbf{X}$ with weights $\mathbf{w}$.

### Iterated version

Repeat Steps 2–3 until $\max|\beta^{(i)} - \beta^{(i-1)}| < \text{tol}$.

`iterated_glejser` is the preferred FWLS choice when the iterative reweighting
framework is explicitly desired.  For asymptotically efficient estimation,
`quadratic_mle` (Section 4) achieves the MLE directly via joint L-BFGS.

### Standard errors

Same as Harvey: WLS sandwich conditional on the estimated weights.

### References

Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American
Statistical Association*, 64(325), 316–323.

---

## 4. MLE heteroskedasticity estimators — exponential, quadratic, linear

Three estimators share a common MLE framework, differing only in the link
function that maps the variance equation parameters to $\sigma_i^2$.  They
correspond to the SAS `PROC AUTOREG HETERO` link options.

### Models

| Function | Link | Variance | Origin | Associated test |
|---|---|---|---|---|
| `exponential_mle` | EXP | $\sigma_i^2 = \exp(\gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma})$ | Harvey (1976) | Harvey LM test |
| `quadratic_mle` | SQUARE | $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ | Glejser (1969) | Glejser test |
| `linear_mle` | LINEAR | $\sigma_i^2 = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ | Breusch & Pagan (1979) | Breusch-Pagan test (`HypothesisTests.BreuschPaganTest`) |

In all cases $\mathbf{z}_i$ is the user-supplied auxiliary regressor vector
**without** a constant; $\gamma_0$ is the intercept added internally.  The
default is $\mathbf{z}_i = \tilde{\mathbf{x}}_i$ (the mean regressors without
their intercept column).

### Log-likelihood

All three maximise the normal log-likelihood jointly over
$\theta = (\boldsymbol{\beta}, \gamma_0, \boldsymbol{\gamma})$:

$$\ell(\theta) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\sum_i \log\sigma_i^2(\theta) - \frac{1}{2}\sum_i \frac{(y_i - \mathbf{x}_i^\top\boldsymbol{\beta})^2}{\sigma_i^2(\theta)}$$

For the exponential link the log-variance is linear in the parameters so the
objective is globally concave in $\boldsymbol{\gamma}$ for fixed
$\boldsymbol{\beta}$.  For the quadratic and linear links the objective can be
non-concave; the warm start (see below) is important for reliable convergence.

### Numerical stability

The quadratic and linear link functions require $\sigma_i > 0$ and
$\sigma_i^2 > 0$ respectively.  The two links are handled differently.

**Exponential link.**  $\sigma_i^2 = \exp(\mathbf{z}_i^\top\boldsymbol{\gamma})$ is strictly
positive for any $\boldsymbol{\gamma}$, so no clamping or smoothing is needed.  The
Harvey FWLS warm start also carries slope information (the auxiliary log-linear
regression directly estimates each element of $\boldsymbol{\gamma}$), so L-BFGS starts
in a well-identified region and never explores infeasible directions.

**Quadratic link.**  $\hat{\sigma}_i = \max(\mathbf{z}_i^\top\hat{\boldsymbol{\gamma}},\, 10^{-10})$.
A hard clamp suffices here because the Glejser FWLS warm start estimates $\boldsymbol{\gamma}$
from $|\hat{u}|$, which is non-negative, so the warm start already carries slope
information with $\hat{\sigma}_i > 0$ everywhere.  L-BFGS starts in the feasible
region and rarely crosses zero during the line search; the clamp is almost never
triggered, so its non-smoothness has no practical effect.  Using softplus instead
would be unnecessary: it requires two transcendental evaluations per observation
versus a single comparison, and ForwardDiff propagates derivatives through both,
adding cost with no benefit.

**Linear link.**  Rather than a hard clamp, the implementation uses the
**softplus** function:

$$\tilde{\sigma}_i^2 = \log\!\left(1 + e^{\,\mathbf{z}_i^\top\boldsymbol{\gamma}}\right)$$

Softplus is smooth and strictly positive everywhere, so ForwardDiff computes exact,
continuous gradients throughout the L-BFGS line search.  The linear link is the
problematic case: its warm start is deliberately flat ($\gamma_0 = \bar{u}^2$, all
slopes $= 0$) to guarantee a feasible starting point, but this gives L-BFGS no
directional information about the slopes.  The optimizer immediately explores slope
directions that push some $\sigma_i^2$ negative, hitting the clamp on the very
first line search.  A hard clamp creates a kink at zero: the gradient reported by
ForwardDiff is accurate within each piece but the non-smoothness causes L-BFGS to
make poor curvature estimates when the iterate crosses the boundary, which can
prevent convergence — particularly on Windows x86 and macOS ARM where
floating-point evaluation order differs.  For $\sigma_i^2$ well above zero,
$\log(1 + e^x) \approx x$, so the effective model is indistinguishable from the
true linear variance function in the well-specified region.

### Optimisation

Minimisation of the negative log-likelihood using **L-BFGS** (Optim.jl) with
gradients supplied by **ForwardDiff.jl** automatic differentiation.  The full
parameter vector $\theta = (\boldsymbol{\beta}, \gamma_0, \boldsymbol{\gamma})$
is optimised jointly; the mean and variance equations are not concentrated.

**Choice of L-BFGS over concentrated likelihood.**  Unlike Beach-MacKinnon or
Joint HARE — where concentrating out $\boldsymbol{\beta}$ analytically reduces
the problem dimension substantially — here the mean and variance parameters
enter the gradient in a coupled way that does not yield a cheap closed-form
profile.  Joint L-BFGS with AD is straightforward and fast for the parameter
dimensions typical of these models.

### Warm starts

Good starting values are essential for the quadratic and linear links.

| Link | Warm start |
|---|---|
| Exponential | Two-step Harvey FWLS: $\boldsymbol{\beta}_0$ from OLS, $\boldsymbol{\gamma}_0$ from `harvey_weights` |
| Quadratic | Two-step Glejser FWLS: $\boldsymbol{\beta}_0$ from OLS, $\boldsymbol{\gamma}_0$ from `glejser_weights` |
| Linear | Constant variance: $\boldsymbol{\beta}_0$ from OLS, $\gamma_0 = \bar{u}^2$, $\boldsymbol{\gamma} = \mathbf{0}$ |

The constant-variance start for the linear link is always feasible (all
$\sigma_i^2 > 0$), avoiding the risk of OLS-of-$\hat{u}^2$ yielding negative
predicted variances at some observations.

### Standard errors

The full observed information matrix is computed by `ForwardDiff.hessian` of
the negative log-likelihood at the optimum $\hat{\theta}$.  The vcov of the
complete parameter vector is $H^{-1}$.  The $\boldsymbol{\beta}$ and
$\boldsymbol{\gamma}$ blocks are then extracted:

$$\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) = H^{-1}[1{:}k,\;1{:}k], \qquad
  \widehat{\mathrm{Var}}(\hat{\boldsymbol{\gamma}}) = H^{-1}[k{+}1{:}\mathrm{end},\;k{+}1{:}\mathrm{end}]$$

This is the full-information MLE standard error, accounting for joint
estimation of mean and variance parameters.  It is asymptotically efficient
under correct specification and normality, in contrast to the FWLS standard
errors in Sections 2–3 which are conditional on the estimated weights.

### References

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American
Statistical Association*, 64(325), 316–323.

SAS Institute Inc. (2015). *SAS/ETS 14.1 User's Guide: The AUTOREG Procedure*.
SAS Institute Inc., Cary, NC.

---

## 5. Groupwise heteroscedasticity

### Model

$$y_i = \mathbf{x}_i^\top\boldsymbol{\beta} + u_i, \qquad \mathrm{Var}(u_i) = \sigma_g^2 \;\text{ for all } i \in g$$

Observations are divided into $G$ groups.  The variance is constant within each
group but unrestricted across groups.  Group membership is supplied by the user
as a vector of labels (any type: integers, strings, …).

### Procedure

**Step 1.** OLS to get residuals $\hat{u}_i$.

**Step 2.** Estimate group variances: $\hat{\sigma}_g^2 = \dfrac{1}{n_g}\displaystyle\sum_{i \in g} \hat{u}_i^2$.

**Step 3.** WLS with weights $w_i = 1/\hat{\sigma}_g^2$.

### Iterated version and MLE equivalence

The MLE first-order condition for $\sigma_g^2$ — obtained by differentiating the
normal log-likelihood — is exactly $\hat{\sigma}_g^2 = (1/n_g)\sum_{i\in g}\hat{u}_i^2$.
Therefore the iterated FWLS (repeating Steps 2–3 until $\boldsymbol{\beta}$ converges)
is the EM algorithm for this model and converges to the joint MLE of
$(\boldsymbol{\beta}, \sigma_1^2, \ldots, \sigma_G^2)$.  No separate MLE function is needed.

### Standard errors

WLS vcov conditional on $\hat{\sigma}_g^2$, identical in form to the Harvey FWLS
standard errors.

### References

Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, §9.7.2.

---

## 6. Prais-Winsten / Cochrane-Orcutt — AR(1) autocorrelation

### Model

$$y_t = \mathbf{x}_t^\top\boldsymbol{\beta} + u_t, \qquad u_t = \rho\,u_{t-1} + e_t, \quad e_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$

### Two-step procedure

**Step 1.** OLS; estimate $\rho$ by the Cochrane-Orcutt moment estimator
(Section 1.1).

**Step 2.** Apply Prais-Winsten transformation (Section 1.2); OLS on the
transformed system.

### Iterated procedure

Alternate between:
- re-estimating $\rho$ from the residuals $\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$,
- re-transforming and re-fitting by OLS,

until $|\rho^{(i)} - \rho^{(i-1)}| < \text{tol}$.

### Choice: Prais-Winsten over Cochrane-Orcutt

Hildreth-Lu and Beach-MacKinnon use the Cochrane-Orcutt convention (dropping the
first observation) because their objectives are defined that way in their
respective papers.  The Prais-Winsten estimator retains the first observation,
which is asymptotically more efficient (the gain is $O(1/n)$ but can matter
in small samples).

### Standard errors

OLS vcov on the transformed system, conditional on $\hat{\rho}$.

### References

Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial correlation.
*Cowles Commission Discussion Paper*, No. 383.

Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the American
Statistical Association*, 44(245), 32–61.

---

## 7. Hildreth-Lu (1960) — grid search over rho

### Procedure

For each $\rho$ in a uniform grid of `nrho = 200` points on $(-0.99, 0.99)$:

1. Apply the Cochrane-Orcutt quasi-difference (first observation dropped):
   $y^*_t = y_t - \rho\,y_{t-1}$, $\mathbf{X}^*_t = \mathbf{X}_t - \rho\,\mathbf{X}_{t-1}$, $t = 2, \ldots, n$.
2. OLS on the transformed system; record RSS.

Select the $\rho$ that minimises RSS.

### Choices

- **Grid vs optimisation.** A grid search is guaranteed to find the global
  minimum over the search space and is robust when the RSS surface has multiple
  local minima.  The cost is resolution: with 200 points the grid spacing is
  $\approx 0.01$ in $\rho$.  If higher precision is needed, `nrho` can be increased.
- **First observation dropped.**  This follows the original Hildreth-Lu
  convention and is consistent with their derivation.  The effective sample size
  is therefore $n - 1$.
- **Search interval $(-0.99, 0.99)$.**  Values at $\pm 1$ make the transformation
  degenerate; $\pm 0.99$ avoids numerical issues while covering the practically
  relevant range.

### Standard errors

OLS vcov on the transformed system at the selected $\rho$, conditional on $\hat{\rho}$.

### References

Hildreth, C., & Lu, J. Y. (1960). Demand relations with autocorrelated
disturbances. *Michigan State University Agricultural Experiment Station
Technical Bulletin*, No. 276.

---

## 8. Beach-MacKinnon (1978) — exact MLE for AR(1)

### Log-likelihood

The exact log-likelihood for the AR(1) model (including the Jacobian for the
first observation) is, up to constants:

$$L(\rho) = \frac{n}{2}\log(1-\rho^2) - \frac{n}{2}\log\!\left(\frac{S(\rho)}{n}\right)$$

where $S(\rho) = \|\mathbf{y}^* - \mathbf{X}^*\hat{\boldsymbol{\beta}}(\rho)\|^2$
is the residual sum of squares at the GLS estimate $\hat{\boldsymbol{\beta}}(\rho)$
obtained after the Prais-Winsten transformation at $\rho$.  Concentrating out
$\boldsymbol{\beta}$ and $\sigma^2$ analytically leaves a univariate problem in $\rho$.

The term $\tfrac{1}{2}\log(1-\rho^2)$ is the Jacobian contributed by the first
observation under the stationary AR(1) distribution.  Dropping it (as in the
Prais-Winsten/Cochrane-Orcutt objective) gives an approximation that is
$O(1/n)$ away from the exact likelihood.

### Maximisation

Implemented as minimisation of the negative concentrated log-likelihood via
**Brent's method** on the interval $(-0.999, 0.999)$.  Brent's method is chosen
because the problem is univariate, the function is smooth but not necessarily
unimodal, and Brent's method combines the robustness of bisection with the
speed of inverse quadratic interpolation.

### Recovery of beta and sigma^2

Once $\hat{\rho}$ is found, $\hat{\boldsymbol{\beta}}$ and $\hat{\sigma}^2$ are
recovered by OLS on the Prais-Winsten-transformed system at $\hat{\rho}$.

### Standard errors

Conditional on $\hat{\rho}$: the OLS vcov on the transformed system.

### References

Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for
regression with autocorrelated errors. *Econometrica*, 46(1), 51–58.

---

## 9. Sequential HARE — AR(1) + Harvey heteroskedasticity

### Model

$$\begin{aligned}
y_t &= \mathbf{x}_t^\top\boldsymbol{\beta} + u_t \\
u_t &= \rho\,u_{t-1} + \sigma_t\,e_t, \quad e_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0,1) \\
\log\sigma_t^2 &= \mathbf{x}_t^\top\boldsymbol{\gamma}
\end{aligned}$$

### Two-step procedure

**Step 1.** OLS; estimate $\rho$ by the Cochrane-Orcutt moment estimator.

**Step 2.** Prais-Winsten transformation → residuals $\hat{u}^{\mathrm{pw}}$ from OLS on the
transformed system.

**Step 3.** Harvey auxiliary regression of $\log(\hat{u}^{\mathrm{pw}}_t)^2 - c$ on the **original**
$\mathbf{X}$ (not the transformed $\mathbf{X}^*$).  Rationale: the PW innovations satisfy
$\hat{u}^{\mathrm{pw}}_t \approx \sigma_t\,e_t$ with $\log\sigma_t^2 = \mathbf{x}_t^\top\boldsymbol{\gamma}$
in the original space, so the correct auxiliary regressors are the untransformed $\mathbf{x}_t$.

**Step 4.** WLS on the PW-transformed system $(\mathbf{X}^*, \mathbf{y}^*)$ with Harvey weights
$w_t = 1/\exp(\mathbf{x}_t^\top\hat{\boldsymbol{\gamma}})$.

### Iterated version

Alternate between Steps 1–4 until
$\max\!\left(\max|\beta^{(i)} - \beta^{(i-1)}|,\;|\rho^{(i)} - \rho^{(i-1)}|\right) < \text{tol}$.

### References

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*, 42(3),
579–590.

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

---

## 10. Joint HARE — simultaneous MLE for AR(1) + Harvey heteroskedasticity

### Model

Same as the Sequential HARE model (Section 9).

### Log-likelihood

The exact log-likelihood (up to the constant $-\tfrac{n}{2}\log(2\pi)$) is:

$$\ell(\rho, \boldsymbol{\gamma}, \boldsymbol{\beta}) =
  \tfrac{1}{2}\log(1-\rho^2)
  - \tfrac{1}{2}\sum_t \log\sigma_t^2
  - \tfrac{1}{2}\!\left(\frac{\sqrt{1-\rho^2}\;u_1}{\sigma_1}\right)^{\!2}
  - \tfrac{1}{2}\sum_{t=2}^{n}\!\left(\frac{u_t - \rho\,u_{t-1}}{\sigma_t}\right)^{\!2}$$

where $u_t = y_t - \mathbf{x}_t^\top\boldsymbol{\beta}$ and
$\sigma_t = \sqrt{\exp(\mathbf{z}_t^\top\boldsymbol{\gamma})}$.

### Two-step procedure

**Step 1 (concentrated MLE).** Profile out $\boldsymbol{\beta}$ analytically at each function
evaluation: given $(\rho, \boldsymbol{\gamma})$, apply the doubly-transformed system

$$y^{**}_t = y^*_t / \sigma_t, \qquad \mathbf{X}^{**}_t = \mathbf{X}^*_t / \sigma_t$$

where $\mathbf{y}^*$, $\mathbf{X}^*$ is the Prais-Winsten transformation at $\rho$, and compute
$\hat{\boldsymbol{\beta}}(\rho, \boldsymbol{\gamma}) = (\mathbf{X}^{**\top}\mathbf{X}^{**})^{-1}\mathbf{X}^{**\top}\mathbf{y}^{**}$
by OLS.  The objective is then a function of $(\rho, \boldsymbol{\gamma})$ only.

**Step 2.** Minimise the negative concentrated log-likelihood over
$(\mathrm{atanh}(\rho), \boldsymbol{\gamma})$ using **L-BFGS** (Optim.jl).  The unconstrained
reparametrisation $\rho = \tanh(\xi)$ enforces $|\rho| < 1$ without box
constraints.  Gradients are computed by **ForwardDiff.jl** automatic
differentiation.  Once convergence is reached, $\boldsymbol{\beta}$ is recovered by one OLS
step on the doubly-transformed system.

### Iterated procedure

Coordinate descent:

1. With $\boldsymbol{\beta}$ fixed, minimise the negative log-likelihood over $(\rho, \boldsymbol{\gamma})$ via
   L-BFGS.
2. With $(\rho, \boldsymbol{\gamma})$ fixed, update $\boldsymbol{\beta}$ by one OLS step on the doubly-transformed
   system.

Repeat until $\max\!\left(\max|\beta^{(i)} - \beta^{(i-1)}|,\;|\rho^{(i)} - \rho^{(i-1)}|\right) < \text{tol}$.

### Standard errors for gamma

The vcov of $\hat{\boldsymbol{\gamma}}$ is the $(2{:}\mathrm{end},\;2{:}\mathrm{end})$ block of the
inverse Hessian of the negative log-likelihood, evaluated at the optimum.  The
Hessian is computed by ForwardDiff.  Index 1 corresponds to
$\mathrm{atanh}(\hat{\rho})$, whose uncertainty is summarised separately via the
delta-method formula $\mathrm{SE}(\hat{\rho}) = \sqrt{(1-\hat{\rho}^2)^2/n}$
shown in the display (consistent with the Fisher information for a stationary AR(1)).

### Standard errors for beta

The OLS vcov on the doubly-transformed system, conditional on
$(\hat{\rho}, \hat{\boldsymbol{\gamma}})$.

### Choices

- **L-BFGS over Nelder-Mead.** The log-likelihood is smooth and gradients are
  available cheaply via AD, making a quasi-Newton method much faster than a
  derivative-free method for this problem size.
- **Concentrated likelihood.** Profiling out $\boldsymbol{\beta}$ reduces the dimension of the
  optimisation from $k + 1 + p$ to $1 + p$ (where $k$ is the number of regressors
  and $p$ the number of variance parameters), which substantially improves
  convergence.
- **ForwardDiff over FiniteDiff.** Automatic differentiation is exact to machine
  precision and avoids step-size tuning; the cost is proportional to the number
  of parameters, which is small here.

### References

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*, 42(3),
579–590.

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for
regression with autocorrelated errors. *Econometrica*, 46(1), 51–58.

---

## 11. Heteroskedasticity LM tests

### Harvey test (`harvey_test`)

Tests $H_0\colon \boldsymbol{\gamma} = \mathbf{0}$ in the exponential variance model
$\log\sigma_i^2 = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$.

**Procedure.** OLS residuals $\hat{u}_i$ → auxiliary regression of
$\log\hat{u}_i^2 - c$ on $\mathbf{Z}$ (same bias correction as in `harvey_weights`) →
$\mathrm{LM} = n R^2 \sim \chi^2(p)$ under $H_0$, where $p = \dim(\boldsymbol{\gamma})$
(slope parameters only, excluding intercept).

**Equivalence to SAS.** SAS `PROC AUTOREG HETERO TEST=LM` computes the same
statistic.

### Glejser test (`glejser_test`)

Tests $H_0\colon \boldsymbol{\gamma} = \mathbf{0}$ in the quadratic variance model
$\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$.

**Procedure.** OLS residuals $\hat{u}_i$ → auxiliary regression of $|\hat{u}_i|$
on $\mathbf{Z}$ → $\mathrm{LM} = n R^2 \sim \chi^2(p)$ under $H_0$.  The
$\sqrt{\pi/2}$ correction used in estimation is scale-invariant and cancels in
$R^2$; it does not affect the test statistic.

### Borrowed tests: Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey

Four additional diagnostic tests are provided as thin wrappers around
`HypothesisTests.jl`, which supplies the underlying computations:

| HARE function | Wraps | Null hypothesis |
|---|---|---|
| `breusch_pagan_test` | `WhiteTest(type=:linear)` | $\sigma_i^2$ constant (linear link) |
| `white_test` | `WhiteTest(type=:White)` | $\sigma_i^2$ constant (general) |
| `durbin_watson_test` | `DurbinWatsonTest` | no first-order serial correlation |
| `breusch_godfrey_test` | `BreuschGodfreyTest` | no serial correlation up to order $p$ |

Each wrapper runs OLS internally and accepts the same `(X, y; intercept=true)`
and `(formula, data)` calling conventions as `harvey_test` and `glejser_test`,
so users do not need to manage residuals manually.

**Why snake\_case names instead of the `HypothesisTests` CamelCase names?**

`HypothesisTests.jl` already exports `BreuschPaganTest`, `WhiteTest`,
`DurbinWatsonTest`, and `BreuschGodfreyTest` as `(X, e)` constructors, where
`X` is the full design matrix (with intercept) and `e` is a pre-computed
residual vector.  HARE's convention is `(X, y)`, where `X` excludes the
intercept and `y` is the raw response — the OLS step is handled internally.
Both signatures are `(AbstractMatrix, AbstractVector)` positionally, so Julia
cannot distinguish them by dispatch: defining HARE methods under the same
CamelCase names would silently override the `HypothesisTests` constructors for
those types, breaking any code that calls the original API directly (type
piracy).  Snake\_case names (`breusch_pagan_test`, etc.) are a different
namespace that coexists cleanly with `HypothesisTests`, and the distinction
also makes the source of each name unambiguous to readers of the code.

### Notes on $\gamma_0$ (intercept)

The FWLS estimator of $\gamma_0$ has residual variance
$\mathrm{Var}[\log\chi^2(1)] = \pi^2/2 \approx 4.93$, which is large.  Monte Carlo
at $N = 10{,}000$ (200 replications) shows:

| Estimator | RMSE($\hat{\gamma}_0$) |
|---|---|
| `two_step_harvey` (FWLS) | 0.021 |
| `exponential_mle` (MLE) | 0.013 |

The MLE is ~40% more efficient for $\gamma_0$.  Stata's `hetregress` in MLE mode
matches `exponential_mle`; in two-step mode it likely omits the bias correction
(not documented), making its $\gamma_0$ biased in addition to being less efficient.

---

## 12. Standard errors: common conventions

**FWLS estimators** (`two_step_harvey`, `two_step_glejser`, Prais-Winsten,
Sequential HARE) report **conditional** standard errors — conditional on the
estimated nuisance parameters $(\rho, \boldsymbol{\gamma})$.  This matches the standard
econometric textbook treatment (Greene, 2018; Judge et al., 1985) and produces
valid inference asymptotically.  Unconditional (sandwich-corrected) standard
errors that account for two-step estimation uncertainty are not currently
implemented.

**MLE estimators** (`exponential_mle`, `quadratic_mle`, `linear_mle`, Joint
HARE, Beach-MacKinnon) report standard errors from the **inverse observed
information matrix**, computed by ForwardDiff at the optimum.  For the three
heteroskedastic MLE estimators the full joint Hessian over $(\boldsymbol{\beta},
\boldsymbol{\gamma})$ is used, so the standard errors already account for the
joint estimation of mean and variance parameters — no conditioning is needed.

The inference machinery supports both a $t$-distribution (default) and a normal
distribution for p-values and confidence intervals.  The $t$ approximation is
recommended for small samples.

---

## References (consolidated)

Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for
regression with autocorrelated errors. *Econometrica*, 46(1), 51–58.

Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and
random coefficient variation. *Econometrica*, 47(5), 1287–1294.

Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression
to relationships containing auto-correlated error terms. *Journal of the
American Statistical Association*, 44(245), 32–61.

Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American
Statistical Association*, 64(325), 316–323.

Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Hildreth, C., & Lu, J. Y. (1960). Demand relations with autocorrelated
disturbances. *Michigan State University Agricultural Experiment Station
Technical Bulletin*, No. 276.

Judge, G. G., Griffiths, W. E., Hill, R. C., Lütkepohl, H., & Lee, T. C.
(1985). *The Theory and Practice of Econometrics* (2nd ed.). Wiley.

Oberhofer, W., & Kmenta, J. (1974). A general procedure for obtaining maximum
likelihood estimates in generalized regression models. *Econometrica*, 42(3),
579–590.

Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial correlation.
*Cowles Commission Discussion Paper*, No. 383.
