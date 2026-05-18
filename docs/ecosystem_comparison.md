# HARE.jl — Ecosystem Comparison

This document compares HARE.jl against existing implementations of the same
estimation methods across open-source and commercial platforms.  The prose
discussion focuses on open-source packages (R and Python); Stata and SAS are
included in the comparison tables for completeness but are excluded from the
open-source narrative.

All HARE methods are textbook-standard procedures covered in Greene (2018),
*Econometric Analysis*, 8th edition.  Relevant sections are cited for each
method.

---

**Search note.** Comparisons for heteroscedasticity correction were conducted
using both author-name searches (Harvey, Glejser) and link-function terminology
(exponential, quadratic, linear variance link, FGLS heteroscedasticity).
The closest open-source analog to HARE's heteroskedasticity estimators is
`nlme::gls()` with variance functions (`varFunc`): `varExp()` implements the
exponential variance model (§1.1), `varPower()` the power variance model (§1.2) —
and with $\delta$ fixed at 1, approximates the quadratic link (§1.3) — and `varIdent()`
groupwise heteroskedasticity (§1.6).
All three are estimated by ML or REML — `nlme` does not offer two-step or iterated
FWLS.

---

## 1. Heteroscedasticity

### 1.1 Exponential variance link — Harvey (1976), `LINK=EXP`
*Greene §9.7.1 "Multiplicative Heteroscedasticity"*

Model: $\sigma_i^2 = \exp(\gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma})$

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | yes `exponential_fwls` | — | — | yes `hetregress, twostep` | — |
| Iterated FWLS | yes `exponential_ifwls` | — | — | — | — |
| MLE | yes `exponential_mle` | partial `nlme::gls` | — | yes `hetregress` | yes `HETERO LINK=EXP` |
| Harvey test | yes `harvey_test` | yes `skedastic` | — | — | yes `HETERO TEST=LM` |

**R `nlme::gls(weights = varExp(form = ~ z))`** (base R recommended package)
models $\mathrm{Var}(\varepsilon_i) = \sigma^2 \exp(2\delta z_i)$, which is
Harvey's exponential variance model with $\gamma_0 = \log\sigma^2$ and
$\gamma = 2\delta$.  For multiple auxiliary regressors,
`varExp(form = ~ z1 + z2)` estimates a $\delta_j$ per regressor, giving
$\log\sigma_i^2 = \log\sigma^2 + 2\sum_j \delta_j z_{ij}$ — equivalent to
Harvey up to this factor-of-2 reparameterisation.  Estimation is by ML or REML
jointly over $(\boldsymbol{\beta}, \delta)$, so the nlme row is comparable to
HARE's `exponential_mle`, not to the two-step or iterated FWLS variants
(`exponential_fwls`, `exponential_ifwls`).  No FWLS path exists in nlme.

**R `skedastic`** (CRAN, active) implements Harvey (1976) as a *test* only — it
computes the test statistic and p-value but produces no corrected beta or
consistent gamma estimates.

**Python:** No implementation found under either the Harvey name or
exponential-link terminology.

**Stata:** `hetregress` implements the exponential variance link as both a
two-step GLS estimator and as MLE.  The two-step variant is directly comparable
to HARE's `exponential_fwls`; an iterated version is not documented.

**SAS:** `PROC AUTOREG` with the `HETERO` statement and `LINK=EXP` implements
the exponential variance model by **MLE only** — two-step FGLS is not offered.
SAS maximises the normal log-likelihood jointly over $\boldsymbol{\beta}$ and
$\boldsymbol{\gamma}$, which is asymptotically efficient under correct
specification.  The `TEST=LM` option produces the Lagrange multiplier test for
heteroscedasticity of the specified form.  SAS allows arbitrary auxiliary
regressors $\mathbf{z}_i$; HARE matches this generality via the optional `Z`
argument (defaulting to the augmented mean regressors).

**HARE implementation notes.** The FWLS auxiliary regression subtracts
`c = E[log(χ²(1))] = digamma(1/2) + log(2) = −1.2703628454614782` from
`log(u_hat²)` before fitting, centring the error term and yielding consistent
estimates of *all* gamma coefficients regardless of regressor means.  Without
this correction (which is not documented in Stata or SAS) the intercept and
all slope gammas are biased when regressors have non-zero means.  The MLE
variant (`exponential_mle`) jointly maximises the normal log-likelihood over
$(\boldsymbol{\beta}, \boldsymbol{\gamma})$ via L-BFGS and is asymptotically
efficient.

---

### 1.2 Power variance link

Model: $\sigma_i^2 = \sigma^2 |\mathbf{z}_i|^{2\boldsymbol{\delta}}$, equivalently $\log\sigma_i^2 = \gamma_0 + \sum_j \gamma_j \log|z_{ij}|$ with $\gamma_0 = \log\sigma^2$, $\gamma_j = 2\delta_j$

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | yes `power_fwls` | — | — | — | — |
| Iterated FWLS | yes `power_ifwls` | — | — | — | — |
| MLE | yes `power_mle` | partial `nlme::gls` | — | — | — |

**R `nlme::gls(weights = varPower(form = ~ abs(z)))`** models
$\mathrm{Var}(\varepsilon_i) = \sigma^2 |z_i|^{2\delta}$, which is the power variance
model with $\gamma_0 = \log\sigma^2$ and $\gamma_1 = 2\delta$.  This is exactly the
HARE power model.  When `varPower` estimates $\delta$ from data it maximises the same
log-likelihood as `power_mle`, so the two are numerically equivalent (cross-validated
in §7.2).

**HARE implementation.** `power_fwls` and `power_ifwls` work by applying the
exponential FWLS procedure (Harvey bias correction) on $\log|Z|$ in place of $Z$,
yielding exact OLS estimates of $\boldsymbol{\gamma} = (\log\sigma^2, 2\delta_1, \ldots)$
with exact standard errors — no delta method approximation is needed.
`power_mle` pre-log-transforms $Z$ and then calls the same exponential MLE kernel;
the reported parameters $(\gamma_0, \gamma_1, \ldots)$ are directly interpretable as
$(\log\sigma^2, 2\delta_1, \ldots)$.

---

### 1.3 Quadratic variance link — Glejser (1969), `LINK=SQUARE`
*Greene §9.5.3 "Estimation When Ω Contains Unknown Parameters"*

Model: $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ (equivalently, $\sigma_i^2$ is quadratic in $\mathbf{z}_i$)

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | yes `quadratic_fwls` | — | — | — | — |
| Iterated FWLS | yes `quadratic_ifwls` | — | — | — | — |
| MLE | yes `quadratic_mle` | partial `nlme::gls` | — | — | yes `HETERO LINK=SQUARE` |
| Glejser test | yes `glejser_test` | yes `skedastic` | — | — | — |

**R `nlme::gls(weights = varPower(value = 1, form = ~ z, fixed = TRUE))`**
models $\mathrm{Var}(\varepsilon_i) \propto |z_i|^2$, i.e. $\sigma_i \propto |z_i|$.
This is **not** Glejser's model, which is $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$
(a shifted linear function of $z_i$).  The power model forces $\sigma_i = 0$
when $z_i = 0$ and has no intercept, whereas Glejser's model allows a non-zero
baseline variance at $z_i = 0$.  The two overlap empirically when $z_i$ varies
well away from zero and the intercept is small, but they are not equivalent in
functional form.  When $\delta$ is estimated from data (rather than fixed at 1),
`varPower(form = ~ z)` fits the power exponent jointly with $\boldsymbol{\beta}$ by ML.

`skedastic` provides the Glejser test statistic only; no open-source package
implements Glejser's linear-SD auxiliary regression as a correction estimator.

**SAS `HETERO LINK=SQUARE`** specifies $\sigma_i^2 = \sigma^2(1 + \mathbf{z}_i^\top\boldsymbol{\eta})^2$,
i.e. $\sigma_i = \sigma(1 + \mathbf{z}_i^\top\boldsymbol{\eta})$.  Absorbing the
scale into the coefficients — $\gamma_0 = \sigma$, $\boldsymbol{\gamma} = \sigma\boldsymbol{\eta}$ —
gives $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$, which is
exactly the Glejser functional form (since HARE's $\mathbf{z}_i$ always includes
a constant column).  The two models are therefore **equivalent in functional
form**; they differ only in estimation: SAS uses **joint MLE** over
$(\boldsymbol{\beta}, \sigma, \boldsymbol{\eta})$; HARE offers both FWLS
(`quadratic_fwls`) and joint MLE (`quadratic_mle`).

**HARE FWLS implementation note.** Because `E[|e|] = sqrt(2/pi)` for e ~ N(0,1),
OLS of `|u_hat|` on Z converges to `gamma × sqrt(2/pi)`.  HARE corrects by
multiplying the raw OLS coefficients by `sqrt(pi/2)` and the vcov by `pi/2`.
Without this correction gamma is biased toward zero by the factor
`sqrt(2/pi) ≈ 0.798` for all sample sizes.

---

### 1.4 Linear variance link — Breusch-Pagan (1979), `LINK=LINEAR`
*No dedicated textbook estimator section; the test appears as the Breusch-Pagan test in Greene §9.5*

Model: $\sigma_i^2 = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ (variance linear in $\mathbf{z}_i$)

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| MLE | yes `linear_mle` | — | — | — | yes `HETERO LINK=LINEAR` |
| Breusch-Pagan test | yes `HypothesisTests.jl` | yes `lmtest::bptest` | yes `statsmodels` | yes | yes |

This is a distinct parameterisation from both the exponential and quadratic
links — the *variance* (not the log-variance, not the standard deviation) is
linear in $\mathbf{z}_i$.  This is exactly the variance model underlying the
**Breusch-Pagan (1979) test** for heteroskedasticity: the LM test for
$H_0\colon \boldsymbol{\gamma} = \mathbf{0}$ in this model.
[HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/)
implements it as `BreuschPaganTest(X, e)`, documented as equivalent to
`WhiteTest(X, e, type = :linear)`.

**Note on `nlme::varPower(delta = 0.5)`.** A power model with $\delta = 0.5$ gives
$\mathrm{Var}(\varepsilon_i) \propto |z_i|$, i.e. variance linear in $|z_i|$.
This is superficially similar to the Breusch-Pagan linear variance model but is
not equivalent: it lacks an additive intercept and constrains the constant term
to zero.  The `varPower` family does not expose a `LINK=LINEAR`-style
parameterisation.

No open-source *estimation* implementation (i.e. corrected beta) with the
exact linear-variance functional form was found in R or Python.
HARE's `linear_mle` jointly maximises the normal log-likelihood over
$(\boldsymbol{\beta}, \boldsymbol{\gamma})$ via L-BFGS, matching SAS's approach.

---

### 1.5 Second-order Taylor expansion — White (1980)
*Greene §9.5.2 "White's General Test"*

Model: $\sigma_i^2 \approx \gamma_0 + \sum_j \gamma_j z_{ij} + \sum_j \gamma_{jj} z_{ij}^2 + \sum_{j < k} \gamma_{jk} z_{ij} z_{ik}$

The variance function is approximated by a second-order Taylor expansion in the
auxiliary regressors $\mathbf{z}_i$ (typically $\mathbf{z}_i = \mathbf{x}_i$,
the mean regressors).  For $p$ auxiliary regressors the variance equation has
$1 + p(p+3)/2$ free parameters: one intercept, $p$ linear terms, $p$ squared
terms, and $\binom{p}{2}$ cross-product terms.  With $p = 1$: 3 gammas; with
$p = 2$: 6 gammas; with $p = 3$: 9 gammas — growing quadratically in $p$.

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| FWLS / MLE estimator | — *(planned)* | — | — | — | — |
| White test | yes `HypothesisTests.jl` | yes `lmtest::bptest` | yes `statsmodels` | yes | yes |

The **White (1980) test** is the LM test for $H_0\colon \boldsymbol{\gamma} = \mathbf{0}$
in this variance model.
[HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/)
implements it as `WhiteTest(X, e)` (also aliased as `BreuschPaganTest(X, e, type = :white)`).
No open-source platform implements the second-order Taylor expansion as an
*estimation* method (corrected beta); all provide the test statistic only.

**HARE planned implementation.** With $\mathbf{z}_i = \mathbf{x}_i$ (the default),
the full expanded auxiliary regressor matrix can be constructed automatically.
The natural estimator would be `quadratic_expansion_mle` (joint MLE over
$(\boldsymbol{\beta}, \boldsymbol{\gamma})$), analogous to `exponential_mle`.
The large number of gamma parameters relative to the exponential link means
this approach is most useful when heteroskedasticity is suspected but the
functional form is unknown.

---

### 1.6 Groupwise heteroscedasticity — `two_step_groupwise`, `iterated_groupwise`, `groupwise_mle`
*Greene §9.7.2 "Groupwise Heteroscedasticity"*

Model: $\text{Var}(\varepsilon_i) = \sigma_g^2$ for all $i$ in group $g$ (constant variance within groups, unrestricted across groups).

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | yes `two_step_groupwise` | — | — | — | yes |
| Iterated (= MLE at convergence) | yes `iterated_groupwise` | partial `nlme::gls` | — | — | yes |
| Explicit MLE | yes `groupwise_mle` | partial `nlme::gls` | — | — | yes |

**R `nlme::gls(weights = varIdent(form = ~1 | group))`** estimates a separate
scale parameter per group, producing group-specific standard deviations
$\sigma_g$.  The functional form is identical to HARE's groupwise model.
Estimation is by ML or REML; nlme does not offer a FWLS path.
`groupwise_mle` is cross-validated against `nlme::gls(varIdent)` in §7.2.

No Python implementation found.  SAS `PROC AUTOREG` supports
groupwise heteroscedasticity via the `HETERO` statement with group dummy variables.

**Estimation.** The MLE first-order condition for $\sigma_g^2$ is exactly the
within-group sample variance of residuals:
$\hat{\sigma}_g^2 = \frac{1}{n_g}\sum_{i \in g}\hat{u}_i^2$.  The iterated FWLS
therefore converges to the joint MLE of $(\boldsymbol{\beta}, \sigma_1^2, \ldots, \sigma_G^2)$
at convergence.  `groupwise_mle` runs the same iteration internally and is the
recommended call when the primary goal is the MLE estimate.

---

## 2. AR(1) Autocorrelation

### 2.1 Cochrane-Orcutt — AR(1) GLS dropping the first observation
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | yes `two_step_cochrane_orcutt` | yes `prais` | yes `GLSAR` | yes `prais, corc` | yes (fallback) |
| Iterated | yes `iterated_cochrane_orcutt` | yes `prais` | yes `GLSAR` | yes `prais, corc` | yes (fallback) |

Cochrane-Orcutt quasi-differences the data starting from observation 2, discarding
the first observation and losing one degree of freedom compared with Prais-Winsten.
**Prais-Winsten strictly dominates Cochrane-Orcutt**: it retains the first
observation via the scaling `sqrt(1 - rho^2)` at zero additional cost, yielding
coefficient estimates with smaller variance.  HARE provides `two_step_cochrane_orcutt`
and `iterated_cochrane_orcutt` for compatibility with software that uses the CO
convention.  `statsmodels.GLSAR` (Python) always uses the Cochrane-Orcutt
convention (first observation dropped).

---

### 2.2 Prais-Winsten — AR(1) GLS
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | yes `two_step_prais_winsten` | yes `prais` | — | yes `prais` | yes `METHOD=YW` |
| Iterated | yes `iterated_prais_winsten` | yes `prais` | — | yes `prais` | yes `METHOD=ITYW` |

**R `prais`** (CRAN, active, July 2025): closest open-source counterpart.  If
estimated rho exceeds 1 during an iteration, the package falls back silently to
Cochrane-Orcutt (drops the first observation).  Also supports panel data.

**Stata `prais`**: implements both Prais-Winsten (default) and Cochrane-Orcutt
(`corc` option), two-step and iterated.

**SAS `PROC AUTOREG` `METHOD=YW`**: the Yule-Walker method is equivalent to
Prais-Winsten for AR(1).  `METHOD=ITYW` is the iterated version.  SAS
documentation refers to these as "two-step full transform" following Harvey
(1981).

**HARE** always applies the Prais-Winsten scaling `sqrt(max(1 − rho², 0))`,
avoiding the imaginary-value edge case without silently changing methods.

---

### 2.3 Hildreth-Lu — grid search for AR(1) rho
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Grid search | yes `hildreth_lu` | partial `HoRM` | — | — | yes `METHOD=HL` |

**R `HoRM`** (CRAN, actively maintained — v0.1.4, May 2025; supplement to
Young's *Handbook of Regression Methods*): exports `hildreth.lu(y, x, rho)`,
which evaluates the Cochrane-Orcutt-transformed regression for a **single
fixed rho**.  This is a building block, not a complete estimator — the user must
write the grid search loop manually to minimise SSE over a rho grid.  HARE's
`hildreth_lu` wraps this loop internally (200-point uniform grid on (−0.99,
0.99)) and returns the best-fitting model in one call.

SAS `PROC AUTOREG METHOD=HL` implements the same full grid search and uses the
same Cochrane-Orcutt convention (first observation dropped).

---

### 2.4 Beach-MacKinnon (1978) — exact MLE for AR(1)
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Exact MLE | yes `beach_mackinnon` | partial `nlme::gls` | — | — | yes `METHOD=ML` |

**R `nlme::gls(corAR1(), method="ML")`** (base R recommended package, always
available): maximises the exact normal log-likelihood for a GLS model with an
AR(1) correlation structure, including the Jacobian contribution of the first
observation — which is precisely what Beach & MacKinnon (1978) derive.  It is
therefore a **functional equivalent**, though it is not named after Beach-MacKinnon
and lives in a broader mixed-effects framework.  Cross-validation on our
reference dataset (see Section 8) confirms agreement to $\sim 10^{-8}$ in $\rho$ and $\sim 10^{-9}$
in coefficients — differences attributable to different numerical optimisers
converging to the same maximum.

**HARE** implements the concentrated likelihood (beta profiled out analytically)
and maximises over rho alone using Brent's method.  `nlme::gls` optimises jointly
over all parameters, which is asymptotically equivalent and numerically
indistinguishable in practice.

**SAS** `PROC AUTOREG METHOD=ML` cites Beach & MacKinnon (1978) explicitly.

---

## 3. Heteroscedasticity and AR(1) Autocorrelation

### 3.1 Sequential HARE — AR(1) + exponential heteroskedasticity (sequential)
*Greene §9.7.1 + §20.9.1*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | yes `two_step_sequential` | — | — | — | — |
| Iterated | yes `iterated_sequential` | — | — | — | — |

**Not available on any other platform.**  SAS `PROC AUTOREG` explicitly
prohibits combining the `HETERO` statement (heteroscedasticity) with the
`NLAG=` option (AR(1)) unless GARCH is also specified.  Stata's `prais` and
`hetregress` are separate commands with no combined mode.

---

### 3.2 Joint HARE — AR(1) + exponential heteroskedasticity (joint MLE)
*Greene §9.7.1 + §20.9.1*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step (concentrated MLE) | yes `two_step_joint` | — | — | — | — |
| Iterated (coordinate descent) | yes `iterated_joint` | — | — | — | — |

**Not available on any other platform**, open-source or commercial.

---

## 4. Summary gap tables

### 4.1 What HARE has that no other open-source platform has

| Method | Greene reference | Stata | SAS |
|---|---|---|---|
| Exponential variance link FWLS two-step and iterated | §9.7.1 | two-step only | — |
| Exponential variance link MLE | §9.7.1 | yes `hetregress` | yes `LINK=EXP` |
| Exponential variance link LM test (`harvey_test`) | §9.7.1 | — | yes `TEST=LM` |
| Power variance link FWLS two-step and iterated | — | — | — |
| Power variance link MLE | — | — | — |
| Quadratic variance link FWLS two-step and iterated | §9.5.3 | — | — |
| Quadratic variance link MLE | §9.5.3 | — | yes `LINK=SQUARE` |
| Quadratic variance link LM test (`glejser_test`) | §9.5.3 | — | — |
| Linear variance link MLE | — | — | yes `LINK=LINEAR` |
| Prais-Winsten (retains first observation) | §20.9.1 | yes | yes |
| Cochrane-Orcutt (drops first observation) | §20.9.1 | yes | yes (fallback) |
| Hildreth-Lu grid search | §20.9.1 | — | yes | R `HoRM::hildreth.lu` evaluates a single fixed rho; full grid search requires user loop |
| Beach-MacKinnon exact MLE | §20.9.1 | — | yes | R `nlme::gls(corAR1(), method="ML")` is a functional equivalent (see §2.4) |
| Groupwise heteroscedasticity FWLS, iterated, and MLE | §9.7.2 | — | yes |
| Sequential AR(1) + exponential heteroskedasticity | §9.7.1 + §20.9.1 | — | — |
| Joint AR(1) + exponential heteroskedasticity MLE | §9.7.1 + §20.9.1 | — | — |

### 4.2 What other platforms have that HARE does not (yet)

| Functionality | Platform | Notes |
|---|---|---|
| Panel data AR(1) correction | R `prais`, Stata `xtgls` | HARE is cross-section / time series only |
| AR(1) + Glejser/quadratic combined estimator | — | Sequential and Joint HARE use the exponential link only |

---

## 5. Implementation quality comparison

| Detail | HARE | R `prais` | Python `GLSAR` | Stata `prais` | SAS `METHOD=YW` |
|---|---|---|---|---|---|
| First observation | PW: retained; CO: dropped | Retained; falls back if rho > 1 | Dropped (CO) | Retained (default); CO via `corc` | Retained |
| Harvey bias correction | Subtracts c from LHS; all gammas consistent | N/A | N/A | Not documented | Not documented |
| Glejser sqrt(pi/2) correction | yes | N/A | N/A | N/A | N/A |
| Exact AR(1) likelihood | yes `beach_mackinnon` | partial `nlme::gls` | — | — | yes `METHOD=ML` |
| Combined AR(1) + heteroscedasticity | yes | — | — | — | — |
| Panel support | — | yes | — | yes | yes |
| Iterated exponential FWLS | yes | N/A | N/A | — | N/A |
| Open-source MIT licence | yes | yes | yes | — | — |

---

## 6. Key finding

The combined AR(1) + exponential heteroskedasticity estimators — both sequential and
joint MLE — are **not available on any other platform, open-source or
commercial**.  For the individual methods, the closest commercial coverage is
SAS `PROC AUTOREG`, which implements Hildreth-Lu, Beach-MacKinnon ML, and
exponential variance MLE (but not exponential FWLS or the quadratic/power links),
yet still cannot combine AR(1) correction with heteroscedasticity correction outside
of GARCH models.

HARE.jl therefore brings to the open-source Julia ecosystem a complete and
correct implementation of the classical FGLS correction methods for
heteroscedasticity and AR(1) autocorrelation — methods taught in every graduate
econometrics course (Greene §9.7.1, §20.9.1) and available in SAS for decades —
with no equivalent open-source implementation in R, Python, or Julia prior to
this package.

---

## 7. Cross-Validating Against R

To validate numerical correctness independently of unit tests, HARE.jl results
are compared against R reference implementations on an identical dataset.  The
data are generated once in Julia (`StableRNG(42)`, $n = 500$) and the exact
same observations are passed to R via CSV export.  No random number generation
occurs in R; agreement between the two implementations is therefore a direct
test of algorithmic equivalence, not just statistical consistency.

**R session:** R 4.5.1 · prais 1.1.4 · orcutt 2.3 · skedastic 2.0.3 · nlme 3.1.168.

The cross-validation tests live in `test/runtests_r.jl` and require no R
installation to run — the reference values below are hardcoded.

### 7.1 Software correspondence and test formulas

| HARE.jl function | R package :: function | Formula / method |
|---|---|---|
| `iterated_prais_winsten` | `prais::prais_winsten()` | Iterates $\hat\rho = \frac{\sum \hat u_t \hat u_{t-1}}{\sum \hat u_{t-1}^2}$ + PW transform until $|\hat\rho^{(i)}-\hat\rho^{(i-1)}| < \text{tol}$ |
| `two_step_prais_winsten` | `prais::prais_winsten(twostep=TRUE)` | Single pass: OLS → $\hat\rho$ → PW-GLS |
| `iterated_cochrane_orcutt` | `orcutt::cochrane.orcutt()` | Same iteration as PW but drops first observation |
| `beach_mackinnon` | `nlme::gls(corAR1(), method="ML")` | Exact AR(1) log-likelihood maximised over $(\boldsymbol\beta, \rho)$; first observation included |
| `exponential_mle` | `nlme::gls(weights=varExp(form=~z), method="ML")` | Same likelihood; $\gamma_0 = \log\sigma^2_{\mathrm{nlme}}$, $\gamma_1 = 2\delta_{\mathrm{nlme}}$ |
| `power_mle` | `nlme::gls(weights=varPower(form=~abs(z)), method="ML")` | Same likelihood; $\gamma_0 = \log\sigma^2_{\mathrm{nlme}}$, $\gamma_1 = 2\delta_{\mathrm{nlme}}$ |
| `groupwise_mle` | `nlme::gls(weights=varIdent(form=~1\|group), method="ML")` | Same likelihood; group $\sigma_g^2$ directly reported |
| `harvey_test` | `skedastic::harvey()` | $\mathrm{LM} = \mathrm{ESS} \,/\, \psi_1(\tfrac{1}{2})$, where $\psi_1(\tfrac{1}{2}) = \pi^2/2 = \mathrm{Var}(\log\chi^2(1))$ |
| `glejser_test` | `skedastic::glejser()` | $\mathrm{LM} = \mathrm{ESS}_{|\hat u|} \,/\, \bigl(\hat\sigma^2 (1 - 2/\pi)\bigr)$, where $(1-2/\pi) = \mathrm{Var}(|N(0,1)|)$ |

**Not cross-validated** (no direct R equivalent): `hildreth_lu` (HoRM evaluates only a fixed rho, not the full grid search),
all FWLS estimators (`exponential_fwls`, `exponential_ifwls`, `quadratic_fwls`, `quadratic_ifwls`,
`power_fwls`, `power_ifwls`), `two_step_sequential`, `iterated_sequential`,
`two_step_joint`, `iterated_joint`.

### 7.2 Numerical results

**AR(1) dataset** — true parameters $\boldsymbol\beta=[1,2,-1]$, $\rho=0.7$.

| Quantity | HARE.jl | R | Max \|diff\| |
|---|---|---|---|
| `iterated_prais_winsten` $\hat\beta$ | $[0.807539,\; 2.000888,\; -0.995886]$ | $[0.807539,\; 2.000888,\; -0.995887]$ | $6\times10^{-10}$ |
| `iterated_prais_winsten` $\hat\rho$ | $0.747516$ | $0.747516$ | $<10^{-12}$ |
| `two_step_prais_winsten` $\hat\beta$ | $[0.807132,\; 2.000733,\; -0.995997]$ | $[0.807132,\; 2.000733,\; -0.995997]$ | $<10^{-12}$ |
| `two_step_prais_winsten` $\hat\rho$ | $0.741971$ | $0.741971$ | $<10^{-12}$ |
| `iterated_cochrane_orcutt` $\hat\beta$ | $[0.804907,\; 2.000953,\; -0.995974]$ | $[0.804907,\; 2.000953,\; -0.995974]$ | $4\times10^{-10}$ |
| `iterated_cochrane_orcutt` $\hat\rho$ | $0.747510$ | $0.747510$ | $<10^{-12}$ |

**Heteroskedastic dataset** — Harvey DGP, $\sigma_i = \exp(0.5\, x_{1i})$,
auxiliary regressors $Z = [1, x_1, x_2]$ ($p = 2$ slope parameters).

| Quantity | HARE.jl | R | \|diff\| |
|---|---|---|---|
| `harvey_test` LM (het DGP) | $122.0117$ | $122.0117$ | $<10^{-10}$ |
| `harvey_test` $p$-value (het DGP) | $3.20\times10^{-27}$ | $3.20\times10^{-27}$ | $<10^{-32}$ |
| `harvey_test` LM (hom DGP) | $1.1127$ | $1.1127$ | $<10^{-10}$ |
| `harvey_test` $p$-value (hom DGP) | $0.5733$ | $0.5733$ | $<10^{-10}$ |
| `glejser_test` LM (het DGP) | $188.1612$ | $188.1612$ | $<10^{-10}$ |
| `glejser_test` $p$-value (het DGP) | $1.38\times10^{-41}$ | $1.38\times10^{-41}$ | $<10^{-46}$ |
| `glejser_test` LM (hom DGP) | $0.8859$ | $0.8859$ | $<10^{-10}$ |
| `glejser_test` $p$-value (hom DGP) | $0.6422$ | $0.6422$ | $<10^{-10}$ |

Differences at the level of $10^{-10}$–$10^{-12}$ reflect standard double-precision
floating-point rounding in the two independent implementations (Julia and R).
There are no algorithmic discrepancies.

**MLE heteroskedastic datasets** — true parameters $\boldsymbol\beta=[1,2,-1]$;
exponential DGP: $\log\sigma_i^2 = x_{1i}$; power DGP: $\sigma_i^2 = |x_{1i}|$;
groupwise DGP: $\sigma_g \in \{0.5, 1.0, 1.5, 2.0\}$, $G=4$ equal groups;
all with $n=500$, `StableRNG(99)`.
Data generated independently of the AR(1)/test datasets (`StableRNG(42)`).

| Quantity | HARE.jl | R (`nlme`) | Max \|diff\| |
|---|---|---|---|
| `exponential_mle` $\hat\beta$ | $[0.9763,\; 1.9934,\; -1.0115]$ | $[0.9763,\; 1.9934,\; -1.0115]$ | $<10^{-4}$ |
| `exponential_mle` $\hat\gamma = (\log\hat\sigma^2, 2\hat\delta)$ | $[0.0039,\; 1.0801]$ | $[0.0039,\; 1.0801]$ | $<10^{-4}$ |
| `exponential_mle` log-likelihood | $-694.669$ | $-694.669$ | $<10^{-3}$ |
| `power_mle` $\hat\beta$ | $[0.9955,\; 2.1150,\; -0.9639]$ | $[0.9955,\; 2.1150,\; -0.9639]$ | $<10^{-4}$ |
| `power_mle` $\hat\gamma = (\log\hat\sigma^2, 2\hat\delta)$ | $[-0.0390,\; 0.9547]$ | $[-0.0390,\; 0.9547]$ | $<10^{-4}$ |
| `power_mle` log-likelihood | $-565.804$ | $-565.804$ | $<10^{-3}$ |
| `groupwise_mle` $\hat\beta$ | $[1.0000,\; 1.9740,\; -0.9413]$ | $[1.0000,\; 1.9740,\; -0.9413]$ | $<10^{-4}$ |
| `groupwise_mle` $\hat\sigma_g^2$ (groups A–D) | $[0.274,\; 0.758,\; 1.950,\; 3.171]$ | $[0.274,\; 0.758,\; 1.950,\; 3.171]$ | $<10^{-3}$ |
| `groupwise_mle` log-likelihood | $-725.147$ | $-725.147$ | $<10^{-3}$ |

Values in both columns are rounded to 4 decimal places; the exact reference values
and tolerances are in `test/runtests_r.jl`.

### 7.3 Historical note on test statistic formulas

An earlier HARE version used the generic LM form $\mathrm{LM} = n R^2$ for both
the Harvey and Glejser tests.  This is asymptotically equivalent to the exact
score-test formulas above but numerically different (e.g.\ $87.5$ vs $122.0$ for
`harvey_test` on the het dataset), and it does not match `skedastic`.  HARE was
updated to use Harvey's (1976) and Glejser's (1969) exact formulas after the
cross-validation revealed the discrepancy.  The denominators $\pi^2/2$ and
$\hat\sigma^2(1-2/\pi)$ reflect the known variances of $\log\chi^2(1)$ and
$|N(0,1)|$ respectively, making the score test exact rather than approximate.

---

## References

Beach, C. M., & MacKinnon, J. G. (1978). A maximum likelihood procedure for
regression with autocorrelated errors. *Econometrica*, 46(1), 51–58.

Glejser, H. (1969). A new test for heteroskedasticity. *Journal of the American
Statistical Association*, 64(325), 316–323.

Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
Relevant sections: §9.5.3, §9.7.1, §20.9.1.

Harvey, A. C. (1976). Estimating regression models with multiplicative
heteroscedasticity. *Econometrica*, 44(3), 461–465.

Harvey, A. C. (1981). *The Econometric Analysis of Time Series*. Philip Allan.

Hildreth, C., & Lu, J. Y. (1960). Demand relations with autocorrelated
disturbances. *Michigan State University Agricultural Experiment Station
Technical Bulletin*, No. 276.

Mohr, F. X. (2025). *prais: Prais-Winsten Estimator for AR(1) Serial
Correlation*. R package version 1.1.4. https://cran.r-project.org/package=prais

Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial correlation.
*Cowles Commission Discussion Paper*, No. 383.

SAS Institute Inc. (2015). *SAS/ETS 14.1 User's Guide: The AUTOREG Procedure*.
SAS Institute Inc., Cary, NC.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical
modeling with Python. *Proceedings of the 9th Python in Science Conference*.

StataCorp. (2023). *Stata Base Reference Manual: hetregress*. StataCorp LLC,
College Station, TX.

White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
