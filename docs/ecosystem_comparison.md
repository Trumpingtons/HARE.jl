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
The link-function framing is more likely to surface implementations that do not
cite the original papers; no additional packages were found.

---

## 1. Heteroscedasticity

### 1.1 Exponential variance link — Harvey (1976), `LINK=EXP`
*Greene §9.7.1 "Multiplicative Heteroscedasticity"*

Model: $\sigma_i^2 = \exp(\gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma})$

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | ✓ `two_step_harvey` | — | — | ✓ `hetregress, twostep` | — |
| Iterated FWLS | ✓ `iterated_harvey` | — | — | — | — |
| MLE | ✓ `exponential_mle` | — | — | ✓ `hetregress` | ✓ `HETERO LINK=EXP` |
| Harvey test | ✓ `harvey_test` | ✓ `skedastic` | — | — | ✓ `HETERO TEST=LM` |

**R:** `skedastic` (CRAN, active) implements Harvey (1976) as a *test* only — it
computes the test statistic and p-value but produces no corrected beta or
consistent gamma estimates.  No R package implementing the exponential variance
link as a correction estimator was found under either the Harvey name or
link-function terminology.

**Python:** No implementation found under either the Harvey name or
exponential-link terminology.

**Stata:** `hetregress` implements the exponential variance link as both a
two-step GLS estimator and as MLE.  The two-step variant is directly comparable
to HARE's `two_step_harvey`; an iterated version is not documented.

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

### 1.2 Quadratic variance link — Glejser (1969), `LINK=SQUARE`
*Greene §9.5.3 "Estimation When Ω Contains Unknown Parameters"*

Model: $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ (equivalently, $\sigma_i^2$ is quadratic in $\mathbf{z}_i$)

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | ✓ `two_step_glejser` | — | — | — | — |
| Iterated FWLS | ✓ `iterated_glejser` | — | — | — | — |
| MLE | ✓ `quadratic_mle` | — | — | — | ✓ `HETERO LINK=SQUARE` |
| Glejser test | ✓ `glejser_test` | ✓ `skedastic` | — | — | — |

No open-source platform implements Glejser's auxiliary regression as a
correction estimator.  `skedastic` (R) provides the test statistic only under
the Glejser name; no implementation was found under quadratic-link or
linear-SD terminology either.

**SAS `HETERO LINK=SQUARE`** specifies $\sigma_i^2 = \sigma^2(1 + \mathbf{z}_i^\top\boldsymbol{\eta})^2$,
i.e. $\sigma_i = \sigma(1 + \mathbf{z}_i^\top\boldsymbol{\eta})$.  Absorbing the
scale into the coefficients — $\gamma_0 = \sigma$, $\boldsymbol{\gamma} = \sigma\boldsymbol{\eta}$ —
gives $\sigma_i = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$, which is
exactly the Glejser functional form (since HARE's $\mathbf{z}_i$ always includes
a constant column).  The two models are therefore **equivalent in functional
form**; they differ only in estimation: SAS uses **joint MLE** over
$(\boldsymbol{\beta}, \sigma, \boldsymbol{\eta})$; HARE offers both FWLS
(`two_step_glejser`) and joint MLE (`quadratic_mle`).

**HARE FWLS implementation note.** Because `E[|ε|] = sqrt(2/π)` for ε ~ N(0,1),
OLS of `|u_hat|` on Z converges to `gamma × sqrt(2/π)`.  HARE corrects by
multiplying the raw OLS coefficients by `sqrt(π/2)` and the vcov by `π/2`.
Without this correction gamma is biased toward zero by the factor
`sqrt(2/π) ≈ 0.798` for all sample sizes.

---

### 1.3 Linear variance link — Breusch-Pagan (1979), `LINK=LINEAR`
*No dedicated textbook estimator section; the test appears as the Breusch-Pagan test in Greene §9.5*

Model: $\sigma_i^2 = \gamma_0 + \mathbf{z}_i^\top\boldsymbol{\gamma}$ (variance linear in $\mathbf{z}_i$)

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| MLE | ✓ `linear_mle` | — | — | — | ✓ `HETERO LINK=LINEAR` |
| Breusch-Pagan test | ✓ `HypothesisTests.jl` | ✓ `lmtest::bptest` | ✓ `statsmodels` | ✓ | ✓ |

This is a distinct parameterisation from both the exponential and quadratic
links — the *variance* (not the log-variance, not the standard deviation) is
linear in $\mathbf{z}_i$.  This is exactly the variance model underlying the
**Breusch-Pagan (1979) test** for heteroskedasticity: the LM test for
$H_0\colon \boldsymbol{\gamma} = \mathbf{0}$ in this model.
[HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/)
implements it as `BreuschPaganTest(X, e)`, documented as equivalent to
`WhiteTest(X, e, type = :linear)`.

No open-source *estimation* implementation (i.e. corrected beta) was found in R
or Python.  HARE's `linear_mle` jointly maximises the normal log-likelihood over
$(\boldsymbol{\beta}, \boldsymbol{\gamma})$ via L-BFGS, matching SAS's approach.

---

### 1.4 Second-order Taylor expansion — White (1980)
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
| White test | ✓ `HypothesisTests.jl` | ✓ `lmtest::bptest` | ✓ `statsmodels` | ✓ | ✓ |

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

### 1.5 Groupwise heteroscedasticity — `two_step_groupwise`, `iterated_groupwise`
*Greene §9.7.2 "Groupwise Heteroscedasticity"*

Model: $\text{Var}(\varepsilon_i) = \sigma_g^2$ for all $i$ in group $g$ (constant variance within groups, unrestricted across groups).

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step FWLS | ✓ `two_step_groupwise` | — | — | — | ✓ |
| Iterated (= MLE at convergence) | ✓ `iterated_groupwise` | — | — | — | ✓ |

No open-source implementation found in R or Python.  SAS `PROC AUTOREG` supports
groupwise heteroscedasticity via the `HETERO` statement with group dummy variables.

**Estimation.** The MLE first-order condition for $\sigma_g^2$ is exactly the
within-group sample variance of residuals:
$\hat{\sigma}_g^2 = \frac{1}{n_g}\sum_{i \in g}\hat{u}_i^2$.  The iterated FWLS
therefore converges to the joint MLE of $(\boldsymbol{\beta}, \sigma_1^2, \ldots, \sigma_G^2)$
at convergence — no separate MLE function is needed.

---

## 2. AR(1) Autocorrelation

### 2.1 Cochrane-Orcutt — AR(1) GLS dropping the first observation
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | ✓ `two_step_cochrane_orcutt` | ✓ `prais` | ✓ `GLSAR` | ✓ `prais, corc` | ✓ (fallback) |
| Iterated | ✓ `iterated_cochrane_orcutt` | ✓ `prais` | ✓ `GLSAR` | ✓ `prais, corc` | ✓ (fallback) |

Cochrane-Orcutt quasi-differences the data starting from observation 2, discarding
the first observation and losing one degree of freedom compared with Prais-Winsten.
**Prais-Winsten strictly dominates Cochrane-Orcutt**: it retains the first
observation via the scaling `sqrt(1 − ρ²)` at zero additional cost, yielding
coefficient estimates with smaller variance.  HARE provides `two_step_cochrane_orcutt`
and `iterated_cochrane_orcutt` for compatibility with software that uses the CO
convention.  `statsmodels.GLSAR` (Python) always uses the Cochrane-Orcutt
convention (first observation dropped).

---

### 2.2 Prais-Winsten — AR(1) GLS
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | ✓ `two_step_prais_winsten` | ✓ `prais` | — | ✓ `prais` | ✓ `METHOD=YW` |
| Iterated | ✓ `iterated_prais_winsten` | ✓ `prais` | — | ✓ `prais` | ✓ `METHOD=ITYW` |

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
| Grid search | ✓ `hildreth_lu` | — | — | — | ✓ `METHOD=HL` |

Only HARE and SAS `PROC AUTOREG` implement Hildreth-Lu.  SAS documentation
describes it as "a more primitive version of ULS that omits the first
transformed residual" — the same Cochrane-Orcutt convention used in HARE.
HARE uses a uniform grid of 200 points on (−0.99, 0.99).

---

### 2.4 Beach-MacKinnon (1978) — exact MLE for AR(1)
*Greene §20.9.1 "AR(1) Disturbances"*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Exact MLE | ✓ `beach_mackinnon` | — | — | — | ✓ `METHOD=ML` |

SAS `PROC AUTOREG` `METHOD=ML` maximises the exact likelihood including the
Jacobian term for the first observation; Beach & MacKinnon (1978) is cited in
the PROC AUTOREG references, confirming the correspondence.  No open-source
equivalent exists in R or Python.

**HARE** implements the concentrated likelihood (beta profiled out) and
maximises over rho alone using Brent's method, which is equivalent to
Beach-MacKinnon's approach.

---

## 3. Heteroscedasticity and AR(1) Autocorrelation

### 3.1 Sequential HARE — AR(1) + exponential heteroskedasticity (sequential)
*Greene §9.7.1 + §20.9.1*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step | ✓ `two_step_sequential` | — | — | — | — |
| Iterated | ✓ `iterated_sequential` | — | — | — | — |

**Not available on any other platform.**  SAS `PROC AUTOREG` explicitly
prohibits combining the `HETERO` statement (heteroscedasticity) with the
`NLAG=` option (AR(1)) unless GARCH is also specified.  Stata's `prais` and
`hetregress` are separate commands with no combined mode.

---

### 3.2 Joint HARE — AR(1) + exponential heteroskedasticity (joint MLE)
*Greene §9.7.1 + §20.9.1*

| | HARE.jl | R | Python | Stata | SAS |
|---|---|---|---|---|---|
| Two-step (concentrated MLE) | ✓ `two_step_joint` | — | — | — | — |
| Iterated (coordinate descent) | ✓ `iterated_joint` | — | — | — | — |

**Not available on any other platform**, open-source or commercial.

---

## 4. Summary gap tables

### 4.1 What HARE has that no other open-source platform has

| Method | Greene reference | Stata | SAS |
|---|---|---|---|
| Exponential variance link FWLS two-step and iterated | §9.7.1 | two-step only | — |
| Exponential variance link MLE | §9.7.1 | ✓ `hetregress` | ✓ `LINK=EXP` |
| Exponential variance link LM test (`harvey_test`) | §9.7.1 | — | ✓ `TEST=LM` |
| Quadratic variance link FWLS two-step and iterated | §9.5.3 | — | — |
| Quadratic variance link MLE | §9.5.3 | — | ✓ `LINK=SQUARE` |
| Quadratic variance link LM test (`glejser_test`) | §9.5.3 | — | — |
| Linear variance link MLE | — | — | ✓ `LINK=LINEAR` |
| Prais-Winsten (retains first observation) | §20.9.1 | ✓ | ✓ |
| Cochrane-Orcutt (drops first observation) | §20.9.1 | ✓ | ✓ (fallback) |
| Hildreth-Lu grid search | §20.9.1 | — | ✓ |
| Beach-MacKinnon exact MLE | §20.9.1 | — | ✓ |
| Groupwise heteroscedasticity FWLS and iterated MLE | §9.7.2 | — | ✓ |
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
| Glejser sqrt(π/2) correction | ✓ | N/A | N/A | N/A | N/A |
| Exact AR(1) likelihood | ✓ `beach_mackinnon` | — | — | — | ✓ `METHOD=ML` |
| Combined AR(1) + heteroscedasticity | ✓ | — | — | — | — |
| Panel support | — | ✓ | — | ✓ | ✓ |
| Iterated Harvey FWLS | ✓ | N/A | N/A | — | N/A |
| Open-source MIT licence | ✓ | ✓ | ✓ | — | — |

---

## 6. Key finding

The combined AR(1) + Harvey heteroskedasticity estimators — both sequential and
joint MLE — are **not available on any other platform, open-source or
commercial**.  For the individual methods, the closest commercial coverage is
SAS `PROC AUTOREG`, which implements Hildreth-Lu, Beach-MacKinnon ML, and
Harvey MLE (but not Harvey FWLS or Glejser), yet still cannot combine AR(1)
correction with heteroscedasticity correction outside of GARCH models.

HARE.jl therefore brings to the open-source Julia ecosystem a complete and
correct implementation of the classical FGLS correction methods for
heteroscedasticity and AR(1) autocorrelation — methods taught in every graduate
econometrics course (Greene §9.7.1, §20.9.1) and available in SAS for decades —
with no equivalent open-source implementation in R, Python, or Julia prior to
this package.

---

## 7. Suggested next steps for HARE

In priority order based on the gaps identified above:

1. **Panel data Prais-Winsten** — to match `prais` (R) and Stata `xtgls`.

2. **AR(1) + Glejser/quadratic combined estimator** — absent from all platforms.
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
