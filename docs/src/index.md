# HARE.jl

**Heteroskedasticity and Autocorrelation Estimators** -- feasible GLS
corrections for linear regression models.

HARE.jl provides classical estimators and tests covering heteroskedasticity,
AR(1) autocorrelation, and their combination:

| Error structure | Estimators / Tests |
|:----------------|:-----------|
| Exponential variance (Harvey) | [`two_step_harvey`](@ref), [`exponential_mle`](@ref), [`harvey_test`](@ref) |
| Quadratic variance (Glejser) | [`two_step_glejser`](@ref), [`quadratic_mle`](@ref), [`glejser_test`](@ref) |
| Linear variance | [`linear_mle`](@ref) |
| Groupwise heteroscedasticity | [`two_step_groupwise`](@ref), [`iterated_groupwise`](@ref) |
| AR(1) autocorrelation | [`two_step_prais_winsten`](@ref), [`iterated_prais_winsten`](@ref), [`two_step_cochrane_orcutt`](@ref), [`iterated_cochrane_orcutt`](@ref), [`hildreth_lu`](@ref) |
| AR(1) + heteroskedasticity | [`two_step_sequential`](@ref), [`iterated_sequential`](@ref), [`two_step_joint`](@ref), [`iterated_joint`](@ref) |
| AR(1) exact MLE | [`beach_mackinnon`](@ref) |

All estimators return typed result structs that implement the
[StatsBase](https://juliastats.org/StatsBase.jl/stable/) interface:
`coef`, `stderror`, `vcov`, `residuals`, `fitted`, `predict`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Trumpingtons/HARE.jl")
```

## Quick Start

Pass regressors **without** a constant column -- the intercept is added automatically:

```julia
using HARE, Random

Random.seed!(1234)
n   = 500
x1  = randn(n); x2 = randn(n)
X = hcat(x1, x2)
y   = hcat(ones(n), X) * [1.0, 2.0, -1.0] .+ exp.(0.5 .* x1) .* randn(n)

m = iterated_harvey(X, y)
coef(m)      # coefficient vector
stderror(m)  # standard errors
vcov(m)      # full covariance matrix
residuals(m) # in-sample residuals
```

The `@formula` interface mirrors GLM.jl:

```julia
using HARE
data = (y = y, x1 = x1, x2 = x2)
m = iterated_harvey(@formula(y ~ x1 + x2), data)
```

## Contents

```@contents
Pages = ["tutorial.md", "api.md"]
Depth = 2
```
