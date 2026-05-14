# HARE.jl

**Heteroskedasticity and Autocorrelation Estimators** — feasible GLS
corrections for linear regression models.

HARE.jl provides eight classical estimators covering three error structures:

| Error structure | Estimators |
|:----------------|:-----------|
| Multiplicative heteroskedasticity | [`two_step_harvey`](@ref), [`iterated_harvey`](@ref) |
| AR(1) autocorrelation | [`two_step_prais_winsten`](@ref), [`iterated_prais_winsten`](@ref), [`hildreth_lu`](@ref) |
| AR(1) + heteroskedasticity | [`two_step_haegls`](@ref), [`iterated_haegls`](@ref) |
| AR(1) exact MLE | [`beach_mackinnon`](@ref) |

All estimators return typed result structs that implement the
[StatsBase](https://juliastats.org/StatsBase.jl/stable/) interface:
`coef`, `stderror`, `vcov`, `residuals`, `fitted`, `predict`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/YOUR_GITHUB_USERNAME/HARE.jl")
```

## Quick Start

```julia
using HARE, Random

Random.seed!(1234)
n  = 500
X  = hcat(ones(n), randn(n), randn(n))
y  = X * [1.0, 2.0, -1.0] .+ exp.(0.5 .* X[:,2]) .* randn(n)

m = iterated_harvey(X, y)
coef(m)      # coefficient vector
stderror(m)  # standard errors
vcov(m)      # full covariance matrix
residuals(m) # in-sample residuals
```

The `@formula` interface mirrors GLM.jl:

```julia
using HARE
data = (y = y, x1 = X[:,2], x2 = X[:,3])
m = iterated_harvey(@formula(y ~ x1 + x2), data)
```

## Contents

```@contents
Pages = ["tutorial.md", "api.md"]
Depth = 2
```
