# API Reference

```@docs
HARE
```

## Result Types

```@docs
HAREModel
HarveyResult
GlejserResult
PraisWinstenResult
CochranOrcuttResult
HildrethLuResult
SequentialResult
JointResult
BeachMacKinnonResult
HeteroMLEResult
GroupwiseResult
WaldTestResult
LRTestResult
```

## Heteroskedasticity FWLS Estimators

```@docs
two_step_harvey
iterated_harvey
two_step_glejser
iterated_glejser
```

## Heteroskedasticity MLE Estimators

```@docs
exponential_mle
quadratic_mle
linear_mle
```

## Groupwise Heteroscedasticity

```@docs
two_step_groupwise
iterated_groupwise
```

## AR(1) Estimators

```@docs
two_step_prais_winsten
iterated_prais_winsten
two_step_cochrane_orcutt
iterated_cochrane_orcutt
hildreth_lu
```

## Sequential AR(1) + Heteroskedasticity Estimators

```@docs
two_step_sequential
iterated_sequential
```

## Joint AR(1) + Heteroskedasticity Estimators

```@docs
two_step_joint
iterated_joint
```

## Exact MLE

```@docs
beach_mackinnon
```

## Heteroskedasticity Tests

```@docs
HarveyTest
GlejserTest
```

## Model Comparison Tests

```@docs
wald_test
StatsModels.lrtest
```
