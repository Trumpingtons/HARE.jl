# API Reference

```@docs
HARE
```

## Result Types

```@docs
HAREModel
FWLSResult
MLEResult
AR1Result
SequentialResult
JointResult
GroupwiseResult
HarveyTestResult
GlejserTestResult
WaldTestResult
LRTestResult
```

## Heteroskedasticity FWLS Estimators

```@docs
exponential_fwls
exponential_ifwls
quadratic_fwls
quadratic_ifwls
linear_fwls
linear_ifwls
power_fwls
power_ifwls
```

## Heteroskedasticity MLE Estimators

```@docs
exponential_mle
quadratic_mle
linear_mle
power_mle
```

## Groupwise Heteroscedasticity

```@docs
two_step_groupwise
iterated_groupwise
groupwise_mle
```

## AR(1) Estimators

```@docs
two_step_prais_winsten
iterated_prais_winsten
two_step_cochrane_orcutt
iterated_cochrane_orcutt
hildreth_lu
beach_mackinnon
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

## Heteroskedasticity Tests

```@docs
harvey_test
glejser_test
breusch_pagan_test
white_test
durbin_watson_test
breusch_godfrey_test
```

## Model Comparison Tests

```@docs
wald_test
StatsModels.lrtest
```
