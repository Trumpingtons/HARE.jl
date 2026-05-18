"""
HARE - Heteroskedasticity and Autocorrelation Estimators

Classical feasible GLS correction methods for linear regression models,
including multiplicative heteroskedasticity (Harvey 1976), AR(1) serial
correlation (Prais-Winsten, Hildreth-Lu, Cochrane-Orcutt), and joint
correction via the Sequential HARE estimator (Oberhofer-Kmenta 1974) and
the Beach-MacKinnon (1978) exact MLE.
"""
module HARE

using LinearAlgebra
using Statistics
using Random
using StatsModels
using Tables
using GLM
using Optim
using ForwardDiff
using Distributions
using HypothesisTests: BreuschPaganTest, WhiteTest, BreuschGodfreyTest, DurbinWatsonTest
import StatsBase
import StatsAPI
using StatsAPI: pvalue, dof

export HAREModel
export FWLSResult
export AR1Result, AR1Result, AR1Result, SequentialResult, JointResult, AR1Result, MLEResult, GroupwiseResult
export tstat, pvalues, sigma2
export wald_test, WaldTestResult, LRTestResult
export HarveyTestResult, GlejserTestResult
export harvey_test, glejser_test
export breusch_pagan_test, white_test, durbin_watson_test, breusch_godfrey_test
export pvalue, dof
export exponential_fwls, exponential_ifwls
export quadratic_fwls, quadratic_ifwls
export linear_fwls, linear_ifwls
export power_fwls, power_ifwls
export two_step_prais_winsten, iterated_prais_winsten
export two_step_cochrane_orcutt, iterated_cochrane_orcutt
export hildreth_lu
export two_step_sequential, iterated_sequential
export two_step_joint, iterated_joint
export beach_mackinnon
export exponential_mle, quadratic_mle, linear_mle, power_mle
export two_step_groupwise, iterated_groupwise, groupwise_mle

include("types.jl")
include("interfaces.jl")
include("show.jl")
include("auxiliary.jl")
include("tests.jl")
include("fwls.jl")
include("mle.jl")

include("prais_winsten.jl")
include("cochrane_orcutt.jl")
include("hildreth_lu.jl")
include("beach_mackinnon.jl")
include("sequential.jl")
include("joint.jl")

end # module HARE
