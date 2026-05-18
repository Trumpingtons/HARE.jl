# Asymptotic consistency tests for HARE.jl estimators.
#
# Strategy: data are generated with N = 1_000_000 observations (StableRNG(123))
# so that sampling noise is small enough to check parameter recovery at atol=0.01.
# These tests are intentionally kept in a separate file because they take ~60 s
# and would slow down the normal development loop unnecessarily.
#
# Run standalone:
#   julia --project test/runtests_consistency.jl

using Test
using HARE
using StableRNGs
using StatsBase

b_true = [1.0, 2.0, -1.0]

@testset "consistency (large n)" begin
    rng = StableRNG(123)
    N   = 1_000_000
    x1L = randn(rng, N);  x2L = randn(rng, N)
    XL  = hcat(x1L, x2L)
    XfL = hcat(ones(N), XL)
    epsL = randn(rng, N)

    # Harvey: log(σ²) = 0·1 + 1·x1 + 0·x2
    gamma_h = [0.0, 1.0, 0.0]
    sigL_h  = exp.(0.5 .* (XfL * gamma_h))
    yL_h    = XfL * b_true .+ sigL_h .* randn(rng, N)
    mh = exponential_fwls(XL, yL_h)
    @test coef(mh)  ≈ b_true  atol=0.01
    @test mh.gamma  ≈ gamma_h atol=0.01

    mmle = exponential_mle(XL, yL_h)
    @test coef(mmle)  ≈ b_true  atol=0.01
    @test mmle.gamma  ≈ gamma_h atol=0.01
    @test mmle.converged

    # Glejser: σ = 0.5 + 0.3·|x1|  (correctly specified Z supplied)
    gamma_g = [0.5, 0.3]
    sigL_g  = 0.5 .+ 0.3 .* abs.(x1L)
    yL_g    = XfL * b_true .+ sigL_g .* randn(rng, N)
    ZL_g    = hcat(abs.(x1L))
    mg = quadratic_fwls(XL, yL_g; Z = ZL_g)
    @test coef(mg)  ≈ b_true  atol=0.01
    @test mg.gamma  ≈ gamma_g atol=0.01

    # Power: σ²_i = 1·|x1|^{2·0.5} = |x1|  (σ²=1, δ₁=0.5)
    sigL_p = abs.(x1L) .^ 0.5
    yL_p   = XfL * b_true .+ sigL_p .* randn(rng, N)
    ZL_p   = hcat(abs.(x1L))

    mp_fwls = power_fwls(XL, yL_p; Z = ZL_p)
    @test coef(mp_fwls)    ≈ b_true  atol=0.01
    @test mp_fwls.gamma[1] ≈ 0.0     atol=0.01   # log σ² = log(1) = 0
    @test mp_fwls.gamma[2] ≈ 1.0     atol=0.01   # 2δ₁ = 2×0.5 = 1

    mp_mle = power_mle(XL, yL_p; Z = ZL_p)
    @test coef(mp_mle)    ≈ b_true   atol=0.01
    @test mp_mle.gamma[1] ≈ 0.0      atol=0.01
    @test mp_mle.gamma[2] ≈ 1.0      atol=0.01
    @test mp_mle.converged

    # AR(1): ρ = 0.7
    uL_ar = zeros(N)
    for t in 2:N; uL_ar[t] = 0.7 * uL_ar[t-1] + epsL[t]; end
    yL_ar = XfL * b_true .+ uL_ar

    mpw = two_step_prais_winsten(XL, yL_ar)
    @test coef(mpw) ≈ b_true  atol=0.01
    @test mpw.rho   ≈ 0.7     atol=0.01

    mco = two_step_cochrane_orcutt(XL, yL_ar)
    @test coef(mco) ≈ b_true  atol=0.01
    @test mco.rho   ≈ 0.7     atol=0.01

    mhl = hildreth_lu(XL, yL_ar)
    @test coef(mhl) ≈ b_true  atol=0.01
    @test mhl.rho   ≈ 0.7     atol=0.01

    mbm = beach_mackinnon(XL, yL_ar)
    @test coef(mbm) ≈ b_true  atol=0.01
    @test mbm.rho   ≈ 0.7     atol=0.01

    # Combined AR(1) + Harvey heteroskedasticity: ρ = 0.7, γ = [0, 1, 0]
    uL_hae = zeros(N)
    for t in 2:N; uL_hae[t] = 0.7 * uL_hae[t-1] + sigL_h[t] * epsL[t]; end
    yL_hae = XfL * b_true .+ uL_hae

    ms = two_step_sequential(XL, yL_hae)
    @test coef(ms) ≈ b_true  atol=0.01
    @test ms.rho   ≈ 0.7     atol=0.01
    @test ms.gamma ≈ gamma_h atol=0.01

    mj = two_step_joint(XL, yL_hae)
    @test coef(mj) ≈ b_true  atol=0.01
    @test mj.rho   ≈ 0.7     atol=0.01
    @test mj.gamma ≈ gamma_h atol=0.01

    # Groupwise: 4 groups each with a known variance
    # σ²_g = [1.0, 2.0, 3.0, 4.0]
    sigma2_true = [1.0, 2.0, 3.0, 4.0]
    groupsL = repeat([1, 2, 3, 4], inner = N ÷ 4)
    sigL_gw = [sqrt(sigma2_true[groupsL[i]]) for i in 1:N]
    yL_gw   = XfL * b_true .+ sigL_gw .* randn(rng, N)

    mgw2 = two_step_groupwise(XL, yL_gw, groupsL)
    @test coef(mgw2)  ≈ b_true      atol=0.01
    @test mgw2.sigma2 ≈ sigma2_true rtol=0.05

    mgwi = iterated_groupwise(XL, yL_gw, groupsL)
    @test coef(mgwi)  ≈ b_true      atol=0.01
    @test mgwi.sigma2 ≈ sigma2_true rtol=0.05
    @test loglikelihood(mgwi) >= loglikelihood(mgw2) - 1e-6
end
