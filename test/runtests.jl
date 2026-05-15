using Test
using HARE
using Random
using LinearAlgebra
using StatsBase
using StatsModels
using GLM
using Distributions

Random.seed!(1234)

# Shared test data
n   = 200
x1  = randn(n)
x2  = randn(n)
X = hcat(x1, x2)
k   = 3   # intercept + x1 + x2 after augmentation

b_true = [1.0, 2.0, -1.0]

sigma     = exp.(0.5 .* x1)
y_het     = hcat(ones(n), X) * b_true .+ sigma .* randn(n)
data_het  = (y = y_het, x1 = x1, x2 = x2)

sigma_lin  = 0.5 .+ 0.3 .* abs.(x1)
y_glej     = hcat(ones(n), X) * b_true .+ sigma_lin .* randn(n)
data_glej  = (y = y_glej, x1 = x1, x2 = x2)

eps_t  = randn(n)
u_ar   = zeros(n)
for t in 2:n; u_ar[t] = 0.7 * u_ar[t-1] + eps_t[t]; end
y_ar     = hcat(ones(n), X) * b_true .+ u_ar
data_ar  = (y = y_ar, x1 = x1, x2 = x2)

u_hae = zeros(n)
for t in 2:n; u_hae[t] = 0.7 * u_hae[t-1] + sigma[t] * eps_t[t]; end
y_hae     = hcat(ones(n), X) * b_true .+ u_hae
data_hae  = (y = y_hae, x1 = x1, x2 = x2)

# Helper: check the full StatsBase interface on any HAREModel
function check_interface(m, y, n, k; check_r2_bounds=true)
    @test m isa HAREModel

    # RegressionModel interface
    @test m isa StatsBase.RegressionModel
    @test islinear(m)

    # core coefficient interface
    @test length(coef(m))      == k
    @test size(vcov(m))        == (k, k)
    @test length(stderror(m))  == k
    @test length(coefnames(m)) == k
    @test all(stderror(m) .> 0)

    # sample / dof
    @test nobs(m)         == n
    @test dof(m)          == k
    @test dof_residual(m) == n - k

    # fitted / residuals / response / predict
    @test length(residuals(m)) == n
    @test length(fitted(m))    == n
    @test length(predict(m))   == n
    @test residuals(m) .+ fitted(m) ≈ y
    @test response(m)          ≈ y
    @test fitted(m)            ≈ predict(m)

    # matrix predict (X already has intercept column)
    Xtest = hcat(ones(3), randn(3), randn(3))
    @test length(predict(m, Xtest)) == 3
    @test predict(m, Xtest) ≈ Xtest * coef(m)

    # inference
    @test length(tstat(m))    == k
    @test tstat(m)            ≈ coef(m) ./ stderror(m)
    @test length(pvalues(m))  == k
    @test all(0 .<= pvalues(m) .<= 1)
    @test length(pvalues(m; dist=:normal)) == k
    @test all(0 .<= pvalues(m; dist=:normal) .<= 1)
    ci = confint(m)
    @test size(ci) == (k, 2)
    @test all(ci[:, 1] .< coef(m) .< ci[:, 2])
    ci90 = confint(m; level=0.90)
    @test all(ci90[:, 2] .- ci90[:, 1] .< ci[:, 2] .- ci[:, 1])   # tighter at 90%

    # fit diagnostics
    @test rss(m)    ≈ sum(abs2, residuals(m))
    @test sigma2(m) ≈ rss(m) / dof_residual(m)
    check_r2_bounds && @test 0 <= r2(m) <= 1
    @test adjr2(m)  <= r2(m)
    @test adjr2(m)  ≈ 1 - (1 - r2(m)) * (nobs(m) - 1) / dof_residual(m)
end

@testset "two_step_harvey" begin
    m = two_step_harvey(X, y_het)
    @test m isa HarveyResult
    check_interface(m, y_het, n, k)
    @test m.iterations == 1
    @test m.converged  == true
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test m.gamma_vcov           ≈ m.gamma_vcov'

    m2 = two_step_harvey(@formula(y ~ x1 + x2), data_het)
    @test coef(m)       ≈ coef(m2)
    @test m.gamma       ≈ m2.gamma
    @test m.gamma_vcov  ≈ m2.gamma_vcov
end

@testset "iterated_harvey" begin
    m = iterated_harvey(X, y_het)
    @test m isa HarveyResult
    check_interface(m, y_het, n, k)
    @test m.iterations >= 1
    @test m.converged
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)

    m2 = iterated_harvey(@formula(y ~ x1 + x2), data_het)
    @test coef(m)  ≈ coef(m2)
    @test m.gamma  ≈ m2.gamma
end

@testset "two_step_glejser" begin
    m = two_step_glejser(X, y_glej)
    @test m isa GlejserResult
    check_interface(m, y_glej, n, k)
    @test m.iterations == 1
    @test m.converged  == true
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test m.gamma_vcov           ≈ m.gamma_vcov'

    m2 = two_step_glejser(@formula(y ~ x1 + x2), data_glej)
    @test coef(m)       ≈ coef(m2)
    @test m.gamma       ≈ m2.gamma
    @test m.gamma_vcov  ≈ m2.gamma_vcov

    # separate auxiliary regressors Z (must include constant)
    Z = hcat(ones(n), abs.(x1))
    mz = two_step_glejser(X, y_glej; Z=Z)
    @test mz isa GlejserResult
    check_interface(mz, y_glej, n, k)
    @test length(mz.gamma)        == 2
    @test size(mz.gamma_vcov)     == (2, 2)
    @test all(diag(mz.gamma_vcov) .> 0)
end

@testset "iterated_glejser" begin
    m = iterated_glejser(X, y_glej)
    @test m isa GlejserResult
    check_interface(m, y_glej, n, k)
    @test m.iterations >= 1
    @test m.converged
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)

    m2 = iterated_glejser(@formula(y ~ x1 + x2), data_glej)
    @test coef(m)  ≈ coef(m2)
    @test m.gamma  ≈ m2.gamma

    # separate auxiliary regressors Z (must include constant)
    Z = hcat(ones(n), abs.(x1))
    mz = iterated_glejser(X, y_glej; Z=Z)
    @test mz isa GlejserResult
    check_interface(mz, y_glej, n, k)
    @test length(mz.gamma)        == 2
    @test size(mz.gamma_vcov)     == (2, 2)
    @test all(diag(mz.gamma_vcov) .> 0)
end

@testset "two_step_prais_winsten" begin
    m = two_step_prais_winsten(X, y_ar)
    @test m isa PraisWinstenResult
    check_interface(m, y_ar, n, k)
    @test -1 < m.rho < 1
    @test m.iterations == 1
    @test m.converged

    m2 = two_step_prais_winsten(@formula(y ~ x1 + x2), data_ar)
    @test coef(m) ≈ coef(m2)
    @test m.rho ≈ m2.rho
end

@testset "iterated_prais_winsten" begin
    m = iterated_prais_winsten(X, y_ar)
    @test m isa PraisWinstenResult
    check_interface(m, y_ar, n, k)
    @test -1 < m.rho < 1
    @test m.converged

    m2 = iterated_prais_winsten(@formula(y ~ x1 + x2), data_ar)
    @test coef(m) ≈ coef(m2)
    @test m.rho ≈ m2.rho
end

@testset "hildreth_lu" begin
    m = hildreth_lu(X, y_ar)
    @test m isa HildrethLuResult
    check_interface(m, y_ar, n, k)
    @test -1 < m.rho < 1
    @test m.rss > 0
    @test m.iterations == 200   # default nrho
    @test m.converged

    m2 = hildreth_lu(@formula(y ~ x1 + x2), data_ar)
    @test coef(m) ≈ coef(m2)
    @test m.rho ≈ m2.rho
end

@testset "two_step_sequential" begin
    m = two_step_sequential(X, y_hae)
    @test m isa SequentialResult
    check_interface(m, y_hae, n, k)
    @test -1 < m.rho < 1
    @test m.iterations == 1
    @test m.converged
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test m.gamma_vcov           ≈ m.gamma_vcov'
    # gamma is estimated against original X, so slope on x1 should be positive
    # (true log-variance slope is 1.0)
    @test m.gamma[2] > 0

    m2 = two_step_sequential(@formula(y ~ x1 + x2), data_hae)
    @test coef(m)       ≈ coef(m2)
    @test m.rho         ≈ m2.rho
    @test m.gamma       ≈ m2.gamma
    @test m.gamma_vcov  ≈ m2.gamma_vcov
end

@testset "iterated_sequential" begin
    m = iterated_sequential(X, y_hae)
    @test m isa SequentialResult
    check_interface(m, y_hae, n, k)
    @test -1 < m.rho < 1
    @test m.converged
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test m.gamma[2] > 0

    m2 = iterated_sequential(@formula(y ~ x1 + x2), data_hae)
    @test coef(m)  ≈ coef(m2)
    @test m.rho    ≈ m2.rho
    @test m.gamma  ≈ m2.gamma
end

@testset "two_step_joint" begin
    m = two_step_joint(X, y_hae)
    @test m isa JointResult
    check_interface(m, y_hae, n, k)
    @test -1 < m.rho < 1
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test m.gamma_vcov           ≈ m.gamma_vcov'
    @test isfinite(m.loglik)
    @test m.iterations == 1
    @test m.converged

    @test loglikelihood(m) == m.loglik

    m2 = two_step_joint(@formula(y ~ x1 + x2), data_hae)
    @test coef(m)   ≈ coef(m2)   atol=1e-4
    @test m.rho     ≈ m2.rho     atol=1e-4
    @test m.gamma   ≈ m2.gamma   atol=1e-4

    # separate auxiliary regressors Z (must include constant)
    Z = hcat(ones(n), x1)
    mz = two_step_joint(X, y_hae; Z=Z)
    @test mz isa JointResult
    @test length(mz.gamma)        == 2
    @test size(mz.gamma_vcov)     == (2, 2)
    @test all(diag(mz.gamma_vcov) .> 0)
    check_interface(mz, y_hae, n, k)
end

@testset "iterated_joint" begin
    m = iterated_joint(X, y_hae)
    @test m isa JointResult
    check_interface(m, y_hae, n, k)
    @test -1 < m.rho < 1
    @test length(m.gamma)        == k
    @test size(m.gamma_vcov)     == (k, k)
    @test all(diag(m.gamma_vcov) .> 0)
    @test isfinite(m.loglik)
    @test m.iterations >= 1
    @test m.converged

    @test loglikelihood(m) == m.loglik

    m2 = iterated_joint(@formula(y ~ x1 + x2), data_hae)
    @test coef(m)  ≈ coef(m2)  atol=1e-4
    @test m.rho    ≈ m2.rho    atol=1e-4
    @test m.gamma  ≈ m2.gamma  atol=1e-4

    # separate auxiliary regressors Z (must include constant)
    Z = hcat(ones(n), x1)
    mz = iterated_joint(X, y_hae; Z=Z)
    @test mz isa JointResult
    @test length(mz.gamma)        == 2
    @test size(mz.gamma_vcov)     == (2, 2)
    @test all(diag(mz.gamma_vcov) .> 0)
    check_interface(mz, y_hae, n, k)
end

@testset "beach_mackinnon" begin
    m = beach_mackinnon(X, y_ar)
    @test m isa BeachMacKinnonResult
    check_interface(m, y_ar, n, k)
    @test -1 < m.rho < 1
    @test isfinite(m.loglik)
    @test m.converged

    @test loglikelihood(m) == m.loglik
    @test isfinite(loglikelihood(m))

    m2 = beach_mackinnon(@formula(y ~ x1 + x2), data_ar)
    @test coef(m) ≈ coef(m2)
    @test m.rho ≈ m2.rho
end

@testset "two-step vs iterated consistency" begin
    @test coef(two_step_glejser(X, y_glej))        ≈ coef(iterated_glejser(X, y_glej))        atol=0.05
    @test coef(two_step_harvey(X, y_het))           ≈ coef(iterated_harvey(X, y_het))           atol=0.05
    @test coef(two_step_prais_winsten(X, y_ar))     ≈ coef(iterated_prais_winsten(X, y_ar))     atol=0.05
    @test coef(two_step_sequential(X, y_hae))       ≈ coef(iterated_sequential(X, y_hae))       atol=0.05
    @test coef(two_step_joint(X, y_hae))            ≈ coef(iterated_joint(X, y_hae))            atol=0.05
end

@testset "formula metadata and predict(newdata)" begin
    estimators = [
        ("two_step_glejser",       (f, d) -> two_step_glejser(f, d)),
        ("iterated_glejser",       (f, d) -> iterated_glejser(f, d)),
        ("two_step_harvey",        (f, d) -> two_step_harvey(f, d)),
        ("iterated_harvey",        (f, d) -> iterated_harvey(f, d)),
        ("two_step_prais_winsten", (f, d) -> two_step_prais_winsten(f, d)),
        ("iterated_prais_winsten", (f, d) -> iterated_prais_winsten(f, d)),
        ("hildreth_lu",            (f, d) -> hildreth_lu(f, d)),
        ("two_step_sequential",    (f, d) -> two_step_sequential(f, d)),
        ("iterated_sequential",    (f, d) -> iterated_sequential(f, d)),
        ("two_step_joint",         (f, d) -> two_step_joint(f, d)),
        ("iterated_joint",         (f, d) -> iterated_joint(f, d)),
        ("beach_mackinnon",        (f, d) -> beach_mackinnon(f, d)),
    ]

    newdata = (x1 = randn(5), x2 = randn(5))
    Xnew    = hcat(ones(5), newdata.x1, newdata.x2)

    for (name, fit) in estimators
        @testset "$name" begin
            mf = fit(@formula(y ~ x1 + x2), data_ar)
            mm = fit(X, y_ar)

            # formula fit: named coefnames
            @test coefnames(mf) == ["(Intercept)", "x1", "x2"]

            # matrix fit with intercept=true: default coefnames include (Intercept)
            @test coefnames(mm) == ["(Intercept)", "x1", "x2"]

            # formula metadata
            @test !isnothing(formula(mf))
            @test termnames(mf) == ["(Intercept)", "x1", "x2"]
            @test responsename(mf) == "y"
            @test isnothing(formula(mm))
            @test isnothing(termnames(mm))
            @test isnothing(responsename(mm))

            # predict with newdata
            y_formula = predict(mf, newdata)
            y_matrix  = predict(mf, Xnew)
            @test length(y_formula) == 5
            @test y_formula ≈ y_matrix

            # matrix-fitted model should throw on newdata predict
            @test_throws ArgumentError predict(mm, newdata)
        end
    end
end

@testset "wald_test" begin
    m = iterated_prais_winsten(X, y_ar)
    kk = length(coef(m))

    # joint test: all slope coefficients zero
    R = [0.0 1.0 0.0; 0.0 0.0 1.0]
    w = wald_test(m, R)
    @test w isa WaldTestResult
    @test w.df == 2
    @test w.df_residual == dof_residual(m)
    @test w.stat > 0
    @test 0 <= w.pvalue <= 1

    # test with explicit r vector
    w2 = wald_test(m, R, zeros(2))
    @test w.stat ≈ w2.stat

    # single restriction matches tstat^2 ~ F(1, n-k)
    R1 = reshape([0.0, 1.0, 0.0], 1, kk)
    w1 = wald_test(m, R1)
    @test w1.df == 1
    @test w1.stat ≈ tstat(m)[2]^2 atol=1e-10

    # dimension mismatch errors
    @test_throws DimensionMismatch wald_test(m, ones(2, kk+1))
    @test_throws DimensionMismatch wald_test(m, R, zeros(3))

    # works for all HAREModel subtypes
    for mfit in [two_step_glejser(X, y_glej),
                 two_step_harvey(X, y_het),
                 two_step_prais_winsten(X, y_ar),
                 hildreth_lu(X, y_ar),
                 two_step_sequential(X, y_hae),
                 two_step_joint(X, y_hae),
                 beach_mackinnon(X, y_ar)]
        @test wald_test(mfit, R) isa WaldTestResult
    end
end

@testset "lrtest" begin
    m_r = beach_mackinnon(X[:, 1:1], y_ar)   # intercept + x1
    m_u = beach_mackinnon(X,         y_ar)   # intercept + x1 + x2

    lr = lrtest(m_r, m_u)
    @test lr isa LRTestResult
    @test lr.df == 1
    @test lr.stat >= 0
    @test 0 <= lr.pvalue <= 1
    @test lr.stat ≈ 2 * (loglikelihood(m_u) - loglikelihood(m_r))

    # wrong order should throw
    @test_throws ArgumentError lrtest(m_u, m_r)
end

@testset "inference: t vs normal distribution" begin
    m = two_step_prais_winsten(X, y_ar)

    pv_t = pvalues(m; dist=:t)
    pv_n = pvalues(m; dist=:normal)
    @test all(pv_t .>= pv_n)          # t has heavier tails -> larger p-values

    ci_t = confint(m; dist=:t)
    ci_n = confint(m; dist=:normal)
    widths_t = ci_t[:, 2] .- ci_t[:, 1]
    widths_n = ci_n[:, 2] .- ci_n[:, 1]
    @test all(widths_t .>= widths_n)   # t intervals are wider
end
