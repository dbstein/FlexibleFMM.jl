# Kernel-independent operators in isolation. With N points per circle the expansion
# carries N/2 harmonics and the nearest interaction-list box sees the source circle
# at radius ratio r1/r2 = 0.547, so the error floor is about 0.547^(N/2):
# 1.4e-8 at N = 60, 8e-14 at N = 100.
using FlexibleFMM: KernelIndependent, level_operators, pinv_rtol, FactoredPinv, apply!,
    conditioning, OperatorCache, fit_box!, get_operators!, has_base_width

rng = MersenneTwister(5)

@testset "M2L on all 40 offsets: $(name)" for (name, K) in (
        ("Laplace", Laplace2D()),
        ("Laplace dipole test (non-symmetric)", LaplaceDipoleTest(0.7)),
        ("Yukawa", Yukawa2D(3.0)),
        ("Helmholtz", Helmholtz2D(4.0)))
    check_m2l_all_offsets(rng, KernelIndependent(K; nequiv=60), 0.37, 1e-7)
end

@testset "convergence in nequiv" begin
    e = [maximum(m2l_errors(rng, KernelIndependent(Laplace2D(); nequiv=n), 0.5)) for n in (16, 32, 64, 100)]
    @test e[1] > 1e2 * e[2] > 1e2 * e[3] > 1e2 * e[4]
    @test e[4] < 1e-11
end

@testset "M2M and L2L: $(name)" for (name, K) in (("Laplace", Laplace2D()),
        ("non-symmetric", LaplaceDipoleTest(0.4)), ("Helmholtz", Helmholtz2D(3.0)))
    m = KernelIndependent(K; nequiv=60)
    check_m2m(rng, m, 0.61, 1e-7)
    check_l2l(rng, m, 0.61, 1e-7)
end

@testset "pseudo-inverse and conditioning" begin
    A = randn(rng, 8, 8)
    P = pinv_rtol(A, 1e-14)
    @test FlexibleFMM.rank(P) == 8 && norm(Matrix(P) * A - I) < 1e-10
    x = randn(rng, 8, 3)
    @test apply!(similar(x), P, x, similar(x)) ≈ A \ x
    @test apply!(zeros(8), P, x[:, 1], zeros(8)) ≈ A \ x[:, 1]
    B = randn(rng, ComplexF64, 6, 6)
    P = pinv_rtol(B, 1e-14)
    @test norm(Matrix(P) * B - I) < 1e-10
    C = A[:, 1:3] * A[1:3, :]
    P = pinv_rtol(C, 1e-10)
    @test FlexibleFMM.rank(P) == 3
    @test norm(C * Matrix(P) * C - C) < 1e-10
    # Laplace scale degeneracy: the monopole check potential vanishes at r2 = 1
    m = KernelIndependent(Laplace2D(); nequiv=50)
    wdeg = 2 / (4 - sqrt(2))
    @test conditioning(level_operators(m, wdeg))[1] < 1e-10
    for w in (0.5, 0.6, 1.0, 1.3)
        c = conditioning(level_operators(m, w))
        @test c[1] > 1e-8 && c[2] > 1e-8
        @test c[3] == 0 && c[4] == 0
    end
end

@testset "allocation-free bodies" begin
    for K in (Laplace2D(), Helmholtz2D(2.0))
        check_allocation_free(rng, KernelIndependent(K; nequiv=40), 0.3)
    end
end

@testset "operator cache and box fitting" begin
    m = KernelIndependent(Laplace2D(); nequiv=20)
    c = OperatorCache(m)
    @test !has_base_width(c)
    pts = uniform_points(rng, 50)
    o, W, e = fit_box!(c, pts, (P2(0, 0), 1.0))
    @test (o, W, e) == (P2(0, 0), 1.0, 0) && c.W0 == 1.0
    o2, W2, e2 = fit_box!(c, 0.9 .* pts, (P2(0, 0), 0.9))
    @test W2 == 1.0 && e2 == 0 && all(o2 .<= 0)
    o3, W3, e3 = fit_box!(c, pts, (P2(-0.2, -0.2), 1.6))
    @test W3 == 2.0 && e3 == 1 && all(o3 .<= -0.2) && all(o3 .+ W3 .>= 1.4)
    o4, W4, e4 = fit_box!(c, 0.1 .* pts, (P2(0, 0), 0.25))
    @test W4 == 0.25 && e4 == -2
    @test_throws ArgumentError fit_box!(c, pts, (P2(0.5, 0.5), 1.0))
    for _ in 1:20
        q = circle_points(rng, 30; center=P2(5rand(rng), 5rand(rng)), radius=0.3 + rand(rng))
        o5, W5, e5 = fit_box!(c, q, nothing)
        @test all(all(o5 .<= x .<= o5 .+ W5) for x in q)
        @test W5 == ldexp(c.W0, e5)
    end
    @test length(c) == 0
    a = get_operators!(c, 0); b = get_operators!(c, 0)
    @test a === b && length(c) == 1
    @test get_operators!(c, -3).width == 0.125
end
