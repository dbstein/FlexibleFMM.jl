# End-to-end: FMM against direct O(N²) sums with real kernels, for both methods
# (the black-box method runs through the driver with no driver changes).
using FlexibleFMM: nlevels, nnodes

rng = MersenneTwister(6)

function fmm_vs_direct(m, pts, τ, targets; ppl=40, bbox=nothing)
    K = m.kernel
    f = FMM(m, pts; ppl=ppl, bbox=bbox)
    st = build_expansions(f, pts, τ)
    us = evaluate_at_sources(f, st)
    ut = evaluate(f, st, targets)
    es = max_relative_diff(us, direct_self(K, pts, τ))
    et = max_relative_diff(ut, direct(K, pts, τ, targets))
    return f, es, et
end

methods_for(K) = (("KI nequiv=60", KernelIndependent(K; nequiv=60), 1e-7),
                  ("BB p=14", BlackBox(K; p=14), 1e-7))

@testset "Laplace, $name, $mname" for (name, gen) in pairs(DISTRIBUTIONS),
                                      (mname, m, tol) in methods_for(Laplace2D())
    N = 6000
    pts = gen(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 3000)
    f, es, et = fmm_vs_direct(m, pts, τ, targets; bbox=(P2(0, 0), 1.0))
    @test nlevels(f) >= 3
    @test es < tol
    @test et < tol
end

@testset "non-symmetric kernel end to end, $mname" for (mname, m, tol) in methods_for(LaplaceDipoleTest(0.8))
    N = 5000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 2000)
    f, es, et = fmm_vs_direct(m, pts, τ, targets; bbox=(P2(0, 0), 1.0), ppl=20)
    @test es < tol && et < tol
end

@testset "accuracy improves with the expansion order" begin
    N = 4000
    pts = uniform_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 1000)
    e = [fmm_vs_direct(KernelIndependent(Laplace2D(); nequiv=n), pts, τ, targets; bbox=(P2(0, 0), 1.0))[3]
         for n in (24, 48, 96)]
    @test e[1] > 30e[2] > 30e[3]
    @test e[3] < 1e-11
    e = [fmm_vs_direct(BlackBox(Laplace2D(); p=p), pts, τ, targets; bbox=(P2(0, 0), 1.0))[3]
         for p in (6, 10, 16)]
    @test e[1] > 10e[2] > 10e[3]
    @test e[3] < 1e-9
end

@testset "tiny trees: one and two levels, $mname" for (mname, m, tol) in methods_for(Laplace2D())
    K = Laplace2D()
    pts = uniform_points(rng, 30)
    τ = randn(rng, 30)
    f = FMM(m, pts; ppl=50, bbox=(P2(0, 0), 1.0))
    @test nlevels(f) == 1
    st = build_expansions(f, pts, τ)
    @test evaluate_at_sources(f, st) ≈ direct_self(K, pts, τ)
    targets = [P2(0.3, 0.3), P2(0.9, 0.1)]
    @test evaluate(f, st, targets) ≈ direct(K, pts, τ, targets)
    f2 = FMM(m, pts; ppl=10, bbox=(P2(0, 0), 1.0))
    @test nlevels(f2) >= 2
    st2 = build_expansions(f2, pts, τ)
    @test evaluate_at_sources(f2, st2) ≈ direct_self(K, pts, τ)
end

@testset "user boxes, random widths, matrix input" begin
    K = Laplace2D()
    N = 3000
    pts = circle_points(rng, N; center=P2(0, 0), radius=1.0)
    τ = randn(rng, N)
    targets = [3 * P2(rand(rng) - 0.5, rand(rng) - 0.5) for _ in 1:1000]
    for W in (3.1, 4.0, 5.7, 9.3)
        f, es, et = fmm_vs_direct(KernelIndependent(K; nequiv=60), pts, τ, targets;
                                  bbox=(P2(-W / 2, -W / 2), W), ppl=30)
        @test es < 1e-7 && et < 1e-7
    end
    M = reduce(hcat, pts)
    m = KernelIndependent(K; nequiv=60)
    f = FMM(m, M; ppl=30, bbox=(P2(-2, -2), 4.0))
    st = build_expansions(f, M, τ)
    @test max_relative_diff(evaluate(f, st, reduce(hcat, targets)), direct(K, pts, τ, targets)) < 1e-7
end

@testset "state reuse and errors" begin
    K = Laplace2D()
    m = KernelIndependent(K; nequiv=40)
    pts = uniform_points(rng, 2000)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    τ1 = randn(rng, 2000); τ2 = randn(rng, 2000)
    st = build_expansions(f, pts, τ1)
    u1 = evaluate_at_sources(f, st)
    build_expansions!(st, f, pts, τ2)
    u2 = evaluate_at_sources(f, st)
    @test max_relative_diff(u2, direct_self(K, pts, τ2)) < 1e-7
    @test !(u1 ≈ u2)
    @test_throws DimensionMismatch build_expansions(f, pts, randn(rng, 10))
    @test_throws ArgumentError evaluate(f, st, [P2(2.0, 2.0)])
    st1 = build_expansions(f, pts, τ1)
    st12 = build_expansions(f, pts, τ1 .+ 2τ2)
    @test evaluate_at_sources(f, st12) ≈ evaluate_at_sources(f, st1) .+ 2u2
    @test occursin("FMM", string(f))
end
