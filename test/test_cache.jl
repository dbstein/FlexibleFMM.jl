# Operator reuse across FMM instances through a shared cache.
using FlexibleFMM: nlevels

rng = MersenneTwister(12)

@testset "shared cache across boxes" begin
    K = Laplace2D()
    for m in (KernelIndependent(K; nequiv=50), BlackBox(K; p=10))
        cache = OperatorCache(m)
        pts1 = uniform_points(rng, 3000)
        τ1 = randn(rng, 3000)
        f1 = FMM(m, pts1; ppl=30, bbox=(P2(0, 0), 1.0), cache=cache)
        n1 = length(cache)
        @test n1 == nlevels(f1)
        # a box of width 0.5 reuses the finer widths; only one new exponent appears at most
        pts2 = 0.5 .* uniform_points(rng, 2000)
        τ2 = randn(rng, 2000)
        f2 = FMM(m, pts2; ppl=30, bbox=(P2(0, 0), 0.5), cache=cache)
        @test f2.tree.W0 == 0.5
        @test length(cache) <= n1 + 1
        @test f2.ops[1] === f1.ops[2]
        # a bigger box snaps to width 2 and adds exactly one width
        pts3 = 1.5 .* uniform_points(rng, 2000)
        f3 = FMM(m, pts3; ppl=30, bbox=(P2(0, 0), 1.5), cache=cache)
        @test f3.tree.W0 == 2.0 && f3.ops[2] === f1.ops[1]
        # results are identical to a fresh cache
        g2 = FMM(m, pts2; ppl=30, bbox=(P2(0, 0), 0.5))
        targets = 0.5 .* uniform_points(rng, 500)
        u = evaluate(f2, build_expansions(f2, pts2, τ2), targets)
        v = evaluate(g2, build_expansions(g2, pts2, τ2), targets)
        @test u == v
        @test max_relative_diff(u, direct(K, pts2, τ2, targets)) < 1e-6
        # a cache belongs to its method
        @test_throws ArgumentError FMM(KernelIndependent(K; nequiv=20), pts1; cache=cache)
    end
end
