# Differentiability: with a fixed tree the FMM is a generic function of the source
# positions, the densities and the target positions, so ForwardDiff duals with one
# partial give directional derivatives. References are direct sums on the same duals.
const DualT = ForwardDiff.Dual{:fmm,Float64,1}
dual(x, v) = DualT(x, ForwardDiff.Partials((v,)))
value(x) = ForwardDiff.value(x)
partial(x) = ForwardDiff.partials(x, 1)
value(z::Complex) = Complex(value(real(z)), value(imag(z)))
partial(z::Complex) = Complex(partial(real(z)), partial(imag(z)))

rng = MersenneTwister(11)

@testset "directional derivatives, $(kname), $(mname)" for (kname, K) in (("Laplace", Laplace2D()),
                                                                         ("Helmholtz", Helmholtz2D(4.0))),
                                                            (mname, m) in (("KI", KernelIndependent(K; nequiv=60)),
                                                                           ("BB", BlackBox(K; p=14)))
    N = 3000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 800)
    V = [P2(randn(rng), randn(rng)) for _ in 1:N]          # perturbation of the sources
    Wt = [P2(randn(rng), randn(rng)) for _ in 1:800]       # perturbation of the targets
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    # sources
    sd = [SVector(dual(p[1], v[1]), dual(p[2], v[2])) for (p, v) in zip(pts, V)]
    st = build_expansions(f, sd, τ)
    u = evaluate(f, st, targets)
    uref = direct(K, sd, τ, targets)
    @test max_relative_diff(value.(u), value.(uref)) < 1e-7
    @test max_relative_diff(partial.(u), partial.(uref)) < 1e-6
    us = evaluate_at_sources(f, st)
    usref = direct_self(K, sd, τ)
    @test max_relative_diff(value.(us), value.(usref)) < 1e-7
    @test max_relative_diff(partial.(us), partial.(usref)) < 1e-6
    # targets
    st0 = build_expansions(f, pts, τ)
    td = [SVector(dual(t[1], w[1]), dual(t[2], w[2])) for (t, w) in zip(targets, Wt)]
    ut = evaluate(f, st0, td)
    utref = direct(K, pts, τ, td)
    @test max_relative_diff(partial.(ut), partial.(utref)) < 1e-6
    # densities (linearity)
    τd = [dual(t, randn(rng)) for t in τ]
    std = build_expansions(f, pts, τd)
    uτ = evaluate(f, std, targets)
    @test max_relative_diff(partial.(uτ), direct(K, pts, partial.(τd), targets)) < 1e-7
end
