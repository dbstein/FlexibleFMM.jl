# Single precision end to end: kernels with Float32 constants, Float32 tree,
# Float32 operators and expansions, and Dual{Float32} positions (the combination a
# Metal GPU would run). References are Float64 direct sums.
using FlexibleFMM: floattype, kernel_eltype, nlevels, conditioning

const P2f = SVector{2,Float32}
rng = MersenneTwister(13)
to64(x::SVector{2,Float32}) = SVector{2,Float64}(x)

@testset "precision is preserved by the kernels" begin
    s = P2f(0.1, 0.2); t = P2f(0.7, -0.3)
    @test Laplace2D(Float32)(s, t) isa Float32
    @test Helmholtz2D(3f0)(s, t) isa ComplexF32
    @test Yukawa2D(2f0)(s, t) isa Float32
    @test LaplaceDipoleTest(0.5f0)(s, t) isa Float32
    @test floattype(Laplace2D(Float32)) == Float32 && floattype(Helmholtz2D(3f0)) == Float32
    @test abs(Laplace2D(Float32)(s, t) - Laplace2D()(to64(s), to64(t))) < 1e-6
    @test abs(Helmholtz2D(3f0)(s, t) - Helmholtz2D(3.0)(to64(s), to64(t))) < 1e-6
    @test abs(Yukawa2D(2f0)(s, t) - Yukawa2D(2.0)(to64(s), to64(t))) < 1e-6
    # duals keep the precision too
    Df = ForwardDiff.Dual{:f,Float32,1}
    d(x) = Df(x, ForwardDiff.Partials((1f0,)))
    sd = SVector(d(0.1f0), d(0.2f0))
    for K in (Laplace2D(Float32), Yukawa2D(2f0), LaplaceDipoleTest(0.5f0))
        @test K(sd, t) isa Df
    end
    @test Helmholtz2D(3f0)(sd, t) isa Complex{Df}
    @test kernel_eltype(Laplace2D(Float32), P2f, P2f) == Float32
end

@testset "Float32 FMM, $(name)" for (name, m, tol) in (
        ("Laplace KI40", KernelIndependent(Laplace2D(Float32); nequiv=40), 1e-4),
        ("Laplace BB8", BlackBox(Laplace2D(Float32); p=8), 1e-4),
        ("Helmholtz KI40", KernelIndependent(Helmholtz2D(5f0); nequiv=40), 1e-4),
        ("Yukawa BB8", BlackBox(Yukawa2D(3f0); p=8), 1e-4))
    N = 5000
    pts = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:N]
    τ = randn(rng, Float32, N)
    targets = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:2000]
    f = FMM(m, pts; ppl=40, bbox=(P2f(0, 0), 1f0))
    @test f.tree.W0 isa Float32 && f.tree.origin isa P2f
    @test nlevels(f) >= 3
    @test eltype(f.ops[3].M2M) == FlexibleFMM.operator_eltype(m)
    @test real(eltype(f.ops[3].M2M)) == Float32
    st = build_expansions(f, pts, τ)
    @test real(eltype(st.multipoles)) == Float32
    us = evaluate_at_sources(f, st)
    ut = evaluate(f, st, targets)
    @test real(eltype(us)) == Float32 && real(eltype(ut)) == Float32
    K64 = m.kernel isa Laplace2D ? Laplace2D() : m.kernel isa Helmholtz2D ? Helmholtz2D(Float64(m.kernel.k)) : Yukawa2D(Float64(m.kernel.k))
    p64 = to64.(pts); τ64 = Float64.(τ)
    @test max_relative_diff(us, direct_self(K64, p64, τ64)) < tol
    @test max_relative_diff(ut, direct(K64, p64, τ64, to64.(targets))) < tol
    if m isa KernelIndependent
        c = conditioning(f.ops[3])
        @test c[3] > 0        # in single precision the default truncation drops modes
    end
end

@testset "Dual{Float32} directional derivative through the FMM" begin
    Df = ForwardDiff.Dual{:g,Float32,1}
    d(x, v) = Df(x, ForwardDiff.Partials((v,)))
    N = 3000
    pts = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:N]
    V = [P2f(randn(rng, Float32), randn(rng, Float32)) for _ in 1:N]
    τ = randn(rng, Float32, N)
    targets = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:500]
    m = KernelIndependent(Laplace2D(Float32); nequiv=40)
    f = FMM(m, pts; ppl=30, bbox=(P2f(0, 0), 1f0))
    sd = [SVector(d(p[1], v[1]), d(p[2], v[2])) for (p, v) in zip(pts, V)]
    st = build_expansions(f, sd, τ)
    @test eltype(st.multipoles) == Df
    u = evaluate(f, st, targets)
    @test eltype(u) == Df
    uref = direct(Laplace2D(), [SVector(ForwardDiff.Dual{:g,Float64,1}(Float64(p[1]), ForwardDiff.Partials((Float64(v[1]),))),
                                        ForwardDiff.Dual{:g,Float64,1}(Float64(p[2]), ForwardDiff.Partials((Float64(v[2]),))))
                                for (p, v) in zip(pts, V)], Float64.(τ), to64.(targets))
    @test max_relative_diff(ForwardDiff.value.(u), ForwardDiff.value.(uref)) < 1e-4
    @test max_relative_diff(ForwardDiff.partials.(u, 1), ForwardDiff.partials.(uref, 1)) < 1e-3
end
