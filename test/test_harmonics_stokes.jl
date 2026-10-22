# The Stokes harmonic method: derivative coefficient sets against automatic
# differentiation, the operators on all 316 offsets for stokeslets, stresslets and
# doublets, and the FMM end to end (velocity, pressure, velocity gradient, Float32,
# dual positions, the Stokes plan).
using FlexibleFMM: StokesHarmonics, SphericalHarmonics, level_operators, order, l2t, apply!,
    far_offsets, multipole_length, partial_local_length

rng = MersenneTwister(34)
const KSH = Stokeslet3D()
stresslets3(rng, K, n) = [Stresslet(randn(rng, P3), normalize(randn(rng, P3))) for _ in 1:n]
doublets3(rng, K, n) = [Doublet(randn(rng, P3)) for _ in 1:n]
mixed_stokes3(rng, K, n) = [(randn(rng, P3), Stresslet(randn(rng, P3), normalize(randn(rng, P3))), Doublet(randn(rng, P3))) for _ in 1:n]

@testset "derivatives of the local expansions" begin
    p = 9
    m = StokesHarmonics(KSH; p=p)
    @test order(m) == p && multipole_length(m) == 4 * (p + 1)^2
    for w in (1.0, 0.3)
        ops = level_operators(m, w)
        L = randn(rng, 4 * (p + 1)^2)
        x = P3(0.2, -0.3, 0.1) * w
        u(y) = l2t(m, ops, L, y)
        @test l2t(m, ops, L, x, Gradient()) ≈ ForwardDiff.jacobian(u, x) rtol = 1e-12
        # the velocity of the representation is divergence free
        @test abs(tr(l2t(m, ops, L, x, Gradient()))) < 1e-12 * norm(l2t(m, ops, L, x, Gradient()))
        # the scalar gradient of harmonics.jl agrees with the coefficient route
        sh = SphericalHarmonics(Laplace3D(); p=p)
        Ls = randn(rng, (p + 1)^2)
        @test l2t(sh, ops, Ls, x, Gradient()) ≈ ForwardDiff.gradient(y -> l2t(sh, ops, Ls, y), x) rtol = 1e-12
    end
    @test_throws ArgumentError StokesHarmonics(KSH; p=0)
end

@testset "operators on all 316 offsets, p = 16" begin
    m = StokesHarmonics(KSH; p=16)
    e = check_m2l_all_offsets(rng, m, 0.37, 1e-4)
    es = check_m2l_all_offsets(rng, m, 0.37, 1e-3; density=stresslets3)
    ed = check_m2l_all_offsets(rng, m, 0.37, 1e-3; density=doublets3)
    @info "Stokes harmonics p = 16, worst offset error: stokeslets $(round(e, sigdigits=2)), stresslets $(round(es, sigdigits=2)), doublets $(round(ed, sigdigits=2))"
    check_m2m(rng, m, 0.61, 1e-7)
    check_m2m(rng, m, 0.61, 1e-7; density=stresslets3)
    check_m2m(rng, m, 0.61, 1e-7; density=doublets3)
    check_l2l(rng, m, 0.61, 1e-4)
    check_l2l(rng, m, 0.61, 1e-4; density=mixed_stokes3)
    check_allocation_free(rng, m, 0.3)
end

@testset "FMM end to end: $(name)" for (name, gen) in ((:uniform, uniform_points3), (:helix, helix_points), (:mixed, mixed_points3))
    N = 5000
    pts = gen(rng, N)
    targets = uniform_points3(rng, 400)
    m = StokesHarmonics(KSH; p=16)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    for (dname, τ) in (("stokeslets", [randn(rng, P3) for _ in 1:N]), ("mixed", mixed_stokes3(rng, KSH, N)))
        st = build_expansions(f, pts, τ)
        tol = dname == "stokeslets" ? 1e-5 : 1e-4
        @test mrd(evaluate_at_sources(f, st), direct_self(KSH, pts, τ)) < tol
        @test mrd(evaluate(f, st, targets), direct(KSH, pts, τ, targets)) < tol
        @test mrd(evaluate(f, st, targets; output=Pressure()), direct(KSH, pts, τ, targets; output=Pressure())) < 10tol
        @test mrd(evaluate(f, st, targets; output=Gradient()), direct(KSH, pts, τ, targets; output=Gradient())) < 10tol
    end
    if name == :uniform
        @test_throws ArgumentError build_expansions(f, pts, [Rotlet(randn(rng, P3)) for _ in 1:N])
        @test_throws ArgumentError build_expansions(f, pts, randn(rng, N))
    end
end

@testset "Stokes plan and Float32" begin
    N = 3000
    pts = helix_points(rng, N); tg = uniform_points3(rng, 200)
    fs = [randn(rng, P3) for _ in 1:N]; g = [0.01 * randn(rng, P3) for _ in 1:N]
    plan = StokesFMMPlan(pts; targets=tg, method=:harmonics, ppl=30)
    @test plan.fmm.method isa StokesHarmonics && order(plan.fmm.method) == 16
    u = zeros(P3, 200)
    stokes_fmm_eval!(u, plan; stokeslets=fs, doublets=g)
    τ = [(fs[i], Doublet(g[i])) for i in 1:N]
    @test mrd(u, direct(KSH, pts, τ, tg)) < 1e-5
    Kf = Stokeslet3D(Float32)
    ptsf = [SVector{3,Float32}(x) for x in pts]
    ff = FMM(StokesHarmonics(Kf; p=16), ptsf; ppl=40, bbox=(SVector{3,Float32}(0, 0, 0), 1f0))
    τf = [SVector{3,Float32}(x) for x in fs]
    stf = build_expansions(ff, ptsf, τf)
    uf = evaluate_at_sources(ff, stf)
    @test eltype(uf) == SVector{3,Float32}
    @test mrd(uf, direct_self(KSH, pts, fs)) < 1e-3
end

@testset "dual positions (the slender-body pattern)" begin
    D1 = ForwardDiff.Dual{:dhs,Float64,1}
    dual3(p, v) = SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],))),
                          D1(p[3], ForwardDiff.Partials((v[3],))))
    N = 2000
    pts = helix_points(rng, N)
    sd = [dual3(p, randn(rng, P3)) for p in pts]
    τ = [(randn(rng, P3), Doublet(0.01 * randn(rng, P3))) for _ in 1:N]
    f = FMM(StokesHarmonics(KSH; p=16), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, sd, τ)
    us = evaluate_at_sources(f, st)
    usref = direct_self(KSH, sd, τ)
    vals(u) = [ForwardDiff.value.(x) for x in u]
    parts(u) = [ForwardDiff.partials.(x, 1) for x in u]
    @test mrd(vals(us), vals(usref)) < 1e-5
    @test mrd(parts(us), parts(usref)) < 1e-4
end
