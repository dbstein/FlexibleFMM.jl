# Square surfaces and spectral translations, exercised in 2D (the code is
# dimension generic): the lattice bookkeeping, spectral versus dense agreement on
# every offset, the operator checks, end-to-end sums for scalar, complex,
# non-symmetric and Stokes kernels, duals and complex densities through the FFT
# part-splitting, the Stokes augmentations, shared operators for a homogeneous
# kernel, and the accuracy against the margin d and the circle method.
using FlexibleFMM: CubeKernelIndependent, SquareKernelIndependent, cube_boundary_points, level_operators, far_offsets,
    multipole_length, local_length, partial_local_length, expansion_length, nlevels,
    _nparts, _cpart, _cassemble, SpectralM2L, apply!, conditioning, locate, evaluate_target,
    s2m!, l2t, center, nsurface

rng = MersenneTwister(17)
const DH2 = ForwardDiff.Dual{:square,Float64,2}

@testset "lattice bookkeeping" begin
    for (D, p) in ((2, 2), (2, 5), (2, 8), (3, 4), (3, 6))
        pts, embed = cube_boundary_points(Val(D), p, 0.7)
        @test length(pts) == p^D - (p - 2)^D
        @test allunique(embed) && all(1 .<= embed .<= (2p)^D)
        @test all(maximum(abs, x) ≈ 0.7 for x in pts)       # all on the boundary
        @test all(minimum(x) >= -0.7 - 1e-15 && maximum(x) <= 0.7 + 1e-15 for x in pts)
        @test pts[2][1] > pts[1][1]                          # coordinate 1 fastest
    end
    @test nsurface(SquareKernelIndependent(Laplace2D(); p=8)) == 28
    @test nsurface(CubeKernelIndependent{3}(Laplace2D(); p=6)) == 152
    @test_throws ArgumentError SquareKernelIndependent(Laplace2D(); p=1)
    @test_throws ArgumentError SquareKernelIndependent(Laplace2D(); d=1.5)
end

@testset "parts of expansion scalars round trip" begin
    d = DH2(1.5, ForwardDiff.Partials((0.25, -2.0)))
    for x in (1.5, 1.5 + 0.5im, d, Complex(d, 2d))
        S = typeof(x)
        n = _nparts(S)
        @test _cassemble(S, ntuple(k -> _cpart(x, k), n)) == x
    end
    @test _nparts(Float64) == 1 && _nparts(ComplexF64) == 1 && _nparts(DH2) == 3 && _nparts(Complex{DH2}) == 3
end

@testset "spectral M2L equals the dense kernel matrix: $(name)" for (name, K) in (
        ("Laplace", Laplace2D()), ("Helmholtz", Helmholtz2D(4.0)),
        ("non-symmetric", LaplaceDipoleTest(0.7)), ("Stokes", Stokeslet2D()))
    m = SquareKernelIndependent(K; p=6, d=0.1)
    w = 0.37
    ops = level_operators(m, w)
    small = ops.extra.small
    N = length(small)
    for (f, off) in enumerate(far_offsets(Val(2)))
        o = SVector{2,Float64}(off) * w
        dense = FlexibleFMM._multipole_to_check_matrix(K, small .+ Ref(o), o, w, small, zero(o))
        x = randn(rng, multipole_length(m))
        out = zeros(eltype(dense), partial_local_length(m))
        apply!(out, ops.M2L[f], x, 2.0)
        @test out ≈ 2.0 * (dense * x) rtol = 1e-12
        # dual and complex expansions go through the part splitting
        xd = [DH2(x[i], ForwardDiff.Partials((rand(rng), rand(rng)))) for i in eachindex(x)]
        outd = zeros(promote_type(eltype(dense), DH2), partial_local_length(m))
        apply!(outd, ops.M2L[f], xd)
        @test outd ≈ dense * xd
        xc = x .+ im .* randn(rng, length(x))
        outc = zeros(promote_type(eltype(dense), ComplexF64), partial_local_length(m))
        apply!(outc, ops.M2L[f], xc)
        @test outc ≈ dense * xc
    end
end

@testset "operator checks: $(name)" for (name, K) in (("Laplace", Laplace2D()),
        ("non-symmetric", LaplaceDipoleTest(0.6)), ("Helmholtz", Helmholtz2D(3.0)))
    m = SquareKernelIndependent(K; p=12, d=0.1)
    @test expansion_length(m) == 44
    check_m2l_all_offsets(rng, m, 0.37, 1e-6)
    check_m2m(rng, m, 0.61, 1e-6)
    check_l2l(rng, m, 0.61, 1e-6)
    check_allocation_free(rng, m, 0.3; m2l=false)
end

@testset "Stokes operator checks" begin
    # Stokes is about 100× less accurate than Laplace at equal p (two components and the
    # log part), doublets another 10×: p = 16 (60 points) for 1e-6 on all of them
    m = SquareKernelIndependent(Stokeslet2D(); p=16, d=0.1)
    @test multipole_length(m) == 121 && local_length(m) == 120 && partial_local_length(m) == 121
    check_m2l_all_offsets(rng, m, 0.37, 1e-6)
    check_m2m(rng, m, 0.61, 1e-6)
    check_l2l(rng, m, 0.61, 1e-6)
    stresslets(rng, K, n) = [Stresslet(P2(randn(rng), randn(rng)), normalize(P2(randn(rng), randn(rng)))) for _ in 1:n]
    doublets(rng, K, n) = [Doublet(P2(randn(rng), randn(rng))) for _ in 1:n]
    check_m2l_all_offsets(rng, m, 0.37, 1e-6; density=stresslets)
    check_m2l_all_offsets(rng, m, 0.37, 1e-6; density=doublets)
    check_m2m(rng, m, 0.61, 1e-6; density=stresslets)
    check_l2l(rng, m, 0.61, 1e-6; density=stresslets)
end

@testset "end to end: $(name)" for (name, K, dens, tol) in (
        ("Laplace", Laplace2D(), n -> randn(rng, n), 1e-7),
        ("Laplace, complex densities", Laplace2D(), n -> randn(rng, ComplexF64, n), 1e-7),
        ("Helmholtz k=5", Helmholtz2D(5.0), n -> randn(rng, n), 1e-7),
        ("non-symmetric", LaplaceDipoleTest(0.8), n -> randn(rng, n), 1e-7),
        ("dipole sources", Laplace2D(), n -> [Dipole(P2(randn(rng), randn(rng))) for _ in 1:n], 1e-6))
    N = 5000
    pts = mixed_points(rng, N)
    τ = dens(N)
    targets = uniform_points(rng, 2000)
    m = SquareKernelIndependent(K; p=10, d=0.1)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    @test nlevels(f) >= 4
    st = build_expansions(f, pts, τ)
    @test mrd(evaluate(f, st, targets), direct(K, pts, τ, targets)) < tol
    @test mrd(evaluate_at_sources(f, st), direct_self(K, pts, τ)) < tol
    @test mrd(evaluate(f, st, targets; output=Gradient()), direct(K, pts, τ, targets; output=Gradient())) < 10tol
end

@testset "Stokes end to end with the augmentations" begin
    N = 5000
    pts = mixed_points(rng, N)
    stk = [P2(randn(rng), randn(rng)) for _ in 1:N]
    nu = [normalize(P2(randn(rng), randn(rng))) for _ in 1:N]
    dub = [P2(randn(rng), randn(rng)) for _ in 1:N]
    τ = [(stk[i], Stresslet(stk[i], nu[i]), Doublet(dub[i])) for i in 1:N]
    targets = uniform_points(rng, 2000)
    K = Stokeslet2D()
    m = SquareKernelIndependent(K; p=12, d=0.1)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test mrd(evaluate(f, st, targets), direct(K, pts, τ, targets)) < 1e-5
    @test mrd(evaluate_at_sources(f, st), direct_self(K, pts, τ)) < 1e-5
    @test mrd(evaluate(f, st, targets; output=Pressure()), direct(K, pts, τ, targets; output=Pressure())) < 1e-5
    # through the front end
    plan = StokesFMMPlan(pts; targets=targets, method=:square, p=12, ppl=40)
    vt = Vector{P2}(undef, 2000)
    stokes_fmm_eval!(vt, plan; stokeslets=stk, stresslets=stk, stressvecs=nu, doublets=dub)
    @test mrd(vt, direct(K, pts, τ, targets)) < 1e-5
end

@testset "dual positions through the spectral translation" begin
    N = 3000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    V = [P2(randn(rng), randn(rng)) for _ in 1:N]
    targets = uniform_points(rng, 500)
    K = Laplace2D()
    m = SquareKernelIndependent(K; p=10, d=0.1)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    D1 = ForwardDiff.Dual{:g1,Float64,1}
    sd = [SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],)))) for (p, v) in zip(pts, V)]
    st = build_expansions(f, sd, τ)
    u = evaluate(f, st, targets)
    uref = direct(K, sd, τ, targets)
    @test mrd(ForwardDiff.value.(u), ForwardDiff.value.(uref)) < 1e-7
    @test mrd(ForwardDiff.partials.(u, 1), ForwardDiff.partials.(uref, 1)) < 1e-6
    # complex kernel with dual positions: Complex{Dual} expansions
    H = Helmholtz2D(3.0)
    mh = SquareKernelIndependent(H; p=10, d=0.1)
    fh = FMM(mh, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    sth = build_expansions(fh, sd, τ)
    uh = evaluate(fh, sth, targets)
    uhref = direct(H, sd, τ, targets)
    @test eltype(uh) == Complex{D1}
    @test mrd([ForwardDiff.partials(real(z), 1) + im * ForwardDiff.partials(imag(z), 1) for z in uh],
              [ForwardDiff.partials(real(z), 1) + im * ForwardDiff.partials(imag(z), 1) for z in uhref]) < 1e-6
end

@testset "shared operators for a homogeneous kernel, Float32, determinism" begin
    m = SquareKernelIndependent(HomogeneousDipole2D(); p=8, d=0.1)
    pts = uniform_points(rng, 4000)
    τ = randn(rng, 4000)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    @test all(f.ops[l].M2L[3].tables[1] === f.ops[1].M2L[3].tables[1] for l in 2:nlevels(f))
    st = build_expansions(f, pts, τ)
    targets = uniform_points(rng, 1000)
    @test max_relative_diff(evaluate(f, st, targets), direct(HomogeneousDipole2D(), pts, τ, targets)) < 1e-7
    @test evaluate(f, st, targets) == evaluate(f, build_expansions(f, pts, τ), targets)
    mf = SquareKernelIndependent(Laplace2D(Float32); p=8, d=0.1)
    ptsf = [SVector{2,Float32}(p) for p in pts]
    ff = FMM(mf, ptsf; ppl=30, bbox=(SVector{2,Float32}(0, 0), 1f0))
    stf = build_expansions(ff, ptsf, Float32.(τ))
    uf = evaluate(ff, stf, [SVector{2,Float32}(t) for t in targets])
    @test eltype(uf) == Float32
    @test max_relative_diff(uf, direct(Laplace2D(), pts, τ, targets)) < 1e-4
end

@testset "accuracy against the margin and against circles" begin
    # worst case: a unit charge at the near corner of the source box at offset (2, 0),
    # targets on the facing edge of the target box
    K = Laplace2D()
    w = 1.0
    off = SVector(2, 0)
    cs = SVector{2,Float64}(off) * w
    xs = [cs + P2(-0.5w, 0.5w)]; τ = [1.0]
    ts = [P2(0.5w, 0.5w), P2(0.5w, 0.0), P2(0.5w, -0.5w), P2(0.25w, 0.5w)]
    uex = direct(K, xs, τ, ts)
    worst(m) = begin
        ops = level_operators(m, w)
        f = findfirst(==(off), far_offsets(Val(2)))
        M = multipole_of(m, ops, xs, τ, cs)
        pL = zeros(partial_local_length(m)); apply!(pL, ops.M2L[f], M)
        L = local_of(m, ops, pL)
        maximum(abs.([l2t(m, ops, L, t) for t in ts] .- uex)) / maximum(abs.(uex))
    end
    rows = String[]
    for p in (6, 8, 10, 12)
        N = 4p - 4
        eg = [worst(SquareKernelIndependent(K; p=p, d=d)) for d in (0.05, 0.1, 0.2)]
        ec = worst(KernelIndependent(K; nequiv=N))
        push!(rows, "p = $p ($N points): square d=0.05 $(round(eg[1], sigdigits=2)), d=0.1 $(round(eg[2], sigdigits=2)), d=0.2 $(round(eg[3], sigdigits=2)); circles $(round(ec, sigdigits=2))")
        @test eg[2] < 10 * ec + 1e-13     # the square is not much worse than the circle at equal point count
    end
    @info "Laplace worst-case translation error, square surfaces vs circles\n" * join(rows, "\n")
    @test worst(SquareKernelIndependent(K; p=12, d=0.1)) < 2e-8
    @test worst(SquareKernelIndependent(K; p=12, d=0.05)) < 1e-8
end
