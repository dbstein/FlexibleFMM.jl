# The expansion passes on the KernelAbstractions CPU backend against the CPU driver:
# multipoles, locals (the dense class translations against the spectral ones) and
# the self evaluation, for scalar and Stokes kernels in 3D and circles in 2D.
rng = MersenneTwister(45)

@testset "device passes, $(name)" for (name, K, dens, m, tol) in (
        ("Laplace 3D", Laplace3D(), n -> randn(rng, n), CubeKernelIndependent{3}(Laplace3D(); p=6), 1e-11),
        ("Stokes 3D", Stokeslet3D(),
         n -> [(randn(rng, P3), Stresslet(randn(rng, P3), normalize(randn(rng, P3))), Doublet(randn(rng, P3))) for _ in 1:n],
         CubeKernelIndependent{3}(Stokeslet3D(); p=5), 1e-10),
        ("Yukawa 3D", Yukawa3D(2.0), n -> randn(rng, n), CubeKernelIndependent{3}(Yukawa3D(2.0); p=5), 1e-11),
        ("Helmholtz 3D (complex)", Helmholtz3D(2.0), n -> randn(rng, n), CubeKernelIndependent{3}(Helmholtz3D(2.0); p=5), 1e-10),
        ("Laplace 2D, circles", Laplace2D(), n -> randn(rng, n), KernelIndependent(Laplace2D(); nequiv=24), 1e-11))
    D = FlexibleFMM.dimension(m)
    N = 4000
    pts = D == 3 ? mixed_points3(rng, N) : mixed_points(rng, N)
    τ = dens(N)
    f = FMM(m, pts; ppl=30, bbox=(zero(eltype(pts)), 1.0))
    st = build_expansions(f, pts, τ)
    @test FlexibleFMM.nlevels(f) >= 5
    plan = DevicePlan(f)
    ds = (sources=st.sources[1], τ=st.τ[1])
    device_build_expansions!(plan, ds; sorted=true)
    @test mrd(vec(plan.multipoles), vec(st.multipoles)) < tol
    @test mrd(vec(plan.plocals), vec(st.plocals)) < tol
    @test mrd(vec(plan.locals), vec(st.locals)) < 100tol     # the downward pseudo-inverse amplifies the dense-vs-spectral difference
    u = similar(evaluate_at_sources(f, st))
    device_matvec!(u, plan, ds; sorted=true)
    @test mrd(u, evaluate_at_sources(f, st)) < tol
    @test mrd(u, direct_self(K, pts, τ)) < 1e-4
end

@testset "device passes with low-rank class operators, $(name)" for (name, K, dens, m, tol, errtol) in (
        ("Laplace 3D, tol 1e-8", Laplace3D(), n -> randn(rng, n), CubeKernelIndependent{3}(Laplace3D(); p=8), 1e-8, 1e-7),
        ("Stokes 3D, tol 1e-6", Stokeslet3D(),
         n -> [(randn(rng, P3), Stresslet(randn(rng, P3), normalize(randn(rng, P3))), Doublet(randn(rng, P3))) for _ in 1:n],
         CubeKernelIndependent{3}(Stokeslet3D(); p=8), 1e-6, 1e-5))
    N = 4000
    pts = uniform_points3(rng, N)
    τ = dens(N)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f; tol=tol)
    @test !isempty(plan.M2Lb[3]) && all(size(B, 1) < size(B, 2) ÷ 2 for B in plan.M2Lb[3])   # ranks well below n
    ds = (sources=st.sources[1], τ=st.τ[1])
    u = similar(evaluate_at_sources(f, st))
    device_matvec!(u, plan, ds; sorted=true)
    @test mrd(u, evaluate_at_sources(f, st)) < errtol       # the compression sits below the method's accuracy
    @test mrd(u, direct_self(K, pts, τ)) < 10errtol
end

@testset "review: covariance, per-level ranks, complex densities" begin
    using FlexibleFMM: covariant, M2LOperator
    @test covariant(Laplace3D()) && covariant(Stokeslet3D()) && covariant(Yukawa3D(2.0)) && covariant(Laplace2D())
    @test !covariant(LaplaceDipoleTest(0.8)) && !covariant((s, t) -> 1.0)
    rngr = MersenneTwister(77)
    # a translation-invariant kernel without rotational covariance takes one operator per offset
    K = LaplaceDipoleTest(0.8)
    pts = [rand(rngr, SVector{2,Float64}) for _ in 1:700]
    τ = randn(rngr, 700)
    f = FMM(SquareKernelIndependent(K; p=8), pts; ppl=30, bbox=(SVector(0.0, 0.0), 1.0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f)
    @test length(plan.M2Lc[3]) == 40
    out = zeros(700)
    device_matvec!(out, plan, (sources=st.sources[1], τ=st.τ[1]); sorted=true)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-10
    fc = FMM(SquareKernelIndependent(Laplace2D(); p=8), pts; ppl=30, bbox=(SVector(0.0, 0.0), 1.0))
    @test length(DevicePlan(fc).M2Lc[3]) == 7
    # low-rank workspaces follow each level's ranks for a kernel built per width
    pts3 = uniform_points3(rngr, 3000)
    τ3 = randn(rngr, 3000)
    fy = FMM(CubeKernelIndependent{3}(Yukawa3D(20.0); p=4), pts3; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    sty = build_expansions(fy, pts3, τ3)
    plany = DevicePlan(fy; tol=1e-5)
    outy = zeros(3000)
    device_matvec!(outy, plany, (sources=sty.sources[1], τ=sty.τ[1]); sorted=true)
    @test mrd(outy, evaluate_at_sources(fy, sty)) < 1e-4
    # complex densities with a real kernel: the plan's element type
    τc = randn(rngr, ComplexF64, 3000)
    fl = FMM(CubeKernelIndependent{3}(Laplace3D(); p=6), pts3; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    stc = build_expansions(fl, pts3, τc)
    planc = DevicePlan(fl; eltype=ComplexF64)
    @test eltype(planc.multipoles) == ComplexF64
    outc = zeros(ComplexF64, 3000)
    device_matvec!(outc, planc, (sources=stc.sources[1], τ=stc.τ[1]); sorted=true)
    @test mrd(outc, evaluate_at_sources(fl, stc)) < 1e-10
    @test_throws ArgumentError DevicePlan(fl; eltype=Float32)
end

@testset "device locals of the real leaves only" begin
    rngl = MersenneTwister(92)
    pts = mixed_points3(rngl, 4000)
    τ = randn(rngl, 4000)
    f = FMM(CubeKernelIndependent{3}(Laplace3D(); p=6), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f)
    ds = (sources=st.sources[1], τ=st.τ[1])
    device_build_expansions!(plan, ds; sorted=true)
    # the pseudo-inverse amplifies roundoff differences of the two paths (1e-10 relative
    # in the locals at this order); the fields agree
    out = zeros(4000)
    device_matvec!(out, plan, ds; sorted=true)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-10
    nonleaf = [i for i in 1:length(f.tree.leaf) if !f.tree.leaf[i]]
    @test all(i -> all(iszero, view(plan.locals, :, i)), nonleaf)
    # one column per real leaf, rounded up to the padding grid of the GEMM buffers
    @test all(l -> size(plan.Ldn[l], 2) == FlexibleFMM._pad_cols(max(1, length(f.leaf_nodes[l]))), 1:plan.nlevels)
end

@testset "the device-built lists equal the host's, and inputs in the original order" begin
    using FlexibleFMM: class_lists, far_offsets, pair_range
    rng = MersenneTwister(31)
    pts = mixed_points3(rng, 20000)
    box = (P3(0, 0, 0), 1.0)
    for m in (CubeKernelIndependent{3}(Laplace3D(); p=4), SphericalHarmonics(Laplace3D(); p=6))
        f = FMM(m, pts; ppl=60, bbox=box)
        plan = DevicePlan(f)
        tree = f.tree
        ops3 = f.ops[min(3, FlexibleFMM.nlevels(f))]
        sc = m isa SphericalHarmonics ? ops3.extra.sc :                 # the classes the plan uses
             FlexibleFMM.class_operators(m.kernel, ops3.extra.small, ops3.extra.embed, ops3.width, ops3.extra.p)[1]
        nclass = length(sc.reps)
        for l in 3:FlexibleFMM.nlevels(f)
            il = tree.ilists[l]
            li = plan.lists[l]
            lo = first(tree.level_range[l])
            # the (target, source, offset) triples
            host = sort!([(il.tgt[q], il.src[q], fi) for fi in eachindex(sc.class) for q in pair_range(il, fi)])
            dev = sort!([(lo + Int(li.rtgt[r]) - 1, Int(li.src[p]), Int(li.f[p])) for r in 1:length(li.rtgt) for p in li.rptr[r]:li.rptr[r+1]-1])
            @test host == dev
            # the class lists: the same pairs per class (the device orders a class's targets
            # by node, the host by Morton key; each target's pairs are consecutive in both)
            cl = class_lists(il, sc.class, nclass, lo)
            pos = 1
            for c in 1:nclass
                n = length(cl[c])
                @test sort(collect(zip(Int.(li.src[pos:pos+n-1]), Int.(li.f[pos:pos+n-1])))) == sort(collect(zip(Int.(cl[c].src), Int.(cl[c].f))))
                pos += n
            end
            # the chunks cover each class's pairs once, with the runs of their pairs
            for ch in plan.chunks[l]
                @test ch.npk <= ch.len && ch.r1 <= ch.r2
                @test li.rptr[ch.r1] <= ch.a && (ch.r2 == length(li.rtgt) || li.rptr[ch.r2+1] > ch.a + ch.npk - 1)
            end
            @test sum(ch.npk for ch in plan.chunks[l]; init=0) == length(il.tgt)
        end
        # the matvec from inputs in the original order equals the one from the tree's order
        τ = randn(rng, length(pts))
        st = build_expansions(f, pts, τ)
        u1 = device_matvec!(similar(τ), plan, (sources=st.sources[1], τ=st.τ[1]); sorted=true)
        u2 = device_matvec!(similar(τ), plan, (sources=pts, τ=τ))
        @test u1 == u2
        @test mrd(u2, evaluate_at_sources(f, st)) < 1e-10
    end
end

@testset "plan robustness: bounds, non-finite inputs, padded buffers, shared device operators" begin
    rngr = MersenneTwister(93)
    # the size-agnostic bounds against the host's, around the work-item count
    for n in (1, 7, 16384, 16385, 50003), T in (Float64, Float32)
        xs = [rand(rngr, SVector{3,T}) .- T(0.5) for _ in 1:n]
        lo, hi = FlexibleFMM._device_bounds(xs)
        @test lo == reduce((a, b) -> min.(a, b), xs) && hi == reduce((a, b) -> max.(a, b), xs)
    end
    @test_throws ArgumentError FlexibleFMM._device_bounds(P3[])
    # the padding grid: at least the count, at most an eighth more, 16 at least, a fixed point,
    # eight values per octave
    pad = FlexibleFMM._pad_cols
    @test all(n -> pad(n) >= n && pad(pad(n)) == pad(n), 1:5000) && pad(1) == 16
    @test all(n -> pad(n) <= n + n ÷ 8, 17:100_000)
    @test length(unique(pad.(17:2^20))) == 8 * 16
    # a non-finite point is reported as such, on the host and from device points
    m = CubeKernelIndependent{3}(Laplace3D(); p=4)
    N = 3000
    pts = mixed_points3(rngr, N)
    τ = randn(rngr, N)
    box = (P3(0, 0, 0), 1.0)
    for bad in (P3(NaN, 0.5, 0.5), P3(0.5, Inf, 0.5))
        q = copy(pts)
        q[17] = bad
        e1 = try FMM(m, q; ppl=40, bbox=box); nothing catch e; e end
        e2 = try DevicePlan(m, q; ppl=40, bbox=box); nothing catch e; e end
        @test e1 isa ArgumentError && occursin("finite", e1.msg)
        @test e2 isa ArgumentError && occursin("finite", e2.msg)
    end
    # a plan recovers from a non-finite density: the next application is a clean plan's, bitwise
    # (the backends that read a GEMM's output at β = 0, Metal's, are checked in bench/gpu/metal_checks.jl)
    cache = OperatorCache(m)
    plan = DevicePlan(m, pts; ppl=40, bbox=box, cache=cache)
    ds = (sources=pts, τ=τ)
    u0 = device_matvec!(zeros(N), plan, ds)
    τbad = copy(τ)
    τbad[5] = NaN
    @test any(isnan, device_matvec!(zeros(N), plan, (sources=pts, τ=τbad)))
    @test device_matvec!(zeros(N), plan, ds) == u0
    # plans built with one cache share the operators' device copies and add nothing to it;
    # the rescaled levels of a homogeneous kernel share one copy of their common matrices
    n1 = length(cache.device)
    plan2 = DevicePlan(m, pts; ppl=40, bbox=box, cache=cache)
    L = plan.nlevels
    @test length(cache.device) == n1
    @test all(l -> plan2.M2M[l] === plan.M2M[l] && plan2.Uh_dn[l] === plan.Uh_dn[l] && plan2.invS_up[l] === plan.invS_up[l], 1:L)
    @test all(l -> all(plan2.M2Lc[l] .=== plan.M2Lc[l]), 3:L)
    @test all(l -> plan.M2M[l] === plan.M2M[1] && plan.L2L[l] === plan.L2L[1], 1:L)
    # a plan from a fresh cache gives the same result, bitwise
    @test device_matvec!(zeros(N), DevicePlan(m, pts; ppl=40, bbox=box), ds) == u0
    # the low-rank factors are computed once per tolerance
    mS = CubeKernelIndependent{3}(Stokeslet3D(); p=5)
    cS = OperatorCache(mS)
    τS = [randn(rngr, P3) for _ in 1:N]
    pa = DevicePlan(mS, pts; ppl=40, bbox=box, cache=cS, tol=1e-6)
    pb = DevicePlan(mS, pts; ppl=40, bbox=box, cache=cS, tol=1e-6)
    pq = DevicePlan(mS, pts; ppl=40, bbox=box, cache=cS, tol=1e-4)
    @test pb.M2Lb[3][1] === pa.M2Lb[3][1] && pq.M2Lb[3][1] !== pa.M2Lb[3][1]
    ua = device_matvec!(similar(τS), pa, (sources=pts, τ=τS))
    @test device_matvec!(similar(τS), pb, (sources=pts, τ=τS)) == ua
    # the harmonic methods: the class representatives, and the Stokes method's dense M2M and L2L
    for mh in (SphericalHarmonics(Laplace3D(); p=8), StokesHarmonics(Stokeslet3D(); p=6))
        ch = OperatorCache(mh)
        h1 = DevicePlan(mh, pts; ppl=40, bbox=box, cache=ch)
        nh = length(ch.device)
        h2 = DevicePlan(mh, pts; ppl=40, bbox=box, cache=ch)
        @test length(ch.device) == nh
        @test all(h2.M2Lc[3] .=== h1.M2Lc[3]) && all(l -> h2.M2M[l] === h1.M2M[l] && h2.L2L[l] === h1.L2L[l], 1:h1.nlevels)
    end
end
