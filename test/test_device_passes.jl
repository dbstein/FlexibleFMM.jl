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
    ds = (sources=st.sources, τ=st.τ)
    device_build_expansions!(plan, ds)
    @test mrd(vec(plan.multipoles), vec(st.multipoles)) < tol
    @test mrd(vec(plan.plocals), vec(st.plocals)) < tol
    @test mrd(vec(plan.locals), vec(st.locals)) < 100tol     # the downward pseudo-inverse amplifies the dense-vs-spectral difference
    u = similar(evaluate_at_sources(f, st))
    device_matvec!(u, plan, ds)
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
    ds = (sources=st.sources, τ=st.τ)
    u = similar(evaluate_at_sources(f, st))
    device_matvec!(u, plan, ds)
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
    device_matvec!(out, plan, (sources=st.sources, τ=st.τ))
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
    device_matvec!(outy, plany, (sources=sty.sources, τ=sty.τ))
    @test mrd(outy, evaluate_at_sources(fy, sty)) < 1e-4
    # complex densities with a real kernel: the plan's element type
    τc = randn(rngr, ComplexF64, 3000)
    fl = FMM(CubeKernelIndependent{3}(Laplace3D(); p=6), pts3; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    stc = build_expansions(fl, pts3, τc)
    planc = DevicePlan(fl; eltype=ComplexF64)
    @test eltype(planc.multipoles) == ComplexF64
    outc = zeros(ComplexF64, 3000)
    device_matvec!(outc, planc, (sources=stc.sources, τ=stc.τ))
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
    ds = (sources=st.sources, τ=st.τ)
    device_build_expansions!(plan, ds)
    # the pseudo-inverse amplifies roundoff differences of the two paths (1e-10 relative
    # in the locals at this order); the fields agree
    out = zeros(4000)
    device_matvec!(out, plan, ds)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-10
    nonleaf = [i for i in 1:length(f.tree.leaf) if !f.tree.leaf[i]]
    @test all(i -> all(iszero, view(plan.locals, :, i)), nonleaf)
    @test all(l -> size(plan.Ldn[l], 2) == max(1, length(f.leaf_nodes[l])), 1:plan.nlevels)
end
