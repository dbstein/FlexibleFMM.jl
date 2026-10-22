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
