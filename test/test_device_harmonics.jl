# The device passes and evaluation for the harmonic methods on the CPU backend:
# the multipoles of the S2M kernel against the CPU pass, the full matvec and the
# evaluation at targets (potential, gradient; velocity, pressure, velocity
# gradient) against the CPU results, dense and low-rank representatives, Float32.
using FlexibleFMM: DeviceHarmonic, DeviceSurface, nlevels

rng = MersenneTwister(45)
const KDH = Laplace3D()
const KDS = Stokeslet3D()

@testset "Laplace: $(label)" for (label, m) in (("dense p = 10", SphericalHarmonics(KDH; p=10)),
                                              ("low rank p = 12", SphericalHarmonics(KDH; p=12, tol=1e-9)))
    N = 4000
    pts = mixed_points3(rng, N)
    τ = randn(rng, N)
    targets = uniform_points3(rng, 300)
    f = FMM(m, pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f)
    @test plan.dev.far isa DeviceHarmonic && !plan.pinv && plan.nb == 1
    @test length(plan.M2Lc[3]) == 34
    ds = (sources=st.sources, τ=st.τ)
    device_build_expansions!(plan, ds; sorted=true)
    @test maximum(abs, plan.multipoles .- st.multipoles) < 1e-13 * maximum(abs, st.multipoles)
    @test maximum(abs, plan.locals .- st.locals) < 1e-11 * maximum(abs, st.locals)
    out = zeros(N)
    device_matvec!(out, plan, ds; sorted=true)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-11
    nodes = Int32.(locate(f.tree, targets))
    dsl = (locals=plan.locals, sources=ds.sources, τ=ds.τ)
    u = zeros(300)
    evaluate!(u, plan.dev, dsl, targets, nodes)
    @test mrd(u, evaluate(f, st, targets)) < 1e-11
    g = zeros(P3, 300)
    evaluate!(g, plan.dev, dsl, targets, nodes; output=Gradient())
    @test mrd(g, evaluate(f, st, targets; output=Gradient())) < 1e-10
    # dipoles through the device S2M
    τd = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:N]
    std = build_expansions(f, pts, τd)
    outd = zeros(N)
    device_matvec!(outd, plan, (sources=std.sources, τ=std.τ); sorted=true)
    @test mrd(outd, evaluate_at_sources(f, std)) < 1e-11
end

@testset "Stokes: $(label)" for (label, m) in (("dense p = 8", StokesHarmonics(KDS; p=8)),
                                             ("low rank p = 10", StokesHarmonics(KDS; p=10, tol=1e-9)))
    N = 3000
    pts = helix_points(rng, N)
    τ = [(randn(rng, P3), Stresslet(randn(rng, P3), normalize(randn(rng, P3))), Doublet(randn(rng, P3))) for _ in 1:N]
    targets = uniform_points3(rng, 200)
    f = FMM(m, pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f)
    @test plan.nb == 4 && all(size(G) == ((m.p + 1)^2, 4len) for (len, G) in plan.Gc)
    ds = (sources=st.sources, τ=st.τ)
    device_build_expansions!(plan, ds; sorted=true)
    @test maximum(abs, plan.multipoles .- st.multipoles) < 1e-13 * maximum(abs, st.multipoles)
    @test maximum(abs, plan.locals .- st.locals) < 1e-11 * maximum(abs, st.locals)
    out = zeros(P3, N)
    device_matvec!(out, plan, ds; sorted=true)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-11
    nodes = Int32.(locate(f.tree, targets))
    dsl = (locals=plan.locals, sources=ds.sources, τ=ds.τ)
    for (output, T) in ((Potential(), P3), (Pressure(), Float64), (Gradient(), SMatrix{3,3,Float64,9}))
        u = zeros(T, 200)
        evaluate!(u, plan.dev, dsl, targets, nodes; output=output)
        @test mrd(u, evaluate(f, st, targets; output=output)) < 1e-10
    end
end

@testset "Float32 and the kernel-independent path unchanged" begin
    N = 3000
    pts = [SVector{3,Float32}(x) for x in uniform_points3(rng, N)]
    τ = randn(rng, Float32, N)
    f = FMM(SphericalHarmonics(Laplace3D(Float32); p=12), pts; ppl=40, bbox=(SVector{3,Float32}(0, 0, 0), 1f0))
    st = build_expansions(f, pts, τ)
    plan = DevicePlan(f)
    @test eltype(plan.multipoles) == Float32
    out = zeros(Float32, N)
    device_matvec!(out, plan, (sources=st.sources, τ=st.τ); sorted=true)
    @test mrd(out, evaluate_at_sources(f, st)) < 1e-5
    fc = FMM(CubeKernelIndependent{3}(KDH; p=6), P3.(pts); ppl=40, bbox=(P3(0, 0, 0), 1.0))
    stc = build_expansions(fc, P3.(pts), Float64.(τ))
    planc = DevicePlan(fc)
    @test planc.dev.far isa DeviceSurface && planc.pinv && planc.nb == 1
    outc = zeros(N)
    device_matvec!(outc, planc, (sources=stc.sources, τ=stc.τ); sorted=true)
    @test mrd(outc, evaluate_at_sources(fc, stc)) < 1e-11
    @test_throws ArgumentError DevicePlan(FMM(BlackBox(Laplace2D(); p=6), [rand(rng, SVector{2,Float64}) for _ in 1:500]; ppl=30))
end
