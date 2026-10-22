# Device evaluation on the KernelAbstractions CPU backend: the kernel reproduces the
# threaded evaluation for every output and density type, at targets and at the
# sources, in Float32 and with dual positions.
using FlexibleFMM: locate
rng = MersenneTwister(44)
dvals(u) = [ForwardDiff.value.(x) for x in u]
dparts(u) = [ForwardDiff.partials.(x, 1) for x in u]

@testset "device evaluation, $(name)" for (name, K, dens, m) in (
        ("Laplace 3D", Laplace3D(), n -> randn(rng, n), CubeKernelIndependent{3}(Laplace3D(); p=6)),
        ("Stokes 3D", Stokeslet3D(),
         n -> [(randn(rng, P3), Stresslet(randn(rng, P3), normalize(randn(rng, P3))), Doublet(randn(rng, P3))) for _ in 1:n],
         CubeKernelIndependent{3}(Stokeslet3D(); p=6)),
        ("Laplace 2D, circles", Laplace2D(), n -> randn(rng, n), KernelIndependent(Laplace2D(); nequiv=30)))
    D = FlexibleFMM.dimension(m)
    N = 3000
    pts = D == 3 ? mixed_points3(rng, N) : mixed_points(rng, N)
    τ = dens(N)
    tg = D == 3 ? uniform_points3(rng, 700) : uniform_points(rng, 700)
    f = FMM(m, pts; ppl=30, bbox=(zero(eltype(pts)), 1.0))
    st = build_expansions(f, pts, τ)
    dev = DeviceEvaluator(f)                      # `to = identity`: the CPU backend
    ds = device_state(st)
    nodes = Int32.(locate(f.tree, tg))
    outputs = K isa Stokeslet3D ? (Potential(), Pressure(), Gradient()) : (Potential(), Gradient())
    for out in outputs
        ref = evaluate(f, st, tg; output=out)
        u = similar(ref)
        evaluate!(u, dev, ds, tg, nodes; output=out)
        @test mrd(u, ref) < 1e-13
        refs = evaluate_at_sources(f, st; output=out)
        us = similar(refs)
        evaluate_at_sources!(us, dev, ds; output=out)
        @test mrd(us, refs) < 1e-13
    end
    @test_throws DimensionMismatch evaluate!(similar(tg, Float64, 10), dev, ds, tg, nodes)
end

@testset "device evaluation in Float32 and with dual positions" begin
    N = 3000
    pts = uniform_points3(rng, N)
    τ = randn(rng, N)
    tg = uniform_points3(rng, 500)
    # Float32 kernel and points
    Kf = Laplace3D(Float32)
    pts32 = SVector{3,Float32}.(pts); tg32 = SVector{3,Float32}.(tg)
    f = FMM(CubeKernelIndependent{3}(Kf; p=6), pts32; ppl=30, bbox=(zero(SVector{3,Float32}), 1f0))
    st = build_expansions(f, pts32, Float32.(τ))
    dev = DeviceEvaluator(f); ds = device_state(st)
    u = zeros(Float32, 500)
    evaluate!(u, dev, ds, tg32, Int32.(locate(f.tree, tg32)))
    @test eltype(u) == Float32 && mrd(u, evaluate(f, st, tg32)) < 1e-5
    # dual sources: directional derivatives through the kernel
    D1 = ForwardDiff.Dual{:dev,Float64,1}
    V = [randn(rng, P3) for _ in 1:N]
    sd = [SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],))),
                  D1(p[3], ForwardDiff.Partials((v[3],)))) for (p, v) in zip(pts, V)]
    f = FMM(CubeKernelIndependent{3}(Laplace3D(); p=6), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, sd, τ)
    dev = DeviceEvaluator(f); ds = device_state(st)
    ref = evaluate(f, st, tg)
    u = similar(ref)
    evaluate!(u, dev, ds, tg, Int32.(locate(f.tree, tg)))
    @test mrd(dvals(u), dvals(ref)) < 1e-13 && mrd(dparts(u), dparts(ref)) < 1e-13
    refs = evaluate_at_sources(f, st)
    us = similar(refs)
    evaluate_at_sources!(us, dev, ds)
    @test mrd(dvals(us), dvals(refs)) < 1e-13 && mrd(dparts(us), dparts(refs)) < 1e-13
end

@testset "Int32 index storage with the identity mover" begin
    rngi = MersenneTwister(3)
    pts = [rand(rngi, SVector{2,Float64}) for _ in 1:500]
    f = FMM(KernelIndependent(Laplace2D(); nequiv=20), pts; ppl=30, bbox=(SVector(0.0, 0.0), 1.0))
    dev = DeviceEvaluator(f)
    @test eltype(dev.colleagues) == Int32 && eltype(dev.plo) == Int32 && eltype(dev.point_node) == Int32
end
