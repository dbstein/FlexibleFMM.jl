# Homogeneous kernels: the translation operators of every width are scalar multiples
# of one reference set, which the cache shares between levels. None of the built-in
# 2D kernels is homogeneous, so two test kernels exercise the path: a harmonic
# x-dipole kernel (degree -1) for the kernel-independent method and 1/r² (degree -2,
# not a PDE kernel) for the black-box method.
using FlexibleFMM: AbstractKernel, floattype, homogeneity, level_operators, rescale, far_offsets,
    nlevels, OperatorCache, get_operators!, fit_box!, operator_eltype, conditioning, Matrix

struct HomogeneousDipole2D <: AbstractKernel end
@inline function (::HomogeneousDipole2D)(s::SVector{2}, t::SVector{2})
    d = t - s
    return d[1] / (2π * sum(abs2, d))
end
FlexibleFMM.homogeneity(::HomogeneousDipole2D) = -1

struct InverseSquare2D <: AbstractKernel end
@inline (::InverseSquare2D)(s::SVector{2}, t::SVector{2}) = 1 / sum(abs2, t - s)
FlexibleFMM.homogeneity(::InverseSquare2D) = -2

rng = MersenneTwister(16)

@testset "homogeneity identities" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3); λ = 3.7
    @test HomogeneousDipole2D()(λ * s, λ * t) ≈ λ^-1 * HomogeneousDipole2D()(s, t)
    @test InverseSquare2D()(λ * s, λ * t) ≈ λ^-2 * InverseSquare2D()(s, t)
    @test homogeneity(Laplace2D()) === nothing && homogeneity(Stokeslet2D()) === nothing
    # the dipole kernel is harmonic
    H = ForwardDiff.hessian(tt -> HomogeneousDipole2D()(s, SVector{2}(tt)), [0.6, -0.4])
    @test abs(tr(H)) < 1e-10 * norm(H)
end

@testset "rescaled operators equal freshly built ones: $(name)" for (name, m) in (
        ("KI, dipole kernel", KernelIndependent(HomogeneousDipole2D(); nequiv=40)),
        ("BB, 1/r²", BlackBox(InverseSquare2D(); p=10)))
    ref = level_operators(m, 1.0)
    for w in (0.25, 0.125, 2.0)
        fresh = level_operators(m, w)
        resc = rescale(ref, m, w)
        @test resc.width == w
        @test resc.M2M === ref.M2M && resc.L2L === ref.L2L && resc.M2L[1].A === ref.M2L[1].A
        @test resc.M2M * resc.m2m_scale ≈ fresh.M2M
        @test resc.L2L * resc.l2l_scale ≈ fresh.L2L
        @test all(resc.M2L[f].A * resc.m2l_scale ≈ fresh.M2L[f].A for f in eachindex(far_offsets(Val(2))))
        if m isa KernelIndependent
            @test resc.extra.small ≈ fresh.extra.small && resc.extra.large ≈ fresh.extra.large
            @test Matrix(resc.extra.P_up) * resc.extra.pinv_scale ≈ Matrix(fresh.extra.P_up)
            @test Matrix(resc.extra.P_down) * resc.extra.pinv_scale ≈ Matrix(fresh.extra.P_down)
        else
            @test resc.extra.nodes ≈ fresh.extra.nodes && resc.extra.scale ≈ fresh.extra.scale
            @test resc.m2m_scale == 1 && resc.l2l_scale == 1
        end
    end
    @test_throws ArgumentError rescale(level_operators(KernelIndependent(Laplace2D(); nequiv=20), 1.0),
                                       KernelIndependent(Laplace2D(); nequiv=20), 0.5)
end

@testset "the cache shares one matrix set across levels and boxes" begin
    m = KernelIndependent(HomogeneousDipole2D(); nequiv=40)
    c = OperatorCache(m)
    pts = uniform_points(rng, 4000)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0), cache=c)
    @test nlevels(f) >= 4
    for l in 2:nlevels(f)
        @test f.ops[l].M2L[7].A === f.ops[1].M2L[7].A
        @test f.ops[l].width == ldexp(1.0, 1 - l)
    end
    # a second box twice as large reuses the same matrices too
    g = FMM(m, 2 .* pts; ppl=30, bbox=(P2(0, 0), 2.0), cache=c)
    # the kernel-independent M2M carries its pseudo-inverse and is width-invariant
    @test g.ops[1].M2M === f.ops[1].M2M && g.ops[1].m2m_scale ≈ (m isa KernelIndependent ? 1.0 : 2.0^-1)
end

@testset "end to end with shared operators: $(name)" for (name, m, tol) in (
        ("KI, dipole kernel", KernelIndependent(HomogeneousDipole2D(); nequiv=60), 1e-6),
        ("BB, 1/r²", BlackBox(InverseSquare2D(); p=14), 1e-6))
    K = m.kernel
    N = 5000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 2000)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test max_relative_diff(evaluate(f, st, targets), direct(K, pts, τ, targets)) < tol
    @test max_relative_diff(evaluate_at_sources(f, st), direct_self(K, pts, τ)) < tol
    # the same with a box fitted to a different width, i.e. a non-reference exponent everywhere
    g = FMM(m, pts; ppl=40, bbox=(P2(-0.3, -0.3), 1.7), cache=f.cache)
    stg = build_expansions(g, pts, τ)
    @test max_relative_diff(evaluate(g, stg, targets), direct(K, pts, τ, targets)) < tol
end

@testset "dimension checks" begin
    m = KernelIndependent(Laplace2D(); nequiv=20)
    @test FlexibleFMM.dimension(m) == 2
    pts3 = [SVector(rand(rng), rand(rng), rand(rng)) for _ in 1:100]
    @test_throws ArgumentError FMM(m, pts3; ppl=10)
    @test_throws ArgumentError StokesFMMPlan(pts3; method=:ki)     # circles are 2D
end
