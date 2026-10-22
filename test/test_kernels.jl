using FlexibleFMM: kernel_eltype, _sqnorm

rng = MersenneTwister(1)

@testset "values and symmetry" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3)
    r = norm(t - s)
    @test Laplace2D()(s, t) ≈ -log(r) / (2π)
    @test Laplace2D()(s, t) == Laplace2D()(t, s)
    K = Helmholtz2D(3.0)
    @test K(s, t) ≈ (im / 4) * SpecialFunctions.hankelh1(0, 3r)
    @test K(s, t) == K(t, s)
    Y = Yukawa2D(2.0)
    @test Y(s, t) ≈ SpecialFunctions.besselk(0, 2r) / (2π)
    @test Y(s, t) == Y(t, s)
    N = LaplaceDipoleTest(0.5)
    @test N(s, t) ≈ -log(r) / (2π) + 0.5 * (t[1] - s[1]) / (2π * r^2)
    @test N(s, t) != N(t, s)
    @test N(s, t) + N(t, s) ≈ 2 * Laplace2D()(s, t)   # dipole part is odd
    # translation invariance
    d = P2(0.33, -0.21)
    for K in (Laplace2D(), Helmholtz2D(3.0), Yukawa2D(2.0), LaplaceDipoleTest(0.5))
        @test K(s + d, t + d) ≈ K(s, t)
    end
end

@testset "eltypes" begin
    @test kernel_eltype(Laplace2D(), P2, P2) == Float64
    @test kernel_eltype(Helmholtz2D(1.0), P2, P2) == ComplexF64
    @test kernel_eltype(Yukawa2D(1.0), P2, P2) == Float64
    @test kernel_eltype(LaplaceDipoleTest(1.0), P2, P2) == Float64
    @test Helmholtz2D(2) isa Helmholtz2D{Float64}
    @test FlexibleFMM.floattype(Laplace2D()) == Float64
end

@testset "harmonicity of the test kernel" begin
    # Laplacian in t of LaplaceDipoleTest vanishes away from the source
    K = LaplaceDipoleTest(0.7)
    s = P2(0.1, 0.2)
    f(t) = K(s, SVector{2}(t))
    H = ForwardDiff.hessian(f, [0.6, -0.4])
    @test abs(tr(H)) < 1e-10 * norm(H)
end

@testset "kernel_matrix and direct sums" begin
    ns, nt = 7, 5
    sources = uniform_points(rng, ns)
    targets = uniform_points(rng, nt)
    τ = randn(rng, ns)
    for K in (Laplace2D(), Helmholtz2D(2.0), Yukawa2D(1.5), LaplaceDipoleTest(0.3))
        A = kernel_matrix(K, sources, targets)
        @test size(A) == (nt, ns)
        @test all(A[j, i] == K(sources[i], targets[j]) for i in 1:ns, j in 1:nt)
        u = direct(K, sources, τ, targets)
        @test u ≈ A * τ
        out = similar(u)
        @test direct!(out, K, sources, τ, targets) ≈ u
        # self sum excludes the diagonal exactly
        B = kernel_matrix(K, sources, sources)
        for i in 1:ns
            B[i, i] = 0
        end
        @test direct_self(K, sources, τ) ≈ B * τ
    end
    # complex density with a real kernel
    τc = randn(rng, ComplexF64, ns)
    @test direct(Laplace2D(), sources, τc, targets) ≈ kernel_matrix(Laplace2D(), sources, targets) * τc
    @test_throws DimensionMismatch direct!(zeros(nt + 1), Laplace2D(), sources, τ, targets)
end

@testset "Dual positions" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3); v = P2(0.3, -0.8)
    for K in (Laplace2D(), Helmholtz2D(3.0), Yukawa2D(2.0), LaplaceDipoleTest(0.5))
        # directional derivative w.r.t. the target, against central differences
        g = ForwardDiff.derivative(ε -> K(s, t + ε * v), 0.0)
        h = 1e-6
        fd = (K(s, t + h * v) - K(s, t - h * v)) / (2h)
        @test abs(g - fd) < 1e-7 * (1 + abs(g))
        # and w.r.t. the source
        g2 = ForwardDiff.derivative(ε -> K(s + ε * v, t), 0.0)
        fd2 = (K(s + h * v, t) - K(s - h * v, t)) / (2h)
        @test abs(g2 - fd2) < 1e-7 * (1 + abs(g2))
    end
end

@testset "allocation free" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3)
    for K in (Laplace2D(), Helmholtz2D(3.0), Yukawa2D(2.0), LaplaceDipoleTest(0.5))
        K(s, t)
        @test @allocated(K(s, t)) == 0
    end
end

@testset "review: callable kernels, and kernels that reject coincident points" begin
    using FlexibleFMM: floattype, masks_self
    plain(s, t) = 1 / norm(t - s)
    @test floattype(plain) == Float64 && !masks_self(plain) && masks_self(Laplace2D())
    guarded(s, t) = (r = norm(t - s); r == 0 && error("coincident"); -log(r) / 2π)
    rngk = MersenneTwister(5)
    pts = [rand(rngk, SVector{2,Float64}) for _ in 1:300]
    τ = randn(rngk, 300)
    @test direct_self(guarded, pts, τ) ≈ direct_self(Laplace2D(), pts, τ)
    @test direct_self(guarded, pts[1:1], τ[1:1]) == [0.0]
    f = FMM(KernelIndependent(guarded; nequiv=30), pts; ppl=20, bbox=(SVector(0.0, 0.0), 1.0))
    st = build_expansions(f, pts, τ)
    @test mrd(evaluate_at_sources(f, st), direct_self(Laplace2D(), pts, τ)) < 1e-6
    fb = FMM(BlackBox(plain; p=6), pts; ppl=20, bbox=(SVector(0.0, 0.0), 1.0))
    @test fb isa FMM
end

@testset "the FastMath policy: relaxed bodies agree with the precise ones" begin
    using FlexibleFMM: gradient_t, hessian_t, pressure_kernel, source_field, target_field, policy, PreciseMath, FastMath
    close(a, b, tol) = maximum(abs.(a .- b)) <= tol * max(maximum(abs.(b)), floatmin(eltype(b)))
    for (T, tol) in ((Float64, 1e-12), (Float32, 1f-5))
        KL, KS = Laplace3D(T), Stokeslet3D(T)
        @test policy(KL) === PreciseMath() && policy(fastmath(KL)) === FastMath()
        @test fastmath(fastmath(KS)) === fastmath(KS) && fastmath(Laplace2D()) === Laplace2D()
        fL, fS = fastmath(KL), fastmath(KS)
        for _ in 1:20
            s, t = rand(rng, SVector{3,T}), rand(rng, SVector{3,T})
            f, ν, g = randn(rng, SVector{3,T}), randn(rng, SVector{3,T}), randn(rng, SVector{3,T})
            @test close(fL(s, t), KL(s, t), tol) && close(gradient_t(fL, s, t), gradient_t(KL, s, t), tol) &&
                  close(hessian_t(fL, s, t), hessian_t(KL, s, t), tol)
            @test close(fS(s, t), KS(s, t), tol) && close(pressure_kernel(fS, s, t), pressure_kernel(KS, s, t), tol)
            for τ in (f, Stresslet(f, ν), Doublet(g), Rotlet(g))
                @test close(source_field(fS, s, t, τ), source_field(KS, s, t, τ), tol)
                @test close(target_field(fS, s, t, τ, Gradient()), target_field(KS, s, t, τ, Gradient()), tol)
                τ isa Stresslet && continue
                @test close(target_field(fS, s, t, τ, Pressure()), target_field(KS, s, t, τ, Pressure()), tol)
            end
            # the stresslet pressure is the difference of two terms, so the relaxed
            # arithmetic deviates relative to the terms, not to the result (1e-4 of
            # the result seen in Float32)
            d = t - s; ir = 1 / norm(d)
            terms = 4 * KS.c * (abs(dot(f, ν)) * ir^3 + 3 * abs(dot(d, f) * dot(d, ν)) * ir^5)
            pf, pp = target_field(fS, s, t, Stresslet(f, ν), Pressure()), target_field(KS, s, t, Stresslet(f, ν), Pressure())
            @test abs(pf - pp) <= tol * terms
        end
    end
    # dual numbers (the relaxed root's derivative) and allocation
    s, f = rand(rng, SVector{3,Float64}), randn(rng, SVector{3,Float64})
    t = SVector(ForwardDiff.Dual(0.7, 1.0), ForwardDiff.Dual(0.1, 0.0), ForwardDiff.Dual(0.5, 0.0))
    ud, up = source_field(fastmath(Stokeslet3D()), s, t, f), source_field(Stokeslet3D(), s, t, f)
    @test close(ForwardDiff.value.(ud), ForwardDiff.value.(up), 1e-12) && close(ForwardDiff.partials.(ud, 1), ForwardDiff.partials.(up, 1), 1e-12)
    alloc(K, s, t, f) = @allocated source_field(K, s, t, f)
    tf = rand(rng, SVector{3,Float64})
    alloc(fastmath(Stokeslet3D()), s, tf, f)
    @test alloc(fastmath(Stokeslet3D()), s, tf, f) == 0
end
