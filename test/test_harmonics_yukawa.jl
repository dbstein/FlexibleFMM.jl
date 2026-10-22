# The Yukawa harmonic method: the radial factors against SpecialFunctions, the
# generating identity, the operators by projection on all 316 offsets (charges and
# dipoles, two values of κw), M2M, L2L, convergence in p, allocation-free bodies, and
# the FMM end to end (potential, gradient, dipoles, low rank, dual positions).
using FlexibleFMM: SphericalHarmonics, _yukawa_u!, _yukawa_v!, _sphere_quadrature, level_operators, order,
    LowRankM2L, PermutedM2L, l2t, nlevels
using SpecialFunctions: besseli, besselk

rng = MersenneTwister(52)
dipoles3y(rng, K, n) = [Dipole(randn(rng, P3)) for _ in 1:n]
mixed3y(rng, K, n) = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:n]

@testset "radial factors" begin
    dfact(n) = prod(1:2:2n-1; init=1.0)                      # (2n−1)!!, 1 for n ≤ 0
    for z in (1e-6, 0.03, 0.7, 2.5, 9.0, 40.0)
        u = zeros(21); v = zeros(21)
        _yukawa_u!(u, z^2, 20); _yukawa_v!(v, z, 20)
        for n in 0:20
            i_n = sqrt(π / (2z)) * besseli(n + 0.5, z)
            k_n = sqrt(π / (2z)) * besselk(n + 0.5, z)
            @test u[n+1] ≈ dfact(n + 1) * i_n / z^n rtol = 1e-12
            @test v[n+1] ≈ (2 / π) * z^(n + 1) * k_n / dfact(n) rtol = 1e-12
        end
    end
    # the generating identity e^{−κr}/r = Σ_n u_n(κ r_<) v_n(κ r_>) (2n+1) P_n(cos γ)... through the harmonics:
    # the expansion in regular × irregular real bases matches the kernel
    κ = 2.0; x = P3(0.6, -0.5, 0.8); y = P3(0.1, 0.2, -0.05)
    p = 30
    tb = FlexibleFMM.harmonic_tables(p, Float64)
    br = zeros((p + 1)^2); bi = zeros((p + 1)^2); u = zeros(p + 1); v = zeros(p + 1)
    FlexibleFMM._yukawa_regular!(br, y, κ, p, tb, u)
    FlexibleFMM._yukawa_irregular!(bi, x, κ, p, tb, v)
    # Σ_j conj R_j(y) I_j(x) with the real bases: the regular one is {R⁰, 2Re R, 2Im R}, the irregular
    # {I⁰, 2Re I, −2Im I}, and 2 Re(conj(R) I) = 2(Re R Re I + Im R Im I): the pairs' products are halved
    # and the imaginary ones enter with the irregular basis' sign undone
    s = 0.0
    for n in 0:p
        s += br[n*n+1] * bi[n*n+1]
        for m in 1:n
            s += (br[n*n+2m] * bi[n*n+2m] - br[n*n+2m+1] * bi[n*n+2m+1]) / 2
        end
    end
    @test s ≈ exp(-κ * norm(x - y)) / norm(x - y) rtol = 1e-12
    pts, wts = _sphere_quadrature(8)
    @test sum(wts) ≈ 4π && all(abs(norm(x) - 1) < 1e-14 for x in pts)
    @test_throws ArgumentError SphericalHarmonics(Yukawa3D(1.0); p=61)
end

# At κw = 5 the series converge slowly relative to their own far field (the
# polynomial factors of k_n for κ̂r ≳ p), which is exponentially small against the
# near field, so the relative tolerances are looser there.
@testset "operators on all 316 offsets, κw = $(round(κ * w, digits=2))" for (κ, w, tc, td, tm) in ((3.0, 0.37, 1e-5, 2e-4, 1e-7), (10.0, 0.5, 2e-3, 2e-3, 1e-6))
    K = Yukawa3D(κ)
    m = SphericalHarmonics(K; p=16)
    @test order(m) == 16 && m isa SphericalHarmonics{<:Yukawa3D}
    ops = level_operators(m, w)
    @test ops.extra.κ̂ ≈ κ * w && length(ops.extra.reps) == 34 && ops.M2L[1] isa PermutedM2L
    e = check_m2l_all_offsets(rng, m, w, tc)
    ed = check_m2l_all_offsets(rng, m, w, td; density=dipoles3y)
    @info "Yukawa harmonics p = 16, κw = $(κ * w): worst offset error charges $(round(e, sigdigits=2)), dipoles $(round(ed, sigdigits=2))"
    check_m2m(rng, m, w, tm)
    check_m2m(rng, m, w, 10tm; density=dipoles3y)
    check_l2l(rng, m, w, tc)
    check_l2l(rng, m, w, td; density=mixed3y)
    check_allocation_free(rng, m, w; m2l=false)
end

@testset "convergence in p and the Laplace limit" begin
    K = Yukawa3D(2.0)
    e = [maximum(m2l_errors(MersenneTwister(3), SphericalHarmonics(K; p=p), 0.5)) for p in (6, 10, 14, 18)]
    @info "Yukawa harmonics worst offset error vs p: " * join(["p = $p: $(round(x, sigdigits=2))" for (p, x) in zip((6, 10, 14, 18), e)], ", ")
    @test e[1] > 5e[2] > 5e[3] > 5e[4] && e[4] < 1e-4
    # a tiny κ reproduces the Laplace operators (the kernels themselves differ by κr ≈ 5e-11·|o|)
    oy = level_operators(SphericalHarmonics(Yukawa3D(1e-10); p=8), 0.5)
    ol = level_operators(SphericalHarmonics(Laplace3D(); p=8), 0.5)
    @test maximum(abs, oy.M2M .- ol.M2M) < 1e-9 * maximum(abs, ol.M2M)
    @test maximum(abs, oy.L2L .- ol.L2L) < 1e-9 * maximum(abs, ol.L2L)
    @test all(maximum(abs, oy.extra.reps[c].A .- ol.extra.reps[c].A) < 1e-9 * maximum(abs, ol.extra.reps[c].A) for c in 1:34)
    # low rank
    ml = SphericalHarmonics(K; p=12, tol=1e-8)
    @test level_operators(ml, 0.5).M2L[1] isa PermutedM2L{LowRankM2L{Float64}}
end

@testset "FMM end to end: $(name)" for (name, gen) in ((:uniform, uniform_points3), (:helix, helix_points), (:mixed, mixed_points3))
    K = Yukawa3D(4.0)
    N = 5000
    pts = gen(rng, N)
    τ = randn(rng, N)
    targets = uniform_points3(rng, 400)
    m = SphericalHarmonics(K; p=16)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test mrd(evaluate_at_sources(f, st), direct_self(K, pts, τ)) < 1e-5
    @test mrd(evaluate(f, st, targets), direct(K, pts, τ, targets)) < 1e-5
    @test mrd(evaluate(f, st, targets; output=Gradient()), direct(K, pts, τ, targets; output=Gradient())) < 1e-4
    if name == :uniform
        τd = mixed3y(rng, K, N)
        std = build_expansions(f, pts, τd)
        @test mrd(evaluate(f, std, targets), direct(K, pts, τd, targets)) < 1e-4
        @test mrd(evaluate(f, std, targets; output=Gradient()), direct(K, pts, τd, targets; output=Gradient())) < 1e-3
        fl = FMM(SphericalHarmonics(K; p=16, tol=1e-7), pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
        stl = build_expansions(fl, pts, τ)
        @test mrd(evaluate(fl, stl, targets), evaluate(f, st, targets)) < 1e-6
        @test length(f.cache) == nlevels(f)                       # one operator set per width
        @test_throws ArgumentError DevicePlan(f)
    end
end

@testset "dual positions" begin
    K = Yukawa3D(3.0)
    D1 = ForwardDiff.Dual{:dy,Float64,1}
    dual3(p, v) = SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],))),
                          D1(p[3], ForwardDiff.Partials((v[3],))))
    N = 2000
    pts = helix_points(rng, N)
    sd = [dual3(p, randn(rng, P3)) for p in pts]
    τ = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:N]
    targets = uniform_points3(rng, 300)
    f = FMM(SphericalHarmonics(K; p=16), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, sd, τ)
    u = evaluate(f, st, targets)
    uref = direct(K, sd, τ, targets)
    vals(u) = [ForwardDiff.value.(x) for x in u]
    parts(u) = [ForwardDiff.partials.(x, 1) for x in u]
    @test mrd(vals(u), vals(uref)) < 1e-4
    @test mrd(parts(u), parts(uref)) < 1e-3
end

@testset "review: derivatives at box centres" begin
    K = Yukawa3D(3.0)
    m = SphericalHarmonics(K; p=8)
    ops = level_operators(m, 0.5)
    L = randn(rng, 81)
    # the potential and gradient are differentiable at the centre (no |x̂| in the way)
    g0 = ForwardDiff.gradient(x -> l2t(m, ops, L, x), P3(0, 0, 0))
    @test all(isfinite, g0) && g0 ≈ l2t(m, ops, L, P3(0, 0, 0), Gradient())
    H0 = ForwardDiff.jacobian(x -> l2t(m, ops, L, x, Gradient()), P3(0, 0, 0))
    @test all(isfinite, H0)
    # through the public API: a target at a leaf centre with dual coordinates
    N = 1500
    pts = uniform_points3(rng, N); τ = randn(rng, N)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    i = findfirst(f.tree.leaf)
    c = FlexibleFMM.center(f.tree, i)
    D1 = ForwardDiff.Dual{:dc,Float64,1}
    cd = SVector(D1(c[1], ForwardDiff.Partials((1.0,))), D1(c[2], ForwardDiff.Partials((0.0,))), D1(c[3], ForwardDiff.Partials((0.0,))))
    u = evaluate(f, st, [cd])
    @test isfinite(ForwardDiff.value(u[1])) && isfinite(ForwardDiff.partials(u[1], 1))
    @test ForwardDiff.partials(u[1], 1) ≈ evaluate(f, st, [c]; output=Gradient())[1][1] rtol = 1e-8
end
