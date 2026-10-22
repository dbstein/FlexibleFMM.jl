# The analytic Laplace method: the Hobson solid harmonics and their identities, the
# Schmidt-normalised real storage, the operators on all 316 offsets (charges and
# dipoles), convergence in p, rescaling, allocation-free bodies, and the FMM end to
# end (potential, gradient, dipoles, Float32, dual positions).
using FlexibleFMM: SphericalHarmonics, regular_hobson, irregular_hobson, cidx, tidx, scaling_factors,
    harmonic_tables, _fold_regular, complex_coefficients, real_coefficients, real_operator,
    level_operators, order, rescale, multipole_field, nlevels

rng = MersenneTwister(21)
const KH = Laplace3D()
dipoles3(rng, K, n) = [Dipole(randn(rng, P3)) for _ in 1:n]
mixed3(rng, K, n) = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:n]

@testset "Hobson solid harmonics" begin
    x = P3(0.3, -0.5, 0.4); r = norm(x); p = 7
    Y = regular_hobson(x, p); Θ = irregular_hobson(x, p)
    xy = complex(x[1], x[2])
    @test Y[cidx(0, 0)] == 1 && Y[cidx(1, 0)] ≈ x[3] && Y[cidx(1, 1)] ≈ -xy / 2 && Y[cidx(2, 2)] ≈ xy^2 / 8
    @test Y[cidx(2, 0)] ≈ (3x[3]^2 - r^2) / 4 && Y[cidx(2, 1)] ≈ -x[3] * xy / 2
    @test Θ[cidx(0, 0)] ≈ 1 / r && Θ[cidx(1, 0)] ≈ x[3] / r^3 && Θ[cidx(1, 1)] ≈ -xy / r^3
    for n in 0:p, m in -n:n
        @test Θ[cidx(n, m)] ≈ factorial(n - abs(m)) * factorial(n + abs(m)) * Y[cidx(n, m)] / r^(2n + 1)
        m > 0 && @test Y[cidx(n, -m)] == (-1)^m * conj(Y[cidx(n, m)])
        m > 0 && @test Θ[cidx(n, -m)] == (-1)^m * conj(Θ[cidx(n, m)])
    end
    # harmonic, by automatic differentiation of the real and imaginary parts
    for n in 0:p, m in 0:n, part in (real, imag), f in (regular_hobson, irregular_hobson)
        H = ForwardDiff.hessian(t -> part(f(t, p)[cidx(n, m)]), x)
        @test abs(tr(H)) < 1e-10 * (1 + maximum(abs, H))
    end
    # the generating identity 1/|x − y| = Σ conj Υ(y) Θ(x), |y| < |x|
    y = P3(0.1, 0.05, -0.08)
    Yy = regular_hobson(y, 30); Θx = irregular_hobson(x, 30)
    @test abs(sum(conj(Yy[k]) * Θx[k] for k in 1:31^2) - 1 / norm(x - y)) < 1e-13 / norm(x - y)
    # the regular addition theorem
    a = P3(0.2, 0.1, -0.3); b = P3(-0.1, 0.4, 0.2)
    Ya, Yb, Yab = regular_hobson(a, p), regular_hobson(b, p), regular_hobson(a + b, p)
    for n in 0:p, m in -n:n
        acc = 0.0im
        for k in 0:n, l in -k:k
            abs(m - l) <= n - k && (acc += Ya[cidx(k, l)] * Yb[cidx(n - k, m - l)])
        end
        @test acc ≈ Yab[cidx(n, m)] atol = 1e-15
    end
    # the irregular translation theorem Θ_n^m(a + b) = Σ (−1)^k conj Υ_k^l(b) Θ_{n+k}^{m+l}(a), |b| < |a|
    a = P3(1.5, -0.8, 1.1); b = P3(0.1, 0.2, -0.15); q = 40
    Θa = irregular_hobson(a, q); Yb = regular_hobson(b, q); Θab = irregular_hobson(a + b, 3)
    for n in 0:3, m in -n:n
        acc = 0.0im
        for k in 0:q-n, l in -k:k
            acc += (-1)^k * conj(Yb[cidx(k, l)]) * Θa[cidx(n + k, m + l)]
        end
        @test acc ≈ Θab[cidx(n, m)] rtol = 1e-12
    end
    # the derivative identities
    at(k, m) = abs(m) <= k ? Y[cidx(k, m)] : 0.0im
    for n in 1:p, m in -n:n
        g = ForwardDiff.gradient(t -> real(regular_hobson(t, p)[cidx(n, m)]), x) +
            im * ForwardDiff.gradient(t -> imag(regular_hobson(t, p)[cidx(n, m)]), x)
        @test g[3] ≈ at(n - 1, m) atol = 1e-14
        @test g[1] - im * g[2] ≈ -at(n - 1, m - 1) atol = 1e-14
        @test g[1] + im * g[2] ≈ at(n - 1, m + 1) atol = 1e-14
    end
end

@testset "Schmidt recurrences and real storage" begin
    p = 9
    tb = harmonic_tables(p, Float64); s = scaling_factors(p)
    x = P3(0.25, -0.4, 0.35)
    Y = regular_hobson(x, p)
    vals = zeros(ComplexF64, (p + 1)^2)
    _fold_regular(nothing, x, p, tb) do _, n, m, vr, vi
        vals[cidx(n, m)] = complex(vr, vi)
        nothing
    end
    for n in 0:p, m in 0:n
        @test vals[cidx(n, m)] ≈ s[cidx(n, m)] * Y[cidx(n, m)] rtol = 1e-13
        @test abs(vals[cidx(n, m)]) <= norm(x)^n * (1 + 1e-12)          # |Υ̃| ≤ r^n
    end
    v = randn(rng, (p + 1)^2)
    @test real_coefficients(complex_coefficients(v, p), p) == v
    c = complex_coefficients(v, p)
    for n in 0:p, m in 1:n
        @test c[cidx(n, -m)] == (-1)^m * conj(c[cidx(n, m)])
    end
    # a complex operator commuting with the conjugate symmetry (a phase per m) and its real matrix
    A = Diagonal([cis(0.3m) * (1 + 0.1n) for n in 0:p for m in -n:n])
    @test real_operator(Matrix(A), p) * v ≈ real_coefficients(A * c, p)
end

@testset "operators on all 316 offsets, p = 16" begin
    m = SphericalHarmonics(KH; p=16)
    @test order(m) == 16
    @test_throws ArgumentError SphericalHarmonics(KH; p=0)
    e = check_m2l_all_offsets(rng, m, 0.37, 1e-4)
    ed = check_m2l_all_offsets(rng, m, 0.37, 3e-4; density=dipoles3)
    em = check_m2l_all_offsets(rng, m, 0.37, 3e-4; density=mixed3)
    @info "spherical harmonics p = 16, worst offset error: charges $(round(e, sigdigits=2)), dipoles $(round(ed, sigdigits=2)), both $(round(em, sigdigits=2))"
    check_m2m(rng, m, 0.61, 1e-8)
    check_m2m(rng, m, 0.61, 1e-8; density=dipoles3)
    check_l2l(rng, m, 0.61, 1e-5)
    check_l2l(rng, m, 0.61, 1e-5; density=dipoles3)
    check_allocation_free(rng, m, 0.3)
end

@testset "convergence in p" begin
    ps = (6, 10, 14, 18)
    e = [maximum(m2l_errors(rng, SphericalHarmonics(KH; p=p), 0.5)) for p in ps]
    @info "spherical harmonics worst offset error vs p: " * join(["p = $p: $(round(x, sigdigits=2))" for (p, x) in zip(ps, e)], ", ")
    @test e[1] > 5e[2] > 5e[3] > 5e[4]
    @test e[4] < 1e-4
end

@testset "rescale" begin
    m = SphericalHarmonics(KH; p=8)
    ref = level_operators(m, 0.5)
    o2 = level_operators(m, 2.0)
    r2 = rescale(ref, m, 2.0)
    @test r2.M2M === ref.M2M && r2.L2L === ref.L2L
    @test r2.m2l_scale ≈ 0.25 && r2.extra.scale ≈ o2.extra.scale
    @test o2.M2M ≈ ref.M2M && o2.L2L ≈ ref.L2L
    @test o2.M2L[7].A ≈ r2.M2L[7].A * r2.m2l_scale
end

@testset "FMM end to end: $(name)" for (name, gen) in pairs(DISTRIBUTIONS3)
    N = 6000
    pts = gen(rng, N)
    τ = randn(rng, N)
    targets = uniform_points3(rng, 500)
    m = SphericalHarmonics(KH; p=16)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test mrd(evaluate_at_sources(f, st), direct_self(KH, pts, τ)) < 1e-5
    @test mrd(evaluate(f, st, targets), direct(KH, pts, τ, targets)) < 1e-5
    @test mrd(evaluate(f, st, targets; output=Gradient()), direct(KH, pts, τ, targets; output=Gradient())) < 1e-4
    if name == :uniform
        τd = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:N]
        std = build_expansions(f, pts, τd)
        @test mrd(evaluate(f, std, targets), direct(KH, pts, τd, targets)) < 1e-5
        @test mrd(evaluate(f, std, targets; output=Gradient()), direct(KH, pts, τd, targets; output=Gradient())) < 1e-4
        τc = randn(rng, ComplexF64, N)
        stc = build_expansions(f, pts, τc)
        @test eltype(stc.multipoles) == ComplexF64
        @test mrd(evaluate(f, stc, targets), direct(KH, pts, τc, targets)) < 1e-5
        @test_throws ArgumentError build_expansions(f, pts, [randn(rng, P3) for _ in 1:N])
        @test_throws ArgumentError evaluate(f, st, targets; output=Pressure())
    end
end

@testset "Float32" begin
    Kf = Laplace3D(Float32)
    N = 4000
    pts = [SVector{3,Float32}(x) for x in uniform_points3(rng, N)]
    τ = randn(rng, Float32, N)
    m = SphericalHarmonics(Kf; p=16)
    f = FMM(m, pts; ppl=40, bbox=(SVector{3,Float32}(0, 0, 0), 1f0))
    st = build_expansions(f, pts, τ)
    @test eltype(st.multipoles) == Float32
    u = evaluate_at_sources(f, st)
    @test eltype(u) == Float32
    @test mrd(u, direct_self(KH, P3.(pts), Float64.(τ))) < 1e-4
end

@testset "dual positions" begin
    D1 = ForwardDiff.Dual{:dh,Float64,1}
    dual3(p, v) = SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],))),
                          D1(p[3], ForwardDiff.Partials((v[3],))))
    N = 2000
    pts = helix_points(rng, N)
    V = [randn(rng, P3) for _ in 1:N]
    targets = uniform_points3(rng, 300)
    sd = [dual3(p, v) for (p, v) in zip(pts, V)]
    τ = [(randn(rng), Dipole(randn(rng, P3))) for _ in 1:N]
    f = FMM(SphericalHarmonics(KH; p=16), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, sd, τ)
    u = evaluate(f, st, targets)
    uref = direct(KH, sd, τ, targets)
    vals(u) = [ForwardDiff.value.(x) for x in u]
    parts(u) = [ForwardDiff.partials.(x, 1) for x in u]
    @test mrd(vals(u), vals(uref)) < 1e-5
    @test mrd(parts(u), parts(uref)) < 1e-4
end
