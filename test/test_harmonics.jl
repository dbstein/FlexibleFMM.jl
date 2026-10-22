# The analytic Laplace method: the Hobson solid harmonics and their identities, the
# Schmidt-normalised real storage, the operators on all 316 offsets (charges and
# dipoles), convergence in p, rescaling, allocation-free bodies, and the FMM end to
# end (potential, gradient, dipoles, Float32, dual positions).
using FlexibleFMM: SphericalHarmonics, regular_hobson, irregular_hobson, cidx, tidx, scaling_factors,
    harmonic_tables, _fold_regular, complex_coefficients, real_coefficients, real_operator,
    level_operators, order, rescale, multipole_field, nlevels, LowRankM2L, M2LOperator, workspace_rows, apply!,
    harmonic_symmetry_classes, d4h_group, harmonic_canonical_offset, _d4h_factors, check_harmonic_classes,
    PermutedM2L, _harmonic_m2l, scaling_factors, permute, permute_t, m2l_level!, AbstractMethod, LevelOperators,
    InteractionList, upward_pass!, downward_pass!, expansion_eltype

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
    check_allocation_free(rng, m, 0.3; m2l=false)             # the per-offset apply! permutes with buffers
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
    @test o2.M2L[7].op.A ≈ r2.M2L[7].op.A * r2.m2l_scale
    @test o2.M2L[7].idx_t == r2.M2L[7].idx_t && o2.extra.sc.reps == r2.extra.sc.reps
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

@testset "low-rank translations" begin
    A = randn(rng, 30, 30)
    F = svd(A); r = 12
    op = LowRankM2L{Float64}(F.U[:, 1:r] .* transpose(F.S[1:r]), F.Vt[1:r, :])
    Ar = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    x = randn(rng, 30); X = randn(rng, 30, 7)
    @test apply!(zeros(30), op, x) ≈ Ar * x
    @test apply!(zeros(30, 7), op, X, 2.0) ≈ 2Ar * X
    work = zeros(20, 9); out = zeros(30, 7)
    @test apply!(out, op, X, 1.5, work) ≈ 1.5Ar * X
    @test @allocated(apply!(out, op, X, 1.5, work)) == 0
    @test apply!(zeros(30), op, x, 1.0, zeros(15)) ≈ Ar * x
    @test workspace_rows(op) == r && workspace_rows(M2LOperator(A)) == 0 && size(op) == (30, 30)
    @test apply!(zeros(30, 7), M2LOperator(A), X, 1.0, work) ≈ A * X
    @test_throws ArgumentError SphericalHarmonics(KH; p=16, tol=1.5)
    # the method with tol: ranks, accuracy against the dense operators, rescaling, the driver
    m = SphericalHarmonics(KH; p=16, tol=1e-7)
    md = SphericalHarmonics(KH; p=16)
    @test m isa SphericalHarmonics{<:Laplace3D,Float64,LowRankM2L{Float64}}
    @test md isa SphericalHarmonics{<:Laplace3D,Float64,M2LOperator{Float64}}
    ops = level_operators(m, 0.5)
    ranks = [workspace_rows(o) for o in ops.M2L]
    @info "p = 16, tol = 1e-7: ranks $(minimum(ranks))–$(maximum(ranks)), mean $(round(sum(ranks) / length(ranks), digits=1)) of 289"
    @test maximum(ranks) < 289 && minimum(ranks) < 60
    e = check_m2l_all_offsets(MersenneTwister(77), m, 0.37, 1e-5)          # the same sample for both
    ed = maximum(m2l_errors(MersenneTwister(77), md, 0.37))
    e6 = maximum(m2l_errors(MersenneTwister(77), SphericalHarmonics(KH; p=16, tol=1e-6), 0.37))
    @info "p = 16: worst offset error dense $(round(ed, sigdigits=2)), tol = 1e-7 $(round(e, sigdigits=2)), tol = 1e-6 $(round(e6, sigdigits=2))"
    @test e < 2ed + 1e-7
    @test e6 < 2ed + 1e-6
    r2 = rescale(ops, m, 1.0)
    @test r2.M2L[5] === ops.M2L[5] && r2.m2l_scale ≈ 0.5
    N = 6000
    pts = uniform_points3(rng, N); τ = randn(rng, N); targets = uniform_points3(rng, 300)
    f = FMM(m, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test haskey(st.scratch, :lowrank)
    @test mrd(evaluate(f, st, targets), direct(KH, pts, τ, targets)) < 1e-5
    fd = FMM(md, pts; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    std = build_expansions(fd, pts, τ)
    @test mrd(evaluate(f, st, targets), evaluate(fd, std, targets)) < 1e-6
    build_expansions!(st, f, pts, τ)                       # the workspace is reused
    @test mrd(evaluate(f, st, targets), evaluate(fd, std, targets)) < 1e-6
    mf = SphericalHarmonics(Laplace3D(Float32); p=12, tol=1f-5)
    @test level_operators(mf, 0.5f0).M2L[1] isa PermutedM2L{LowRankM2L{Float32}}
end

@testset "symmetry classes of the harmonic basis" begin
    G = d4h_group()
    @test length(G) == 16 && allunique([(R.perm, R.sign) for R in G])
    @test harmonic_canonical_offset(SVector(-1, 3, -2)) == SVector(3, 1, 2)
    # the factorisation R = Rot_z(kπ/2) ∘ Refl_y ∘ Refl_z reproduces the element
    rot(k) = ([1 0; 0 1], [0 -1; 1 0], [-1 0; 0 -1], [0 1; -1 0])[k+1]
    for R in G
        k, εy, εz = _d4h_factors(R)
        M = [rot(k) * [1 0; 0 (εy ? -1 : 1)] zeros(Int, 2); 0 0 (εz ? -1 : 1)]
        x = SVector(2, 5, 7)
        @test R(x) == SVector{3}(M * Vector(x))
    end
    sc = harmonic_symmetry_classes(6)
    @test length(sc.reps) == 34 && length(sc.class) == 316
    @test sort([count(==(c), sc.class) for c in 1:34]) == sort(vcat(fill(16, 9), fill(8, 19), fill(4, 4), fill(2, 2)))
    # the permutations within (n, m): Pᵀ undoes P on each side, and the two sides agree up to transposition
    v = randn(rng, 49)
    for f in (1, 50, 200, 316)
        @test sort(sc.idx[f]) == 1:49 && sort(sc.idx_t[f]) == 1:49
        Ps = zeros(49, 49); Pt = zeros(49, 49)
        for i in 1:49
            Ps[sc.idx[f][i], i] = sc.sgn[f][i]; Pt[sc.idx_t[f][i], i] = sc.sgn_t[f][i]
        end
        @test Ps' * Ps ≈ I && Pt' * Pt ≈ I
    end
    # every dense operator from its class representative
    @test check_harmonic_classes(6) < 1e-13
    @test check_harmonic_classes(16) < 1e-13
    # the permuted operator applies like the dense one
    p = 10
    s = scaling_factors(2p); A = zeros(ComplexF64, 121, 121)
    ops = level_operators(SphericalHarmonics(KH; p=p), 1.0)
    for f in (7, 123, 316)
        o = ops.extra.offs[f]
        D = _harmonic_m2l(o, p, KH.c, s, A)
        x = randn(rng, 121); X = randn(rng, 121, 3)
        @test apply!(zeros(121), ops.M2L[f], x) ≈ D * x
        @test apply!(zeros(121, 3), ops.M2L[f], X, 2.0) ≈ 2D * X
    end
    @test length(ops.extra.reps) == 34 && ops.M2L[1].op === ops.extra.reps[ops.extra.sc.class[1]]
    # the class-wise level translation equals the generic per-offset one
    N = 3000
    pts = mixed_points3(rng, N); τ = randn(rng, N)
    for m in (SphericalHarmonics(KH; p=8), SphericalHarmonics(KH; p=8, tol=1e-9))
        f = FMM(m, pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
        st = build_expansions(f, pts, τ)
        st2 = FMMState(f, Float64, Float64, Float64)
        st2.sources .= st.sources; st2.τ .= st.τ
        upward_pass!(st2, f)
        @test st2.multipoles ≈ st.multipoles
        for l in 3:nlevels(f)
            lr = f.tree.level_range[l]
            pL1 = zeros(81, length(lr)); pL2 = zeros(81, length(lr))
            m2l_level!(pL1, m, f.ops[l], st.multipoles, f.tree.ilists[l], first(lr) - 1, st)
            invoke(m2l_level!, Tuple{AbstractMatrix,AbstractMethod,LevelOperators,AbstractMatrix,InteractionList,Int,FMMState},
                   pL2, m, f.ops[l], st.multipoles, f.tree.ilists[l], first(lr) - 1, st2)
            @test maximum(abs, pL1 .- pL2) < 1e-12 * maximum(abs, pL2)      # the GEMM shapes differ: roundoff only
        end
    end
end

@testset "one order at a time" begin
    using FlexibleFMM: _fold_chain, _accumulate_order!, _accumulate!, _zero_order!, harmonic_tables
    p = 9
    tb = harmonic_tables(p, Float64)
    x = P3(0.3, -0.4, 0.25)
    full = zeros(ComplexF64, (p + 1)^2)
    _fold_regular(nothing, x, p, tb) do _, n, m, vr, vi
        full[cidx(n, m)] = complex(vr, vi); nothing
    end
    for m in 0:p
        _fold_chain(nothing, x, m, p, tb) do _, n, mm, vr, vi
            @test complex(vr, vi) ≈ full[cidx(n, mm)] rtol = 1e-13
            nothing
        end
    end
    # the per-order accumulation equals the full one for charges, dipoles and tuples
    for τ in (1.7, Dipole(P3(0.5, -1.2, 0.8)), (0.6, Dipole(P3(-0.3, 0.9, 1.1))))
        a = zeros((p + 1)^2); _accumulate!(a, x, p, tb, τ)
        b = fill(NaN, (p + 1)^2)
        for m in 0:p
            _zero_order!(b, p, m)
            _accumulate_order!(b, x, p, tb, τ, m)
        end
        @test b ≈ a atol = 1e-14 * maximum(abs, a)
    end
end
