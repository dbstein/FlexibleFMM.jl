# Stokes: the kernel and its singularities (identities checked by automatic
# differentiation), block assembly, both methods through the operator checks and
# end to end, pressure and velocity-gradient outputs, dual positions, and the
# JellySim-shaped plan API.
using FlexibleFMM: ncomp, nlevels, locate, evaluate_target, expansion_length,
    multipole_length, local_length, partial_local_length, operator_eltype, conditioning,
    level_operators

rng = MersenneTwister(15)
const K2 = Stokeslet2D()

randvec(rng, n) = [P2(randn(rng), randn(rng)) for _ in 1:n]
randunit(rng, n) = [normalize(P2(randn(rng), randn(rng))) for _ in 1:n]
dvals(u) = [ForwardDiff.value.(x) for x in u]
dparts(u) = [ForwardDiff.partials.(x, 1) for x in u]

# velocity u(t), pressure p(t) and the Stokes residual μ∇²u - ∇p, by ForwardDiff
function stokes_residual(K, s, τ, t)
    u(tt) = source_field(K, s, tt, τ)
    p(tt) = target_field(K, s, tt, τ, Pressure())
    J = ForwardDiff.jacobian(u, t)
    lap = SVector(tr(ForwardDiff.hessian(tt -> u(tt)[1], t)), tr(ForwardDiff.hessian(tt -> u(tt)[2], t)))
    return tr(J), lap - ForwardDiff.gradient(p, t)
end

@testset "Stokes singularities satisfy the Stokes equations" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3)
    f = P2(0.4, -1.1); ν = normalize(P2(0.3, 0.8)); g = P2(-0.6, 0.2)
    @test ncomp(K2) == 2
    @test K2(s, t) isa SMatrix{2,2,Float64}
    @test K2(s, t) ≈ transpose(K2(s, t))
    @test K2(s, t) ≈ K2(t, s)
    for τ in (f, Stresslet(f, ν), Doublet(g), Rotlet(0.7), (f, Stresslet(f, ν), Doublet(g), Rotlet(0.7)))
        div, res = stokes_residual(K2, s, τ, t)
        @test abs(div) < 1e-10
        @test norm(res) < 1e-9
        # velocity gradient output = Jacobian of the field
        @test target_field(K2, s, t, τ, Gradient()) ≈ ForwardDiff.jacobian(tt -> source_field(K2, s, tt, τ), t)
    end
    # the stokeslet pressure kernel
    @test target_field(K2, s, t, f, Pressure()) ≈ sum(pressure_kernel(K2, s, t) .* f)
    # the doublet is half the Laplacian of the Stokeslet and the rotlet an antisymmetric derivative
    Gfun(ss) = K2(ss, t) * g
    lapG = SVector(tr(ForwardDiff.hessian(ss -> Gfun(ss)[1], s)), tr(ForwardDiff.hessian(ss -> Gfun(ss)[2], s)))
    @test source_field(K2, s, t, Doublet(g)) ≈ lapG / 2
    ω = 0.7
    dG = ForwardDiff.jacobian(ss -> vec(K2(ss, t)), s)          # d(G_ij)/d(s_k), rows (i,j) column-major
    Gk(i, j, k) = dG[i + 2(j - 1), k]
    rot = SVector((-ω * (Gk(i, 1, 2) - Gk(i, 2, 1)) for i in 1:2)...)
    @test source_field(K2, s, t, Rotlet(ω)) ≈ rot
    # precision
    sf = SVector{2,Float32}(s); tf = SVector{2,Float32}(t)
    Kf = Stokeslet2D(Float32)
    @test Kf(sf, tf) isa SMatrix{2,2,Float32}
    @test source_field(Kf, sf, tf, Doublet(SVector{2,Float32}(g))) isa SVector{2,Float32}
    @test target_field(Kf, sf, tf, SVector{2,Float32}(f), Pressure()) isa Float32
    for τ in (f, Stresslet(f, ν), Doublet(g), Rotlet(0.7))
        source_field(K2, s, t, τ)
        @test @allocated(source_field(K2, s, t, τ)) == 0
        @test @allocated(target_field(K2, s, t, τ, Pressure())) == 0
    end
    @test @allocated(target_field(K2, s, t, f, Gradient())) == 0
end

@testset "block kernel matrix and direct sums" begin
    srcs = uniform_points(rng, 5); tgts = uniform_points(rng, 3)
    A = kernel_matrix(K2, srcs, tgts)
    @test size(A) == (6, 10)
    for i in 1:5, j in 1:3, a in 1:2, b in 1:2
        @test A[(a - 1) * 3 + j, (b - 1) * 5 + i] == K2(srcs[i], tgts[j])[a, b]
    end
    f = randvec(rng, 5)
    u = direct(K2, srcs, f, tgts)
    @test eltype(u) == P2
    flat = A * vcat(getindex.(f, 1), getindex.(f, 2))
    @test [u[j][a] for a in 1:2 for j in 1:3] ≈ flat
    @test direct_self(K2, srcs, f)[2] ≈ sum(K2(srcs[i], srcs[2]) * f[i] for i in 1:5 if i != 2)
end

stresslets(rng, K, n) = [Stresslet(P2(randn(rng), randn(rng)), normalize(P2(randn(rng), randn(rng)))) for _ in 1:n]
doublets(rng, K, n) = [Doublet(P2(randn(rng), randn(rng))) for _ in 1:n]
rotlets(rng, K, n) = [Rotlet(randn(rng)) for _ in 1:n]

@testset "operators: $(mname)" for (mname, m) in (("KI nequiv=60", KernelIndependent(K2; nequiv=60)),
                                                  ("BB p=14", BlackBox(K2; p=14)))
    @test expansion_length(m) == 2 * (m isa KernelIndependent ? 60 : 196)
    @test operator_eltype(m) == Float64
    if m isa KernelIndependent
        @test multipole_length(m) == 121 && local_length(m) == 120 && partial_local_length(m) == 121
        ops = level_operators(m, 0.37)
        c = conditioning(ops)
        # the augmented 120 × 121 upward map has full row rank (the single-layer null
        # vector lies in its column space); the pinned downward map is full rank too
        @test c[3] == 0 && c[4] == 0
        @test FlexibleFMM.rank(ops.extra.P_up) == 120 && FlexibleFMM.rank(ops.extra.P_down) == 120
    end
    check_m2l_all_offsets(rng, m, 0.37, 1e-7)
    check_m2m(rng, m, 0.61, 1e-7)
    check_l2l(rng, m, 0.61, 1e-7)
    check_allocation_free(rng, m, 0.3)
    for (dname, density) in (("doublets", doublets), ("rotlets", rotlets))
        check_m2l_all_offsets(rng, m, 0.37, 1e-6; density=density)
        check_m2m(rng, m, 0.61, 1e-6; density=density)
        check_l2l(rng, m, 0.61, 1e-6; density=density)
    end
    if m isa KernelIndependent
        # stresslets carry a point source of strength -(f·ν), hence the augmented multipole
        check_m2l_all_offsets(rng, m, 0.37, 1e-6; density=stresslets)
        check_m2m(rng, m, 0.61, 1e-6; density=stresslets)
        check_l2l(rng, m, 0.61, 1e-6; density=stresslets)
    end
end

@testset "end to end, $(mname)" for (mname, m) in (("KI nequiv=60", KernelIndependent(K2; nequiv=60)),
                                                    ("BB p=14", BlackBox(K2; p=14)))
    N = 5000
    pts = mixed_points(rng, N)
    tg = uniform_points(rng, 2000)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    @test nlevels(f) >= 4
    stk = randvec(rng, N); dub = randvec(rng, N); rot = [Rotlet(randn(rng)) for _ in 1:N]
    for (name, τ) in (("stokeslets", stk), ("doublets", Doublet.(dub)), ("rotlets", rot),
                      ("stokeslets+doublets", [(stk[i], Doublet(dub[i])) for i in 1:N]))
        st = build_expansions(f, pts, τ)
        @test eltype(st.multipoles) == Float64
        @test mrd(evaluate(f, st, tg), direct(K2, pts, τ, tg)) < 1e-5
        @test mrd(evaluate_at_sources(f, st), direct_self(K2, pts, τ)) < 1e-5
        @test mrd(evaluate(f, st, tg; output=Gradient()), direct(K2, pts, τ, tg; output=Gradient())) < 1e-4
    end
    if m isa KernelIndependent
        strs = [Stresslet(stk[i], normalize(dub[i])) for i in 1:N]
        comb = [(stk[i], strs[i], Doublet(dub[i])) for i in 1:N]
        for (name, τ) in (("stresslets", strs), ("all three", comb))
            st = build_expansions(f, pts, τ)
            @test mrd(evaluate(f, st, tg), direct(K2, pts, τ, tg)) < 1e-5
            @test mrd(evaluate_at_sources(f, st), direct_self(K2, pts, τ)) < 1e-5
            # pressure: correct everywhere thanks to the pinned downward solve
            @test mrd(evaluate(f, st, tg; output=Pressure()), direct(K2, pts, τ, tg; output=Pressure())) < 1e-5
            @test mrd(evaluate_at_sources(f, st; output=Pressure()), direct_self(K2, pts, τ; output=Pressure())) < 1e-5
        end
    else
        @test_throws ArgumentError build_expansions(f, pts, [Stresslet(stk[i], dub[i]) for i in 1:N])
        st = build_expansions(f, pts, stk)
        @test_throws ArgumentError evaluate(f, st, tg; output=Pressure())
    end
end

@testset "dual positions through a stokeslet+doublet FMM (the SBT pattern)" begin
    DualS = ForwardDiff.Dual{:stk,Float64,1}
    d(x, v) = DualS(x, ForwardDiff.Partials((v,)))
    N = 3000
    pts = circle_points(rng, N; radius=0.35)
    V = randvec(rng, N)
    stk = randvec(rng, N); dub = randvec(rng, N)
    w = 0.5 .+ rand(rng, N)
    tg = uniform_points(rng, 500)
    m = KernelIndependent(K2; nequiv=60)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    sd = [SVector(d(p[1], v[1]), d(p[2], v[2])) for (p, v) in zip(pts, V)]
    τ = [(w[i] * stk[i], Doublet(w[i] * dub[i])) for i in 1:N]
    st = build_expansions(f, sd, τ)
    @test eltype(st.multipoles) == DualS
    u = evaluate(f, st, tg)
    uref = direct(K2, sd, τ, tg)
    @test mrd(dvals(u), dvals(uref)) < 1e-6
    @test mrd(dparts(u), dparts(uref)) < 1e-5
    us = evaluate_at_sources(f, st)
    usref = direct_self(K2, sd, τ)
    @test mrd(dparts(us), dparts(usref)) < 1e-5
end

@testset "JellySim-shaped API" begin
    N = 4000
    pts = mixed_points(rng, N)
    tg = uniform_points(rng, 1500)
    stk = randvec(rng, N); strs = randvec(rng, N); nu = randunit(rng, N); dub = randvec(rng, N)
    w = rand(rng, N)
    ref(τ, targets; output=Velocity()) = direct(K2, pts, τ, targets; output=output)
    refself(τ) = direct_self(K2, pts, τ)
    for method in (:ki, :bb)
        plan_t = StokesFMMPlan(pts; targets=tg, method=method, nequiv=60, p=14, ppl=40)
        plan_s = StokesFMMPlan(pts; method=method, nequiv=60, p=14, ppl=40)
        @test plan_t.targets == tg && plan_s.targets === nothing
        vt = Vector{P2}(undef, 1500); vs = Vector{P2}(undef, N)
        # stokeslets with weights
        τ = [(w[i] * stk[i],) for i in 1:N]
        stokes_fmm_eval!(vt, plan_t; stokeslets=stk, weights=w)
        @test mrd(vt, ref(τ, tg)) < 1e-5
        stokes_fmm_eval!(vs, plan_s; stokeslets=stk, weights=w)
        @test mrd(vs, refself(τ)) < 1e-5
        # stokeslets + doublets, source and target in one pass
        τ2 = [(stk[i], Doublet(dub[i])) for i in 1:N]
        stokes_fmm_eval_src_trg!(vs, vt, plan_t; stokeslets=stk, doublets=dub)
        @test mrd(vs, refself(τ2)) < 1e-5 && mrd(vt, ref(τ2, tg)) < 1e-5
        # state reuse with a different density set allocates a new state silently
        stokes_fmm_eval!(vt, plan_t; doublets=dub)
        @test mrd(vt, ref(Doublet.(dub), tg)) < 1e-5
        if method === :ki
            τ3 = [(stk[i], Stresslet(strs[i], nu[i]), Doublet(dub[i])) for i in 1:N]
            stokes_fmm_eval!(vt, plan_t; stokeslets=stk, stresslets=strs, stressvecs=nu, doublets=dub)
            @test mrd(vt, ref(τ3, tg)) < 1e-5
            pt = zeros(1500)
            stokes_fmm_eval!(pt, plan_t; stokeslets=stk, stresslets=strs, stressvecs=nu, output=Pressure())
            @test mrd(pt, ref([(stk[i], Stresslet(strs[i], nu[i])) for i in 1:N], tg; output=Pressure())) < 1e-5
        end
        @test_throws ArgumentError stokes_fmm_eval!(vt, plan_t; stresslets=strs)
        @test_throws ArgumentError stokes_fmm_eval!(vt, plan_t)
        @test_throws DimensionMismatch stokes_fmm_eval!(vt, plan_t; stokeslets=stk[1:10])
        @test_throws ArgumentError stokes_fmm_eval_src_trg!(vs, vt, plan_s; stokeslets=stk)
    end
    # one-shot forms, matrix input, self by identity of the arrays
    M = reduce(hcat, pts)
    u1 = StokesFMMKernel(M; stokeslets=stk, nequiv=60, ppl=40)
    @test mrd(u1, refself([(s,) for s in stk])) < 1e-5
    u2 = StokesFMMKernel(pts; targets=pts, stokeslets=stk, doublets=dub, nequiv=60, ppl=40)
    @test mrd(u2, refself([(stk[i], Doublet(dub[i])) for i in 1:N])) < 1e-5
    u3 = StokesFMMKernel(pts; targets=tg, stokeslets=stk, nequiv=60, ppl=40)
    @test mrd(u3, ref([(s,) for s in stk], tg)) < 1e-5
    # dual positions and weights through the one-shot form
    DualS = ForwardDiff.Dual{:api,Float64,1}
    d(x, v) = DualS(x, ForwardDiff.Partials((v,)))
    V = randvec(rng, N)
    sd = [SVector(d(p[1], v[1]), d(p[2], v[2])) for (p, v) in zip(pts, V)]
    wd = [d(w[i], randn(rng)) for i in 1:N]
    ud = StokesFMMKernel(sd; targets=tg, stokeslets=stk, doublets=dub, weights=wd, nequiv=60, ppl=40)
    τd = [(wd[i] * stk[i], Doublet(wd[i] * dub[i])) for i in 1:N]
    udref = direct(K2, sd, τd, tg)
    @test eltype(ud) == SVector{2,DualS}
    @test mrd(dparts(ud), dparts(udref)) < 1e-5
    # Float32 plan
    ptsf = [SVector{2,Float32}(p) for p in pts]; stkf = [SVector{2,Float32}(s) for s in stk]
    uf = StokesFMMKernel(ptsf; stokeslets=stkf, nequiv=40, ppl=40, precision=Float32)
    @test eltype(uf) == SVector{2,Float32}
    @test mrd(uf, refself([(s,) for s in stk])) < 1e-2      # single precision: ~5e-3 for Stokes
end

@testset "allocation-free Stokes bodies" begin
    N = 2000
    pts = uniform_points(rng, N)
    stk = randvec(rng, N); dub = randvec(rng, N)
    τ = [(stk[i], Doublet(dub[i])) for i in 1:N]
    for m in (KernelIndependent(K2; nequiv=40), BlackBox(K2; p=10))
        f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
        st = build_expansions(f, pts, τ)
        x = P2(0.37, 0.61)
        i = locate(f.tree, x)
        for out in (Velocity(), Gradient())
            evaluate_target(f, st, x, i, 0, out)
            @test @allocated(evaluate_target(f, st, x, i, 0, out)) == 0
        end
        if m isa KernelIndependent
            evaluate_target(f, st, x, i, 0, Pressure())
            @test @allocated(evaluate_target(f, st, x, i, 0, Pressure())) == 0
        end
    end
end
