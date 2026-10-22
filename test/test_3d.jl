# Three dimensions: the 3D kernels (Laplace, Helmholtz, Yukawa, and Stokes in
# JellySim's conventions), the tree and interaction lists on 3D point sets, the cube
# method's operators over all 316 far offsets, end-to-end comparisons with direct
# sums, analytic sphere tests, dual positions, Float32, and the Stokes front end.
# check_tree_invariants, check_coverage and the operator checks are the 2D ones,
# which are generic in the dimension.
using FlexibleFMM: nlevels, nnodes, level_operators, far_offsets, apply!, partial_local_length,
    multipole_length, local_length, nsurface, nlarge, ncomp, Tree, kernel_eltype

rng = MersenneTwister(33)
const K3 = Laplace3D()
const KS3 = Stokeslet3D()
rand3(rng) = randn(rng, P3)
unit3(rng) = normalize(randn(rng, P3))
vals(u) = [ForwardDiff.value.(x) for x in u]
parts(u) = [ForwardDiff.partials.(x, 1) for x in u]

@testset "3D kernels" begin
    s = P3(0.1, 0.2, -0.1); t = P3(0.7, -0.3, 0.4); r = norm(t - s)
    @test K3(s, t) ≈ 1 / (4π * r)
    @test K3(s, t) == K3(t, s)
    H = Helmholtz3D(2.5); Y = Yukawa3D(2.5)
    @test H(s, t) ≈ cis(2.5r) / (4π * r)
    @test Y(s, t) ≈ exp(-2.5r) / (4π * r)
    @test kernel_eltype(K3, P3, P3) == Float64 && kernel_eltype(H, P3, P3) == ComplexF64
    @test Helmholtz3D(2) isa Helmholtz3D{Float64} && Yukawa3D(1f0) isa Yukawa3D{Float32}
    @test homogeneity(K3) == -1 && homogeneity(KS3) == -1 && homogeneity(Y) === nothing
    # the PDEs by automatic differentiation, and the hand-written derivatives against it
    for (K, c) in ((K3, 0.0), (H, 2.5^2), (Y, -2.5^2))
        Hr = ForwardDiff.hessian(tt -> real(K(s, tt)), t); Hi = ForwardDiff.hessian(tt -> imag(K(s, tt)), t)
        @test abs(tr(Hr) + im * tr(Hi) + c * K(s, t)) < 1e-12 * abs(K(s, t))
        g = ForwardDiff.gradient(tt -> real(K(s, tt)), t) + im * ForwardDiff.gradient(tt -> imag(K(s, tt)), t)
        @test gradient_t(K, s, t) ≈ g
        @test hessian_t(K, s, t) ≈ Hr + im * Hi
    end
    # Stokes: the kernel, its singularities and their pressures satisfy the Stokes equations
    f = P3(0.4, -1.1, 0.3); ν = unit3(rng); g = P3(-0.6, 0.2, 0.9); ω = P3(0.2, -0.5, 0.7)
    @test ncomp(KS3, Val(3)) == 3
    @test KS3(s, t) isa SMatrix{3,3,Float64}
    @test KS3(s, t) ≈ transpose(KS3(s, t)) && KS3(s, t) ≈ KS3(t, s)
    @test KS3(s, t) ≈ (one(SMatrix{3,3}) + (t - s) * transpose(t - s) / r^2) / (8π * r)
    function stokes_residual3(K, s, τ, t)
        u(tt) = source_field(K, s, tt, τ)
        p(tt) = target_field(K, s, tt, τ, Pressure())
        J = ForwardDiff.jacobian(u, t)
        lap = SVector((tr(ForwardDiff.hessian(tt -> u(tt)[a], t)) for a in 1:3)...)
        return tr(J), lap - ForwardDiff.gradient(p, t)
    end
    for τ in (f, Stresslet(f, ν), Doublet(g), Rotlet(ω), (f, Stresslet(f, ν), Doublet(g), Rotlet(ω)))
        div, res = stokes_residual3(KS3, s, τ, t)
        @test abs(div) < 1e-12 && norm(res) < 1e-12
        @test target_field(KS3, s, t, τ, Gradient()) ≈ ForwardDiff.jacobian(tt -> source_field(KS3, s, tt, τ), t)
    end
    @test target_field(KS3, s, t, f, Pressure()) ≈ sum((t - s) .* f) / (4π * r^3)
    # the doublet is half the Laplacian of the Stokeslet, the rotlet its antisymmetric derivative
    Gfun(ss) = KS3(ss, t) * g
    lapG = SVector((tr(ForwardDiff.hessian(ss -> Gfun(ss)[a], s)) for a in 1:3)...)
    @test source_field(KS3, s, t, Doublet(g)) ≈ lapG / 2
    dG = ForwardDiff.jacobian(ss -> vec(KS3(ss, t)), s)          # d(G_il)/d(s_k), rows (i,l) column-major
    Gk(i, l, k) = dG[i + 3(l - 1), k]
    ε(i, j, k) = (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)) ? 1 : (i, j, k) in ((3, 2, 1), (2, 1, 3), (1, 3, 2)) ? -1 : 0
    rot = SVector((sum(ε(j, k, l) * ω[j] * Gk(i, l, k) / 2 for j in 1:3, k in 1:3, l in 1:3) for i in 1:3)...)
    @test source_field(KS3, s, t, Rotlet(ω)) ≈ rot
    # JellySim's kernels, transcribed from src/hydro/kernels/stokes/single_point.jl
    ir = 1 / r; dn = (t - s) * ir
    @test source_field(KS3, s, t, f) ≈ (dot(dn, f) * dn + f) * (ir / (8π))
    @test source_field(KS3, s, t, Stresslet(f, ν)) ≈ dn * (dot(dn, f) * dot(dn, ν) * ir * ir * (-3 / (4π)))
    @test source_field(KS3, s, t, Doublet(g)) ≈ (-3 * dot(dn, g) * dn + g) * (ir^3 / (8π))
    # precision and allocations
    Kf = Stokeslet3D(Float32); sf = SVector{3,Float32}(s); tf = SVector{3,Float32}(t)
    @test Kf(sf, tf) isa SMatrix{3,3,Float32}
    @test Laplace3D(Float32)(sf, tf) isa Float32
    for τ in (f, Stresslet(f, ν), Doublet(g), Rotlet(ω))
        source_field(KS3, s, t, τ)
        @test @allocated(source_field(KS3, s, t, τ)) == 0
        @test @allocated(target_field(KS3, s, t, τ, Pressure())) == 0
    end
    @test @allocated(target_field(KS3, s, t, f, Gradient())) == 0
    @test @allocated(gradient_t(K3, s, t)) == 0 && @allocated(hessian_t(Y, s, t)) == 0
    # block matrices and direct sums
    xs = uniform_points3(rng, 7); ts = uniform_points3(rng, 5)
    A = kernel_matrix(KS3, xs, ts)
    @test size(A) == (15, 21)
    τv = [rand3(rng) for _ in 1:7]
    u = direct(KS3, xs, τv, ts)
    @test [u[j][a] for a in 1:3 for j in 1:5] ≈ A * [τv[i][a] for a in 1:3 for i in 1:7]
end

@testset "tree and lists on 3D points: $name" for (name, gen) in pairs(DISTRIBUTIONS3)
    for (N, ppl) in ((300, 4), (3000, 30))
        pts = gen(rng, N)
        check_tree_invariants(Tree(pts; ppl=ppl), pts)
    end
    check_coverage(Tree(gen(rng, 400); ppl=4))
end

@testset "3D tree edge cases" begin
    pts = uniform_points3(rng, 10)
    tree = Tree(pts; ppl=10)
    @test nlevels(tree) == 1 && nnodes(tree) == 1
    # eight octant clusters: exactly two levels, nine nodes, no ghosts
    ppl = 6
    pts = vec([P3(0.25 + 0.5a, 0.25 + 0.5b, 0.25 + 0.5c) + 0.01 * rand3(rng)
               for a in 0:1, b in 0:1, c in 0:1, _ in 1:ppl])
    tree = Tree(pts; ppl=ppl, bbox=(P3(0, 0, 0), 1.0))
    @test nlevels(tree) == 2 && nnodes(tree) == 9 && count(tree.ghost) == 0
    check_tree_invariants(tree, pts)
    t3 = Tree(mixed_points3(rng, 3000); ppl=8)
    @test count(t3.ghost) > 0 && nlevels(t3) >= 4
    @test Tree(reduce(hcat, pts); ppl=ppl, bbox=(P3(0, 0, 0), 1.0)).perm == tree.perm
    @test_throws ArgumentError FMM(KernelIndependent(Laplace2D(); nequiv=20), pts; ppl=10)
end

stokes3(rng, K, n) = [(rand3(rng), Stresslet(rand3(rng), unit3(rng)), Doublet(rand3(rng))) for _ in 1:n]
stresslets3(rng, K, n) = [Stresslet(rand3(rng), unit3(rng)) for _ in 1:n]
doublets3(rng, K, n) = [Doublet(rand3(rng)) for _ in 1:n]
rotlets3(rng, K, n) = [Rotlet(rand3(rng)) for _ in 1:n]

@testset "cube operators over all 316 offsets: $(name)" for (name, K, tol) in (
        ("Laplace", K3, 2e-6), ("Yukawa k=3", Yukawa3D(3.0), 1e-5), ("Helmholtz k=2", Helmholtz3D(2.0), 5e-6))
    m = CubeKernelIndependent{3}(K; p=8, d=0.1)
    @test nsurface(m) == 296 && nlarge(m) == 16^3 - 14^3
    @test local_length(m) == nlarge(m) && multipole_length(m) == 296
    check_m2l_all_offsets(rng, m, 0.37, tol)
    check_m2m(rng, m, 0.61, tol)
    check_l2l(rng, m, 0.61, tol)
    check_allocation_free(rng, m, 0.3; m2l=false)
end

@testset "cube operators, Stokes" begin
    m = CubeKernelIndependent{3}(KS3; p=8, d=0.1)
    @test multipole_length(m) == 3 * 296 + 1 && partial_local_length(m) == 3 * 296 + 1
    @test local_length(m) == 3 * nlarge(m)
    check_m2l_all_offsets(rng, m, 0.37, 5e-5)                 # stokeslets
    check_m2m(rng, m, 0.61, 5e-5)
    check_l2l(rng, m, 0.61, 5e-5)
    check_allocation_free(rng, m, 0.3; m2l=false)
    for (dname, density, tol) in (("stresslets", stresslets3, 2e-4), ("rotlets", rotlets3, 2e-4),
                                  ("doublets", doublets3, 2e-3))
        check_m2l_all_offsets(rng, m, 0.37, tol; density=density)
        check_m2m(rng, m, 0.61, tol; density=density)
        check_l2l(rng, m, 0.61, tol; density=density)
    end
end

function fmm_vs_direct3(m, pts, τ, targets; ppl=40, output=Potential())
    K = m.kernel
    f = FMM(m, pts; ppl=ppl, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    es = mrd(evaluate_at_sources(f, st; output=output), direct_self(K, pts, τ; output=output))
    et = mrd(evaluate(f, st, targets; output=output), direct(K, pts, τ, targets; output=output))
    return f, es, et
end

@testset "Laplace end to end, $name" for (name, gen) in pairs(DISTRIBUTIONS3)
    N = 5000
    pts = gen(rng, N)
    τ = randn(rng, N)
    targets = uniform_points3(rng, 2000)
    f, es, et = fmm_vs_direct3(CubeKernelIndependent{3}(K3; p=8, d=0.1), pts, τ, targets)
    @test nlevels(f) >= 3
    @test es < 1e-6 && et < 1e-6
end

@testset "Helmholtz, Yukawa, gradients and dipoles end to end" begin
    N = 4000
    pts = sphere_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points3(rng, 1000)
    for (K, tol) in ((Helmholtz3D(3.0), 5e-6), (Yukawa3D(2.0), 2e-6))
        f, es, et = fmm_vs_direct3(CubeKernelIndependent{3}(K; p=8, d=0.1), pts, τ, targets)
        @test es < tol && et < tol
    end
    m = CubeKernelIndependent{3}(K3; p=8, d=0.1)
    f, es, et = fmm_vs_direct3(m, pts, τ, targets; output=Gradient())
    @test es < 1e-5 && et < 1e-5
    τd = [Dipole(rand3(rng)) for _ in 1:N]
    f, es, et = fmm_vs_direct3(m, pts, τd, targets)
    @test es < 1e-5 && et < 1e-5
end

@testset "Stokes end to end with the augmentations" begin
    N = 4000
    pts = mixed_points3(rng, N)
    stk = [rand3(rng) for _ in 1:N]; nu = [unit3(rng) for _ in 1:N]; dub = [rand3(rng) for _ in 1:N]
    τ = [(stk[i], Stresslet(stk[i], nu[i]), Doublet(dub[i])) for i in 1:N]
    targets = uniform_points3(rng, 1500)
    m = CubeKernelIndependent{3}(KS3; p=8, d=0.1)
    # The tight cluster makes this adversarial for the doublets: a single far box can
    # dominate a target's doublet velocity, so their 1e-3 worst-case translation error
    # (3D accuracy table below) shows through, as does the spurious pressure of the
    # density representing them (a doublet has none). Uniform and sphere layouts give
    # 1e-6 to 1e-10 on the same quantities (see the uniform case below).
    f, es, et = fmm_vs_direct3(m, pts, τ, targets)
    @test nlevels(f) >= 6
    @test es < 1e-5 && et < 3e-4
    fp, esp, etp = fmm_vs_direct3(m, pts, τ, targets; output=Pressure())
    @test esp < 1e-3 && etp < 1e-2
    @info "3D Stokes on the sphere-plus-cluster layout, p = 8: velocity (self, targets) $(round(es, sigdigits=2)), $(round(et, sigdigits=2)); pressure $(round(esp, sigdigits=2)), $(round(etp, sigdigits=2))"
    f, es, et = fmm_vs_direct3(m, pts, τ, targets; output=Gradient())
    @test es < 1e-5 && et < 1e-5
    ptu = uniform_points3(rng, N)
    f, es, et = fmm_vs_direct3(m, ptu, τ, targets)
    @test es < 1e-6 && et < 1e-6
    f, es, et = fmm_vs_direct3(m, ptu, τ, targets; output=Pressure())
    @test es < 2e-5 && et < 2e-5
    τr = [Rotlet(rand3(rng)) for _ in 1:N]
    f, es, et = fmm_vs_direct3(m, pts, τr, targets)
    @test es < 1e-4 && et < 1e-4
end

@testset "tiny trees in 3D" begin
    m = CubeKernelIndependent{3}(K3; p=6, d=0.1)
    pts = uniform_points3(rng, 30); τ = randn(rng, 30); targets = uniform_points3(rng, 20)
    f, es, et = fmm_vs_direct3(m, pts, τ, targets; ppl=30)
    @test nlevels(f) == 1 && es < 1e-13 && et < 1e-13
    f, es, et = fmm_vs_direct3(m, pts, τ, targets; ppl=10)
    @test es < 1e-5 && et < 1e-5
end

function gausslegendre(n)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end
# Gauss–Legendre in cos θ times the trapezoid rule in φ: exact for spherical harmonics
# up to degree about 2nθ, so the layer potentials of smooth densities converge
# geometrically in nθ at points away from the surface.
function sphere_quadrature(c, a, nθ, nφ)
    x, wg = gausslegendre(nθ)
    pts = P3[]; wts = Float64[]; nrm = P3[]
    for i in 1:nθ, j in 0:nφ-1
        cθ = x[i]; sθ = sqrt(1 - cθ^2); φ = 2π * j / nφ
        n = P3(sθ * cos(φ), sθ * sin(φ), cθ)
        push!(pts, c + a * n); push!(wts, a^2 * wg[i] * 2π / nφ); push!(nrm, n)
    end
    return pts, wts, nrm
end

@testset "analytic: charged sphere, translating sphere, double layer" begin
    c = P3(0.5, 0.5, 0.5); a = 0.3
    sp, sw, sn = sphere_quadrature(c, a, 40, 80)
    @test sum(sw) ≈ 4π * a^2
    ext = filter(x -> norm(x - c) > 0.45, uniform_points3(rng, 3000))[1:300]
    int = filter(x -> norm(x - c) < 0.2, [c + 0.4 * (rand(rng, P3) .- 0.5) for _ in 1:3000])[1:300]
    # Laplace, unit surface charge: a²/r outside, a inside
    f = FMM(CubeKernelIndependent{3}(K3; p=8, d=0.1), sp; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    @test nlevels(f) >= 4
    st = build_expansions(f, sp, sw)
    @test mrd(evaluate(f, st, ext), [a^2 / norm(x - c) for x in ext]) < 1e-6
    @test mrd(evaluate(f, st, int), fill(a, length(int))) < 1e-6
    # Stokes: the uniform traction F/(4πa²) is the single layer of a sphere translating
    # with U = F/(6πa): outside (1 + a²/6 ∇²)G·F = G·F + (a²/3) D·F, inside U; the
    # pressure is that of a stokeslet outside and zero inside
    F = P3(0.3, -0.7, 1.1)
    fs = [wj * F / (4π * a^2) for wj in sw]
    fS = FMM(CubeKernelIndependent{3}(KS3; p=8, d=0.1), sp; ppl=40, bbox=(P3(0, 0, 0), 1.0))
    stS = build_expansions(fS, sp, fs)
    uan = [source_field(KS3, c, x, F) + (a^2 / 3) * source_field(KS3, c, x, Doublet(F)) for x in ext]
    @test mrd(evaluate(fS, stS, ext), uan) < 1e-5
    @test mrd(evaluate(fS, stS, int), fill(F / (6π * a), length(int))) < 1e-5
    @test mrd(evaluate(fS, stS, ext; output=Pressure()), [target_field(KS3, c, x, F, Pressure()) for x in ext]) < 1e-4
    @test maximum(abs.(evaluate(fS, stS, int; output=Pressure()))) < 1e-4 * norm(F) / a^2
    # the double layer of a constant vector with the outward normal: F₀ inside, 0 outside
    F0 = P3(1.0, -2.0, 0.5)
    dl = [Stresslet(wj * F0, nj) for (wj, nj) in zip(sw, sn)]
    stD = build_expansions(fS, sp, dl)
    @test mrd(evaluate(fS, stD, int), fill(F0, length(int))) < 1e-4
    @test maximum(norm.(evaluate(fS, stD, ext))) < 1e-4 * norm(F0)
end

@testset "dual positions through the 3D FMM" begin
    D1 = ForwardDiff.Dual{:d3,Float64,1}
    dual3(p, v) = SVector(D1(p[1], ForwardDiff.Partials((v[1],))), D1(p[2], ForwardDiff.Partials((v[2],))),
                          D1(p[3], ForwardDiff.Partials((v[3],))))
    N = 2000
    pts = helix_points(rng, N)
    V = [rand3(rng) for _ in 1:N]
    targets = uniform_points3(rng, 400)
    sd = [dual3(p, v) for (p, v) in zip(pts, V)]
    τ = randn(rng, N)
    f = FMM(CubeKernelIndependent{3}(K3; p=8, d=0.1), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    st = build_expansions(f, sd, τ)
    u = evaluate(f, st, targets)
    uref = direct(K3, sd, τ, targets)
    @test mrd(vals(u), vals(uref)) < 1e-6
    @test mrd(parts(u), parts(uref)) < 1e-5
    # the slender-body pattern: stokeslets and doublets at dual positions, self evaluation
    τs = [(rand3(rng), Doublet(0.01 * rand3(rng))) for _ in 1:N]
    fS = FMM(CubeKernelIndependent{3}(KS3; p=8, d=0.1), pts; ppl=30, bbox=(P3(0, 0, 0), 1.0))
    stS = build_expansions(fS, sd, τs)
    us = evaluate_at_sources(fS, stS)
    usref = direct_self(KS3, sd, τs)
    @test mrd(vals(us), vals(usref)) < 1e-5
    @test mrd(parts(us), parts(usref)) < 1e-4
end

@testset "Float32 in 3D" begin
    Kf = Laplace3D(Float32)
    N = 3000
    pts = [SVector{3,Float32}(p) for p in uniform_points3(rng, N)]
    τ = randn(rng, Float32, N)
    targets = [SVector{3,Float32}(p) for p in uniform_points3(rng, 500)]
    f = FMM(CubeKernelIndependent{3}(Kf; p=6, d=0.1), pts; ppl=30, bbox=(SVector{3,Float32}(0, 0, 0), 1f0))
    st = build_expansions(f, pts, τ)
    @test eltype(st.multipoles) == Float32
    u = evaluate(f, st, targets)
    @test eltype(u) == Float32
    @test mrd(u, direct(Kf, pts, τ, targets)) < 1e-3
end

@testset "Stokes front end in 3D" begin
    N = 3000
    srcs = helix_points(rng, N); tg = uniform_points3(rng, 500)
    w = rand(rng, N)
    stk = [rand3(rng) for _ in 1:N]; strs = [rand3(rng) for _ in 1:N]; nu = [unit3(rng) for _ in 1:N]
    dub = [rand3(rng) for _ in 1:N]
    plan = StokesFMMPlan(srcs; targets=tg, ppl=30)                  # defaults in 3D: :cube, p = 8
    @test plan isa StokesFMMPlan{3}
    @test plan.fmm.method isa CubeKernelIndependent{3,<:Stokeslet3D{Float64}}
    @test plan.fmm.method.p == 8 && plan.fmm.method.pc == 16
    τ = stokes_densities(N; weights=w, stokeslets=stk, stresslets=strs, stressvecs=nu, doublets=dub)
    u = Vector{P3}(undef, 500)
    stokes_fmm_eval!(u, plan; weights=w, stokeslets=stk, stresslets=strs, stressvecs=nu, doublets=dub)
    @test mrd(u, direct(KS3, srcs, τ, tg)) < 1e-5
    us = Vector{P3}(undef, N)
    stokes_fmm_eval_src_trg!(us, u, plan; weights=w, stokeslets=stk, doublets=dub)
    τ2 = stokes_densities(N; weights=w, stokeslets=stk, doublets=dub)
    @test mrd(us, direct_self(KS3, srcs, τ2)) < 1e-5 && mrd(u, direct(KS3, srcs, τ2, tg)) < 1e-5
    # one shot, self, matrix input
    u1 = StokesFMMKernel(reduce(hcat, srcs); stokeslets=stk, ppl=30)
    @test mrd(u1, direct_self(KS3, srcs, stk)) < 1e-5
    @test_throws ArgumentError StokesFMMPlan(srcs; method=:ki)
    @test_throws ArgumentError StokesFMMPlan(srcs; method=:bb)
end

@testset "3D accuracy table" begin
    # worst case: a unit source at the near corner of the source box at offset (2, 0, 0),
    # targets on the facing face of the target box
    w = 1.0; off = SVector(2, 0, 0); cs = P3(off)
    xs = [cs + P3(-0.5, 0.5, 0.5)]
    ts = [P3(0.5, 0.5, 0.5), P3(0.5, 0, 0), P3(0.5, -0.5, -0.5), P3(0.5, 0.5, 0), P3(0.25, 0.5, 0.5)]
    function worst(m, τ)
        K = m.kernel
        ops = level_operators(m, w)
        fi = findfirst(==(off), far_offsets(Val(3)))
        M = multipole_of(m, ops, xs, τ, cs)
        pL = zeros(eltype(M), partial_local_length(m)); apply!(pL, ops.M2L[fi], M)
        L = local_of(m, ops, pL)
        return mrd([l2t(m, ops, L, t) for t in ts], direct(K, xs, τ, ts))
    end
    f = P3(0.4, -1.1, 0.3); ν = unit3(rng); g = P3(-0.6, 0.2, 0.9)
    rows = String[]
    for p in (6, 8, 10)
        eL = worst(CubeKernelIndependent{3}(K3; p=p, d=0.1), [1.0])
        line = "p = $p, pc = $(2p) ($(p^3 - (p-2)^3) / $((2p)^3 - (2p-2)^3) points): Laplace $(round(eL, sigdigits=2))"
        if p <= 8
            mS = CubeKernelIndependent{3}(KS3; p=p, d=0.1)
            es = [round(worst(mS, [τ]), sigdigits=2) for τ in (f, Stresslet(f, ν), Doublet(g))]
            line *= "; stokeslet $(es[1]), stresslet $(es[2]), doublet $(es[3])"
        end
        push!(rows, line)
    end
    @info "3D worst-case translation error (corner source, facing face), d = 0.1\n" * join(rows, "\n")
    @test worst(CubeKernelIndependent{3}(K3; p=8, d=0.1), [1.0]) < 5e-6
end
