# General sources (charges, dipoles, tuples) and targets (potential, gradient):
# kernel derivatives, the field functions, the Chebyshev derivative machinery, both
# methods end to end, and the cross-check of the two ways to form a position
# derivative (dual positions versus a dipole FMM).
using FlexibleFMM: density_scalar, chebyshev_values, chebyshev_values_derivatives,
    clenshaw2d, clenshaw2d_gradient, nlevels, AbstractKernel, evaluate_target, locate,
    s2m!, l2t, expansion_length, operator_eltype, center

rng = MersenneTwister(14)
mrd(a, b) = maximum(norm.(a .- b)) / maximum(norm.(b))     # scalars or SVectors
const SDual = ForwardDiff.Dual{:src,Float64,1}
sdual(x, v) = SDual(x, ForwardDiff.Partials((v,)))
spartial(x) = ForwardDiff.partials(x, 1)
T_exact(j, x) = cos(j * acos(x))

# kernels without hand-written derivatives, to exercise the generic ForwardDiff path
struct GenericLaplace <: AbstractKernel end
(::GenericLaplace)(s, t) = Laplace2D()(s, t)
struct GenericHelmholtz <: AbstractKernel
    k::Float64
end
(K::GenericHelmholtz)(s, t) = Helmholtz2D(K.k)(s, t)

@testset "kernel derivatives" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3)
    e1 = P2(1, 0); e2 = P2(0, 1); h = 1e-6
    fdgrad(K) = SVector((K(s, t + h * e1) - K(s, t - h * e1)) / 2h, (K(s, t + h * e2) - K(s, t - h * e2)) / 2h)
    for K in (Laplace2D(), Helmholtz2D(3.0), Yukawa2D(2.0), LaplaceDipoleTest(0.6))
        g = gradient_t(K, s, t)
        @test norm(g - fdgrad(K)) < 1e-7 * (1 + norm(g))
        @test gradient_s(K, s, t) == -g
        # Hessian against ForwardDiff of the gradient, and symmetry
        Hr = ForwardDiff.jacobian(tt -> real.(gradient_t(K, s, tt)), t)
        Hi = ForwardDiff.jacobian(tt -> imag.(gradient_t(K, s, tt)), t)
        H = hessian_t(K, s, t)
        @test norm(H - (Hr + im * Hi)) < 1e-9 * (1 + norm(H))
        @test norm(H - transpose(H)) < 1e-10 * norm(H)
    end
    # hand-written and generic paths agree
    @test gradient_t(Laplace2D(), s, t) ≈ gradient_t(GenericLaplace(), s, t)
    @test hessian_t(Laplace2D(), s, t) ≈ hessian_t(GenericLaplace(), s, t)
    @test gradient_t(Helmholtz2D(3.0), s, t) ≈ gradient_t(GenericHelmholtz(3.0), s, t)
    @test hessian_t(Helmholtz2D(3.0), s, t) ≈ hessian_t(GenericHelmholtz(3.0), s, t)
    # precision is preserved
    sf = SVector{2,Float32}(0.1, 0.2); tf = SVector{2,Float32}(0.7, -0.3)
    @test gradient_t(Laplace2D(Float32), sf, tf) isa SVector{2,Float32}
    @test gradient_t(Helmholtz2D(3f0), sf, tf) isa SVector{2,ComplexF32}
    @test gradient_t(Yukawa2D(2f0), sf, tf) isa SVector{2,Float32}
    @test hessian_t(Laplace2D(Float32), sf, tf) isa SMatrix{2,2,Float32}
    @test hessian_t(Yukawa2D(2f0), sf, tf) isa SMatrix{2,2,Float32}
    for K in (Laplace2D(), Helmholtz2D(3.0), Yukawa2D(2.0))
        gradient_t(K, s, t); hessian_t(K, s, t)
        @test @allocated(gradient_t(K, s, t)) == 0
        @test @allocated(hessian_t(K, s, t)) == 0
    end
end

@testset "source and target fields" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3); d = P2(0.3, -0.5)
    @test Dipole(2.0, P2(0.5, 0.5)).d == P2(1.0, 1.0)
    @test density_scalar(Float64) == Float64
    @test density_scalar(Dipole{2,Float32}) == Float32
    @test density_scalar(Tuple{Float64,Dipole{2,ComplexF64}}) == ComplexF64
    for K in (Laplace2D(), Yukawa2D(2.0), LaplaceDipoleTest(0.4))
        # dipole field = directional derivative of the kernel in the source
        @test source_field(K, s, t, Dipole(d)) ≈ ForwardDiff.derivative(ε -> K(s + ε * d, t), 0.0)
        @test source_field(K, s, t, (1.5, Dipole(d))) ≈ 1.5 * K(s, t) + source_field(K, s, t, Dipole(d))
        # gradient outputs = gradient of the field in the target
        @test target_field(K, s, t, 2.0, Gradient()) ≈ ForwardDiff.gradient(tt -> source_field(K, s, tt, 2.0), t)
        @test target_field(K, s, t, Dipole(d), Gradient()) ≈ ForwardDiff.gradient(tt -> source_field(K, s, tt, Dipole(d)), t)
        @test target_field(K, s, t, (2.0, Dipole(d)), Gradient()) ≈
              target_field(K, s, t, 2.0, Gradient()) + target_field(K, s, t, Dipole(d), Gradient())
        @test target_field(K, s, t, 2.0, Potential()) == source_field(K, s, t, 2.0)
    end
    H = Helmholtz2D(2.0)
    @test source_field(H, s, t, Dipole(d)) ≈ ForwardDiff.derivative(ε -> real(H(s + ε * d, t)), 0.0) +
                                              im * ForwardDiff.derivative(ε -> imag(H(s + ε * d, t)), 0.0)
    gH = target_field(H, s, t, Dipole(d), Gradient())
    @test gH ≈ ForwardDiff.gradient(tt -> real(source_field(H, s, tt, Dipole(d))), t) +
               im * ForwardDiff.gradient(tt -> imag(source_field(H, s, tt, Dipole(d))), t)
    K = Laplace2D()
    for τ in (1.0, Dipole(d), (1.0, Dipole(d)))
        target_field(K, s, t, τ, Gradient())
        @test @allocated(target_field(K, s, t, τ, Gradient())) == 0
        @test @allocated(source_field(K, s, t, τ)) == 0
    end
    # reference sums with dipoles and gradient outputs
    pts = uniform_points(rng, 6); dips = [Dipole(P2(randn(rng), randn(rng))) for _ in 1:6]
    tg = uniform_points(rng, 4)
    @test direct(K, pts, dips, tg) ≈ [sum(source_field(K, p, x, dd) for (p, dd) in zip(pts, dips)) for x in tg]
    @test direct(K, pts, dips, tg; output=Gradient()) ≈ [sum(target_field(K, p, x, dd, Gradient()) for (p, dd) in zip(pts, dips)) for x in tg]
    @test direct_self(K, pts, dips; output=Gradient())[2] ≈ sum(target_field(K, pts[i], pts[2], dips[i], Gradient()) for i in 1:6 if i != 2)
end

@testset "nested duals through the generic derivative fallback" begin
    s = P2(0.1, 0.2); t = P2(0.7, -0.3); d = P2(0.3, -0.5); V = P2(0.2, 0.9)
    g(K) = ε -> source_field(K, s + ε * V, t, Dipole(d))
    @test ForwardDiff.derivative(g(GenericLaplace()), 0.0) ≈ ForwardDiff.derivative(g(Laplace2D()), 0.0)
    h(K) = ε -> target_field(K, s + ε * V, t, 1.3, Gradient())
    @test ForwardDiff.derivative(h(GenericLaplace()), 0.0) ≈ ForwardDiff.derivative(h(Laplace2D()), 0.0)
    q(K) = ε -> real(target_field(K, s + ε * V, t, Dipole(d), Gradient()))
    @test ForwardDiff.derivative(q(GenericHelmholtz(2.0)), 0.0) ≈ ForwardDiff.derivative(q(Helmholtz2D(2.0)), 0.0)
end

@testset "chebyshev derivatives and 2D gradient" begin
    for p in (1, 2, 3, 8)
        for x in (2rand(rng, 3) .- 1)
            tv, dv = chebyshev_values_derivatives(Val(p), x)
            @test tv ≈ chebyshev_values(Val(p), x)
            @test dv ≈ [ForwardDiff.derivative(y -> T_exact(j, y), x) for j in 0:p-1] atol = 1e-12
        end
        c = randn(rng, p^2)
        for q in (2rand(rng, 2) .- 1, 2rand(rng, 2) .- 1)
            x, y = q[1], q[2]
            gx, gy = clenshaw2d_gradient(c, p, x, y)
            @test gx ≈ ForwardDiff.derivative(xx -> clenshaw2d(c, p, xx, y), x) atol = 1e-12
            @test gy ≈ ForwardDiff.derivative(yy -> clenshaw2d(c, p, x, yy), y) atol = 1e-12
        end
    end
    c = randn(rng, 64)
    clenshaw2d_gradient(c, 8, 0.1, 0.2)
    @test @allocated(clenshaw2d_gradient(c, 8, 0.1, 0.2)) == 0
    chebyshev_values_derivatives(Val(8), 0.3)
    @test @allocated(chebyshev_values_derivatives(Val(8), 0.3)) == 0
end

methods_for(K) = (("KI nequiv=60", KernelIndependent(K; nequiv=60)), ("BB p=14", BlackBox(K; p=14)))

@testset "dipoles, gradients and tuples end to end, $(kname), $(mname)" for
        (kname, K) in (("Laplace", Laplace2D()), ("Helmholtz k=5", Helmholtz2D(5.0))),
        (mname, m) in methods_for(K)
    N = 5000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    dips = [Dipole(P2(randn(rng), randn(rng))) for _ in 1:N]
    comb = [(τ[i], dips[i]) for i in 1:N]
    targets = uniform_points(rng, 2000)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    @test nlevels(f) >= 4
    st_c = build_expansions(f, pts, τ)
    st_d = build_expansions(f, pts, dips)
    st_t = build_expansions(f, pts, comb)
    @test eltype(st_d.multipoles) == operator_eltype(m)
    # dipole sources, potential output
    @test mrd(evaluate(f, st_d, targets), direct(K, pts, dips, targets)) < 1e-6
    @test mrd(evaluate_at_sources(f, st_d), direct_self(K, pts, dips)) < 1e-6
    # gradient output for charges and for dipoles (the hypersingular combination)
    ug = evaluate(f, st_c, targets; output=Gradient())
    @test eltype(ug) <: SVector{2}
    @test mrd(ug, direct(K, pts, τ, targets; output=Gradient())) < 1e-6
    @test mrd(evaluate_at_sources(f, st_c; output=Gradient()), direct_self(K, pts, τ; output=Gradient())) < 1e-6
    @test mrd(evaluate(f, st_d, targets; output=Gradient()), direct(K, pts, dips, targets; output=Gradient())) < 1e-5
    # tuple densities: linear in their parts and correct
    @test evaluate(f, st_t, targets) ≈ evaluate(f, st_c, targets) .+ evaluate(f, st_d, targets)
    @test mrd(evaluate(f, st_t, targets), direct(K, pts, comb, targets)) < 1e-6
    @test mrd(evaluate_at_sources(f, st_t; output=Gradient()), direct_self(K, pts, comb; output=Gradient())) < 1e-5
end

@testset "position derivatives: dual positions versus dipole FMM, $(mname)" for (mname, m) in methods_for(Laplace2D())
    N = 3000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    V = [P2(randn(rng), randn(rng)) for _ in 1:N]
    W = [P2(randn(rng), randn(rng)) for _ in 1:800]
    targets = uniform_points(rng, 800)
    f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
    # d/dε Σ K(s_j + εV_j, t) τ_j = Σ τ_j V_j·∇ₛK(s_j, t): a dipole FMM with moments τ_j V_j
    sd = [SVector(sdual(p[1], v[1]), sdual(p[2], v[2])) for (p, v) in zip(pts, V)]
    ud = evaluate(f, build_expansions(f, sd, τ), targets)
    up = evaluate(f, build_expansions(f, pts, [Dipole(τ[j] * V[j]) for j in 1:N]), targets)
    @test mrd(spartial.(ud), up) < 1e-6
    # d/dε u(t + εW) = W·∇u: the gradient output
    st = build_expansions(f, pts, τ)
    td = [SVector(sdual(t[1], w[1]), sdual(t[2], w[2])) for (t, w) in zip(targets, W)]
    utd = evaluate(f, st, td)
    ug = evaluate(f, st, targets; output=Gradient())
    @test mrd(spartial.(utd), [dot(w, g) for (w, g) in zip(W, ug)]) < 1e-6
    # dual dipole moments (derivative with respect to the density of a double layer)
    dd = [Dipole(SVector(sdual(v[1], 1.0), sdual(v[2], 0.0))) for v in V]
    udd = evaluate(f, build_expansions(f, pts, dd), targets)
    @test eltype(udd) == SDual
    @test mrd(spartial.(udd), evaluate(f, build_expansions(f, pts, fill(Dipole(P2(1, 0)), N)), targets)) < 1e-6
end

@testset "Float32 dipoles and gradients" begin
    K = Laplace2D(Float32)
    P2f = SVector{2,Float32}
    sf = P2f(0.1, 0.2); tf = P2f(0.7, -0.3); df = Dipole(P2f(0.3, -0.5))
    @test source_field(K, sf, tf, df) isa Float32
    @test target_field(K, sf, tf, 1f0, Gradient()) isa SVector{2,Float32}
    @test target_field(K, sf, tf, df, Gradient()) isa SVector{2,Float32}
    N = 3000
    pts = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:N]
    dips = [Dipole(P2f(randn(rng, Float32), randn(rng, Float32))) for _ in 1:N]
    targets = [P2f(rand(rng, Float32), rand(rng, Float32)) for _ in 1:500]
    for m in (KernelIndependent(K; nequiv=40), BlackBox(K; p=8))
        f = FMM(m, pts; ppl=40, bbox=(P2f(0, 0), 1f0))
        st = build_expansions(f, pts, dips)
        @test eltype(st.multipoles) == Float32
        u = evaluate(f, st, targets)
        @test eltype(u) == Float32
        ug = evaluate(f, st, targets; output=Gradient())
        @test eltype(ug) == SVector{2,Float32}
        p64 = [SVector{2,Float64}(p) for p in pts]; d64 = [Dipole(SVector{2,Float64}(d.d)) for d in dips]
        t64 = [SVector{2,Float64}(t) for t in targets]
        @test mrd(u, direct(Laplace2D(), p64, d64, t64)) < 1e-3
        @test mrd(ug, direct(Laplace2D(), p64, d64, t64; output=Gradient())) < 1e-3
    end
end

@testset "allocation-free bodies with dipoles and gradients" begin
    N = 2000
    pts = uniform_points(rng, N)
    dips = [Dipole(P2(randn(rng), randn(rng))) for _ in 1:N]
    for m in (KernelIndependent(Laplace2D(); nequiv=40), BlackBox(Laplace2D(); p=10),
              KernelIndependent(Helmholtz2D(3.0); nequiv=40))
        f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
        st = build_expansions(f, pts, dips)
        x = P2(0.37, 0.61)
        i = locate(f.tree, x)
        for out in (Potential(), Gradient())
            evaluate_target(f, st, x, i, 0, out)
            @test @allocated(evaluate_target(f, st, x, i, 0, out)) == 0
        end
        node = f.s2m_nodes[end][1]
        r = f.tree.points[node]
        col = zeros(eltype(st.multipoles), expansion_length(m))
        xs = view(st.sources, r); ts = view(st.τ, r); c = center(f.tree, node)
        s2m!(col, m, f.ops[end], xs, ts, c)
        @test @allocated(s2m!(col, m, f.ops[end], xs, ts, c)) == 0
        l2t(m, f.ops[end], col, P2(0.01, 0.02), Gradient())
        @test @allocated(l2t(m, f.ops[end], col, P2(0.01, 0.02), Gradient())) == 0
    end
end
