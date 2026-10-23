# Determinism of the threaded passes and allocation-free hot bodies.
using FlexibleFMM: evaluate_target, locate, nnodes, s2m!, l2t, apply!, expansion_length,
    operator_eltype, center

rng = MersenneTwister(10)

@testset "threaded results are bitwise reproducible and equal the serial bodies" begin
    N = 5000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 3000)
    for m in (KernelIndependent(Laplace2D(); nequiv=40), BlackBox(Helmholtz2D(3.0); p=10))
        f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
        st = build_expansions(f, pts, τ)
        st2 = build_expansions(f, pts, τ)
        @test st.multipoles == st2.multipoles && st.locals == st2.locals
        u1 = evaluate(f, st, targets)
        u2 = evaluate(f, st, targets)
        @test u1 == u2
        nodes = locate(f.tree, targets)
        u3 = [evaluate_target(f, st, targets[j], nodes[j], 0) for j in eachindex(targets)]
        @test u1 == u3
        s1 = evaluate_at_sources(f, st)
        tree = f.tree
        s3 = similar(s1)
        for j in 1:N
            s3[tree.perm[j]] = evaluate_target(f, st, st.sources[1][j], tree.point_node[j], j)
        end
        @test s1 == s3
    end
end

@testset "allocation-free per-target and per-leaf bodies" begin
    N = 3000
    pts = uniform_points(rng, N)
    τ = randn(rng, N)
    for m in (KernelIndependent(Laplace2D(); nequiv=40), BlackBox(Laplace2D(); p=10),
              KernelIndependent(Helmholtz2D(3.0); nequiv=40))
        f = FMM(m, pts; ppl=30, bbox=(P2(0, 0), 1.0))
        st = build_expansions(f, pts, τ)
        x = P2(0.37, 0.61)
        i = locate(f.tree, x)
        evaluate_target(f, st, x, i, 0)
        @test @allocated(evaluate_target(f, st, x, i, 0)) == 0
        node = f.s2m_nodes[end][1]
        r = f.tree.points[node]
        col = zeros(eltype(st.multipoles), expansion_length(m))
        xs = view(st.sources[1], r); ts = view(st.τ[1], r); c = center(f.tree, node)
        s2m!(col, m, f.ops[end], xs, ts, c)
        @test @allocated(s2m!(col, m, f.ops[end], xs, ts, c)) == 0
    end
end
