# Groups of sources with different density types in one FMM: the per-group tree
# tables, and mixed groups against direct sums per group and output.
using FlexibleFMM: ngroups, nlevels, locate

rng = MersenneTwister(51)

function check_group_tables(t::Tree, groups)
    G = length(groups)
    @test ngroups(t) == G && t.group_sizes == length.(collect(groups))
    nc = 1 << length(t.origin)
    for g in 1:G
        n = t.group_sizes[g]
        @test sort(t.gperm[g]) == 1:n                                          # a permutation of the group
        # childless ranges partition the group, in sorted order; children partition their parent in slot order
        childless = [i for i in eachindex(t.points) if t.child_first[i] == 0]
        rs = sort([t.gpoints[g][i] for i in childless if !isempty(t.gpoints[g][i])]; by=first)   # childless nodes span levels
        @test sum(length, rs; init=0) == n && all(first(rs[k+1]) == last(rs[k]) + 1 for k in 1:length(rs)-1) && first(rs[1]) == 1
        for i in eachindex(t.points)
            cf = t.child_first[i]
            cf == 0 && continue
            cr = [t.gpoints[g][c] for c in cf:cf+nc-1]
            nonempty = filter(!isempty, cr)
            @test sum(length, cr) == length(t.gpoints[g][i])
            isempty(nonempty) || @test first(nonempty[1]) == first(t.gpoints[g][i]) && last(nonempty[end]) == last(t.gpoints[g][i])
        end
        # the group-local sorted points are the group's points in the tree's sorted order
        pts = groups[g]
        @test all(t.gpoint_node[g][k] == locate(t, [pts[t.gperm[g][k]]])[1] for k in 1:min(n, 500))
        # and the global view agrees: node i's group-g points are those of points[i] belonging to the group
        off = sum(t.group_sizes[1:g-1]; init=0)
        for i in rand(rng, eachindex(t.points), 50)
            ingroup = count(j -> off < t.perm[j] <= off + n, t.points[i])
            @test ingroup == length(t.gpoints[g][i])
        end
    end
end

@testset "per-group tree tables, $(name)" for (name, groups, ppl) in (
        ("two 2D groups", (uniform_points(rng, 3000), [0.5 .* p for p in uniform_points(rng, 2000)]), 30),
        ("three 3D groups", (sphere_points(rng, 4000), helix_points(rng, 3000), clustered_points3(rng, 1500)), 40),
        ("a one-point group", (sphere_points(rng, 2000), [P3(0.3, 0.3, 0.3)]), 25))
    t = Tree(groups; ppl=ppl, bbox=(zero(eltype(groups[1])), 1.0))
    check_group_tables(t, groups)
    @test Tree(vcat(groups...); ppl=ppl, bbox=(zero(eltype(groups[1])), 1.0)).perm == t.perm   # the same tree
    t1 = Tree((groups[1],); ppl=ppl, bbox=(zero(eltype(groups[1])), 1.0))
    @test t1.gpoints[1] == t1.points && t1.gperm[1] == t1.perm && t1.gpoint_node[1] == t1.point_node
end

# tolerances at the methods' own accuracy for stresslets (measured: velocities 1.9e-6
# at p = 16 and 5.4e-4 for the cube at p = 6, the cube's double-layer pressure 4.3e-4
# at p = 8; three one-group FMMs give the same orders)
@testset "mixed groups, 3D: $(name)" for (name, m, tol) in (
        ("Stokes harmonics", StokesHarmonics(Stokeslet3D(); p=16), 1e-5),
        ("Stokes cube", CubeKernelIndependent{3}(Stokeslet3D(); p=8), 1e-3))
    K = Stokeslet3D()
    groups = (sphere_points(rng, 3000), helix_points(rng, 2500), clustered_points3(rng, 1000), uniform_points3(rng, 400))
    τs = ([Stresslet(randn(rng, P3), normalize(randn(rng, P3))) for _ in 1:3000],          # a double layer
          [(randn(rng, P3), Doublet(randn(rng, P3))) for _ in 1:2500],                      # stokeslets and doublets
          [randn(rng, P3) for _ in 1:1000],                                                # stokeslets
          nothing)                                                                         # targets only
    box = (P3(0, 0, 0), 1.0)
    f = FMM(m, groups; ppl=40, bbox=box)
    st = build_expansions(f, groups, τs)
    @test ngroups(f.tree) == 4 && ngroups(st) == 4
    # the reference: every source of every group, at every point
    allpts = vcat(groups[1:3]...)
    for g in 1:4
        pts = groups[g]
        u = evaluate_at_sources(f, st, g)
        @test eltype(u) === P3                                                   # concrete with a target-only group
        uref = [sum(FlexibleFMM.source_field(K, allpts[j], pts[i], τ) for (τ, j) in
                    ((τs[1][k], k) for k in 1:3000) if j != (g == 1 ? i : 0)) +
                sum(FlexibleFMM.source_field(K, groups[2][k], pts[i], τs[2][k]) for k in 1:2500 if !(g == 2 && k == i)) +
                sum(FlexibleFMM.source_field(K, groups[3][k], pts[i], τs[3][k]) for k in 1:1000 if !(g == 3 && k == i))
                for i in 1:min(length(pts), 150)]
        @test mrd(u[1:length(uref)], uref) < tol
        # the pressure and the gradient at the mobile-object-like group, against the parts
        if g == 3
            pr = evaluate_at_sources(f, st, g; output=Pressure())
            pref = [sum(FlexibleFMM.target_field(K, groups[1][k], pts[i], τs[1][k], Pressure()) for k in 1:3000) +
                    sum(FlexibleFMM.target_field(K, groups[2][k], pts[i], τs[2][k], Pressure()) for k in 1:2500) +
                    sum(FlexibleFMM.target_field(K, groups[3][k], pts[i], τs[3][k], Pressure()) for k in 1:1000 if k != i)
                    for i in 1:100]
            @test mrd(pr[1:100], pref) < 10tol
        end
    end
    # external targets
    tg = uniform_points3(rng, 200)
    ut = evaluate(f, st, tg)
    utref = [sum(FlexibleFMM.source_field(K, groups[1][k], x, τs[1][k]) for k in 1:3000) +
             sum(FlexibleFMM.source_field(K, groups[2][k], x, τs[2][k]) for k in 1:2500) +
             sum(FlexibleFMM.source_field(K, groups[3][k], x, τs[3][k]) for k in 1:1000) for x in tg]
    @test mrd(ut, utref) < tol
    # the one-group forms are the G = 1 case, bitwise
    f1 = FMM(m, groups[3]; ppl=40, bbox=box)
    st1 = build_expansions(f1, groups[3], τs[3])
    st1t = build_expansions(f1, (groups[3],), (τs[3],))
    @test st1.multipoles == st1t.multipoles && st1.locals == st1t.locals
    @test evaluate_at_sources(f1, st1) == evaluate_at_sources(f1, st1t, 1)
    # a state reused for new densities, and the guard on a target-only group
    build_expansions!(st, f, groups, τs)
    @test_throws ArgumentError build_expansions!(st, f, groups, (τs[1], τs[2], τs[3], τs[3][1:400]))
end

@testset "mixed groups, 2D, kernel-independent and black-box" begin
    K = Laplace2D()
    groups = (uniform_points(rng, 3000), circle_points(rng, 2000))
    τs = (randn(rng, 3000), [Dipole(randn(rng, P2)) for _ in 1:2000])
    box = (P2(0, 0), 1.0)
    for m in (KernelIndependent(K; nequiv=60), BlackBox(K; p=14))              # 1e-13 and 2e-10 measured
        f = FMM(m, groups; ppl=30, bbox=box)
        st = build_expansions(f, groups, τs)
        for g in 1:2
            pts = groups[g]
            u = evaluate_at_sources(f, st, g)
            uref = [sum(FlexibleFMM.source_field(K, groups[1][k], pts[i], τs[1][k]) for k in 1:3000 if !(g == 1 && k == i)) +
                    sum(FlexibleFMM.source_field(K, groups[2][k], pts[i], τs[2][k]) for k in 1:2000 if !(g == 2 && k == i))
                    for i in 1:200]
            @test mrd(u[1:200], uref) < 1e-8
        end
    end
end

@testset "groups through the device passes (CPU backend): $(name)" for (name, m, tol) in (
        ("Stokes harmonics", StokesHarmonics(Stokeslet3D(); p=10), 1e-9),
        ("Stokes cube", CubeKernelIndependent{3}(Stokeslet3D(); p=6), 1e-9))
    groups = (sphere_points(rng, 2500), helix_points(rng, 2000), clustered_points3(rng, 800), uniform_points3(rng, 300))
    τs = ([Stresslet(randn(rng, P3), normalize(randn(rng, P3))) for _ in 1:2500], [(randn(rng, P3), Doublet(randn(rng, P3))) for _ in 1:2000],
          [randn(rng, P3) for _ in 1:800], nothing)
    box = (P3(0, 0, 0), 1.0)
    cache = OperatorCache(m)
    f = FMM(m, groups; ppl=40, bbox=box, cache=cache)
    st = build_expansions(f, groups, τs)
    ucpu = [evaluate_at_sources(f, st, g) for g in 1:4]
    pcpu = evaluate_at_sources(f, st, 3; output=Pressure())
    # the plan from the host tree (uploaded tables) and from the points (device tables)
    plans = (DevicePlan(f), DevicePlan(m, groups; ppl=40, bbox=box, cache=cache))
    for plan in plans
        @test FlexibleFMM.ngroups(plan.dev) == 4
        outs = (similar(ucpu[1]), similar(ucpu[2]), similar(pcpu), similar(ucpu[4]))
        device_matvec!(outs, plan, (sources=groups, τ=τs); outputs=(Potential(), Potential(), Pressure(), Potential()))
        @test mrd(outs[1], ucpu[1]) < tol && mrd(outs[2], ucpu[2]) < tol && mrd(outs[4], ucpu[4]) < tol
        @test mrd(outs[3], pcpu) < tol
        @test_throws ArgumentError device_matvec!(similar(ucpu[1]), plan, (sources=groups, τ=τs))   # one vector for four groups
    end
    # inputs already in the host tree's order (what device_state gives) are valid for the plan
    # from that tree only (the device tree orders a leaf's points differently); an output left out
    ds = device_state(st)
    outs2 = (similar(ucpu[1]), nothing, nothing, nothing)
    device_matvec!(outs2, plans[1], ds; sorted=true)
    @test mrd(outs2[1], ucpu[1]) < tol
    # the device tree's group tables equal the host's
    dt = device_tree(vcat(groups...); ppl=40, bbox=box, group_sizes=collect(length.(groups)))
    t = Tree(dt)
    @test t.group_sizes == f.tree.group_sizes && t.gpoints == f.tree.gpoints
    @test all(sort(t.gperm[g]) == 1:length(groups[g]) for g in 1:4)
end
