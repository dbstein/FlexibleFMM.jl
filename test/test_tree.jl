using FlexibleFMM: Tree, nlevels, nnodes, npoints, center, width, has_children, children,
    locate, locate!, bucket_by_node, square_bbox, child_slot, child_offsets,
    colleague_offsets, self_colleague_slot, is_far_offset, m2l_table_index, npairs, pair_range


# Every structural invariant of the ghost-cell tree, checked by brute force.
function check_tree_invariants(tree, original)
    D = length(eltype(original))
    NC = 1 << D
    N = length(original)
    n = nnodes(tree)
    L = nlevels(tree)
    @test npoints(tree) == N
    @test sort(tree.perm) == 1:N
    sp = original[tree.perm]
    @test vcat(tree.level_range...) == 1:n
    @test all(!isempty, tree.level_range)
    @test all(tree.level[i] == l for l in 1:L for i in tree.level_range[l])
    @test tree.points[1] == 1:N
    @test tree.parent[1] == 0
    @test tree.cell[1] == zero(SVector{D,Int})
    offs = child_offsets(Val(D))
    for i in 1:n
        l = tree.level[i]
        w = width(tree, l)
        @test all(0 .<= tree.cell[i] .< 2^(l - 1))
        # single-rounding edges, bitwise consistent with the centers the builder splits on
        blo = tree.origin + SVector{D,Float64}(tree.cell[i]) * w
        bhi = tree.origin + SVector{D,Float64}(tree.cell[i] .+ 1) * w
        @test all(all(blo .<= sp[j] .<= bhi) for j in tree.points[i])
        @test tree.has_source[i] == !isempty(tree.points[i])
        @test !(tree.ghost[i] && tree.leaf[i])
        if has_children(tree, i)
            ch = children(tree, i)
            @test all(tree.parent[c] == i for c in ch)
            @test all(tree.level[c] == l + 1 for c in ch)
            @test [tree.cell[c] for c in ch] == [2tree.cell[i] + o for o in offs]
            @test first(tree.points[first(ch)]) == first(tree.points[i])
            @test last(tree.points[last(ch)]) == last(tree.points[i])
            @test all(last(tree.points[ch[k]]) + 1 == first(tree.points[ch[k+1]]) for k in 1:NC-1)
            c = center(tree, i)
            @test all(child_slot(sp[j], c) == k for k in 1:NC for j in tree.points[ch[k]])
            @test !tree.ghost[i]
            if tree.leaf[i]
                @test tree.xlist[i]
                @test all(tree.ghost[c] for c in ch)
            else
                @test all(!tree.ghost[c] for c in ch)
            end
        else
            @test !tree.xlist[i]
            @test tree.leaf[i] || tree.ghost[i]
        end
    end
    # sibling groups
    @test isempty(tree.group_parent[1])
    for l in 2:L
        r = tree.level_range[l]
        @test length(r) % NC == 0
        gp = tree.group_parent[l]
        @test length(gp) == length(r) ÷ NC
        for (g, p) in enumerate(gp)
            f = first(r) + (g - 1) * NC
            @test tree.child_first[p] == f
            @test all(tree.parent[f+k] == p for k in 0:NC-1)
        end
    end
    # unique cells and brute-force colleagues
    coffs = colleague_offsets(Val(D))
    for l in 1:L
        r = tree.level_range[l]
        @test allunique(tree.cell[r])
        lookup = Dict(tree.cell[i] => i for i in r)
        for i in r
            @test tree.colleagues[self_colleague_slot(Val(D)), i] == i
            for (k, o) in enumerate(coffs)
                @test tree.colleagues[k, i] == get(lookup, tree.cell[i] + o, 0)
            end
        end
    end
    # real depths (ghost children do not count) and 2:1 balance on colleagues
    rdepth = zeros(Int, n)
    for l in L:-1:1, i in tree.level_range[l]
        if has_children(tree, i) && !tree.leaf[i]
            rdepth[i] = 1 + maximum(rdepth[c] for c in children(tree, i))
        end
    end
    for i in 1:n
        tree.leaf[i] || continue
        for j in view(tree.colleagues, :, i)
            (j == 0 || tree.ghost[j]) && continue
            @test abs(rdepth[i] - rdepth[j]) <= 1
        end
    end
    # adjacency balance between real leaves, on the finest lattice
    leaves = findall(tree.leaf)
    scale(i) = 2^(L - tree.level[i])
    for a in leaves, b in leaves
        a < b || continue
        loa = tree.cell[a] * scale(a); hia = loa .+ scale(a)
        lob = tree.cell[b] * scale(b); hib = lob .+ scale(b)
        if all(loa .<= hib) && all(lob .<= hia)
            @test abs(tree.level[a] - tree.level[b]) <= 1
        end
    end
    # ghost invariants
    for i in 1:n
        if !tree.leaf[i] && !tree.ghost[i]
            for j in view(tree.colleagues, :, i)
                j == 0 && continue
                @test !tree.ghost[j]
                @test has_children(tree, j)
            end
        end
        if tree.leaf[i]
            nonleaf_colleague = any(j != 0 && !tree.leaf[j] && !tree.ghost[j]
                                    for j in view(tree.colleagues, :, i))
            @test tree.xlist[i] == nonleaf_colleague
        end
    end
    # point_node and the locator agree with the builder
    for j in 1:N
        p = tree.point_node[j]
        @test tree.leaf[p]
        @test j in tree.points[p]
        @test locate(tree, sp[j]) == p
    end
    # locate against brute-force containment
    @test all(locate(tree, original[q]) == tree.point_node[invperm(tree.perm)[q]] for q in 1:N)
    return nothing
end

rng = MersenneTwister(3)

@testset "invariants on $name" for (name, gen) in pairs(DISTRIBUTIONS)
    for (N, ppl) in ((300, 4), (2000, 20))
        pts = gen(rng, N)
        tree = Tree(pts; ppl=ppl)
        check_tree_invariants(tree, pts)
    end
end

@testset "ghosts actually occur" begin
    pts = mixed_points(rng, 2000)
    tree = Tree(pts; ppl=8)
    @test count(tree.ghost) > 0
    @test count(tree.xlist) > 0
    @test nlevels(tree) >= 5
end

@testset "edge cases" begin
    # N ≤ ppl: a single node
    pts = uniform_points(rng, 10)
    tree = Tree(pts; ppl=10)
    @test nlevels(tree) == 1 && nnodes(tree) == 1
    @test tree.leaf[1] && !has_children(tree, 1)
    @test all(tree.point_node .== 1)
    @test all(locate(tree, p) == 1 for p in pts)
    check_tree_invariants(tree, pts)
    # exactly two levels: four quadrant clusters of ppl points each
    ppl = 6
    pts = vec([P2(0.25 + 0.5a, 0.25 + 0.5b) + 0.01 * P2(randn(rng), randn(rng))
               for a in 0:1, b in 0:1, _ in 1:ppl])
    tree = Tree(pts; ppl=ppl, bbox=(P2(0, 0), 1.0))
    @test nlevels(tree) == 2 && nnodes(tree) == 5
    @test count(tree.ghost) == 0
    check_tree_invariants(tree, pts)
    # one quadrant: sparse levels, xlist at level 2
    pts = one_quadrant_points(rng, 500)
    tree = Tree(pts; ppl=5, bbox=(P2(0, 0), 1.0))
    check_tree_invariants(tree, pts)
    # the three empty quadrants are refined by level restriction (they are two or more
    # levels shallower than the populated one), so ghosts appear deeper, not on level 2
    @test length(tree.level_range[2]) == 4
    @test count(tree.has_source[tree.level_range[2]]) == 1
    @test all(has_children(tree, i) for i in tree.level_range[2])
    @test count(tree.ghost) > 0
    # coincident points refine forever: max_levels guard
    @test_throws ArgumentError Tree(duplicate_points(20); ppl=5, max_levels=10)
    @test_throws ArgumentError Tree(P2[]; ppl=5)
    @test_throws ArgumentError Tree(pts; ppl=0)
    # points outside an explicit box
    @test_throws ArgumentError Tree(pts; ppl=5, bbox=(P2(0.5, 0.5), 1.0))
    # matrix input
    M = reduce(hcat, pts)
    tm = Tree(M; ppl=5, bbox=(P2(0, 0), 1.0))
    @test tm.perm == tree.perm && tm.cell == tree.cell
end

@testset "square_bbox and box edges" begin
    pts = [P2(0.1, 0.2), P2(0.9, 0.3), P2(0.5, 0.25)]
    origin, W0 = square_bbox(pts)
    @test W0 == 0.8
    @test origin ≈ P2(0.1, 0.25 - 0.4)
    @test square_bbox([P2(0.3, 0.3)]) == (P2(-0.2, -0.2), 1.0)
    for pts in (uniform_points(rng, 50), circle_points(rng, 50), [P2(1e-3, 7.0), P2(1e-3, 7.5)])
        o, w = square_bbox(pts)
        @test all(all(o .<= x .<= o .+ w) for x in pts)
        @test w <= maximum(reduce((a, b) -> max.(a, b), pts) - reduce((a, b) -> min.(a, b), pts)) * (1 + 1e-14)
    end
    # points exactly on all four edges and corners are accepted and located
    pts = vcat(uniform_points(rng, 200), [P2(0, 0), P2(1, 0), P2(0, 1), P2(1, 1), P2(0.5, 0), P2(1, 0.5), P2(0, 0.37)])
    tree = Tree(pts; ppl=5, bbox=(P2(0, 0), 1.0))
    check_tree_invariants(tree, pts)
    @test_throws ArgumentError locate(tree, P2(1.0 + 1e-12, 0.5))
    @test_throws ArgumentError locate(tree, P2(0.5, -1e-12))
    # a point on an internal box edge goes to the lower child, consistently with the builder
    t = Tree(uniform_points(rng, 300); ppl=4, bbox=(P2(0, 0), 1.0))
    q = P2(0.5, 0.5)
    i = locate(t, q)
    @test all(tree.origin .<= q)   # trivial sanity
    @test child_slot(q, center(t, 1)) == 1
end

@testset "locate! and bucketing" begin
    pts = uniform_points(rng, 1000)
    tree = Tree(pts; ppl=10, bbox=(P2(0, 0), 1.0))
    targets = uniform_points(rng, 500)
    nodes = locate(tree, targets)
    @test nodes == [locate(tree, x) for x in targets]
    @test all(tree.leaf[nodes])
    order, ptr = bucket_by_node(nodes, nnodes(tree))
    @test sort(order) == 1:500
    for i in 1:nnodes(tree)
        @test all(nodes[order[q]] == i for q in ptr[i]:ptr[i+1]-1)
    end
    @test ptr[end] == 501
    locate(tree, targets[1])
    @test @allocated(locate(tree, targets[1])) == 0
end

@testset "determinism" begin
    pts = mixed_points(rng, 3000)
    t1 = Tree(pts; ppl=7)
    t2 = Tree(pts; ppl=7)
    @test t1.perm == t2.perm && t1.cell == t2.cell && t1.colleagues == t2.colleagues
    @test string(t1) == string(t2)
    @test occursin("ghosts", string(t1))
end

@testset "review: bounding boxes of bad points" begin
    @test_throws ArgumentError square_bbox([SVector(NaN, 0.0)])
    @test_throws ArgumentError square_bbox([SVector(Inf, 0.0), SVector(0.0, 0.0)])
    @test_throws ArgumentError square_bbox([SVector(1e308, 0.0), SVector(-1e308, 0.0)])
    o, W = square_bbox([SVector(1e308, 0.0), SVector(1.1e308, 0.0)])   # the midpoint does not overflow
    @test isfinite(W) && all(isfinite, o)
end
