# The kernel-free correctness invariant of the FMM structure: every (source terminal,
# target leaf) pair is covered exactly once, either by the near field or by exactly
# one V-list interaction between an ancestor-or-self of each.
using FlexibleFMM: Tree, nnodes, has_children, children, npairs, pair_range,
    far_offsets, m2l_table_index, morton_key

function ancestors_or_self(tree, i)
    out = Int[]
    while i != 0
        push!(out, i)
        i = tree.parent[i]
    end
    return out
end

# V(y): children of the colleagues of parent(y) that are not colleagues of y.
function vlist(tree, y)
    p = tree.parent[y]
    p == 0 && return Set{Int}()
    cols = Set(j for j in view(tree.colleagues, :, y) if j != 0)
    S = Set{Int}()
    for c in view(tree.colleagues, :, p)
        (c == 0 || !has_children(tree, c)) && continue
        for s in children(tree, c)
            s in cols || push!(S, s)
        end
    end
    return S
end

function check_coverage(tree)
    n = nnodes(tree)
    terminals = [i for i in 1:n if !has_children(tree, i) && tree.has_source[i]]
    leaves = findall(tree.leaf)
    vl = [vlist(tree, y) for y in 1:n]
    anc = [ancestors_or_self(tree, i) for i in 1:n]
    bad = 0
    for T in terminals, B in leaves
        near = any(C != 0 && C in anc[T] for C in view(tree.colleagues, :, B))
        cnt = count(X in vl[Y] for X in anc[T], Y in anc[B])
        ok = (near && cnt == 0) || (!near && cnt == 1)
        ok || (bad += 1)
    end
    @test bad == 0
    # the interaction lists are exactly V restricted to sourced sources and non-ghost targets
    pairs = Set{Tuple{Int,Int}}()
    total = 0
    for il in tree.ilists
        for (t, s) in zip(il.tgt, il.src)
            push!(pairs, (t, s))
        end
        total += npairs(il)
    end
    @test length(pairs) == total
    expected = Set((y, x) for y in 1:n if !tree.ghost[y] for x in vl[y] if tree.has_source[x])
    @test pairs == expected
    # offsets and levels are consistent with the CSR layout
    D = length(eltype(tree.cell))
    far = far_offsets(Val(D))
    for (l, il) in enumerate(tree.ilists)
        @test length(il.offset_ptr) == length(far) + 1
        for f in eachindex(far), q in pair_range(il, f)
            t, s = il.tgt[q], il.src[q]
            @test tree.level[t] == l && tree.level[s] == l
            @test tree.cell[s] - tree.cell[t] == far[f]
        end
        # a target appears at most once per offset
        for f in eachindex(far)
            @test allunique(view(il.tgt, pair_range(il, f)))
        end
    end
    @test all(npairs(tree.ilists[l]) == 0 for l in 1:min(2, length(tree.ilists)))
    # the by-target grouping lists the same pairs, each target once, in Morton order
    pairs_t = Set((il.torder[q], il.tsrc[k]) for il in tree.ilists for q in eachindex(il.torder) for k in il.tptr[q]:il.tptr[q+1]-1)
    @test pairs_t == pairs
    @test sum(length(il.tsrc) for il in tree.ilists) == total
    for (l, il) in enumerate(tree.ilists)
        @test allunique(il.torder)
        @test all(tree.cell[il.tsrc[k]] - tree.cell[il.torder[q]] == far[il.tfar[k]]
                  for q in eachindex(il.torder) for k in il.tptr[q]:il.tptr[q+1]-1)
        @test issorted(morton_key.(tree.cell[il.torder], min(l - 1, 64 ÷ D)))
    end
    return nothing
end

rng = MersenneTwister(4)
@testset "coverage on $name" for (name, gen) in pairs(DISTRIBUTIONS)
    pts = gen(rng, 400)
    check_coverage(Tree(pts; ppl=4))
end
@testset "coverage, small ppl and many ghosts" begin
    check_coverage(Tree(mixed_points(rng, 800); ppl=3))
    check_coverage(Tree(uniform_points(rng, 12); ppl=12))
end
