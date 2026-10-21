# Tree construction. A mutable per-level representation is refined, level restricted
# and ghosted, then flattened into a `Tree`. Indices inside a `LevelBuilder` are local
# to that level (parents index the previous level, children the next).

mutable struct LevelBuilder{D}
    cell::Vector{SVector{D,Int}}
    parent::Vector{Int}
    child_first::Vector{Int}
    lo::Vector{Int}
    hi::Vector{Int}
    leaf::Vector{Bool}
    ghost::Vector{Bool}
    xlist::Vector{Bool}
    depth::Vector{Int}
    colleagues::Matrix{Int}
end
LevelBuilder{D}() where {D} = LevelBuilder{D}(SVector{D,Int}[], Int[], Int[], Int[], Int[],
                                              Bool[], Bool[], Bool[], Int[], zeros(Int, 3^D, 0))
nnodes(L::LevelBuilder) = length(L.cell)
@inline npoints(L::LevelBuilder, i::Int) = L.hi[i] - L.lo[i] + 1

function _push_node!(L::LevelBuilder{D}, cell, parent, lo, hi, leaf, ghost) where {D}
    push!(L.cell, cell); push!(L.parent, parent); push!(L.child_first, 0)
    push!(L.lo, lo); push!(L.hi, hi)
    push!(L.leaf, leaf); push!(L.ghost, ghost); push!(L.xlist, false); push!(L.depth, 0)
    return nnodes(L)
end

"""
    split_nodes!(par, which, l, chi, pts, perm, spts, sperm, origin, W0; ghost)

Give each node `i ∈ which` of level `par` (level number `l`) its `2^D` children,
appended to `chi`. The points of each node are counting-sorted in place by child slot
(using `spts`/`sperm` as scratch) so that every child owns a contiguous range. With
`ghost = true` the children are ghosts and the parent remains a leaf.
"""
function split_nodes!(par::LevelBuilder{D}, which::AbstractVector{Int}, l::Int,
                      chi::LevelBuilder{D}, pts::Vector{SVector{D,T}}, perm::Vector{Int},
                      spts::Vector{SVector{D,T}}, sperm::Vector{Int},
                      origin::SVector{D,T}, W0::T; ghost::Bool) where {D,T}
    w = level_width(W0, l)
    offs = child_offsets(Val(D))
    nc = 1 << D
    counts = MVector{nc,Int}(undef)
    starts = MVector{nc,Int}(undef)
    pos = MVector{nc,Int}(undef)
    for i in which
        par.child_first[i] == 0 || error("node already has children")
        lo, hi = par.lo[i], par.hi[i]
        c = _center(origin, par.cell[i], w)
        fill!(counts, 0)
        @inbounds for j in lo:hi
            counts[child_slot(pts[j], c)] += 1
        end
        s = lo
        @inbounds for k in 1:nc
            starts[k] = s
            pos[k] = s
            s += counts[k]
        end
        @inbounds for j in lo:hi
            k = child_slot(pts[j], c)
            spts[pos[k]] = pts[j]
            sperm[pos[k]] = perm[j]
            pos[k] += 1
        end
        @inbounds for j in lo:hi
            pts[j] = spts[j]
            perm[j] = sperm[j]
        end
        par.child_first[i] = nnodes(chi) + 1
        ghost || (par.leaf[i] = false)
        for k in 1:nc
            _push_node!(chi, 2 * par.cell[i] + offs[k], i, starts[k], starts[k] + counts[k] - 1,
                        !ghost, ghost)
        end
    end
    return nothing
end

"""
    tag_colleagues!(L)

Fill `L.colleagues[k, i]` with the node at lattice offset `colleague_offsets(Val(D))[k]`
from node `i`, or 0. Also checks that cells on the level are unique.
"""
function tag_colleagues!(L::LevelBuilder{D}) where {D}
    n = nnodes(L)
    keys = [Tuple(c) for c in L.cell]
    order = sortperm(keys)
    sorted = keys[order]
    for q in 2:n
        sorted[q] == sorted[q-1] && error("duplicate cell on a level")
    end
    offs = colleague_offsets(Val(D))
    L.colleagues = zeros(Int, 3^D, n)
    for i in 1:n
        for (k, o) in enumerate(offs)
            key = Tuple(L.cell[i] + o)
            j = searchsortedfirst(sorted, key)
            if j <= n && sorted[j] == key
                L.colleagues[k, i] = order[j]
            end
        end
    end
    return L
end

"""
    compute_depths!(levels)

`depth[i]` = height of the subtree below `i` (0 for childless nodes).
"""
function compute_depths!(levels::Vector{LevelBuilder{D}}) where {D}
    L = length(levels)
    fill!(levels[L].depth, 0)
    nc = 1 << D
    for l in L-1:-1:1
        lev, nxt = levels[l], levels[l+1]
        for i in 1:nnodes(lev)
            cf = lev.child_first[i]
            if cf == 0
                lev.depth[i] = 0
            else
                m = 0
                for k in 0:nc-1
                    m = max(m, nxt.depth[cf+k])
                end
                lev.depth[i] = m + 1
            end
        end
    end
    return levels
end

"""
    level_restrict!(levels, ...)

Split every leaf that has a colleague whose depth differs from its own by more than
one, until none is left. Leaves on the two deepest levels can never be bad, so only
levels `2:L-2` are examined, and no level is ever added.
"""
function level_restrict!(levels::Vector{LevelBuilder{D}}, pts, perm, spts, sperm, origin, W0) where {D}
    L = length(levels)
    compute_depths!(levels)
    while true
        nsplit = 0
        for l in 2:L-2
            lev = levels[l]
            which = Int[]
            for i in 1:nnodes(lev)
                lev.leaf[i] || continue
                for k in 1:3^D
                    j = lev.colleagues[k, i]
                    if j != 0 && abs(lev.depth[i] - lev.depth[j]) > 1
                        push!(which, i)
                        break
                    end
                end
            end
            isempty(which) && continue
            split_nodes!(lev, which, l, levels[l+1], pts, perm, spts, sperm, origin, W0; ghost=false)
            tag_colleagues!(levels[l+1])
            compute_depths!(levels)
            nsplit += length(which)
        end
        nsplit == 0 && break
    end
    return levels
end

"""
    add_ghosts!(levels, ...)

Two phases: (a) mark every real leaf that has a colleague with (real) children as an
`xlist` leaf, on all levels, before anything is added; (b) give every xlist leaf `2^D`
ghost children. Doing (a) for all levels first is what prevents ghosting from
cascading along a level (a ghosted leaf must not count as "having children" for its
neighbours). Levels `2:L-1` only: the root has no colleagues and the last level has
no deeper neighbours.
"""
function add_ghosts!(levels::Vector{LevelBuilder{D}}, pts, perm, spts, sperm, origin, W0) where {D}
    L = length(levels)
    for l in 2:L-1
        lev = levels[l]
        for i in 1:nnodes(lev)
            lev.leaf[i] || continue
            for k in 1:3^D
                j = lev.colleagues[k, i]
                if j != 0 && lev.child_first[j] > 0
                    lev.xlist[i] = true
                    break
                end
            end
        end
    end
    for l in 2:L-1
        lev = levels[l]
        which = findall(lev.xlist)
        isempty(which) && continue
        split_nodes!(lev, which, l, levels[l+1], pts, perm, spts, sperm, origin, W0; ghost=true)
        tag_colleagues!(levels[l+1])
    end
    return levels
end

"""
    build_levels(pts, perm, origin, W0, ppl, max_levels)

Refine (splitting every node with more than `ppl` points), tag colleagues, level
restrict and add ghosts. `pts` and `perm` are permuted in place.
"""
function build_levels(pts::Vector{SVector{D,T}}, perm::Vector{Int}, origin::SVector{D,T},
                      W0::T, ppl::Int, max_levels::Int) where {D,T}
    N = length(pts)
    spts = similar(pts)
    sperm = similar(perm)
    root = LevelBuilder{D}()
    _push_node!(root, zero(SVector{D,Int}), 0, 1, N, true, false)
    levels = [root]
    while true
        lev = levels[end]
        which = [i for i in 1:nnodes(lev) if npoints(lev, i) > ppl]
        isempty(which) && break
        length(levels) >= max_levels && throw(ArgumentError(
            "tree reached max_levels = $max_levels; more than ppl = $ppl coincident points?"))
        nxt = LevelBuilder{D}()
        split_nodes!(lev, which, length(levels), nxt, pts, perm, spts, sperm, origin, W0; ghost=false)
        push!(levels, nxt)
    end
    foreach(tag_colleagues!, levels)
    level_restrict!(levels, pts, perm, spts, sperm, origin, W0)
    add_ghosts!(levels, pts, perm, spts, sperm, origin, W0)
    return levels
end

"""
    flatten(levels, perm, origin, W0) -> Tree

Convert the per-level builders to global indices, derive `point_node`, the sibling
group tables and the interaction lists.
"""
function flatten(levels::Vector{LevelBuilder{D}}, perm::Vector{Int}, origin::SVector{D,T}, W0::T) where {D,T}
    L = length(levels)
    counts = nnodes.(levels)
    all(>(0), counts) || error("empty level")
    offs = cumsum(vcat(0, counts[1:end-1]))
    level_range = [offs[l]+1:offs[l]+counts[l] for l in 1:L]
    n = sum(counts)
    nc = 1 << D
    cell = Vector{SVector{D,Int}}(undef, n)
    level = Vector{Int}(undef, n)
    parent = Vector{Int}(undef, n)
    child_first = Vector{Int}(undef, n)
    points = Vector{UnitRange{Int}}(undef, n)
    leaf = Vector{Bool}(undef, n)
    ghost = Vector{Bool}(undef, n)
    xlist = Vector{Bool}(undef, n)
    has_source = Vector{Bool}(undef, n)
    colleagues = zeros(Int, 3^D, n)
    point_node = zeros(Int, length(perm))
    for l in 1:L
        lev = levels[l]
        for i in 1:nnodes(lev)
            g = offs[l] + i
            cell[g] = lev.cell[i]
            level[g] = l
            parent[g] = l == 1 ? 0 : offs[l-1] + lev.parent[i]
            child_first[g] = lev.child_first[i] == 0 ? 0 : offs[l+1] + lev.child_first[i]
            points[g] = lev.lo[i]:lev.hi[i]
            leaf[g] = lev.leaf[i]
            ghost[g] = lev.ghost[i]
            xlist[g] = lev.xlist[i]
            has_source[g] = lev.hi[i] >= lev.lo[i]
            for k in 1:3^D
                j = lev.colleagues[k, i]
                colleagues[k, g] = j == 0 ? 0 : offs[l] + j
            end
            if lev.leaf[i]
                for j in lev.lo[i]:lev.hi[i]
                    point_node[j] == 0 || error("point assigned to two leaves")
                    point_node[j] = g
                end
            end
        end
    end
    all(>(0), point_node) || error("point not assigned to a leaf")
    group_parent = Vector{Vector{Int}}(undef, L)
    group_parent[1] = Int[]
    for l in 2:L
        r = level_range[l]
        length(r) % nc == 0 || error("level $l is not made of whole sibling groups")
        gp = [parent[first(r) + (g-1)*nc] for g in 1:length(r) ÷ nc]
        for (g, p) in enumerate(gp)
            f = first(r) + (g - 1) * nc
            child_first[p] == f || error("sibling group does not start at its parent's child_first")
            all(parent[f+k] == p for k in 0:nc-1) || error("sibling group with mixed parents")
        end
        group_parent[l] = gp
    end
    far_index = far_index_table(Val(D))
    ilists = build_interaction_lists(Val(D), level_range, cell, parent, child_first,
                                     colleagues, ghost, has_source, far_index)
    return Tree{D,T}(origin, W0, level_range, cell, level, parent, child_first, points,
                     leaf, ghost, xlist, has_source, colleagues, group_parent, ilists,
                     perm, point_node, far_index)
end

"""
    Tree(points; ppl=50, bbox=nothing, max_levels=30)

Build the ghost-cell tree of `points` (a vector of `SVector`s or a `D × N` matrix),
splitting nodes with more than `ppl` points. `bbox = (origin, W0)` is the square root
box; by default `square_bbox(points)`. All points must lie in the box.
"""
function Tree(points::AbstractVector{SVector{D,T}}; ppl::Integer=50, bbox=nothing,
              max_levels::Integer=30) where {D,T<:AbstractFloat}
    ppl >= 1 || throw(ArgumentError("ppl must be positive"))
    isempty(points) && throw(ArgumentError("no points"))
    origin, W0 = bbox === nothing ? square_bbox(points) : (SVector{D,T}(bbox[1]), T(bbox[2]))
    W0 > 0 || throw(ArgumentError("box width must be positive"))
    for x in points
        all(origin .<= x .<= origin .+ W0) || throw(ArgumentError("point $x lies outside the box"))
    end
    pts = collect(points)
    perm = collect(1:length(pts))
    levels = build_levels(pts, perm, origin, W0, Int(ppl), Int(max_levels))
    return flatten(levels, perm, origin, W0)
end
Tree(points::AbstractMatrix{T}; kwargs...) where {T} = Tree(as_svectors(Val(size(points, 1)), points); kwargs...)
