# Tree construction. A mutable per-level representation is refined, level restricted
# and ghosted, then flattened into a `Tree`. Indices inside a `LevelBuilder` are local
# to that level (parents index the previous level, children the next). The threaded
# phases (the splits, the colleague tagging, the flattening, the lists) run one region
# per level with their bodies in their own functions, and every result is independent
# of the thread count; the restriction scans and depths are cheap and stay serial.

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

Threaded over the nodes when there are many; the few large nodes of the first
levels (the root holds every point) are each sorted chunk-parallel instead. Both
are the same stable counting sort, so the result does not depend on the thread
count or the chunking.
"""
function split_nodes!(par::LevelBuilder{D}, which::AbstractVector{Int}, l::Int,
                      chi::LevelBuilder{D}, pts::Vector{SVector{D,T}}, perm::Vector{Int},
                      spts::Vector{SVector{D,T}}, sperm::Vector{Int},
                      origin::SVector{D,T}, W0::T; ghost::Bool) where {D,T}
    nw = length(which)
    nw == 0 && return nothing
    w = level_width(W0, l)
    nc = 1 << D
    counts = Matrix{Int}(undef, nc, nw)
    starts = Matrix{Int}(undef, nc, nw)
    @inbounds for q in 1:nw
        par.child_first[which[q]] == 0 || error("node already has children")
    end
    if nw < 4 * Threads.nthreads()
        for q in 1:nw
            i = which[q]
            _sort_range!(pts, perm, spts, sperm, par.lo[i], par.hi[i], _center(origin, par.cell[i], w),
                         view(counts, :, q), view(starts, :, q), true)
        end
    else
        _split_many!(par, which, pts, perm, spts, sperm, origin, w, counts, starts)
    end
    _append_children!(par, which, chi, counts, starts, child_offsets(Val(D)), ghost)
    return nothing
end
function _split_many!(par::LevelBuilder{D}, which, pts, perm, spts, sperm, origin, w, counts, starts) where {D}
    Threads.@threads :dynamic for q in eachindex(which)
        i = which[q]
        _sort_range!(pts, perm, spts, sperm, par.lo[i], par.hi[i], _center(origin, par.cell[i], w),
                     view(counts, :, q), view(starts, :, q), false)
    end
    return nothing
end
# ranges with at least this many points are sorted chunk-parallel (a Ref so that the
# tests can force the chunked path on small inputs and compare with the serial one)
const _PARALLEL_SORT_MIN = Ref(100_000)
# Stable counting sort of pts[lo:hi] (and perm) by child slot against the center `c`,
# serial, or chunk-parallel when allowed and the range is large: per-chunk histograms,
# one exclusive scan in (slot, chunk) order, per-chunk stable scatter, copy back.
function _sort_range!(pts::Vector{SVector{D,T}}, perm, spts, sperm, lo::Int, hi::Int, c::SVector{D,T},
                      counts::AbstractVector{Int}, starts::AbstractVector{Int}, chunked::Bool) where {D,T}
    nc = 1 << D
    n = hi - lo + 1
    nth = Threads.nthreads()
    if !chunked || nth == 1 || n < _PARALLEL_SORT_MIN[]
        fill!(counts, 0)
        @inbounds for j in lo:hi
            counts[child_slot(pts[j], c)] += 1
        end
        s = lo
        @inbounds for k in 1:nc
            starts[k] = s
            s += counts[k]
        end
        pos = MVector{nc,Int}(undef)
        @inbounds for k in 1:nc
            pos[k] = starts[k]
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
    else
        chunks = [(lo + (q - 1) * n ÷ (4nth)):(lo + q * n ÷ (4nth) - 1) for q in 1:4nth]
        hist = zeros(Int, nc, length(chunks))
        _chunk_histograms!(hist, pts, chunks, c)
        s = lo
        @inbounds for k in 1:nc
            starts[k] = s
            for q in eachindex(chunks)
                h = hist[k, q]
                hist[k, q] = s
                s += h
            end
            counts[k] = s - starts[k]
        end
        _chunk_scatter!(hist, pts, perm, spts, sperm, chunks, c)
        _chunk_copy!(pts, perm, spts, sperm, chunks)
    end
    return nothing
end
function _chunk_histograms!(hist, pts, chunks, c)
    Threads.@threads :dynamic for q in eachindex(chunks)
        @inbounds for j in chunks[q]
            hist[child_slot(pts[j], c), q] += 1
        end
    end
    return nothing
end
function _chunk_scatter!(hist, pts, perm, spts, sperm, chunks, c)
    Threads.@threads :dynamic for q in eachindex(chunks)
        @inbounds for j in chunks[q]
            k = child_slot(pts[j], c)
            p = hist[k, q]
            spts[p] = pts[j]
            sperm[p] = perm[j]
            hist[k, q] = p + 1
        end
    end
    return nothing
end
function _chunk_copy!(pts, perm, spts, sperm, chunks)
    Threads.@threads :dynamic for q in eachindex(chunks)
        @inbounds for j in chunks[q]
            pts[j] = spts[j]
            perm[j] = sperm[j]
        end
    end
    return nothing
end
# the children of the split nodes, appended to `chi` in the order of `which`, each
# group in slot order (what `_push_node!` did one by one)
function _append_children!(par::LevelBuilder{D}, which, chi::LevelBuilder{D}, counts, starts, offs, ghost::Bool) where {D}
    nc = 1 << D
    n0 = nnodes(chi)
    nnew = nc * length(which)
    for f in (chi.cell, chi.parent, chi.child_first, chi.lo, chi.hi, chi.leaf, chi.ghost, chi.xlist, chi.depth)
        resize!(f, n0 + nnew)
    end
    Threads.@threads :dynamic for q in eachindex(which)
        @inbounds begin
            i = which[q]
            par.child_first[i] = n0 + nc * (q - 1) + 1
            ghost || (par.leaf[i] = false)
            for k in 1:nc
                g = n0 + nc * (q - 1) + k
                chi.cell[g] = 2 * par.cell[i] + offs[k]
                chi.parent[g] = i
                chi.child_first[g] = 0
                chi.lo[g] = starts[k, q]
                chi.hi[g] = starts[k, q] + counts[k, q] - 1
                chi.leaf[g] = !ghost
                chi.ghost[g] = ghost
                chi.xlist[g] = false
                chi.depth[g] = 0
            end
        end
    end
    return nothing
end

"""
    tag_colleagues!(lev, par), tag_root!(root)

Fill `lev.colleagues[k, i]` with the node of `lev` at lattice offset
`colleague_offsets(Val(D))[k]` from node `i`, or 0, from the parent level's tables
(the parent rule). The cell `c' = c + o` exists on the level iff its parent cell
`c' >> 1` exists on the level above and was split, since every level ≥ 2 consists of
whole sibling groups; and `|c'_k − c_k| ≤ 1` gives `⌊c'_k/2⌋ − ⌊c_k/2⌋ ∈ {−1, 0, 1}`
(the arithmetic shift is the floor for `c'_k = −1`), so the parent cell of `c'` sits
at a colleague offset of `parent(i)` and is read from that row; cells outside the
lattice have parent cells outside the lattice above, absent by induction down to
the root, whose row holds only itself. The colleague is then that node's child in
the slot of `c'`'s low bits (`child_offsets`: coordinate 1 least significant). A
ghost parent has no children and finds nothing, as before; a ghost child is tagged
like any node. 3^D lookups per node, no sort, no search; cells are unique per level
by construction (distinct parents have distinct prefixes, siblings distinct low
bits), which the tree tests assert. (The first version sorted a level's cells and
searched 3^D keys per node, re-run on whole levels after every restriction and
ghost split: 0.67 of the 1.1 s build of 10⁷ sphere points, `docs/notes/tree_plan.md`.)
"""
function tag_colleagues!(lev::LevelBuilder{D}, par::LevelBuilder{D}) where {D}
    n = nnodes(lev)
    size(lev.colleagues, 2) == n || (lev.colleagues = Matrix{Int}(undef, 3^D, n))
    _tag_from_parents!(lev.colleagues, lev.cell, lev.parent, par.colleagues, par.child_first, colleague_offsets(Val(D)), n)
    return lev
end
function _tag_from_parents!(col, cell, parent, pcol, pchild, offs, n)
    Threads.@threads :dynamic for i in 1:n
        @inbounds begin
            c = cell[i]
            p = parent[i]
            pc = c .>> 1
            for (k, o) in enumerate(offs)
                cc = c + o
                q = pcol[_offset_index((cc .>> 1) - pc), p]
                col[k, i] = (q == 0 || pchild[q] == 0) ? 0 : pchild[q] + _slot_bits(cc) - 1
            end
        end
    end
    return col
end
# the slot of a parent-level offset in `colleague_offsets` (coordinate 1 fastest), and
# the child slot of a cell's low bits (`child_offsets`)
@inline function _offset_index(d::SVector{D,Int}) where {D}
    k = 1
    @inbounds for j in 1:D
        k += (d[j] + 1) * 3^(j - 1)
    end
    return k
end
@inline function _slot_bits(c::SVector{D,Int}) where {D}
    k = 1
    @inbounds for j in 1:D
        k += (c[j] & 1) << (j - 1)
    end
    return k
end
function tag_root!(root::LevelBuilder{D}) where {D}
    root.colleagues = zeros(Int, 3^D, 1)
    root.colleagues[self_colleague_slot(Val(D)), 1] = 1
    return root
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
            tag_colleagues!(levels[l+1], lev)
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
        tag_colleagues!(levels[l+1], lev)
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
    tag_root!(levels[1])
    for l in 2:length(levels)
        tag_colleagues!(levels[l], levels[l-1])
    end
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
        _flatten_level!(levels[l], l, offs, cell, level, parent, child_first, points, leaf, ghost, xlist,
                        has_source, colleagues, point_node, Val(D))
    end
    all(>(0), point_node) || error("point not assigned to a leaf")
    sum(length(points[g]) for g in 1:n if leaf[g]) == length(perm) || error("point assigned to two leaves")
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
                     perm, point_node, far_index, [length(perm)], [points], [perm], [point_node])
end
# the same tree with the group tables of `sizes` groups
function _with_groups(t::Tree{D,T}, sizes::Vector{Int}) where {D,T}
    sum(sizes) == length(t.perm) || throw(DimensionMismatch("the groups do not add up to the points"))
    gpoints, gperm, gpoint_node = _group_tables(t.level_range, t.child_first, t.points, t.perm, t.point_node, sizes, Val(D))
    return Tree{D,T}(t.origin, t.W0, t.level_range, t.cell, t.level, t.parent, t.child_first, t.points,
                     t.leaf, t.ghost, t.xlist, t.has_source, t.colleagues, t.group_parent, t.ilists,
                     t.perm, t.point_node, t.far_index, sizes, gpoints, gperm, gpoint_node)
end

# one level of `flatten`, threaded over its nodes (leaves write disjoint point ranges)
function _flatten_level!(lev::LevelBuilder{D}, l::Int, offs, cell, level, parent, child_first, points, leaf, ghost, xlist,
                         has_source, colleagues, point_node, ::Val{D}) where {D}
    Threads.@threads :dynamic for i in 1:nnodes(lev)
        @inbounds begin
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
                    point_node[j] = g
                end
            end
        end
    end
    return nothing
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

"""
    Tree(groups::Tuple; ppl=50, bbox=nothing, max_levels=30)

The tree of several groups of points (each a vector of `SVector`s or a `D × N`
matrix), built on their concatenation, with the per-group tables (`gpoints`,
`gperm`, `gpoint_node`): the groups carry different density types in one FMM.
"""
function Tree(groups::Tuple; kwargs...)
    isempty(groups) && throw(ArgumentError("no groups"))
    D = point_dimension(groups[1])
    pts = [as_svectors(Val(D), g) for g in groups]
    all(p -> eltype(p) == eltype(pts[1]), pts) || throw(ArgumentError("the groups must share one point type"))
    t = Tree(vcat(pts...); kwargs...)
    return _with_groups(t, [length(p) for p in pts])
end