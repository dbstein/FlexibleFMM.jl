# The flat, immutable tree. See docs/tree.md.
#
# Nodes carry global indices; `level_range[l]` are the nodes on level `l` (level 1 is
# the root). Children of a node are the `2^D` consecutive nodes starting at
# `child_first` (0 if none), in `child_offsets` slot order, and every level ≥ 2
# consists of whole sibling groups. Three orthogonal flags describe a node:
#   leaf   — terminal for target location (real leaves, including xlist leaves)
#   ghost  — a fake child of an xlist leaf; has sources and a multipole, is never a
#            target and never has children
#   xlist  — a real leaf with a non-leaf colleague; it keeps its targets but its
#            multipole is assembled from its ghost children
# The FMM passes only need `has_children`: childless nodes (real non-xlist leaves and
# ghosts) get multipoles from their sources, nodes with children get them from their
# children.

"""
    InteractionList

M2L pairs of one level in CSR form by far offset: the pairs at far offset `f` (see
`far_offsets`) are `tgt[offset_ptr[f]:offset_ptr[f+1]-1]` and `src[...]`. A target
appears at most once per offset. The same pairs grouped by target, with the targets
in Morton order of their cells: target `torder[q]` has the sources
`tsrc[tptr[q]:tptr[q+1]-1]` at far offsets `tfar[...]`. The dense translation
batches by offset; the spectral one walks targets in Morton order so that a
target's spectrum stays in cache across its pairs and neighbouring targets reuse
their sources.
"""
struct InteractionList
    offset_ptr::Vector{Int}
    tgt::Vector{Int}
    src::Vector{Int}
    torder::Vector{Int}
    tptr::Vector{Int}
    tsrc::Vector{Int}
    tfar::Vector{Int}
end
npairs(il::InteractionList) = length(il.tgt)
@inline pair_range(il::InteractionList, f::Int) = il.offset_ptr[f]:(il.offset_ptr[f+1] - 1)

struct Tree{D,T<:AbstractFloat}
    origin::SVector{D,T}
    W0::T
    level_range::Vector{UnitRange{Int}}
    # per node
    cell::Vector{SVector{D,Int}}
    level::Vector{Int}
    parent::Vector{Int}
    child_first::Vector{Int}
    points::Vector{UnitRange{Int}}
    leaf::Vector{Bool}
    ghost::Vector{Bool}
    xlist::Vector{Bool}
    has_source::Vector{Bool}
    colleagues::Matrix{Int}
    # per level
    group_parent::Vector{Vector{Int}}
    ilists::Vector{InteractionList}
    # points
    perm::Vector{Int}
    point_node::Vector{Int}
    # offset bookkeeping
    far_index::Vector{Int}
    # groups of points (one by default): per group the sizes, the point range of every
    # node in the group's own numbering, the original index within the group of each
    # group-local sorted point, and its leaf; with one group these alias `points`,
    # `perm` and `point_node`
    group_sizes::Vector{Int}
    gpoints::Vector{Vector{UnitRange{Int}}}
    gperm::Vector{Vector{Int}}
    gpoint_node::Vector{Vector{Int}}
end

nlevels(t::Tree) = length(t.level_range)
nnodes(t::Tree) = length(t.cell)
npoints(t::Tree) = length(t.perm)
@inline width(t::Tree, l::Integer) = level_width(t.W0, l)
@inline has_children(t::Tree, i::Integer) = t.child_first[i] > 0
@inline children(t::Tree{D}, i::Integer) where {D} = t.child_first[i]:(t.child_first[i] + (1 << D) - 1)
@inline colleague_slots(::Tree{D}) where {D} = 1:3^D

@inline function _center(origin::SVector{D,T}, cell::SVector{D,Int}, w::T) where {D,T}
    return origin + (SVector{D,T}(cell) .+ T(0.5)) * w
end
"""
    center(tree, i)

Center of node `i`, computed from its lattice cell.
"""
@inline center(t::Tree, i::Integer) = _center(t.origin, t.cell[i], width(t, t.level[i]))

"""
    square_bbox(points) -> (origin, W0)

The square box centred on the points' bounding box with side equal to its largest
extent (or 1 if all points coincide).
"""
function square_bbox(points::AbstractVector{SVector{D,T}}) where {D,T}
    isempty(points) && throw(ArgumentError("no points"))
    all(p -> all(isfinite, p), points) || throw(ArgumentError("the points must be finite"))
    lo = reduce((a, b) -> min.(a, b), points)
    hi = reduce((a, b) -> max.(a, b), points)
    W0 = maximum(hi - lo)
    isfinite(W0) || throw(ArgumentError("the points' extent overflows"))
    W0 > 0 || (W0 = one(T))
    c = lo + (hi - lo) / 2                        # the midpoint without overflowing lo + hi
    origin = c .- W0 / 2
    # rounding can leave an extreme point an ulp outside; widen until all are inside
    for _ in 1:64
        all(origin .<= lo) && all(origin .+ W0 .>= hi) && return origin, W0
        W0 *= 1 + 4eps(T)
        origin = c .- W0 / 2
    end
    throw(ArgumentError("could not fit a box around the points"))
end

function Base.show(io::IO, t::Tree{D,T}) where {D,T}
    print(io, "Tree{$D,$T}: ", npoints(t), " points, ", nnodes(t), " nodes on ",
          nlevels(t), " levels (", count(t.ghost), " ghosts, ", count(t.xlist),
          " xlist leaves, W0 = ", t.W0, ")")
end

ngroups(t::Tree) = length(t.group_sizes)

# The per-group tables of a tree built on the concatenation of `sizes` groups. In
# the sorted order restricted to a group every node's points of that group are one
# contiguous range (the restriction of a contiguous range to a subsequence is
# contiguous): childless nodes get their group ranges from their group counts, in
# sorted order; nodes with children the union of their children's (contiguous, in
# slot order). No sort key changes; the tables are a post-processing of `perm`.
function _group_tables(level_range, child_first, points, perm, point_node, sizes::Vector{Int}, ::Val{D}) where {D}
    G = length(sizes)
    n = length(points)
    N = length(perm)
    starts = cumsum(vcat(0, sizes))                     # group g holds original indices starts[g]+1:starts[g+1]
    gid = Vector{Int}(undef, N)                         # the group of each sorted point
    glocal = Vector{Int}(undef, N)                      # its index in the group's sorted numbering
    counters = zeros(Int, G)
    gperm = [Vector{Int}(undef, sizes[g]) for g in 1:G]
    gpoint_node = [Vector{Int}(undef, sizes[g]) for g in 1:G]
    @inbounds for j in 1:N
        i = perm[j]
        g = searchsortedlast(starts, i - 1)             # starts[g] < i <= starts[g+1]
        g = min(g, G)
        k = (counters[g] += 1)
        gid[j] = g
        glocal[j] = k
        gperm[g][k] = i - starts[g]
        gpoint_node[g][k] = point_node[j]
    end
    gpoints = [Vector{UnitRange{Int}}(undef, n) for _ in 1:G]
    nc = 1 << D
    # childless nodes: the group-local indices of their points are consecutive, so the
    # range is that of the first and last group-g point in the node; nodes with children
    # (processed level by level from the bottom): the union of the children's ranges
    for l in length(level_range):-1:1
        @inbounds for i in level_range[l]
            cf = child_first[i]
            if cf == 0
                r = points[i]
                for g in 1:G
                    lo, hi = typemax(Int), 0
                    for j in r
                        if gid[j] == g
                            lo = min(lo, glocal[j]); hi = max(hi, glocal[j])
                        end
                    end
                    gpoints[g][i] = hi == 0 ? (1:0) : (lo:hi)
                end
            else
                for g in 1:G
                    lo, hi = typemax(Int), 0
                    for c in cf:cf+nc-1
                        rc = gpoints[g][c]
                        isempty(rc) && continue
                        lo = min(lo, first(rc)); hi = max(hi, last(rc))
                    end
                    gpoints[g][i] = hi == 0 ? (1:0) : (lo:hi)
                end
            end
        end
    end
    return gpoints, gperm, gpoint_node
end
