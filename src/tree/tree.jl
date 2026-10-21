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
appears at most once per offset.
"""
struct InteractionList
    offset_ptr::Vector{Int}
    tgt::Vector{Int}
    src::Vector{Int}
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
    lo = reduce((a, b) -> min.(a, b), points)
    hi = reduce((a, b) -> max.(a, b), points)
    W0 = maximum(hi - lo)
    W0 > 0 || (W0 = one(T))
    c = (lo + hi) / 2
    origin = c .- W0 / 2
    # rounding can leave an extreme point an ulp outside; widen until all are inside
    while !(all(origin .<= lo) && all(origin .+ W0 .>= hi))
        W0 *= 1 + 4eps(T)
        origin = c .- W0 / 2
    end
    return origin, W0
end

function Base.show(io::IO, t::Tree{D,T}) where {D,T}
    print(io, "Tree{$D,$T}: ", npoints(t), " points, ", nnodes(t), " nodes on ",
          nlevels(t), " levels (", count(t.ghost), " ghosts, ", count(t.xlist),
          " xlist leaves, W0 = ", t.W0, ")")
end
