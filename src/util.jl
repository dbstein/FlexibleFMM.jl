# Shared enumerations and small helpers.
#
# Everything that assigns a slot to a child, a colleague or an M2L offset goes through
# the functions in this file, so the tree builder, the point locator, the operator
# builders and the interaction-list builder can never disagree on an ordering.

"""
    _offset_tuple(::Val{D}, ::Val{R}) -> NTuple{(2R+1)^D, SVector{D,Int}}

All lattice offsets in `(-R:R)^D`, coordinate 1 varying fastest.
"""
@inline function _offset_tuple(::Val{D}, ::Val{R}) where {D,R}
    B = 2R + 1
    return ntuple(i -> SVector{D,Int}(ntuple(d -> ((i - 1) ÷ B^(d - 1)) % B - R, Val(D))),
                  Val(B^D))
end

"""
    child_offsets(::Val{D}) -> NTuple{2^D, SVector{D,Int}}

Lattice offsets of the `2^D` children of a cell, in slot order. Slot `k` (1-based) has
bit pattern `k-1` with coordinate 1 as the least significant bit, so coordinate 1
varies fastest. (The Python code used coordinate 1 as the most significant bit.)
"""
@inline function child_offsets(::Val{D}) where {D}
    return ntuple(k -> SVector{D,Int}(ntuple(d -> ((k - 1) >> (d - 1)) & 1, Val(D))),
                  Val(1 << D))
end

"""
    child_slot(x, center) -> Int

Slot (1-based, consistent with `child_offsets`) of the child containing the point `x`
of a box with the given center. A coordinate strictly greater than the center's goes
to the upper child; equality goes to the lower child. The tree builder and the point
locator both use this function, so a point is always located in the box it was built
into.
"""
@inline function child_slot(x::SVector{D}, center::SVector{D}) where {D}
    k = 1
    @inbounds for d in 1:D
        k += (x[d] > center[d]) << (d - 1)
    end
    return k
end

"""
    colleague_offsets(::Val{D}) -> NTuple{3^D, SVector{D,Int}}

All lattice offsets in `(-1:1)^D`, coordinate 1 varying fastest. A node's colleague
list has one slot per offset; the self offset is slot `self_colleague_slot(Val(D))`.
"""
@inline colleague_offsets(::Val{D}) where {D} = _offset_tuple(Val(D), Val(1))
@inline self_colleague_slot(::Val{D}) where {D} = (3^D + 1) ÷ 2

"""
    m2l_table_offsets(::Val{D}) -> NTuple{7^D, SVector{D,Int}}
    m2l_table_index(off) -> Int

The full `(-3:3)^D` offset table (coordinate 1 fastest) and the linear index of an
offset in it. The far offsets (Chebyshev norm ≥ 2) are the relative positions, in
units of the box width, of the boxes in an interaction list.
"""
@inline m2l_table_offsets(::Val{D}) where {D} = _offset_tuple(Val(D), Val(3))
@inline function m2l_table_index(off::SVector{D,Int}) where {D}
    k = 1
    @inbounds for d in 1:D
        k += (off[d] + 3) * 7^(d - 1)
    end
    return k
end
@inline is_far_offset(off::SVector{D,Int}) where {D} = maximum(abs, off) >= 2

"""
    far_offsets(::Val{D}) -> Vector{SVector{D,Int}}

The `7^D - 3^D` far offsets, in table order. M2L operators are stored in this order.
"""
far_offsets(::Val{D}) where {D} = [o for o in m2l_table_offsets(Val(D)) if is_far_offset(o)]

"""
    far_index_table(::Val{D}) -> Vector{Int}

Length `7^D`; maps `m2l_table_index(off)` to the position of `off` in
`far_offsets(Val(D))`, or 0 for a near offset.
"""
function far_index_table(::Val{D}) where {D}
    table = zeros(Int, 7^D)
    n = 0
    for o in m2l_table_offsets(Val(D))
        if is_far_offset(o)
            n += 1
            table[m2l_table_index(o)] = n
        end
    end
    return table
end

"""
    level_width(W0, l)

Width of the boxes on level `l` (level 1 is the root box of width `W0`). Exact in
floating point, so equal keys always mean bitwise-equal widths.
"""
@inline level_width(W0::AbstractFloat, l::Integer) = ldexp(W0, 1 - l)

"""
    as_svectors(::Val{D}, X) -> AbstractVector{<:SVector{D}}

Accepts a vector of `SVector{D}` (returned as is) or a `D × N` matrix (copied).
"""
as_svectors(::Val{D}, X::AbstractVector{<:SVector{D}}) where {D} = X
function as_svectors(::Val{D}, X::AbstractMatrix{T}) where {D,T}
    size(X, 1) == D || throw(DimensionMismatch("expected a $D × N matrix, got $(size(X))"))
    return [SVector{D,T}(ntuple(d -> X[d, j], Val(D))) for j in axes(X, 2)]
end

"""
    point_dimension(X)

`D` for a vector of `SVector{D}` or a `D × N` matrix.
"""
point_dimension(::AbstractVector{<:SVector{D}}) where {D} = D
point_dimension(X::AbstractMatrix) = size(X, 1)

"""
    morton_key(cell, nbits) -> UInt64

Interleaved bits of the lattice coordinates (`nbits` each): sorting cells by it
puts spatial neighbours close together.
"""
function morton_key(cell::SVector{D,Int}, nbits::Int) where {D}
    key = UInt64(0)
    for b in 0:nbits-1, k in 1:D
        key |= UInt64((cell[k] >> b) & 1) << (D * b + k - 1)
    end
    return key
end
