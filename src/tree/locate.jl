# Target location: walk from the root with the same `child_slot` predicate the
# builder used, stopping at the first leaf (xlist leaves are leaves, so the walk never
# enters a ghost).

"""
    locate(tree, x) -> node index

The real leaf containing `x`. Throws `ArgumentError` if `x` lies outside the root box.
"""
function locate(t::Tree{D,T}, x::SVector{D}) where {D,T}
    inside = true
    @inbounds for d in 1:D
        inside &= t.origin[d] <= x[d] <= t.origin[d] + t.W0
    end
    inside || throw(ArgumentError("point $x lies outside the tree's box"))
    i = 1
    @inbounds while !t.leaf[i]
        i = t.child_first[i] + child_slot(x, center(t, i)) - 1
    end
    @boundscheck t.ghost[i] && error("location ended on a ghost")
    return i
end

"""
    locate!(out, tree, targets)

`out[j] = locate(tree, targets[j])`, threaded.
"""
function locate!(out::AbstractVector{Int}, t::Tree{D}, targets::AbstractVector{<:SVector{D}}) where {D}
    length(out) == length(targets) || throw(DimensionMismatch("out and targets differ in length"))
    # check containment serially so the error is not wrapped in a task exception
    for x in targets
        all(t.origin .<= x .<= t.origin .+ t.W0) || throw(ArgumentError("point $x lies outside the tree's box"))
    end
    Threads.@threads :dynamic for j in eachindex(targets)
        @inbounds out[j] = locate(t, targets[j])
    end
    return out
end
locate(t::Tree{D}, targets::AbstractVector{<:SVector{D}}) where {D} =
    locate!(Vector{Int}(undef, length(targets)), t, targets)

"""
    bucket_by_node(node, nnodes) -> (order, ptr)

Counting sort of target indices by node: `order[ptr[i]:ptr[i+1]-1]` are the targets in
node `i`.
"""
function bucket_by_node(node::AbstractVector{Int}, nnodes::Int)
    counts = zeros(Int, nnodes)
    for i in node
        counts[i] += 1
    end
    ptr = Vector{Int}(undef, nnodes + 1)
    ptr[1] = 1
    for i in 1:nnodes
        ptr[i+1] = ptr[i] + counts[i]
    end
    pos = ptr[1:nnodes]
    order = Vector{Int}(undef, length(node))
    for j in eachindex(node)
        i = node[j]
        order[pos[i]] = j
        pos[i] += 1
    end
    return order, ptr
end
