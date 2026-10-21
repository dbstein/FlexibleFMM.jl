# Evaluation: for a target in real leaf B, the potential is the local expansion of B
# at the target plus the direct sum over the sources of B's colleagues. Targets are
# located, bucketed by leaf (so consecutive targets share expansions and source
# ranges) and processed in contiguous chunks of that order.

@inline function _direct_range(K, xs, τ, x, r::UnitRange{Int}, skip::Int, acc)
    @inbounds for j in r
        j == skip && continue
        acc += K(xs[j], x) * τ[j]
    end
    return acc
end

"""
    evaluate_target(fmm, state, x, i, skip)

Potential at `x`, located in leaf `i`; the source with tree index `skip` is excluded
(0 for none).
"""
@inline function evaluate_target(f::FMM, state::FMMState, x::SVector{2}, i::Int, skip::Int)
    tree = f.tree
    m = f.method
    K = near_kernel(m)
    l = tree.level[i]
    u = l2t(m, f.ops[l], view(state.locals, :, i), x - center(tree, i))
    @inbounds for k in colleague_slots(tree)
        c = tree.colleagues[k, i]
        c == 0 && continue
        u = _direct_range(K, state.sources, state.τ, x, tree.points[c], skip, u)
    end
    return u
end

function _output_eltype(f::FMM, state::FMMState{S}, ::Type{Tt}) where {S,Tt}
    K = near_kernel(f.method)
    return promote_type(S, kernel_eltype(K, eltype(state.sources), SVector{2,Tt}))
end

"""
    evaluate!(out, fmm, state, targets)

`out[j]` = potential at `targets[j]` (a vector of `SVector{2}` or a `2 × N` matrix).
Targets must lie in the root box.
"""
function evaluate!(out::AbstractVector, f::FMM, state::FMMState, targets)
    ts = as_svectors(Val(2), targets)
    nt = length(ts)
    length(out) == nt || throw(DimensionMismatch("out and targets differ in length"))
    nodes = locate(f.tree, ts)
    order, _ = bucket_by_node(nodes, nnodes(f.tree))
    chunks = _chunks(nt, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for q in ch
            j = order[q]
            out[j] = evaluate_target(f, state, ts[j], nodes[j], 0)
        end
    end
    return out
end

function evaluate(f::FMM, state::FMMState, targets)
    ts = as_svectors(Val(2), targets)
    out = Vector{_output_eltype(f, state, eltype(eltype(ts)))}(undef, length(ts))
    return evaluate!(out, f, state, ts)
end

"""
    evaluate_at_sources!(out, fmm, state)

`out[i]` = potential at source `i` (original order) from all other sources.
"""
function evaluate_at_sources!(out::AbstractVector, f::FMM, state::FMMState)
    tree = f.tree
    N = npoints(tree)
    length(out) == N || throw(DimensionMismatch("expected $N outputs"))
    chunks = _chunks(N, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for j in ch
            out[tree.perm[j]] = evaluate_target(f, state, state.sources[j], tree.point_node[j], j)
        end
    end
    return out
end

function evaluate_at_sources(f::FMM, state::FMMState)
    out = Vector{_output_eltype(f, state, eltype(eltype(state.sources)))}(undef, npoints(f.tree))
    return evaluate_at_sources!(out, f, state)
end
