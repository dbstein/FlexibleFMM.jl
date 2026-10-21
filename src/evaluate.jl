# Evaluation: for a target in real leaf B, the output is the local expansion of B
# at the target plus the direct sum over the sources of B's colleagues. Targets are
# located, bucketed by leaf (so consecutive targets share expansions and source
# ranges) and processed in contiguous chunks of that order. The output type
# (`Potential()`, `Gradient()`) selects the `l2t` and near-field bodies.

@inline function _direct_range(K, xs, τ, x, r::UnitRange{Int}, skip::Int, acc, out)
    @inbounds for j in r
        j == skip && continue
        acc += target_field(K, xs[j], x, τ[j], out)
    end
    return acc
end

"""
    evaluate_target(fmm, state, x, i, skip, out=Potential())

Output `out` at `x`, located in leaf `i`; the source with tree index `skip` is
excluded (0 for none).
"""
@inline function evaluate_target(f::FMM, state::FMMState, x::SVector{2}, i::Int, skip::Int, out=Potential())
    tree = f.tree
    m = f.method
    K = near_kernel(m)
    l = tree.level[i]
    u = l2t(m, f.ops[l], view(state.locals, :, i), x - center(tree, i), out)
    @inbounds for k in colleague_slots(tree)
        c = tree.colleagues[k, i]
        c == 0 && continue
        u = _direct_range(K, state.sources, state.τ, x, tree.points[c], skip, u, out)
    end
    return u
end

function _output_eltype(f::FMM, state::FMMState, ::Type{Tt}, out) where {Tt}
    K = near_kernel(f.method)
    Tnear = Base.promote_op(target_field, typeof(K), eltype(state.sources), SVector{2,Tt},
                            eltype(state.τ), typeof(out))
    Tfar = Base.promote_op(l2t, typeof(f.method), eltype(f.ops), typeof(view(state.locals, :, 1)),
                           SVector{2,Tt}, typeof(out))
    return promote_type(Tnear, Tfar)
end

"""
    evaluate!(out, fmm, state, targets; output=Potential())

`out[j]` = `output` at `targets[j]` (a vector of `SVector{2}` or a `2 × N` matrix).
Targets must lie in the root box.
"""
function evaluate!(out::AbstractVector, f::FMM, state::FMMState, targets; output=Potential())
    ts = as_svectors(Val(2), targets)
    nt = length(ts)
    length(out) == nt || throw(DimensionMismatch("out and targets differ in length"))
    nodes = locate(f.tree, ts)
    order, _ = bucket_by_node(nodes, nnodes(f.tree))
    chunks = _chunks(nt, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for q in ch
            j = order[q]
            out[j] = evaluate_target(f, state, ts[j], nodes[j], 0, output)
        end
    end
    return out
end

function evaluate(f::FMM, state::FMMState, targets; output=Potential())
    ts = as_svectors(Val(2), targets)
    out = Vector{_output_eltype(f, state, eltype(eltype(ts)), output)}(undef, length(ts))
    return evaluate!(out, f, state, ts; output=output)
end

"""
    evaluate_at_sources!(out, fmm, state; output=Potential())

`out[i]` = `output` at source `i` (original order) from all other sources.
"""
function evaluate_at_sources!(out::AbstractVector, f::FMM, state::FMMState; output=Potential())
    tree = f.tree
    N = npoints(tree)
    length(out) == N || throw(DimensionMismatch("expected $N outputs"))
    chunks = _chunks(N, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for j in ch
            out[tree.perm[j]] = evaluate_target(f, state, state.sources[j], tree.point_node[j], j, output)
        end
    end
    return out
end

function evaluate_at_sources(f::FMM, state::FMMState; output=Potential())
    out = Vector{_output_eltype(f, state, eltype(eltype(state.sources)), output)}(undef, npoints(f.tree))
    return evaluate_at_sources!(out, f, state; output=output)
end
