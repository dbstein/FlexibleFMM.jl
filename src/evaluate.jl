# Evaluation: for a target in real leaf B, the output is the local expansion of B
# at the target plus the direct sum over the sources of B's colleagues. Targets are
# located, bucketed by leaf (so consecutive targets share expansions and source
# ranges) and processed in contiguous chunks of that order. The output type
# (`Potential()`, `Gradient()`) selects the `l2t` and near-field bodies.

# The excluded source is masked rather than skipped: a branch-free body vectorises
# (1.5× for Laplace, 1.3× for Stokes measured); its value is computed at r = 0 and
# discarded, never summed.
# the sums accumulate through `_accumulate` (kernels.jl): reassociation allowed, so
# that the loop vectorises across the sources
@inline function _direct_range(K, xs, τ, x, r::UnitRange{Int}, skip::Int, acc, out)
    if masks_self(K)
        @inbounds @simd for j in r
            v = target_field(K, xs[j], x, τ[j], out)
            acc = _accumulate(acc, ifelse(j == skip, zero(v), v))
        end
    else                                            # a kernel that throws at r = 0: skip
        @inbounds for j in r
            j == skip && continue
            acc = _accumulate(acc, target_field(K, xs[j], x, τ[j], out))
        end
    end
    return acc
end

"""
    evaluate_target(fmm, state, x, i, skip, out=Potential())

Output `out` at `x`, located in leaf `i`; the source with tree index `skip` is
excluded (0 for none).
"""
@inline evaluate_target(f::FMM{D}, state::FMMState, x::SVector{D}, i::Int, skip::Int, out=Potential()) where {D} =
    evaluate_target(f, state, x, i, 1, skip, out)
# the target is source `skip` of group `sg` (0 for none): that term is excluded in
# that group's sum only
@inline function evaluate_target(f::FMM{D}, state::FMMState, x::SVector{D}, i::Int, sg::Int, skip::Int, out) where {D}
    tree = f.tree
    m = f.method
    K = near_kernel(m)
    l = tree.level[i]
    u = l2t(m, f.ops[l], view(state.locals, :, i), x - center(tree, i), out)
    @inbounds for k in colleague_slots(tree)
        c = tree.colleagues[k, i]
        c == 0 && continue
        u = _near_groups(K, state.sources, state.τ, tree.gpoints, c, x, sg, skip, u, out, 1)
    end
    return u
end
# the groups' sums over one colleague, a recursion over the tuple of densities
@inline function _near_groups(K, sources, τ::Tuple, gpoints, c, x, sg, skip, u, out, g::Int)
    τg = first(τ)
    if τg !== nothing
        u = _direct_range(K, sources[g], τg, x, gpoints[g][c], g == sg ? skip : 0, u, out)
    end
    return _near_groups(K, sources, Base.tail(τ), gpoints, c, x, sg, skip, u, out, g + 1)
end
@inline _near_groups(K, sources, ::Tuple{}, gpoints, c, x, sg, skip, u, out, g::Int) = u

function _output_eltype(f::FMM{D}, state::FMMState, ::Type{Tt}, out) where {D,Tt}
    K = near_kernel(f.method)
    P = eltype(state.sources[1])
    Tnear = mapreduce(τ -> τ === nothing ? Bool : Base.promote_op(target_field, typeof(K), P, SVector{D,Tt}, eltype(τ), typeof(out)),
                      promote_type, state.τ)
    Tfar = Base.promote_op(l2t, typeof(f.method), eltype(f.ops), typeof(view(state.locals, :, 1)),
                           SVector{D,Tt}, typeof(out))
    return promote_type(Tnear, Tfar)
end

"""
    evaluate!(out, fmm, state, targets; output=Potential())

`out[j]` = `output` at `targets[j]` (a vector of `SVector{2}` or a `2 × N` matrix).
Targets must lie in the root box.
"""
function _check_output(f::FMM, output)
    supports_output(f.method, output) ||
        throw(ArgumentError("output $(typeof(output)) is not available with $(typeof(f.method).name.name)"))
end

function evaluate!(out::AbstractVector, f::FMM{D}, state::FMMState, targets; output=Potential()) where {D}
    _check_output(f, output)
    ts = as_svectors(Val(D), targets)
    nt = length(ts)
    length(out) == nt || throw(DimensionMismatch("out and targets differ in length"))
    nodes = locate(f.tree, ts)
    order, _ = bucket_by_node(nodes, nnodes(f.tree))
    chunks = _chunks(nt, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for q in ch
            j = order[q]
            out[j] = evaluate_target(f, state, ts[j], nodes[j], 0, 0, output)
        end
    end
    return out
end

function evaluate(f::FMM{D}, state::FMMState, targets; output=Potential()) where {D}
    ts = as_svectors(Val(D), targets)
    out = Vector{_output_eltype(f, state, eltype(eltype(ts)), output)}(undef, length(ts))
    return evaluate!(out, f, state, ts; output=output)
end

"""
    evaluate_at_sources!(out, fmm, state, g=1; output=Potential())

`out[i]` = `output` at point `i` of group `g` (in the group's original order) from
all other sources of all groups; with several groups, one call per group, each
with its own output.
"""
function evaluate_at_sources!(out::AbstractVector, f::FMM, state::FMMState, g::Int=1; output=Potential())
    _check_output(f, output)
    tree = f.tree
    1 <= g <= ngroups(tree) || throw(ArgumentError("group $g of $(ngroups(tree))"))
    n = tree.group_sizes[g]
    length(out) == n || throw(DimensionMismatch("expected $n outputs for group $g"))
    gperm = tree.gperm[g]
    gnode = tree.gpoint_node[g]
    xs = state.sources[g]
    chunks = _chunks(n, 8 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for k in ch
            out[gperm[k]] = evaluate_target(f, state, xs[k], gnode[k], g, k, output)
        end
    end
    return out
end

function evaluate_at_sources(f::FMM, state::FMMState, g::Int=1; output=Potential())
    out = Vector{_output_eltype(f, state, eltype(eltype(state.sources[g])), output)}(undef, f.tree.group_sizes[g])
    return evaluate_at_sources!(out, f, state, g; output=output)
end
