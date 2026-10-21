# The driver: a tree plus the operators of its levels, and the expansion state of one
# right-hand side. See docs/framework.md.
#
# Upward pass, levels L..3: the partial multipoles of a level are the `s2m!` of its
# childless nodes with sources and, for the others, M2M of their children (one GEMM
# on the reshaped child-level block); the method then turns the whole level block
# into multipoles. Downward pass, levels 3..L: partial locals are L2L of the parents
# (one GEMM, gathered by group) plus, per far offset, M2L of the listed sources
# (gather → GEMM → scatter-add); the method then turns the block into locals.
# Levels 1 and 2 have neither far field nor multipoles.
#
# Threaded bodies live in their own functions: a variable assigned both inside a
# `Threads.@threads` body and in the enclosing function is shared between the tasks.

struct FMM{D,T,M<:AbstractMethod,O}
    method::M
    tree::Tree{D,T}
    cache::OperatorCache{M,O}
    ops::Vector{O}                   # operators of each level
    s2m_nodes::Vector{Vector{Int}}   # childless nodes with sources, per level
end

"""
    FMM(method, sources; ppl=50, bbox=nothing, cache=OperatorCache(method), max_levels=30)

Build the tree of `sources` (a vector of `SVector{2}` or a `2 × N` matrix) and fetch
the operators of its levels from `cache`. The root box is `bbox = (origin, width)`
(or the points' own square box), snapped to a width the cache knows.
"""
function FMM(method::AbstractMethod, sources; ppl::Integer=50, bbox=nothing,
             cache::OperatorCache=OperatorCache(method), max_levels::Integer=30)
    cache.method === method || throw(ArgumentError("the cache was built for a different method"))
    pts = as_svectors(Val(2), sources)
    origin, W0, e0 = fit_box!(cache, pts, bbox)
    tree = Tree(pts; ppl=ppl, bbox=(origin, W0), max_levels=max_levels)
    L = nlevels(tree)
    ops = [get_operators!(cache, e0 + 1 - l) for l in 1:L]
    s2m_nodes = [[i for i in tree.level_range[l] if !has_children(tree, i) && tree.has_source[i]]
                 for l in 1:L]
    return FMM(method, tree, cache, ops, s2m_nodes)
end

nlevels(f::FMM) = nlevels(f.tree)
expansion_length(f::FMM) = expansion_length(f.method)

function Base.show(io::IO, f::FMM)
    print(io, "FMM(", typeof(f.method).name.name, ", ", f.tree, ")")
end

"""
    FMMState

Expansions of one right-hand side: `multipoles` and `locals` (one column per node),
the sources and densities in tree order, and workspaces.
"""
struct FMMState{S,P<:SVector,Tτ}
    multipoles::Matrix{S}
    locals::Matrix{S}
    sources::Vector{P}
    τ::Vector{Tτ}
    partial::Matrix{S}     # partial expansions of the level being processed
    work1::Matrix{S}
    work2::Matrix{S}
end

function FMMState(f::FMM{D,T}, ::Type{S}, ::Type{Tp}, ::Type{Tτ}) where {D,T,S,Tp,Tτ}
    m = f.method
    rows = max(multipole_length(m), local_length(m), partial_multipole_length(m), partial_local_length(m))
    tree = f.tree
    n = nnodes(tree)
    N = npoints(tree)
    nc = maximum(length, tree.level_range)
    return FMMState{S,SVector{D,Tp},Tτ}(zeros(S, multipole_length(m), n), zeros(S, local_length(m), n),
                                       Vector{SVector{D,Tp}}(undef, N), Vector{Tτ}(undef, N),
                                       Matrix{S}(undef, rows, nc), Matrix{S}(undef, rows, nc),
                                       Matrix{S}(undef, rows, nc))
end

# contiguous chunks of 1:n for threading
function _chunks(n::Int, nchunks::Int)
    nchunks = max(1, min(nchunks, n))
    return [((c - 1) * n ÷ nchunks + 1):(c * n ÷ nchunks) for c in 1:nchunks]
end

# S2M of the listed nodes into the columns of `pM` (local to the level starting at `first`)
function _s2m_level!(pM::AbstractMatrix, m::AbstractMethod, ops, nodes::Vector{Int}, first::Int,
                     tree::Tree, sources::AbstractVector, τ::AbstractVector)
    chunks = _chunks(length(nodes), 4 * Threads.nthreads())
    Threads.@threads :dynamic for ch in chunks
        @inbounds for q in ch
            node = nodes[q]
            pr = tree.points[node]
            s2m!(view(pM, :, node - first + 1), m, ops, view(sources, pr), view(τ, pr),
                 center(tree, node))
        end
    end
    return pM
end

function upward_pass!(state::FMMState{S}, f::FMM) where {S}
    tree = f.tree
    m = f.method
    L = nlevels(tree)
    Nm = multipole_length(m)
    nc = 1 << 2
    Mp = state.multipoles
    fill!(Mp, zero(S))
    pml = partial_multipole_length(m)
    for l in L:-1:3
        lr = tree.level_range[l]
        nl = length(lr)
        ops = f.ops[l]
        pM = view(state.partial, 1:pml, 1:nl)
        fill!(pM, zero(S))
        _s2m_level!(pM, m, ops, f.s2m_nodes[l], first(lr), tree, state.sources, state.τ)
        if l < L
            cr = tree.level_range[l+1]
            ng = length(cr) ÷ nc
            B = reshape(view(Mp, :, cr), nc * Nm, ng)
            tmp = view(state.work1, 1:pml, 1:ng)
            mul!(tmp, ops.M2M, B)
            gp = tree.group_parent[l+1]
            off = first(lr) - 1
            @inbounds for g in 1:ng
                p = gp[g] - off
                for k in 1:pml
                    pM[k, p] = tmp[k, g]
                end
            end
        end
        partial_to_multipole!(view(Mp, :, lr), m, ops, pM, state.work2)
    end
    return state
end

function downward_pass!(state::FMMState{S}, f::FMM) where {S}
    tree = f.tree
    m = f.method
    L = nlevels(tree)
    Nm = multipole_length(m)
    Nl = local_length(m)
    nc = 1 << 2
    Mp = state.multipoles
    Lc = state.locals
    fill!(Lc, zero(S))
    pll = partial_local_length(m)
    for l in 3:L
        lr = tree.level_range[l]
        nl = length(lr)
        ops = f.ops[l]
        pL = view(state.partial, 1:pll, 1:nl)
        if l > 3
            # L2L: parents on level l-1 to the whole level block, in group order
            ng = nl ÷ nc
            gp = tree.group_parent[l]
            G = view(state.work1, 1:Nl, 1:ng)
            @inbounds for g in 1:ng
                p = gp[g]
                for k in 1:Nl
                    G[k, g] = Lc[k, p]
                end
            end
            mul!(reshape(pL, nc * pll, ng), f.ops[l-1].L2L, G)
        else
            fill!(pL, zero(S))
        end
        # M2L, one far offset at a time
        il = tree.ilists[l]
        off = first(lr) - 1
        for fidx in 1:length(ops.M2L)
            pr = pair_range(il, fidx)
            n = length(pr)
            n == 0 && continue
            G = view(state.work1, 1:Nm, 1:n)
            R = view(state.work2, 1:pll, 1:n)
            @inbounds for (q, idx) in enumerate(pr)
                s = il.src[idx]
                for k in 1:Nm
                    G[k, q] = Mp[k, s]
                end
            end
            apply!(R, ops.M2L[fidx], G)
            @inbounds for (q, idx) in enumerate(pr)
                t = il.tgt[idx] - off
                for k in 1:pll
                    pL[k, t] += R[k, q]
                end
            end
        end
        partial_to_local!(view(Lc, :, lr), m, ops, pL, state.work2)
    end
    return state
end

"""
    build_expansions!(state, fmm, sources, τ)

Form all multipole and local expansions for the sources (in the order the tree was
built from, any eltype) and densities `τ`.
"""
function build_expansions!(state::FMMState, f::FMM, sources, τ::AbstractVector)
    tree = f.tree
    pts = as_svectors(Val(2), sources)
    N = npoints(tree)
    length(pts) == N || throw(DimensionMismatch("expected $N sources"))
    length(τ) == N || throw(DimensionMismatch("expected $N densities"))
    supports_density(f.method, eltype(τ)) ||
        throw(ArgumentError("densities of type $(eltype(τ)) are not supported by $(typeof(f.method).name.name)"))
    perm = tree.perm
    @inbounds for j in 1:N
        state.sources[j] = pts[perm[j]]
        state.τ[j] = τ[perm[j]]
    end
    upward_pass!(state, f)
    downward_pass!(state, f)
    return state
end

"""
    build_expansions(fmm, sources, τ) -> FMMState

Allocating version of `build_expansions!`; the expansion eltype is promoted from the
operators, the kernel's output on the source eltype, and `eltype(τ)`.
"""
function build_expansions(f::FMM, sources, τ::AbstractVector)
    pts = as_svectors(Val(2), sources)
    Tp = eltype(eltype(pts))
    Tτ = eltype(τ)
    S = expansion_eltype(f.method, Tp, Tτ)
    state = FMMState(f, S, Tp, Tτ)
    return build_expansions!(state, f, pts, τ)
end
