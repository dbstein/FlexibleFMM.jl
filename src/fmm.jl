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
    leaf_nodes::Vector{Vector{Int}}  # real leaves per level: the only nodes whose locals the evaluation reads
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
    D = point_dimension(sources)
    D == dimension(method) || throw(ArgumentError("$(D)-dimensional points with a $(dimension(method))-dimensional method"))
    pts = as_svectors(Val(D), sources)
    origin, W0, e0 = fit_box!(cache, pts, bbox)
    tree = Tree(pts; ppl=ppl, bbox=(origin, W0), max_levels=max_levels)
    L = nlevels(tree)
    ops = [get_operators!(cache, e0 + 1 - l) for l in 1:L]
    s2m_nodes = [[i for i in tree.level_range[l] if !has_children(tree, i) && tree.has_source[i]]
                 for l in 1:L]
    leaf_nodes = [[i for i in tree.level_range[l] if tree.leaf[i]] for l in 1:nlevels(tree)]
    return FMM(method, tree, cache, ops, s2m_nodes, leaf_nodes)
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
    plocals::Matrix{S}     # partial locals (check data) of every node: what L2L consumes
    sources::Vector{P}
    τ::Vector{Tτ}
    partial::Matrix{S}     # partial expansions of the level being processed, addressed as
                           # flat storage (reshape of a linear-range view): the sibling-group
                           # reshape of a two-range view is not a StridedMatrix and would
                           # send the M2M/L2L products to the generic matmul
    work1::Matrix{S}
    work2::Matrix{S}
    scratch::Dict{Any,Any} # method-specific workspaces (spectral translation)
end

function FMMState(f::FMM{D,T}, ::Type{S}, ::Type{Tp}, ::Type{Tτ}) where {D,T,S,Tp,Tτ}
    m = f.method
    rows = max(multipole_length(m), local_length(m), partial_multipole_length(m), partial_local_length(m))
    tree = f.tree
    n = nnodes(tree)
    N = npoints(tree)
    nc = maximum(length, tree.level_range)
    return FMMState{S,SVector{D,Tp},Tτ}(zeros(S, multipole_length(m), n), zeros(S, local_length(m), n),
                                       zeros(S, partial_local_length(m), n),
                                       Vector{SVector{D,Tp}}(undef, N), Vector{Tτ}(undef, N),
                                       Matrix{S}(undef, rows, nc), Matrix{S}(undef, rows, nc),
                                       Matrix{S}(undef, rows, nc), Dict{Any,Any}())
end

# Default level-wise M2L: per far offset, gather the source multipoles, one GEMM,
# scatter-add into the targets (a target appears at most once per offset). Low-rank
# operators get their intermediate product from a workspace kept in the state.
function m2l_level!(pL::AbstractMatrix, m::AbstractMethod, ops::LevelOperators, Mp::AbstractMatrix,
                    il::InteractionList, off::Int, state::FMMState)
    Nm = size(Mp, 1)
    pll = size(pL, 1)
    rmax = maximum(workspace_rows, ops.M2L)
    work = rmax == 0 ? nothing : _lowrank_workspace(state, rmax, size(state.work1, 2))
    for fidx in 1:length(ops.M2L)
        pr = pair_range(il, fidx)
        n = length(pr)
        n == 0 && continue
        G = view(state.work1, 1:Nm, 1:n)
        Rw = view(state.work2, 1:pll, 1:n)
        @inbounds for (q, idx) in enumerate(pr)
            s = il.src[idx]
            for k in 1:Nm
                G[k, q] = Mp[k, s]
            end
        end
        apply!(Rw, ops.M2L[fidx], G, ops.m2l_scale, work)
        @inbounds for (q, idx) in enumerate(pr)
            t = il.tgt[idx] - off
            for k in 1:pll
                pL[k, t] += Rw[k, q]
            end
        end
    end
    return pL
end

function _lowrank_workspace(state::FMMState{S}, r::Int, n::Int) where {S}
    w = get(state.scratch, :lowrank, nothing)
    if w === nothing || size(w, 1) < r || size(w, 2) < n
        w = Matrix{S}(undef, r, n)
        state.scratch[:lowrank] = w
    end
    return w::Matrix{S}
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

function upward_pass!(state::FMMState{S}, f::FMM{D}) where {S,D}
    tree = f.tree
    m = f.method
    L = nlevels(tree)
    Nm = multipole_length(m)
    nc = 1 << D
    Mp = state.multipoles
    fill!(Mp, zero(S))
    pml = partial_multipole_length(m)
    for l in L:-1:3
        lr = tree.level_range[l]
        nl = length(lr)
        ops = f.ops[l]
        pM = reshape(view(state.partial, 1:pml*nl), pml, nl)
        fill!(pM, zero(S))
        _s2m_level!(pM, m, ops, f.s2m_nodes[l], first(lr), tree, state.sources, state.τ)
        # check data → multipoles of the childless nodes (the parents' columns are zero)
        partial_to_multipole!(view(Mp, :, lr), m, ops, pM, state.work2)
        if l < L
            # M2M: the children's multipoles to the parent's, in group order
            cr = tree.level_range[l+1]
            ng = length(cr) ÷ nc
            B = reshape(view(Mp, :, cr), nc * Nm, ng)
            tmp = view(state.work1, 1:Nm, 1:ng)
            m2m_apply!(tmp, m, ops, B, ops.m2m_scale, state.work2)
            gp = tree.group_parent[l+1]
            @inbounds for g in 1:ng
                p = gp[g]
                for k in 1:Nm
                    Mp[k, p] = tmp[k, g]
                end
            end
        end
    end
    return state
end

function downward_pass!(state::FMMState{S}, f::FMM{D}) where {S,D}
    tree = f.tree
    m = f.method
    L = nlevels(tree)
    nc = 1 << D
    Mp = state.multipoles
    Lc = state.locals
    Pl = state.plocals
    fill!(Lc, zero(S))
    fill!(Pl, zero(S))
    pll = partial_local_length(m)
    for l in 3:L
        lr = tree.level_range[l]
        nl = length(lr)
        ops = f.ops[l]
        pL = reshape(view(state.partial, 1:pll*nl), pll, nl)
        if l > 3
            # L2L: the parents' partial locals (level l-1) to the whole level block, in group order
            ng = nl ÷ nc
            gp = tree.group_parent[l]
            G = view(state.work1, 1:pll, 1:ng)
            @inbounds for g in 1:ng
                p = gp[g]
                for k in 1:pll
                    G[k, g] = Pl[k, p]
                end
            end
            pops = f.ops[l-1]
            l2l_apply!(reshape(pL, nc * pll, ng), m, pops, G, pops.l2l_scale, state.work2)
        else
            fill!(pL, zero(S))
        end
        # M2L for the whole level
        m2l_level!(pL, m, ops, Mp, tree.ilists[l], first(lr) - 1, state)
        copyto!(view(Pl, :, lr), pL)
        # the densities the evaluation reads: those of the level's real leaves only (the
        # propagation runs on the partial locals), gathered from the stored partials,
        # transformed in the level buffer (free once the partials are stored) and
        # scattered into the leaf columns; the other columns of `locals` stay zero
        leaves = f.leaf_nodes[l]
        nq = length(leaves)
        if nq > 0
            G = view(state.work1, 1:pll, 1:nq)
            @inbounds for (q, i) in enumerate(leaves), k in 1:pll
                G[k, q] = Pl[k, i]
            end
            Nl = size(Lc, 1)
            Tq = reshape(view(state.partial, 1:Nl*nq), Nl, nq)
            partial_to_local!(Tq, m, ops, G, state.work2)
            @inbounds for (q, i) in enumerate(leaves), k in 1:Nl
                Lc[k, i] = Tq[k, q]
            end
        end
    end
    return state
end

"""
    build_expansions!(state, fmm, sources, τ)

Form all multipole and local expansions for the sources (in the order the tree was
built from, any eltype) and densities `τ`.
"""
function build_expansions!(state::FMMState, f::FMM{D}, sources, τ::AbstractVector) where {D}
    tree = f.tree
    pts = as_svectors(Val(D), sources)
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
function build_expansions(f::FMM{D}, sources, τ::AbstractVector) where {D}
    pts = as_svectors(Val(D), sources)
    Tp = eltype(eltype(pts))
    Tτ = eltype(τ)
    S = expansion_eltype(f.method, Tp, Tτ)
    state = FMMState(f, S, Tp, Tτ)
    return build_expansions!(state, f, pts, τ)
end
