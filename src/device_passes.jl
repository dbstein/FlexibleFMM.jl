# The expansion passes on a device (KernelAbstractions + the device's GEMM): the
# upward pass as the S2M kernel, the factored pseudo-inverse and the folded M2M
# as GEMMs; the downward pass as L2L GEMMs, the dense translations by symmetry
# class (one operator per class: a signed-permutation gather of the sources'
# multipoles, one GEMM, a target-owned scatter), and the pseudo-inverse to the
# densities the evaluation kernel reads. Real and complex expansions (complex
# GEMMs need the device's BLAS to support them: CUDA yes, Metal no); dual numbers
# take the CPU passes and the device evaluation. All buffers are exact-size device
# arrays, because device GEMMs on views fall back to slow generic code.

"""
    DevicePlan(fmm, to=identity; tol=0, fast=false)

Everything a device needs for the full matvec of a kernel-independent cube (or
circle) method or of a harmonic method (`SphericalHarmonics`, `StokesHarmonics`:
the class representatives of `docs/harmonics.md`, the S2M and evaluation kernels
through the same folds as the CPU, no pseudo-inverse stages): the evaluation data (`DeviceEvaluator`), the per-level operators
(folded M2M and L2L, the pseudo-inverse factors, the class operators of the dense
translations), the per-level node lists, the interaction lists by class built on
the device from the tree's flat arrays (`device_lists.jl`), the signed
permutations of the far offsets, and the work buffers. With `tol > 0`
the class operators of the cube method are replaced by their low-rank factors
(`class_factors`: singular values above `tol` times the largest), two GEMMs per
chunk instead of one, `4·n·k` flops per pair instead of `2·n²`; `tol` should sit
below the accuracy of the order used (`1e-8` for Laplace at `p = 8`, `1e-6` for
Stokes). `verbose` prints the build's stages (evaluator, operators, pair lists, buffers
and uploads).
"""
struct DevicePlan{D,E,M,V,VI,MI,MB,VO}
    dev::E
    nlevels::Int
    lo::Vector{Int}               # first node of each level
    nl::Vector{Int}
    nm::Int                       # multipole length
    pml::Int                      # partial multipole length (check data)
    pll::Int                      # partial local length
    nloc::Int                     # local length (density)
    s2m_nodes::Vector{VI}         # per level: childless nodes with sources (global, Int32)
    leaf_nodes::Vector{VI}        # per level: the real leaves (global), whose locals the evaluation reads
    leaf_cols::Vector{VI}         # the same as level-local columns
    group_parent::Vector{VI}      # per level: parent of each sibling group (global)
    M2M::Vector{M}; m2m_scale::Vector{Float64}
    L2L::Vector{M}; l2l_scale::Vector{Float64}
    Uh_up::Vector{M}; invS_up::Vector{V}; V_up::Vector{M}
    Uh_dn::Vector{M}; invS_dn::Vector{V}; V_dn::Vector{M}
    pinv_scale::Vector{Float64}
    M2Lc::Vector{Vector{M}}; m2l_scale::Vector{Float64}     # class operators per level (dense, or A of A·Bt)
    M2Lb::Vector{Vector{M}}                                  # Bt factors per level (empty vectors: dense)
    Zc::Dict{Tuple{Int,Int},M}                               # per (class, chunk length): Bt·Gc work (rank × nb·len)
    idx::MI; inv::MI; sgn::MB; sgn_t::MB                     # Ns × nfar: the gather applies (idx, sgn), the scatter (inv of idx_t, sgn_t)
    nb::Int                                                  # scalar expansions per box (columns per pair)
    offs::VO                                                 # the far offsets (the Stokes coupling)
    pinv::Bool                                               # kernel-independent: pseudo-inverse stages
    lists::Vector{DeviceLevelLists{VI}}                      # per level: the class lists and their runs, on the device
    chunks::Vector{Vector{ClassChunk}}                       # per level: the pieces of the class lists the GEMMs take
    chunk::Int
    multipoles::M; plocals::M; locals::M                     # global, one column per node
    pM::Vector{M}; tup::Vector{M}; Mup::Vector{M}            # per level: S2M check data and pinv work
    B::Vector{M}; tM::Vector{M}                              # per level: children block and M2M output
    G::Vector{M}; Y::Vector{M}                               # per level: L2L input and output
    pL::Vector{M}; pLq::Vector{M}; tdn::Vector{M}; Ldn::Vector{M}   # per level: partial locals, their leaf columns, pinv work, leaf densities
    Gc::Dict{Int,M}; Yc::Dict{Int,M}                         # per chunk length: class gather and GEMM output
    scratch::Dict{Any,Any}                                   # the inputs gathered into the tree's order, per type
end

"""
    DeviceEvaluator(fmm, to=identity; fast=false)
    DeviceEvaluator(dt::DeviceTree, method, ops, to; fast=false)

The evaluation data from a host FMM (uploaded) or from a device tree and the
levels' operators.
"""
function DeviceEvaluator(dt::DeviceTree{D,T}, m, ops::Vector, to; fast::Bool=false) where {D,T}
    n = nnodes(dt)
    far = _device_far(ops, m, nlevels(dt), to, Val(D))
    K = fast ? fastmath(near_kernel(m)) : near_kernel(m)
    centers = similar(dt.sources, SVector{D,T}, n)
    _centers!(get_backend(dt.level), 256)(centers, dt.cell, dt.level, dt.origin, dt.W0; ndrange=n)
    return DeviceEvaluator{D,typeof(K),typeof(dt.colleagues),typeof(dt.level),typeof(centers),typeof(far)}(
        K, dt.colleagues, dt.level, centers, far, dt.gplo, dt.gphi, dt.gpoint_node, dt.gperm)
end
DeviceEvaluator(f::FMM, to=identity; fast::Bool=false) = DeviceEvaluator(DeviceTree(f.tree, to), f.method, f.ops, to; fast=fast)

# the array mover of the points' array type (`CuArray`, `MtlArray`; `identity` for `Array`s)
_mover(::Array) = identity
_mover(xs::AbstractArray) = typeof(xs).name.wrapper

function DevicePlan(f::FMM{D,T}, to=identity; tol::Real=0, fast::Bool=false, eltype::Type=operator_eltype(f.method),
                    verbose::Bool=false) where {D,T}
    dt = DeviceTree(f.tree, to)
    return _device_plan(dt, f.method, f.ops, [to(Int32.(f.s2m_nodes[l])) for l in 1:nlevels(dt)],
                        [to(Int32.(f.leaf_nodes[l])) for l in 1:nlevels(dt)], to; tol=tol, fast=fast, eltype=eltype, verbose=verbose)
end

"""
    DevicePlan(method, xs; ppl=50, bbox, cache=OperatorCache(method), max_levels=30, tol=0, fast=false, eltype, verbose=false)
    DevicePlan(method, groups::Tuple; …)

The plan from points on the device (`xs` a `CuArray`/`MtlArray` of `SVector`s, or a
tuple of them, one per group of sources with its own density type;
`Array`s run on KernelAbstractions' CPU backend): the tree built there
(`device_tree`), the operators of its levels from `cache` (the root box `bbox =
(origin, width)` snapped to a width the cache knows), the node lists by compaction,
the lists and the chunks as in the plan of a host FMM. Nothing of the tree touches
the host but the level counts.
"""
DevicePlan(m::AbstractMethod, xs::AbstractVector{<:SVector}; kwargs...) = DevicePlan(m, (xs,); kwargs...)
function DevicePlan(m::AbstractMethod, groups::Tuple{Vararg{AbstractVector{SVector{D,T}}}}; ppl::Integer=50, bbox,
                    cache::OperatorCache=OperatorCache(m), max_levels::Integer=30, tol::Real=0, fast::Bool=false,
                    eltype::Type=operator_eltype(m), verbose::Bool=false) where {D,T}
    cache.method === m || throw(ArgumentError("the cache was built for a different method"))
    D == dimension(m) || throw(ArgumentError("$(D)-dimensional points with a $(dimension(m))-dimensional method"))
    isempty(groups) && throw(ArgumentError("no groups"))
    xs = length(groups) == 1 ? groups[1] : vcat(groups...)
    to = _mover(xs)
    t_tree = @elapsed begin
        lo = SVector{D,T}(ntuple(k -> mapreduce(p -> p[k], min, xs), Val(D)))
        hi = SVector{D,T}(ntuple(k -> mapreduce(p -> p[k], max, xs), Val(D)))
        origin, W0, e0 = fit_box!(cache, lo, hi, bbox)
        dt = device_tree(xs; ppl=ppl, bbox=(origin, W0), max_levels=max_levels, group_sizes=[length(g) for g in groups])
        L = nlevels(dt)
        ops = [get_operators!(cache, e0 + 1 - l) for l in 1:L]
        s2m_flag = map((c, h) -> Int32(c == 0 && h != 0), dt.child_first, dt.has_source)
        leaf_flag = map(x -> Int32(x), dt.leaf)
        s2m_nodes = [_compact(s2m_flag[r])[1] .+ Int32(first(r) - 1) for r in dt.level_range]
        leaf_nodes = [_compact(leaf_flag[r])[1] .+ Int32(first(r) - 1) for r in dt.level_range]
    end
    verbose && println("DevicePlan: tree ", round(t_tree, digits=3), " s (", L, " levels, ", nnodes(dt), " nodes)")
    return _device_plan(dt, m, ops, s2m_nodes, leaf_nodes, to; tol=tol, fast=fast, eltype=eltype, verbose=verbose)
end

function _device_plan(dt::DeviceTree{D,T}, m, ops::Vector, s2m_nodes::Vector, leaf_nodes::Vector, to;
                      tol::Real=0, fast::Bool=false, eltype::Type=operator_eltype(m), verbose::Bool=false) where {D,T}
    t_start = time()
    m isa AbstractKernelIndependent || m isa HarmonicMethod ||
        throw(ArgumentError("the device passes are written for the kernel-independent and harmonic methods"))
    K = m.kernel
    L = nlevels(dt)
    nc = 1 << D
    Nm, pml, pll, Nl = multipole_length(m), partial_multipole_length(m), partial_local_length(m), local_length(m)
    R = floattype(K)
    TO = eltype                          # the operators' and expansions' eltype: the kernel's, or complex for
                                         # complex densities (the operators promote; no complex GEMM on Metal)
    TO <: Number && promote_type(TO, operator_eltype(m)) == TO ||
        throw(ArgumentError("eltype must hold the operators' element type " * string(operator_eltype(m))))
    pinv = m isa AbstractKernelIndependent      # the harmonic methods have no partial-to-expansion transforms
    nb = pinv ? 1 : _nblocks(m)                 # scalar expansions per box, translated as columns of one pair
    Ns = Nm ÷ nb                                # rows of the scalar operators and permutations
    t_dev = @elapsed dev = DeviceEvaluator(dt, m, ops, to; fast=fast)
    lo = [first(dt.level_range[l]) for l in 1:L]
    nl = [length(dt.level_range[l]) for l in 1:L]
    # the dense translation operators: one per symmetry class for the cube method
    # (built at the width of level 3 for a homogeneous kernel and scaled, else per
    # level), one per far offset for circles, which have no cube lattice
    t_ops = @elapsed begin
    l3 = min(3, L)
    a = homogeneity(K)
    M2Lc = Vector{Vector{typeof(to(zeros(TO, 1, 1)))}}(undef, L)
    M2Lb = [typeof(to(zeros(TO, 1, 1)))[] for _ in 1:L]
    m2l_scale = ones(L)
    ranks = Int[]
    rankl = [Int[] for _ in 1:L]                # per level: the low-rank class ranks (empty: dense)
    if m isa CubeKernelIndependent && covariant(K)
        e3 = ops[l3].extra
        build(e, w) = tol > 0 ? (r = class_factors(K, e.small, e.embed, w, e.p, tol); (r[1], r[2], r[3], r[4])) :
                                (r = class_operators(K, e.small, e.embed, w, e.p); (r[1], r[2], Matrix{TO}[], Int[]))
        sc, opsc, btc, ranks = build(e3, ops[l3].width)
        dc = [to(Matrix{TO}(A)) for A in opsc]
        db = [to(Matrix{TO}(B)) for B in btc]
        for l in 1:L
            if a === nothing && l >= 3 && l != l3
                _, ol, bl, rl = build(ops[l].extra, ops[l].width)
                M2Lc[l] = [to(Matrix{TO}(A)) for A in ol]
                M2Lb[l] = [to(Matrix{TO}(B)) for B in bl]
                rankl[l] = rl
            else
                M2Lc[l] = dc
                M2Lb[l] = db
                rankl[l] = ranks
                m2l_scale[l] = a === nothing ? 1.0 : (ops[l].width / ops[l3].width)^a
            end
        end
    elseif m isa CubeKernelIndependent          # no covariance: one dense operator per offset, per width
        offs = far_offsets(Val(D))
        nfo = length(offs)
        sc = SymmetryClasses{D}(offs, collect(1:nfo), [collect(1:Nm) for _ in 1:nfo], [ones(Int8, Nm) for _ in 1:nfo])
        dense(l) = [to(Matrix{TO}(_cube_m2l_matrix(K, ops[l].extra, ops[l].width, o))) for o in offs]
        dc = dense(l3)
        for l in 1:L
            if a === nothing && l >= 3 && l != l3
                M2Lc[l] = dense(l)
            else
                M2Lc[l] = dc
                m2l_scale[l] = a === nothing ? 1.0 : (ops[l].width / ops[l3].width)^a
            end
        end
    elseif m isa HarmonicMethod                 # the 34 class representatives, shared by all levels (homogeneous)
        e3 = ops[l3].extra
        sc = e3.sc
        reps = e3.reps
        if reps[1] isa LowRankM2L
            dc = [to(Matrix{TO}(r.U)) for r in reps]
            db = [to(Matrix{TO}(r.Vt)) for r in reps]
            ranks = [size(r.U, 2) for r in reps]
        else
            dc = [to(Matrix{TO}(r.A)) for r in reps]
            db = typeof(to(zeros(TO, 1, 1)))[]
        end
        for l in 1:L
            M2Lc[l] = dc
            M2Lb[l] = db
            rankl[l] = ranks
            m2l_scale[l] = Float64(ops[l].m2l_scale)
        end
    else
        offs = far_offsets(Val(D))
        nfo = length(offs)
        sc = SymmetryClasses{D}(offs, collect(1:nfo), [collect(1:Nm) for _ in 1:nfo], [ones(Int8, Nm) for _ in 1:nfo])
        dc = [to(Matrix{TO}(ops[l3].M2L[fi].A)) for fi in 1:nfo]
        for l in 1:L
            if a === nothing && l >= 3 && l != l3
                M2Lc[l] = [to(Matrix{TO}(ops[l].M2L[fi].A)) for fi in 1:nfo]
            else
                M2Lc[l] = dc                                     # shared reference matrices
                m2l_scale[l] = Float64(ops[l].m2l_scale)
            end
        end
    end
    end
    nfar = length(sc.class)
    idx = Matrix{Int32}(undef, Ns, nfar); inv = similar(idx); sgn = Matrix{Int8}(undef, Ns, nfar); sgn_t = similar(sgn)
    for fi in 1:nfar
        idx[:, fi] = sc.idx[fi]; sgn[:, fi] = sc.sgn[fi]          # the gather applies P_sᵀ
        inv[sc.idx_t[fi], fi] = 1:Ns; sgn_t[:, fi] = sc.sgn_t[fi]  # the scatter applies P_t
    end
    offs = to(SVector{D,R}.(far_offsets(Val(D))))                 # the Stokes coupling Ψ̂ += o·Φ
    # the lists, the class lists and the chunks of every level, built on the device
    # from the tree's flat arrays (device_lists.jl); levels 1 and 2 have no far pairs
    nclass = length(sc.reps)
    chunk = 4096
    VI = typeof(to(Int32[]))
    t_lists = @elapsed begin
        far_index_d = to(Int32.(far_index_table(Val(D))))
        class_d = to(Int32.(sc.class))
        lists = Vector{DeviceLevelLists{VI}}(undef, L)
        chunks = Vector{Vector{ClassChunk}}(undef, L)
        for l in 1:L
            if l >= 3
                lists[l], chunks[l] = _device_level_lists(dt, dev.colleagues, lo[l], nl[l], far_index_d, class_d, nclass, chunk, Val(D))
            else
                lists[l], chunks[l] = DeviceLevelLists(to(Int32[])), ClassChunk[]
            end
        end
    end
    t_up = @elapsed begin
    proto = to(Matrix{TO}(undef, 0, 0))
    mk(r, c) = fill!(similar(proto, r, c), zero(TO))      # zeroed on the device: no host zeros to page in and upload
    # the Stokes harmonic M2M and L2L in their dense form: one large GEMM per level
    # beats the eight child blocks with their kernels on a device
    dense_ops = nb == 4 ? Dict(l => _stokes_dense_operators(ops[l], Ns) for l in 1:L) : Dict{Int,Tuple{Matrix{TO},Matrix{TO}}}()
    m2m_dense(l) = nb == 4 ? dense_ops[l][1] : ops[l].M2M
    l2l_dense(l) = nb == 4 ? dense_ops[l][2] : ops[l].L2L
    lens = unique([ch.len for l in 1:L for ch in chunks[l]])
    isempty(lens) && push!(lens, 64)
    zranks = Dict{Tuple{Int,Int},Int}()             # per (class, chunk length): the rank over the levels
    for l in 1:L, ch in chunks[l]
        r = isempty(rankl[l]) ? 1 : rankl[l][ch.class]
        key = (ch.class, ch.len)
        zranks[key] = max(get(zranks, key, 0), r)
    end
    # the pseudo-inverse factors (placeholders for the harmonic methods)
    pm(get) = pinv ? [to(Matrix{TO}(get(ops[l].extra))) for l in 1:L] : [mk(1, 1) for l in 1:L]
    pv(get) = pinv ? [to(Vector{R}(get(ops[l].extra))) for l in 1:L] : [to(zeros(R, 1)) for l in 1:L]
    rup(l) = pinv ? rank(ops[l].extra.P_up) : 1
    rdn(l) = pinv ? rank(ops[l].extra.P_down) : 1
    nq(l) = max(1, length(leaf_nodes[l]))
    plan = DevicePlan{D,typeof(dev),typeof(to(zeros(TO, 1, 1))),typeof(to(zeros(R, 1))),VI,typeof(to(idx)),typeof(to(sgn)),typeof(offs)}(
        dev, L, lo, nl, Nm, pml, pll, Nl,
        s2m_nodes, leaf_nodes, [leaf_nodes[l] .- Int32(lo[l] - 1) for l in 1:L], dt.group_parent,
        [to(Matrix{TO}(m2m_dense(l))) for l in 1:L], [Float64(ops[l].m2m_scale) for l in 1:L],
        [to(Matrix{TO}(l2l_dense(l))) for l in 1:L], [Float64(ops[l].l2l_scale) for l in 1:L],
        pm(e -> e.P_up.Uh), pv(e -> e.P_up.invS), pm(e -> e.P_up.V),
        pm(e -> e.P_down.Uh), pv(e -> e.P_down.invS), pm(e -> e.P_down.V),
        pinv ? [Float64(ops[l].extra.pinv_scale) for l in 1:L] : ones(L),
        M2Lc, m2l_scale, M2Lb,
        Dict(key => mk(max(1, r), nb * key[2]) for (key, r) in zranks),
        to(idx), to(inv), to(sgn), to(sgn_t), nb, offs, pinv, lists, chunks, chunk,
        mk(Nm, nnodes(dt)), mk(pll, nnodes(dt)), mk(Nl, nnodes(dt)),
        [mk(pml, max(1, length(s2m_nodes[l]))) for l in 1:L],
        [mk(rup(l), max(1, length(s2m_nodes[l]))) for l in 1:L],
        [mk(Nm, max(1, length(s2m_nodes[l]))) for l in 1:L],
        [mk(nc * Nm, l < L ? nl[l+1] ÷ nc : 1) for l in 1:L], [mk(Nm, l < L ? nl[l+1] ÷ nc : 1) for l in 1:L],
        [mk(pll, max(1, nl[l] ÷ nc)) for l in 1:L], [mk(nc * pll, max(1, nl[l] ÷ nc)) for l in 1:L],
        [mk(pll, nl[l]) for l in 1:L], [mk(pll, nq(l)) for l in 1:L], [mk(rdn(l), nq(l)) for l in 1:L], [mk(Nl, nq(l)) for l in 1:L],
        Dict(len => mk(Ns, nb * len) for len in lens), Dict(len => mk(Ns, nb * len) for len in lens), Dict{Any,Any}())
    end
    if verbose
        t_all = time() - t_start
        println("DevicePlan: evaluator ", round(t_dev, digits=3), " s, operators ", round(t_ops, digits=3), " s, pair lists ",
                round(t_lists, digits=3), " s, buffers and uploads ", round(t_up, digits=3), " s, other ",
                round(t_all - t_dev - t_ops - t_lists - t_up, digits=3), " s")
    end
    return plan
end

# multipoles of the childless nodes of the harmonic methods: one work-item per node
# and order m, the order's chain of harmonics.jl on the node's sources (the chains
# of different orders are independent and write disjoint entries)
@kernel function _s2m_harmonic_kernel!(M, @Const(nodes), @Const(centers), @Const(level), far, @Const(gplo), @Const(gphi),
                                       sources, τ, ::Val{nb}) where {nb}
    g = @index(Global, Linear)
    @inbounds begin
        p = far.p
        q = (g - 1) ÷ (p + 1) + 1
        m = (g - 1) % (p + 1)
        node = Int(nodes[q])
        c = centers[node]
        s = far.scales[Int(level[node])]
        col = view(M, :, q)
        N = (p + 1)^2
        for a in 1:nb
            _zero_order!(_block(col, N, a), p, m)
        end
        _s2m_harmonic_groups!(col, sources, τ, gplo, gphi, node, c, s, p, far.tb, m, Val(nb), 1)
    end
end
@inline function _s2m_harmonic_groups!(col, sources::Tuple, τ::Tuple, gplo, gphi, node, c, s, p, tb, m, ::Val{nb}, g::Int) where {nb}
    τg = first(τ)
    if τg !== nothing
        xs = first(sources)
        @inbounds for j in Int(gplo[g, node]):Int(gphi[g, node])
            _device_accumulate!(col, (xs[j] - c) * s, p, tb, τg[j], s, m, Val(nb))
        end
    end
    return _s2m_harmonic_groups!(col, Base.tail(sources), Base.tail(τ), gplo, gphi, node, c, s, p, tb, m, Val(nb), g + 1)
end
@inline _s2m_harmonic_groups!(col, ::Tuple{}, ::Tuple{}, gplo, gphi, node, c, s, p, tb, m, ::Val{nb}, g::Int) where {nb} = col
@inline _device_accumulate!(col, ŷ, p, tb, τ, s, m, ::Val{1}) = _accumulate_order!(col, ŷ, p, tb, _scaled_density(τ, s), m)
@inline _device_accumulate!(col, ŷ, p, tb, τ, s, m, ::Val{4}) = _stokes_accumulate_order!(col, (p + 1)^2, ŷ, p, tb, τ, s, m)

# ---- kernels ----

# check data of the childless nodes: one work-item per (node, check point)
@kernel function _s2m_kernel!(pM, @Const(nodes), @Const(centers), @Const(ysoa), lvl, @Const(gplo), @Const(gphi),
                              sources, τ, K, ::Val{D}, ::Val{nc}) where {D,nc}
    g = @index(Global, Linear)
    @inbounds begin
        N = size(ysoa, 1)
        q = (g - 1) ÷ N + 1
        k = (g - 1) % N + 1
        node = Int(nodes[q])
        c = centers[node]
        y = SVector(ntuple(d -> ysoa[k, d, lvl], Val(D)))
        acc = _s2m_groups_device(K, sources, τ, gplo, gphi, node, c, y, _field_zero(eltype(pM), Val(nc)), 1)
        _store!(view(pM, :, q), k, N, acc)
    end
end
# the check potential of one point over the groups of a node (tuples, unrolled by recursion)
@inline function _s2m_groups_device(K, sources::Tuple, τ::Tuple, gplo, gphi, node, c, y, acc, g::Int)
    τg = first(τ)
    if τg !== nothing
        xs = first(sources)
        @inbounds for j in Int(gplo[g, node]):Int(gphi[g, node])
            acc += source_field(K, xs[j] - c, y, τg[j])
        end
    end
    return _s2m_groups_device(K, Base.tail(sources), Base.tail(τ), gplo, gphi, node, c, y, acc, g + 1)
end
@inline _s2m_groups_device(K, ::Tuple{}, ::Tuple{}, gplo, gphi, node, c, y, acc, g::Int) = acc
# columns of `src` (scaled) into the columns `cols` (global node indices) of `dst`
@kernel function _scatter_cols!(dst, @Const(src), @Const(cols), α)
    g = @index(Global, Linear)
    @inbounds begin
        n = size(src, 1)
        q = (g - 1) ÷ n + 1
        i = (g - 1) % n + 1
        dst[i, Int(cols[q])] = α * src[i, q]
    end
end
# the stacked children output of L2L into the level's partial locals (local columns)
@kernel function _scatter_children!(pL, @Const(Y), pll, nc)
    g = @index(Global, Linear)
    @inbounds begin
        r = size(Y, 1)
        gi = (g - 1) ÷ r + 1
        i = (g - 1) % r + 1
        k = (i - 1) ÷ pll + 1                 # child slot
        e = (i - 1) % pll + 1
        pL[e, (gi - 1) * nc + k] = Y[i, gi]
    end
end
# gather P_sᵀ·m_s for the pairs of a class, the nb scalar expansions of a pair as
# consecutive columns: Gc[i, (q−1)nb + a] = sgn[i, f] · M[(a−1)n + idx[i, f], src[q]]
@kernel function _class_gather!(Gc, @Const(M), @Const(src), @Const(fs), a::Int32, npk::Int32, @Const(idx), @Const(sgn),
                                ::Val{nb}) where {nb}
    g = @index(Global, Linear)
    @inbounds begin
        n = size(Gc, 1)
        col = (g - 1) ÷ n + 1
        i = (g - 1) % n + 1
        q = (col - 1) ÷ nb + 1
        blk = (col - 1) % nb + 1
        if q <= npk
            f = Int(fs[a + q - 1])
            Gc[i, col] = sgn[i, f] * M[(blk - 1) * n + Int(idx[i, f]), Int(src[a + q - 1])]
        else                                    # padding column
            Gc[i, col] = zero(eltype(Gc))
        end
    end
end
# scatter P_t·y_q into the targets, one work-item per (run of the chunk, block a,
# entry j): (P_t y)[j] = sgn_t[inv[j]] · y[inv[j]] over the run's pairs that fall in
# the chunk; the fourth block of the Stokes method also takes the coupling o·Φ of
# the pair's offset
@kernel function _class_scatter!(pL, @Const(Yc), @Const(rtgt), @Const(rptr), @Const(fs), a::Int32, npk::Int32, r1::Int32,
                                 @Const(inv), @Const(sgn), @Const(offs), α, ::Val{nb}) where {nb}
    g = @index(Global, Linear)
    @inbounds begin
        n = size(Yc, 1)
        t = (g - 1) ÷ (n * nb) + 1
        r = (g - 1) % (n * nb)
        blk = r ÷ n + 1
        j = r % n + 1
        run = Int(r1) + t - 1
        acc = zero(eltype(pL))
        for p in max(Int(rptr[run]), Int(a)):min(Int(rptr[run+1]) - 1, Int(a) + Int(npk) - 1)
            f = Int(fs[p])
            i = Int(inv[j, f])
            col = (p - Int(a)) * nb + blk
            v = Yc[i, col]
            if nb == 4 && blk == 4
                o = offs[f]
                v += o[1] * Yc[i, col-3] + o[2] * Yc[i, col-2] + o[3] * Yc[i, col-1]
            end
            acc += sgn[i, f] * v
        end
        pL[(blk - 1) * n + j, Int(rtgt[run])] += α * acc
    end
end

# ---- the passes ----

function _pinv_apply!(out, Uh, invS, V, X, tmp, α)
    mul!(tmp, Uh, X)
    tmp .= invS .* tmp
    mul!(out, V, tmp)
    α == 1 || (out .*= eltype(out)(α))
    return out
end

"""
    device_build_expansions!(plan::DevicePlan, ds; sorted=false)

The upward and downward passes on the device from the device state `ds`
(sources and densities): fills `plan.multipoles`, `plan.plocals` and
`plan.locals`, the last of which the evaluation kernel reads.
"""
function device_build_expansions!(plan::DevicePlan{D}, ds; sorted::Bool=false) where {D}
    ds = sorted ? (sources=_group_tuple(ds.sources), τ=_group_tuple(ds.τ)) : _sorted_inputs(plan, ds)
    dev = plan.dev
    backend = get_backend(plan.multipoles)
    K = dev.kernel
    nc = ncomp(K, Val(D))
    nch = 1 << D
    L = plan.nlevels
    Nm, pml, pll = plan.nm, plan.pml, plan.pll
    Mp = plan.multipoles
    fill!(Mp, zero(eltype(Mp)))
    # upward
    for l in L:-1:3
        nq = length(plan.s2m_nodes[l])
        if nq > 0
            if plan.pinv
                pM = plan.pM[l]
                ysoa = dev.far.ysoa
                _s2m_kernel!(backend, 256)(pM, plan.s2m_nodes[l], dev.centers, ysoa, l, dev.gplo, dev.gphi,
                                           ds.sources, ds.τ, K, Val(D), Val(nc); ndrange=nq * size(ysoa, 1))
                _pinv_apply!(plan.Mup[l], plan.Uh_up[l], plan.invS_up[l], plan.V_up[l], pM, plan.tup[l], plan.pinv_scale[l])
            else
                _s2m_harmonic_kernel!(backend, 256)(plan.Mup[l], plan.s2m_nodes[l], dev.centers, dev.level, dev.far, dev.gplo, dev.gphi,
                                                    ds.sources, ds.τ, Val(plan.nb); ndrange=nq * (dev.far.p + 1))
            end
            _scatter_cols!(backend, 256)(Mp, plan.Mup[l], plan.s2m_nodes[l], one(eltype(Mp)); ndrange=Nm * nq)
        end
        if l < L
            ng = plan.nl[l+1] ÷ nch
            B = plan.B[l]
            copyto!(vec(B), 1, vec(Mp), (plan.lo[l+1] - 1) * Nm + 1, nch * Nm * ng)
            _device_m2m!(plan, l, B, ng, backend)
            _scatter_cols!(backend, 256)(Mp, plan.tM[l], plan.group_parent[l+1], eltype(Mp)(plan.m2m_scale[l]); ndrange=Nm * ng)
        end
    end
    synchronize(backend)
    # downward
    Pl, Lc = plan.plocals, plan.locals
    fill!(Pl, zero(eltype(Pl))); fill!(Lc, zero(eltype(Lc)))
    for l in 3:L
        nl = plan.nl[l]
        pL = plan.pL[l]
        if l > 3
            ng = nl ÷ nch
            G = plan.G[l]
            _gather_cols!(backend, 256)(G, Pl, plan.group_parent[l]; ndrange=pll * ng)
            _device_l2l!(plan, l, G, ng, backend)
            plan.l2l_scale[l-1] == 1 || (plan.Y[l] .*= eltype(pL)(plan.l2l_scale[l-1]))
            _scatter_children!(backend, 256)(pL, plan.Y[l], pll, nch; ndrange=nch * pll * ng)
        else
            fill!(pL, zero(eltype(pL)))
        end
        Ns = Nm ÷ plan.nb
        li = plan.lists[l]
        for ch in plan.chunks[l]
            Gc, Yc = plan.Gc[ch.len], plan.Yc[ch.len]
            _class_gather!(backend, 256)(Gc, Mp, li.src, li.f, Int32(ch.a), Int32(ch.npk), plan.idx, plan.sgn, Val(plan.nb);
                                         ndrange=Ns * plan.nb * ch.len)
            if isempty(plan.M2Lb[l])
                mul!(Yc, plan.M2Lc[l][ch.class], Gc)
            else                                    # low-rank: A·(Bt·Gc)
                Z = plan.Zc[(ch.class, ch.len)]
                mul!(Z, plan.M2Lb[l][ch.class], Gc)
                mul!(Yc, plan.M2Lc[l][ch.class], Z)
            end
            nt = ch.r2 - ch.r1 + 1
            _class_scatter!(backend, 256)(pL, Yc, li.rtgt, li.rptr, li.f, Int32(ch.a), Int32(ch.npk), Int32(ch.r1), plan.inv,
                                          plan.sgn_t, plan.offs, eltype(pL)(plan.m2l_scale[l]), Val(plan.nb); ndrange=pll * nt)
        end
        copyto!(vec(Pl), (plan.lo[l] - 1) * pll + 1, vec(pL), 1, pll * nl)
        # the densities of the level's real leaves only (the propagation runs on the partials)
        nq = length(plan.leaf_nodes[l])
        if nq > 0
            G = plan.pLq[l]
            _gather_cols!(backend, 256)(G, pL, plan.leaf_cols[l]; ndrange=pll * nq)
            if plan.pinv
                _pinv_apply!(plan.Ldn[l], plan.Uh_dn[l], plan.invS_dn[l], plan.V_dn[l], G, plan.tdn[l], plan.pinv_scale[l])
                _scatter_cols!(backend, 256)(Lc, plan.Ldn[l], plan.leaf_nodes[l], one(eltype(Lc)); ndrange=plan.nloc * nq)
            else                                # the locals are the partial locals
                _scatter_cols!(backend, 256)(Lc, G, plan.leaf_nodes[l], one(eltype(Lc)); ndrange=pll * nq)
            end
        end
    end
    synchronize(backend)
    return plan
end
# M2M and L2L of one level: one GEMM (the Stokes harmonic method in its dense
# block form; the blockwise application of m2m_apply! was measured slower here)
_device_m2m!(plan::DevicePlan, l::Int, B, ng::Int, backend) = (mul!(plan.tM[l], plan.M2M[l], B); plan)
_device_l2l!(plan::DevicePlan, l::Int, G, ng::Int, backend) = (mul!(plan.Y[l], plan.L2L[l-1], G); plan)
# columns `cols` (global) of `src` into the columns of `dst`
@kernel function _gather_cols!(dst, @Const(src), @Const(cols))
    g = @index(Global, Linear)
    @inbounds begin
        n = size(dst, 1)
        q = (g - 1) ÷ n + 1
        i = (g - 1) % n + 1
        dst[i, q] = src[i, Int(cols[q])]
    end
end

"""
    device_matvec!(out, plan::DevicePlan, ds; output=Potential(), sorted=false)

Expansions on the device and the self evaluation at the sources, `out` in the
original order.
"""
function device_matvec!(out::AbstractVector, plan::DevicePlan, ds; output=Potential(), sorted::Bool=false)
    ngroups(plan.dev) == 1 || throw(ArgumentError("several groups: pass one output vector per group (a tuple)"))
    return device_matvec!((out,), plan, ds; outputs=(output,), sorted=sorted)[1]
end
"""
    device_matvec!(outs::Tuple, plan, ds; outputs=(Potential(), …), sorted=false)

With groups: `ds = (sources=(xs₁, …), τ=(τ₁, …))` (a group with `nothing` densities
is targets only), one output array per group (`nothing` for a group whose output
is not wanted), one output type per group.
"""
function device_matvec!(outs::Tuple, plan::DevicePlan, ds; outputs::Tuple=ntuple(_ -> Potential(), length(outs)), sorted::Bool=false)
    G = ngroups(plan.dev)
    length(outs) == G == length(outputs) || throw(DimensionMismatch("expected $G outputs"))
    ds = sorted ? (sources=_group_tuple(ds.sources), τ=_group_tuple(ds.τ)) : _sorted_inputs(plan, ds)
    device_build_expansions!(plan, ds; sorted=true)
    dsl = (locals=plan.locals, sources=ds.sources, τ=ds.τ)
    for g in 1:G
        outs[g] === nothing || evaluate_at_sources!(outs[g], plan.dev, dsl, g; output=outputs[g])
    end
    return outs
end
# the sources and densities gathered into the tree's order by `perm`, into buffers
# the plan keeps per element type (`sorted=true` on the entry points takes inputs
# already in that order, as `device_state` provides them)
function _sorted_inputs(plan::DevicePlan, ds)
    srcs = _group_tuple(ds.sources)
    τs = _group_tuple(ds.τ)
    G = ngroups(plan.dev)
    length(srcs) == G == length(τs) || throw(DimensionMismatch("expected $G groups of sources and densities"))
    sources = ntuple(g -> _gather_input!(plan, :sources, g, srcs[g]), G)
    τ = ntuple(g -> _gather_input!(plan, :τ, g, τs[g]), G)
    return (sources=sources, τ=τ)
end
_gather_input!(plan::DevicePlan, name::Symbol, g::Int, ::Nothing) = nothing
function _gather_input!(plan::DevicePlan, name::Symbol, g::Int, x)
    perm = plan.dev.gperm[g]
    length(x) == length(perm) || throw(DimensionMismatch("expected $(length(perm)) entries in group $g"))
    key = (name, g, typeof(x))
    buf = get(plan.scratch, key, nothing)
    if buf === nothing || length(buf) != length(x)
        buf = similar(x)
        plan.scratch[key] = buf
    end
    _gather_perm!(get_backend(buf), 256)(buf, x, perm; ndrange=length(x))
    return buf
end
