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
translations), the per-level node lists, the class pair lists grouped by target,
the signed permutations of the far offsets, and the work buffers. With `tol > 0`
the class operators of the cube method are replaced by their low-rank factors
(`class_factors`: singular values above `tol` times the largest), two GEMMs per
chunk instead of one, `4·n·k` flops per pair instead of `2·n²`; `tol` should sit
below the accuracy of the order used (`1e-8` for Laplace at `p = 8`, `1e-6` for
Stokes).
"""
# The pairs of one class of one level, grouped by target, in a chunk of `chunk`
# columns (padded with dummy pairs that no target reads): the gather fills a fixed
# Nm × chunk buffer, one GEMM, and the scatter runs over the chunk's targets.
struct ClassChunk{VI}
    class::Int
    src::VI        # global source node per column (padded)
    f::VI          # far offset per column
    tgt::VI        # local target index of each target in the chunk
    ptr::VI        # CSR over the columns, per target
end

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
    group_parent::Vector{VI}      # per level: parent of each sibling group (global)
    M2M::Vector{M}; m2m_scale::Vector{Float64}
    L2L::Vector{M}; l2l_scale::Vector{Float64}
    Uh_up::Vector{M}; invS_up::Vector{V}; V_up::Vector{M}
    Uh_dn::Vector{M}; invS_dn::Vector{V}; V_dn::Vector{M}
    pinv_scale::Vector{Float64}
    M2Lc::Vector{Vector{M}}; m2l_scale::Vector{Float64}     # class operators per level (dense, or A of A·Bt)
    M2Lb::Vector{Vector{M}}                                  # Bt factors per level (empty vectors: dense)
    Zc::Vector{M}                                            # per class: Bt·Gc work (rank × chunk)
    idx::MI; inv::MI; sgn::MB; sgn_t::MB                     # Ns × nfar: the gather applies (idx, sgn), the scatter (inv of idx_t, sgn_t)
    nb::Int                                                  # scalar expansions per box (columns per pair)
    offs::VO                                                 # the far offsets (the Stokes coupling)
    pinv::Bool                                               # kernel-independent: pseudo-inverse stages
    chunks::Vector{Vector{ClassChunk{VI}}}                  # per level: the pairs by class in fixed-size chunks
    chunk::Int
    multipoles::M; plocals::M; locals::M                     # global, one column per node
    pM::Vector{M}; tup::Vector{M}; Mup::Vector{M}            # per level: S2M check data and pinv work
    B::Vector{M}; tM::Vector{M}                              # per level: children block and M2M output
    G::Vector{M}; Y::Vector{M}                               # per level: L2L input and output
    pL::Vector{M}; tdn::Vector{M}; Ldn::Vector{M}            # per level: partial locals, pinv work, densities
    Gc::M; Yc::M                                             # class gather and GEMM output (max pairs)
end

function DevicePlan(f::FMM{D,T}, to=identity; tol::Real=0, fast::Bool=false) where {D,T}
    tree = f.tree
    m = f.method
    m isa AbstractKernelIndependent || m isa HarmonicMethod ||
        throw(ArgumentError("the device passes are written for the kernel-independent and harmonic methods"))
    K = m.kernel
    L = nlevels(tree)
    nc = 1 << D
    Nm, pml, pll, Nl = multipole_length(m), partial_multipole_length(m), partial_local_length(m), local_length(m)
    R = floattype(K)
    TO = operator_eltype(m)              # the operators' eltype, complex for a complex kernel
    pinv = m isa AbstractKernelIndependent      # the harmonic methods have no partial-to-expansion transforms
    nb = pinv ? 1 : _nblocks(m)                 # scalar expansions per box, translated as columns of one pair
    Ns = Nm ÷ nb                                # rows of the scalar operators and permutations
    dev = DeviceEvaluator(f, to; fast=fast)
    lo = [first(tree.level_range[l]) for l in 1:L]
    nl = [length(tree.level_range[l]) for l in 1:L]
    # the dense translation operators: one per symmetry class for the cube method
    # (built at the width of level 3 for a homogeneous kernel and scaled, else per
    # level), one per far offset for circles, which have no cube lattice
    l3 = min(3, L)
    a = homogeneity(K)
    M2Lc = Vector{Vector{typeof(to(zeros(TO, 1, 1)))}}(undef, L)
    M2Lb = [typeof(to(zeros(TO, 1, 1)))[] for _ in 1:L]
    m2l_scale = ones(L)
    ranks = Int[]
    if m isa CubeKernelIndependent
        e3 = f.ops[l3].extra
        build(e, w) = tol > 0 ? (r = class_factors(K, e.small, e.embed, w, e.p, tol); (r[1], r[2], r[3], r[4])) :
                                (r = class_operators(K, e.small, e.embed, w, e.p); (r[1], r[2], Matrix{TO}[], Int[]))
        sc, opsc, btc, ranks = build(e3, f.ops[l3].width)
        dc = [to(Matrix{TO}(A)) for A in opsc]
        db = [to(Matrix{TO}(B)) for B in btc]
        for l in 1:L
            if a === nothing && l >= 3 && l != l3
                _, ol, bl, _ = build(f.ops[l].extra, f.ops[l].width)
                M2Lc[l] = [to(Matrix{TO}(A)) for A in ol]
                M2Lb[l] = [to(Matrix{TO}(B)) for B in bl]
            else
                M2Lc[l] = dc
                M2Lb[l] = db
                m2l_scale[l] = a === nothing ? 1.0 : (f.ops[l].width / f.ops[l3].width)^a
            end
        end
    elseif m isa HarmonicMethod                 # the 34 class representatives, shared by all levels (homogeneous)
        e3 = f.ops[l3].extra
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
            m2l_scale[l] = Float64(f.ops[l].m2l_scale)
        end
    else
        offs = far_offsets(Val(D))
        nfo = length(offs)
        sc = SymmetryClasses{D}(offs, collect(1:nfo), [collect(1:Nm) for _ in 1:nfo], [ones(Int8, Nm) for _ in 1:nfo])
        dc = [to(Matrix{TO}(f.ops[l3].M2L[fi].A)) for fi in 1:nfo]
        for l in 1:L
            if a === nothing && l >= 3 && l != l3
                M2Lc[l] = [to(Matrix{TO}(f.ops[l].M2L[fi].A)) for fi in 1:nfo]
            else
                M2Lc[l] = dc                                     # shared reference matrices
                m2l_scale[l] = Float64(f.ops[l].m2l_scale)
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
    # the pairs by class, grouped by target, in chunks
    nclass = length(sc.reps)
    chunk = 4096
    VI = typeof(to(Int32[]))
    chunks = Vector{Vector{ClassChunk{VI}}}(undef, L)
    for l in 1:L
        il = tree.ilists[l]
        pairs = [(il.tgt[q], il.src[q], fi) for fi in eachindex(sc.class) for q in pair_range(il, fi)]
        chunks[l] = ClassChunk{VI}[]
        for c in 1:nclass
            pc = sort([p for p in pairs if sc.class[p[3]] == c]; by=first)
            for k in 1:chunk:length(pc)
                pk = pc[k:min(k + chunk - 1, end)]
                np = length(pk)
                src = fill(Int32(1), chunk); fs = fill(Int32(1), chunk)
                src[1:np] = [p[2] for p in pk]; fs[1:np] = [p[3] for p in pk]
                tg = Int32[]; ptr = Int32[1]
                for (q, p) in enumerate(pk)
                    (q == 1 || p[1] != pk[q-1][1]) && push!(tg, Int32(p[1] - lo[l] + 1))
                    (q > 1 && p[1] != pk[q-1][1]) && push!(ptr, Int32(q))
                end
                push!(ptr, Int32(np + 1))
                push!(chunks[l], ClassChunk{VI}(c, to(src), to(fs), to(tg), to(ptr)))
            end
        end
    end
    mk(r, c) = to(zeros(TO, r, c))
    # the pseudo-inverse factors (placeholders for the harmonic methods)
    pm(get) = pinv ? [to(Matrix{TO}(get(f.ops[l].extra))) for l in 1:L] : [mk(1, 1) for l in 1:L]
    pv(get) = pinv ? [to(Vector{R}(get(f.ops[l].extra))) for l in 1:L] : [to(zeros(R, 1)) for l in 1:L]
    rup(l) = pinv ? rank(f.ops[l].extra.P_up) : 1
    rdn(l) = pinv ? rank(f.ops[l].extra.P_down) : 1
    plan = DevicePlan{D,typeof(dev),typeof(to(zeros(TO, 1, 1))),typeof(to(zeros(R, 1))),VI,typeof(to(idx)),typeof(to(sgn)),typeof(offs)}(
        dev, L, lo, nl, Nm, pml, pll, Nl,
        [to(Int32.(f.s2m_nodes[l])) for l in 1:L],
        [to(Int32.(tree.group_parent[l])) for l in 1:L],
        [to(Matrix{TO}(f.ops[l].M2M)) for l in 1:L], [Float64(f.ops[l].m2m_scale) for l in 1:L],
        [to(Matrix{TO}(f.ops[l].L2L)) for l in 1:L], [Float64(f.ops[l].l2l_scale) for l in 1:L],
        pm(e -> e.P_up.Uh), pv(e -> e.P_up.invS), pm(e -> e.P_up.V),
        pm(e -> e.P_down.Uh), pv(e -> e.P_down.invS), pm(e -> e.P_down.V),
        pinv ? [Float64(f.ops[l].extra.pinv_scale) for l in 1:L] : ones(L),
        M2Lc, m2l_scale, M2Lb, [mk(max(1, k), nb * chunk) for k in (isempty(ranks) ? [1] : ranks)],
        to(idx), to(inv), to(sgn), to(sgn_t), nb, offs, pinv, chunks, chunk,
        mk(Nm, nnodes(tree)), mk(pll, nnodes(tree)), mk(Nl, nnodes(tree)),
        [mk(pml, max(1, length(f.s2m_nodes[l]))) for l in 1:L],
        [mk(rup(l), max(1, length(f.s2m_nodes[l]))) for l in 1:L],
        [mk(Nm, max(1, length(f.s2m_nodes[l]))) for l in 1:L],
        [mk(nc * Nm, l < L ? nl[l+1] ÷ nc : 1) for l in 1:L], [mk(Nm, l < L ? nl[l+1] ÷ nc : 1) for l in 1:L],
        [mk(pll, max(1, nl[l] ÷ nc)) for l in 1:L], [mk(nc * pll, max(1, nl[l] ÷ nc)) for l in 1:L],
        [mk(pll, nl[l]) for l in 1:L], [mk(rdn(l), nl[l]) for l in 1:L], [mk(Nl, nl[l]) for l in 1:L],
        mk(Ns, nb * chunk), mk(Ns, nb * chunk))
    return plan
end

# multipoles of the childless nodes of the harmonic methods: one work-item per node
# and order m, the order's chain of harmonics.jl on the node's sources (the chains
# of different orders are independent and write disjoint entries)
@kernel function _s2m_harmonic_kernel!(M, @Const(nodes), @Const(centers), @Const(level), far, @Const(plo), @Const(phi),
                                       @Const(sources), @Const(τ), ::Val{nb}) where {nb}
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
        for j in Int(plo[node]):Int(phi[node])
            _device_accumulate!(col, (sources[j] - c) * s, p, far.tb, τ[j], s, m, Val(nb))
        end
    end
end
@inline _device_accumulate!(col, ŷ, p, tb, τ, s, m, ::Val{1}) = _accumulate_order!(col, ŷ, p, tb, _scaled_density(τ, s), m)
@inline _device_accumulate!(col, ŷ, p, tb, τ, s, m, ::Val{4}) = _stokes_accumulate_order!(col, (p + 1)^2, ŷ, p, tb, τ, s, m)

# ---- kernels ----

# check data of the childless nodes: one work-item per (node, check point)
@kernel function _s2m_kernel!(pM, @Const(nodes), @Const(centers), @Const(ysoa), lvl, @Const(plo), @Const(phi),
                              @Const(sources), @Const(τ), K, ::Val{D}, ::Val{nc}) where {D,nc}
    g = @index(Global, Linear)
    @inbounds begin
        N = size(ysoa, 1)
        q = (g - 1) ÷ N + 1
        k = (g - 1) % N + 1
        node = Int(nodes[q])
        c = centers[node]
        y = SVector(ntuple(d -> ysoa[k, d, lvl], Val(D)))
        acc = _field_zero(eltype(pM), Val(nc))
        for j in Int(plo[node]):Int(phi[node])
            acc += source_field(K, sources[j] - c, y, τ[j])
        end
        _store!(view(pM, :, q), k, N, acc)
    end
end
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
@kernel function _class_gather!(Gc, @Const(M), @Const(src), @Const(fs), @Const(idx), @Const(sgn), ::Val{nb}) where {nb}
    g = @index(Global, Linear)
    @inbounds begin
        n = size(Gc, 1)
        col = (g - 1) ÷ n + 1
        i = (g - 1) % n + 1
        q = (col - 1) ÷ nb + 1
        a = (col - 1) % nb + 1
        f = Int(fs[q])
        Gc[i, col] = sgn[i, f] * M[(a - 1) * n + Int(idx[i, f]), Int(src[q])]
    end
end
# scatter P_t·y_q into the targets, one work-item per (target of the class, block a,
# entry j): (P_t y)[j] = sgn_t[inv[j]] · y[inv[j]]; the fourth block of the Stokes
# method also takes the coupling o·Φ of the pair's offset
@kernel function _class_scatter!(pL, @Const(Yc), @Const(tgt), @Const(ptr), @Const(fs), @Const(inv), @Const(sgn), @Const(offs), α,
                                 ::Val{nb}) where {nb}
    g = @index(Global, Linear)
    @inbounds begin
        n = size(Yc, 1)
        t = (g - 1) ÷ (n * nb) + 1
        r = (g - 1) % (n * nb)
        a = r ÷ n + 1
        j = r % n + 1
        acc = zero(eltype(pL))
        for q in Int(ptr[t]):Int(ptr[t+1]) - 1
            f = Int(fs[q])
            i = Int(inv[j, f])
            col = (q - 1) * nb + a
            v = Yc[i, col]
            if nb == 4 && a == 4
                o = offs[f]
                v += o[1] * Yc[i, col-3] + o[2] * Yc[i, col-2] + o[3] * Yc[i, col-1]
            end
            acc += sgn[i, f] * v
        end
        pL[(a - 1) * n + j, Int(tgt[t])] += α * acc
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
    device_build_expansions!(plan::DevicePlan, ds)

The upward and downward passes on the device from the device state `ds`
(sources and densities): fills `plan.multipoles`, `plan.plocals` and
`plan.locals`, the last of which the evaluation kernel reads.
"""
function device_build_expansions!(plan::DevicePlan{D}, ds) where {D}
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
                _s2m_kernel!(backend, 256)(pM, plan.s2m_nodes[l], dev.centers, ysoa, l, dev.plo, dev.phi,
                                           ds.sources, ds.τ, K, Val(D), Val(nc); ndrange=nq * size(ysoa, 1))
                _pinv_apply!(plan.Mup[l], plan.Uh_up[l], plan.invS_up[l], plan.V_up[l], pM, plan.tup[l], plan.pinv_scale[l])
            else
                _s2m_harmonic_kernel!(backend, 256)(plan.Mup[l], plan.s2m_nodes[l], dev.centers, dev.level, dev.far, dev.plo, dev.phi,
                                                    ds.sources, ds.τ, Val(plan.nb); ndrange=nq * (dev.far.p + 1))
            end
            _scatter_cols!(backend, 256)(Mp, plan.Mup[l], plan.s2m_nodes[l], one(eltype(Mp)); ndrange=Nm * nq)
        end
        if l < L
            ng = plan.nl[l+1] ÷ nch
            B = plan.B[l]
            copyto!(vec(B), 1, vec(Mp), (plan.lo[l+1] - 1) * Nm + 1, nch * Nm * ng)
            mul!(plan.tM[l], plan.M2M[l], B)
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
            mul!(plan.Y[l], plan.L2L[l-1], G)
            plan.l2l_scale[l-1] == 1 || (plan.Y[l] .*= eltype(pL)(plan.l2l_scale[l-1]))
            _scatter_children!(backend, 256)(pL, plan.Y[l], pll, nch; ndrange=nch * pll * ng)
        else
            fill!(pL, zero(eltype(pL)))
        end
        Ns = Nm ÷ plan.nb
        for ch in plan.chunks[l]
            Gc, Yc = plan.Gc, plan.Yc
            _class_gather!(backend, 256)(Gc, Mp, ch.src, ch.f, plan.idx, plan.sgn, Val(plan.nb); ndrange=Ns * plan.nb * plan.chunk)
            if isempty(plan.M2Lb[l])
                mul!(Yc, plan.M2Lc[l][ch.class], Gc)
            else                                    # low-rank: A·(Bt·Gc)
                Z = plan.Zc[ch.class]
                mul!(Z, plan.M2Lb[l][ch.class], Gc)
                mul!(Yc, plan.M2Lc[l][ch.class], Z)
            end
            nt = length(ch.tgt)
            _class_scatter!(backend, 256)(pL, Yc, ch.tgt, ch.ptr, ch.f, plan.inv, plan.sgn_t, plan.offs, eltype(pL)(plan.m2l_scale[l]),
                                          Val(plan.nb); ndrange=pll * nt)
        end
        copyto!(vec(Pl), (plan.lo[l] - 1) * pll + 1, vec(pL), 1, pll * nl)
        if plan.pinv
            _pinv_apply!(plan.Ldn[l], plan.Uh_dn[l], plan.invS_dn[l], plan.V_dn[l], pL, plan.tdn[l], plan.pinv_scale[l])
            copyto!(vec(Lc), (plan.lo[l] - 1) * plan.nloc + 1, vec(plan.Ldn[l]), 1, plan.nloc * nl)
        else                                    # the locals are the partial locals
            copyto!(vec(Lc), (plan.lo[l] - 1) * plan.nloc + 1, vec(pL), 1, plan.nloc * nl)
        end
    end
    synchronize(backend)
    return plan
end
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
    device_matvec!(out, plan::DevicePlan, ds; output=Potential())

Expansions on the device and the self evaluation at the sources, `out` in the
original order.
"""
function device_matvec!(out::AbstractVector, plan::DevicePlan, ds; output=Potential())
    device_build_expansions!(plan, ds)
    return evaluate_at_sources!(out, plan.dev, (locals=plan.locals, sources=ds.sources, τ=ds.τ); output=output)
end
