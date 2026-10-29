# The ghost-cell tree built on the device from the points there
# (docs/notes/gpu_everything_plan.md §3): Morton keys by the builder's own child-slot
# rule, the backend's `sortperm!`, then the host builder's phases as kernels over
# nodes with the level loop on the host (one small readback per level per pass):
# refinement with the point ranges as prefix runs of the sorted keys (two binary
# searches per child, no point ever moves again), colleagues by the parent rule,
# the 2:1 restriction and the ghosts as flag, compaction and split passes, and the
# flattening. The structural arrays come out identical to the host `Tree`'s (the
# node order per level is the host's: refinement children in parent order,
# restriction children appended per round, ghosts last); only the order of the
# points within a leaf differs (the full key here, the input order there).

"""
    DeviceTree{D,T,…}

The flat tree on the device: the fields of `Tree` as `Int32`/`UInt8` device arrays
(`points` as `plo`/`phi`), plus the sorted points. Built by `device_tree` from
points on the device, or uploaded from a host `Tree` with `DeviceTree(tree, to)`
(without the points). `Tree(dt)` downloads one (and builds the interaction lists).
"""
struct DeviceTree{D,T,VP,VC,VI,VB,AI}
    origin::SVector{D,T}
    W0::T
    level_range::Vector{UnitRange{Int}}
    cell::VC
    level::VI
    parent::VI
    child_first::VI
    plo::VI
    phi::VI
    leaf::VB
    ghost::VB
    xlist::VB
    has_source::VB
    colleagues::AI
    group_parent::Vector{VI}
    perm::VI
    point_node::VI
    sources::VP
    # groups (`docs/tree.md`): per group and node the point range in the group's own
    # numbering (`gplo`/`gphi`, G × nnodes), the original index within the group of each
    # group-local sorted point, and its leaf
    group_sizes::Vector{Int}
    gplo::AI
    gphi::AI
    gperm::Vector{VI}
    gpoint_node::Vector{VI}
end
nnodes(dt::DeviceTree) = length(dt.level)
nlevels(dt::DeviceTree) = length(dt.level_range)
npoints(dt::DeviceTree) = length(dt.perm)
ngroups(dt::DeviceTree) = length(dt.group_sizes)

function DeviceTree(tree::Tree{D,T}, to) where {D,T}
    plo = to(Int32[first(r) for r in tree.points])
    G = ngroups(tree)
    gplo = to(Int32[first(tree.gpoints[g][i]) for g in 1:G, i in eachindex(tree.points)])
    gphi = to(Int32[last(tree.gpoints[g][i]) for g in 1:G, i in eachindex(tree.points)])
    return DeviceTree{D,T,typeof(to(SVector{D,T}[])),typeof(to(SVector{D,Int32}[])),typeof(plo),typeof(to(UInt8[])),typeof(gplo)}(
        tree.origin, tree.W0, tree.level_range, to(SVector{D,Int32}.(tree.cell)), to(Int32.(tree.level)),
        to(Int32.(tree.parent)), to(Int32.(tree.child_first)), plo, to(Int32[last(r) for r in tree.points]),
        to(UInt8.(tree.leaf)), to(UInt8.(tree.ghost)), to(UInt8.(tree.xlist)), to(UInt8.(tree.has_source)),
        to(Int32.(tree.colleagues)), [to(Int32.(gp)) for gp in tree.group_parent], to(Int32.(tree.perm)),
        to(Int32.(tree.point_node)), to(SVector{D,T}[]),
        copy(tree.group_sizes), gplo, gphi, [to(Int32.(gp)) for gp in tree.gperm], [to(Int32.(gn)) for gn in tree.gpoint_node])
end

# the group tables from the sorted order: per group the flags of its points in sorted
# order, their exclusive scan (the group-local index minus one), the node ranges from
# the scan at the range ends, and the group's permutation and leaves by compaction
@kernel function _group_flags!(flag, @Const(perm), lo::Int32, hi::Int32)
    j = @index(Global, Linear)
    @inbounds flag[j] = Int32(lo < perm[j] <= hi)
end
@kernel function _group_ranges!(gplo, gphi, @Const(scan), @Const(flag), @Const(plo), @Const(phi), g::Int32)
    i = @index(Global, Linear)
    @inbounds begin
        lo, hi = plo[i], phi[i]
        if hi < lo
            gplo[g, i] = Int32(1); gphi[g, i] = Int32(0)
        else
            gplo[g, i] = scan[lo] + Int32(1)
            gphi[g, i] = scan[hi] + flag[hi]
        end
    end
end
function _device_group_tables(perm, plo, phi, point_node, sizes::Vector{Int})
    backend = get_backend(perm)
    N = length(perm)
    n = length(plo)
    G = length(sizes)
    starts = cumsum(vcat(0, sizes))
    gplo = similar(plo, G, n)
    gphi = similar(plo, G, n)
    gperm = Vector{typeof(perm)}(undef, G)
    gpoint_node = Vector{typeof(perm)}(undef, G)
    flag = similar(perm)
    scan = similar(perm)
    for g in 1:G
        _group_flags!(backend, 256)(flag, perm, Int32(starts[g]), Int32(starts[g+1]); ndrange=N)
        _exclusive_scan!(scan, flag)
        _group_ranges!(backend, 256)(gplo, gphi, scan, flag, plo, phi, Int32(g); ndrange=n)
        idx, cnt = _compact(flag)
        cnt == sizes[g] || error("group $g: $cnt points found of $(sizes[g])")
        # gathered by kernel: indexing a device array with a device index array checks
        # its bounds by a reduction that Metal compiles anew for every index length
        gp = similar(perm, cnt); gn = similar(perm, cnt)
        if cnt > 0
            _gather_by!(backend, 256)(gp, perm, idx; ndrange=cnt)
            _gather_by!(backend, 256)(gn, point_node, idx; ndrange=cnt)
        end
        gp .-= Int32(starts[g])
        gperm[g] = gp
        gpoint_node[g] = gn
    end
    return gplo, gphi, gperm, gpoint_node
end

# The componentwise bounds of points on a device: one strided pass per work-item over
# at most 16384 work-items, finished on the host. A reduction of the backend
# (`mapreduce`) would do, but Metal compiles its reduction kernels anew for every
# array length, about 0.5 s per new point count. NaN propagates through `min` and
# `max`, so non-finite coordinates show in the bounds.
@kernel function _bounds_strided!(lo, hi, @Const(xs), n::Int32, stride::Int32)
    t = @index(Global, Linear)
    @inbounds begin
        a = xs[t]
        b = a
        j = t + stride
        while j <= n
            y = xs[j]
            a = min.(a, y)
            b = max.(b, y)
            j += stride
        end
        lo[t] = a
        hi[t] = b
    end
end
function _device_bounds(xs::AbstractVector{SVector{D,T}}) where {D,T}
    n = length(xs)
    n > 0 || throw(ArgumentError("no points"))
    nt = min(n, 16384)
    lo = similar(xs, nt)
    hi = similar(xs, nt)
    _bounds_strided!(get_backend(xs), 256)(lo, hi, xs, Int32(n), Int32(nt); ndrange=nt)
    return reduce((a, b) -> min.(a, b), Array(lo)), reduce((a, b) -> max.(a, b), Array(hi))
end

# ---- the per-level builder on the device ----

mutable struct DeviceLevel{VC,VI,VB,AI}
    n::Int
    cell::VC
    parent::VI
    child_first::VI
    lo::VI
    hi::VI
    leaf::VB
    ghost::VB
    xlist::VB
    depth::VI
    colleagues::AI
end
function DeviceLevel(proto_i::VI, proto_c::VC, proto_b::VB, ::Val{D}) where {VI,VC,VB,D}
    return DeviceLevel{VC,VI,VB,typeof(similar(proto_i, 3^D, 0))}(0, similar(proto_c, 0), similar(proto_i, 0), similar(proto_i, 0),
        similar(proto_i, 0), similar(proto_i, 0), similar(proto_b, 0), similar(proto_b, 0), similar(proto_b, 0),
        similar(proto_i, 0), similar(proto_i, 3^D, 0))
end
# room for `nnew` more nodes (the arrays copied; the colleague table is rebuilt by the tagging)
function _grow!(lev::DeviceLevel, nnew::Int)
    n = lev.n + nnew
    grow(a) = (b = similar(a, n); lev.n > 0 && copyto!(b, 1, a, 1, lev.n); b)
    lev.cell = grow(lev.cell); lev.parent = grow(lev.parent); lev.child_first = grow(lev.child_first)
    lev.lo = grow(lev.lo); lev.hi = grow(lev.hi); lev.leaf = grow(lev.leaf); lev.ghost = grow(lev.ghost)
    lev.xlist = grow(lev.xlist); lev.depth = grow(lev.depth)
    lev.n = n
    return lev
end

# the Morton key of a point: its child slots on levels 2..KL+1 by the exact
# comparison against the box centers (the rule of `child_slot` and `locate`), the
# slot of level b in the bits D·(KL − b) .. D·(KL − b) + D − 1
@kernel function _point_keys!(keys, @Const(xs), origin, W0, coffs, ::Val{D}, ::Val{KL}) where {D,KL}
    j = @index(Global, Linear)
    @inbounds begin
        x = xs[j]
        cell = zero(SVector{D,Int})
        w = W0
        key = UInt64(0)
        for b in 1:KL
            k = child_slot(x, _center(origin, cell, w))
            key |= UInt64(k - 1) << (D * (KL - b))
            cell = 2 * cell + coffs[k]
            w *= typeof(w)(0.5)
        end
        keys[j] = key
    end
end
@kernel function _gather_by!(dst, @Const(src), @Const(perm))
    j = @index(Global, Linear)
    @inbounds dst[j] = src[Int(perm[j])]
end
@kernel function _flag_split!(flag, @Const(lo), @Const(hi), ppl::Int32)
    i = @index(Global, Linear)
    @inbounds flag[i] = Int32(hi[i] - lo[i] + Int32(1) > ppl)
end
@kernel function _compact_fill!(idx, @Const(flag), @Const(scan))
    i = @index(Global, Linear)
    @inbounds flag[i] != 0 && (idx[scan[i] + Int32(1)] = Int32(i))
end
# the indices of the set flags, in order, and their number
function _compact(flag)
    n = length(flag)
    scan = similar(flag)
    _exclusive_scan!(scan, flag)
    cnt = n == 0 ? 0 : _last_entry(scan) + _last_entry(flag)
    idx = similar(flag, cnt)
    cnt > 0 && _compact_fill!(get_backend(flag), 256)(idx, flag, scan; ndrange=n)
    return idx, cnt
end
# first j in lo:hi with keys[j] >= x, hi + 1 if none
@inline function _lower_bound(keys, lo::Int32, hi::Int32, x::UInt64)
    @inbounds while lo <= hi
        mid = (lo + hi) >> 1
        if keys[mid] < x
            lo = mid + Int32(1)
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end
# the children of the split nodes `which` (level l), appended from node n0 + 1 of the
# next level: one work-item per child; its point range is the run of its key prefix
# inside the parent's range, found by two binary searches in the sorted keys
@kernel function _children!(ccell, cparent, ccf, clo, chi, cleaf, cghost, cxlist, cdepth, pcf, pleaf,
                            @Const(which), @Const(pcell), @Const(plo), @Const(phi), @Const(keys), coffs, n0::Int32, l::Int32,
                            ghost::UInt8, ::Val{D}, ::Val{KL}) where {D,KL}
    g = @index(Global, Linear)
    @inbounds begin
        nc = 1 << D
        q = (g - 1) ÷ nc + 1
        k = (g - 1) % nc + 1
        i = Int(which[q])
        cc = 2 * pcell[i] + SVector{D,Int32}(coffs[k])
        pk = morton_key(SVector{D,Int}(cc), Int(l))                  # the child's prefix: l slots
        shift = D * (KL - Int(l))
        lo, hi = plo[i], phi[i]
        first = _lower_bound(keys, lo, hi, pk << shift)
        last = _lower_bound(keys, first, hi, (pk + UInt64(1)) << shift) - Int32(1)
        c = Int(n0) + nc * (q - 1) + k
        ccell[c] = cc
        cparent[c] = Int32(i)
        ccf[c] = Int32(0)
        clo[c] = first
        chi[c] = last
        cleaf[c] = ghost == 0 ? UInt8(1) : UInt8(0)
        cghost[c] = ghost
        cxlist[c] = UInt8(0)
        cdepth[c] = Int32(0)
        if k == 1
            pcf[i] = Int32(n0 + nc * (q - 1) + 1)
            ghost == 0 && (pleaf[i] = UInt8(0))
        end
    end
end
function _split!(par::DeviceLevel, which, nw::Int, l::Int, chi::DeviceLevel, keys, ::Val{D}, ::Val{KL}; ghost::Bool) where {D,KL}
    nw == 0 && return nothing
    nc = 1 << D
    n0 = chi.n
    _grow!(chi, nc * nw)
    _children!(get_backend(keys), 256)(chi.cell, chi.parent, chi.child_first, chi.lo, chi.hi, chi.leaf, chi.ghost, chi.xlist,
                                       chi.depth, par.child_first, par.leaf, which, par.cell, par.lo, par.hi, keys,
                                       child_offsets(Val(D)), Int32(n0), Int32(l), UInt8(ghost), Val(D), Val(KL); ndrange=nc * nw)
    return nothing
end
# colleagues by the parent rule (builder.jl `tag_colleagues!`), one work-item per node
@kernel function _tag!(col, @Const(cell), @Const(parent), @Const(pcol), @Const(pchild), noffs, ::Val{D}) where {D}
    i = @index(Global, Linear)
    @inbounds begin
        c = cell[i]
        p = Int(parent[i])
        pc = c .>> 1
        for (k, o) in enumerate(noffs)
            cc = c + SVector{D,Int32}(o)
            q = Int(pcol[_offset_index(SVector{D,Int}((cc .>> 1) - pc)), p])
            col[k, i] = (q == 0 || pchild[q] == 0) ? Int32(0) : pchild[q] + Int32(_slot_bits(SVector{D,Int}(cc)) - 1)
        end
    end
end
function _tag!(lev::DeviceLevel, par::DeviceLevel, ::Val{D}) where {D}
    size(lev.colleagues, 2) == lev.n || (lev.colleagues = similar(lev.colleagues, 3^D, lev.n))
    lev.n > 0 && _tag!(get_backend(lev.cell), 256)(lev.colleagues, lev.cell, lev.parent, par.colleagues, par.child_first,
                                                   colleague_offsets(Val(D)), Val(D); ndrange=lev.n)
    return lev
end
function _tag_root!(root::DeviceLevel, ::Val{D}) where {D}
    root.colleagues = similar(root.colleagues, 3^D, 1)
    fill!(root.colleagues, Int32(0))
    _setindex_device!(vec(root.colleagues), self_colleague_slot(Val(D)), 1)
    return root
end
@kernel function _depth!(depth, @Const(cf), @Const(ndepth), ::Val{nc}) where {nc}
    i = @index(Global, Linear)
    @inbounds begin
        f = Int(cf[i])
        if f == 0
            depth[i] = Int32(0)
        else
            m = Int32(0)
            for k in 0:nc-1
                m = max(m, ndepth[f+k])
            end
            depth[i] = m + Int32(1)
        end
    end
end
function _depths!(levels::Vector{<:DeviceLevel}, ::Val{D}) where {D}
    L = length(levels)
    fill!(levels[L].depth, Int32(0))
    for l in L-1:-1:1
        _depth!(get_backend(levels[l].depth), 256)(levels[l].depth, levels[l].child_first, levels[l+1].depth, Val(1 << D); ndrange=levels[l].n)
    end
    return levels
end
# a leaf is bad if a colleague's depth differs from its own by more than one
@kernel function _flag_bad!(flag, @Const(leaf), @Const(depth), @Const(col))
    i = @index(Global, Linear)
    @inbounds begin
        b = Int32(0)
        if leaf[i] != 0
            di = depth[i]
            for k in 1:size(col, 1)
                j = Int(col[k, i])
                if j != 0 && abs(di - depth[j]) > 1
                    b = Int32(1)
                    break
                end
            end
        end
        flag[i] = b
    end
end
# a real leaf with a colleague that has children is an xlist leaf
@kernel function _flag_xlist!(xlist, @Const(leaf), @Const(cf), @Const(col))
    i = @index(Global, Linear)
    @inbounds begin
        b = UInt8(0)
        if leaf[i] != 0
            for k in 1:size(col, 1)
                j = Int(col[k, i])
                if j != 0 && cf[j] != 0
                    b = UInt8(1)
                    break
                end
            end
        end
        xlist[i] = b
    end
end

# ---- flattening ----
@kernel function _flatten!(cell, level, parent, child_first, plo, phi, leaf, ghost, xlist, has_source, col, point_node,
                           @Const(lcell), @Const(lparent), @Const(lcf), @Const(llo), @Const(lhi), @Const(lleaf), @Const(lghost),
                           @Const(lxlist), @Const(lcol), l::Int32, off::Int32, poff::Int32, noff::Int32)
    i = @index(Global, Linear)
    @inbounds begin
        g = off + Int32(i)
        cell[g] = lcell[i]
        level[g] = l
        parent[g] = l == 1 ? Int32(0) : poff + lparent[i]
        child_first[g] = lcf[i] == 0 ? Int32(0) : noff + lcf[i]
        plo[g] = llo[i]
        phi[g] = lhi[i]
        leaf[g] = lleaf[i]
        ghost[g] = lghost[i]
        xlist[g] = lxlist[i]
        has_source[g] = UInt8(lhi[i] >= llo[i])
        for k in 1:size(col, 1)
            j = lcol[k, i]
            col[k, g] = j == 0 ? Int32(0) : off + j
        end
        if lleaf[i] != 0
            for j in llo[i]:lhi[i]
                point_node[j] = g
            end
        end
    end
end

"""
    device_tree(xs; ppl=50, bbox, max_levels=30) -> DeviceTree

The tree of the points `xs` (a device vector of `SVector`s; `bbox = (origin, W0)`
is required), built on their device: the same structure as `Tree(Array(xs); …)`.
"""
function device_tree(xs::AbstractVector{SVector{D,T}}; ppl::Integer=50, bbox, max_levels::Integer=30,
                     group_sizes::Vector{Int}=[length(xs)]) where {D,T<:AbstractFloat}
    ppl >= 1 || throw(ArgumentError("ppl must be positive"))
    N = length(xs)
    N > 0 || throw(ArgumentError("no points"))
    sum(group_sizes) == N || throw(DimensionMismatch("the groups do not add up to the points"))
    origin, W0 = SVector{D,T}(bbox[1]), T(bbox[2])
    W0 > 0 || throw(ArgumentError("box width must be positive"))
    backend = get_backend(xs)
    KL = min(Int(max_levels) - 1, 63 ÷ D)                    # key levels: D bits each
    maxl = KL + 1
    # keys, the permutation, the sorted keys and points
    keys = similar(xs, UInt64, N)
    _point_keys!(backend, 256)(keys, xs, origin, W0, child_offsets(Val(D)), Val(D), Val(KL); ndrange=N)
    perm = similar(xs, Int32, N)
    sortperm!(perm, keys)
    skeys = similar(keys)
    _gather_by!(backend, 256)(skeys, keys, perm; ndrange=N)
    sources = similar(xs)
    _gather_by!(backend, 256)(sources, xs, perm; ndrange=N)
    # the level builders, from the root
    proto_i, proto_c, proto_b = perm, similar(xs, SVector{D,Int32}, 0), similar(xs, UInt8, 0)
    root = DeviceLevel(proto_i, proto_c, proto_b, Val(D))
    _grow!(root, 1)
    copyto!(root.cell, [zero(SVector{D,Int32})]); copyto!(root.parent, Int32[0]); copyto!(root.child_first, Int32[0])
    copyto!(root.lo, Int32[1]); copyto!(root.hi, Int32[N]); copyto!(root.leaf, UInt8[1]); copyto!(root.ghost, UInt8[0])
    copyto!(root.xlist, UInt8[0]); copyto!(root.depth, Int32[0])
    levels = [root]
    flag = similar(perm, 0)
    while true
        lev = levels[end]
        flag = similar(perm, lev.n)
        _flag_split!(backend, 256)(flag, lev.lo, lev.hi, Int32(ppl); ndrange=lev.n)
        which, nsplit = _compact(flag)
        nsplit == 0 && break
        length(levels) >= maxl && throw(ArgumentError(
            "tree reached max_levels = $maxl; more than ppl = $ppl coincident points?"))
        nxt = DeviceLevel(proto_i, proto_c, proto_b, Val(D))
        _split!(lev, which, nsplit, length(levels), nxt, skeys, Val(D), Val(KL); ghost=false)
        push!(levels, nxt)
    end
    L = length(levels)
    _tag_root!(levels[1], Val(D))
    for l in 2:L
        _tag!(levels[l], levels[l-1], Val(D))
    end
    # 2:1 restriction
    _depths!(levels, Val(D))
    while true
        nsplit = 0
        for l in 2:L-2
            lev = levels[l]
            flag = similar(perm, lev.n)
            _flag_bad!(backend, 256)(flag, lev.leaf, lev.depth, lev.colleagues; ndrange=lev.n)
            which, nb = _compact(flag)
            nb == 0 && continue
            _split!(lev, which, nb, l, levels[l+1], skeys, Val(D), Val(KL); ghost=false)
            _tag!(levels[l+1], lev, Val(D))
            _depths!(levels, Val(D))
            nsplit += nb
        end
        nsplit == 0 && break
    end
    # ghosts: the xlist flags of all levels first, then the splits
    for l in 2:L-1
        lev = levels[l]
        _flag_xlist!(backend, 256)(lev.xlist, lev.leaf, lev.child_first, lev.colleagues; ndrange=lev.n)
    end
    for l in 2:L-1
        lev = levels[l]
        which, nx = _compact(map(x -> Int32(x), lev.xlist))
        nx == 0 && continue
        _split!(lev, which, nx, l, levels[l+1], skeys, Val(D), Val(KL); ghost=true)
        _tag!(levels[l+1], lev, Val(D))
    end
    return _flatten(levels, origin, W0, perm, sources, group_sizes, Val(D))
end

function _flatten(levels::Vector{<:DeviceLevel}, origin::SVector{D,T}, W0::T, perm, sources, group_sizes::Vector{Int}, ::Val{D}) where {D,T}
    L = length(levels)
    counts = [lev.n for lev in levels]
    offs = cumsum(vcat(0, counts[1:end-1]))
    level_range = [offs[l]+1:offs[l]+counts[l] for l in 1:L]
    n = sum(counts)
    nc = 1 << D
    backend = get_backend(perm)
    cell = similar(levels[1].cell, n)
    level = similar(perm, n); parent = similar(perm, n); child_first = similar(perm, n)
    plo = similar(perm, n); phi = similar(perm, n)
    leaf = similar(levels[1].leaf, n); ghost = similar(leaf); xlist = similar(leaf); has_source = similar(leaf)
    col = similar(perm, 3^D, n)
    point_node = similar(perm, length(perm))
    for l in 1:L
        lev = levels[l]
        _flatten!(backend, 256)(cell, level, parent, child_first, plo, phi, leaf, ghost, xlist, has_source, col, point_node,
                                lev.cell, lev.parent, lev.child_first, lev.lo, lev.hi, lev.leaf, lev.ghost, lev.xlist, lev.colleagues,
                                Int32(l), Int32(offs[l]), Int32(l == 1 ? 0 : offs[l-1]), Int32(l == L ? 0 : offs[l+1]); ndrange=lev.n)
    end
    group_parent = [l == 1 ? similar(perm, 0) : parent[first(level_range[l]):nc:last(level_range[l])] for l in 1:L]
    gplo, gphi, gperm, gpoint_node = _device_group_tables(perm, plo, phi, point_node, group_sizes)
    return DeviceTree{D,T,typeof(sources),typeof(cell),typeof(perm),typeof(leaf),typeof(col)}(
        origin, W0, level_range, cell, level, parent, child_first, plo, phi, leaf, ghost, xlist, has_source, col,
        group_parent, perm, point_node, sources, copy(group_sizes), gplo, gphi, gperm, gpoint_node)
end

"""
    Tree(dt::DeviceTree) -> Tree

The host tree with the device tree's structure (downloaded), with the interaction lists.
"""
function Tree(dt::DeviceTree{D,T}) where {D,T}
    cell = SVector{D,Int}.(Array(dt.cell))
    plo, phi = Int.(Array(dt.plo)), Int.(Array(dt.phi))
    child_first = Int.(Array(dt.child_first))
    parent = Int.(Array(dt.parent))
    ghost = Bool.(Array(dt.ghost)); has_source = Bool.(Array(dt.has_source))
    colleagues = Int.(Array(dt.colleagues))
    far_index = far_index_table(Val(D))
    ilists = build_interaction_lists(Val(D), dt.level_range, cell, parent, child_first, colleagues, ghost, has_source, far_index)
    points = [plo[i]:phi[i] for i in eachindex(plo)]
    perm = Int.(Array(dt.perm))
    point_node = Int.(Array(dt.point_node))
    t = Tree{D,T}(dt.origin, dt.W0, dt.level_range, cell, Int.(Array(dt.level)), parent, child_first,
                  points, Bool.(Array(dt.leaf)), ghost, Bool.(Array(dt.xlist)), has_source,
                  colleagues, [Int.(Array(gp)) for gp in dt.group_parent], ilists, perm, point_node, far_index,
                  [length(perm)], [points], [perm], [point_node])
    return ngroups(dt) == 1 ? t : _with_groups(t, dt.group_sizes)
end
