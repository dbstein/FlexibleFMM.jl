# The interaction lists, the class lists and the chunk descriptors of a level, built
# on the device from the tree's flat arrays (docs/notes/gpu_everything_plan.md): one
# work-item per target counts its far pairs (the enumeration of src/tree/lists.jl),
# a prefix sum places them and a fill writes them; the pairs are counted per
# (target, class), placed class-major (the pairs of a class in target order, each
# target's consecutive: the class lists of operators/symmetry.jl) and filled; the
# runs (one per (target, class) with pairs) come from a compaction of the counts,
# and the chunks (at most `chunk` pairs of one class, padded to the power of two
# above their size) from the per-class totals, their run ranges by binary search in
# the run starts. The host reads back three counts and the chunk descriptors per
# level; the lists never leave the device. (The first version cut the chunks on the
# host from `class_lists` and uploaded four arrays per chunk: 0.97 of the 1.15 s
# plan build of 10⁷ points on the M3 Max.)

"""
    DeviceLevelLists{VI}

A level's class lists on the device: `src[q]`, `f[q]` (global source node, far
offset) of pair `q`, class-major with the targets of a class in node order and each
target's pairs consecutive; run `r` (one target of one class) is the level-local
target `rtgt[r]` with the pairs `rptr[r]:rptr[r+1]-1` (`rptr` ends with a sentinel).
"""
struct DeviceLevelLists{VI}
    src::VI
    f::VI
    rtgt::VI
    rptr::VI
end
DeviceLevelLists(v::VI) where {VI} = DeviceLevelLists{VI}(similar(v, 0), similar(v, 0), similar(v, 0), similar(v, 1))

"""
    ClassChunk

A chunk of a level's class lists: the pairs `a:a+npk-1` of class `class`, padded to
`len` columns in the gather buffer, whose targets are the runs `r1:r2`.
"""
struct ClassChunk
    class::Int
    a::Int
    npk::Int
    len::Int
    r1::Int
    r2::Int
end

@inline _is_far(o::SVector{D,Int32}) where {D} = maximum(abs, o) >= 2
@inline function _far_slot(o::SVector{D,Int32}) where {D}            # m2l_table_index on Int32 cells
    k = 1
    @inbounds for d in 1:D
        k += (Int(o[d]) + 3) * 7^(d - 1)
    end
    return k
end

# far pairs of the level-local target q (global node lo + q - 1; none for a ghost):
# the sourced children of the parent's colleagues at a far offset
@kernel function _pair_count!(cnt, @Const(cell), @Const(parent), @Const(cf), @Const(col), @Const(ghost), @Const(hs),
                              lo::Int32, ::Val{D}) where {D}
    q = @index(Global, Linear)
    @inbounds begin
        i = lo + Int32(q) - Int32(1)
        c = Int32(0)
        if ghost[i] == 0
            p = parent[i]
            ci = cell[i]
            for k in 1:size(col, 1)
                pc = col[k, p]
                pc == 0 && continue
                s0 = cf[pc]
                s0 == 0 && continue
                for s in s0:s0+Int32((1 << D) - 1)
                    hs[s] == 0 && continue
                    _is_far(cell[s] - ci) && (c += Int32(1))
                end
            end
        end
        cnt[q] = c
    end
end
@kernel function _pair_fill!(src, far, @Const(ptr), @Const(cell), @Const(parent), @Const(cf), @Const(col), @Const(ghost),
                             @Const(hs), @Const(far_index), lo::Int32, ::Val{D}) where {D}
    q = @index(Global, Linear)
    @inbounds begin
        i = lo + Int32(q) - Int32(1)
        pos = ptr[q]
        if ghost[i] == 0
            p = parent[i]
            ci = cell[i]
            for k in 1:size(col, 1)
                pc = col[k, p]
                pc == 0 && continue
                s0 = cf[pc]
                s0 == 0 && continue
                for s in s0:s0+Int32((1 << D) - 1)
                    hs[s] == 0 && continue
                    o = cell[s] - ci
                    if _is_far(o)
                        src[pos] = s
                        far[pos] = far_index[_far_slot(o)]
                        pos += Int32(1)
                    end
                end
            end
        end
    end
end
# per (target, class) counts of the target's pairs, and the class-major placement
@kernel function _class_count!(cnt, @Const(far), @Const(ptr), @Const(class), nclass::Int32)
    q = @index(Global, Linear)
    @inbounds begin
        for c in 1:nclass
            cnt[q, c] = Int32(0)
        end
        for p in ptr[q]:ptr[q+1]-Int32(1)
            cnt[q, class[far[p]]] += Int32(1)
        end
    end
end
@kernel function _class_fill!(csrc, cfar, cursor, @Const(src), @Const(far), @Const(ptr), @Const(class))
    q = @index(Global, Linear)
    @inbounds for p in ptr[q]:ptr[q+1]-Int32(1)
        c = class[far[p]]
        pos = cursor[q, c] + Int32(1)
        cursor[q, c] = pos
        csrc[pos] = src[p]
        cfar[pos] = far[p]
    end
end
# the runs: entry e of the class-major (target, class) table with pairs is run
# rscan[e] + 1 (exclusive scan of the flags), of target (e - 1) % nl + 1, starting at pair start[e] + 1
@kernel function _run_fill!(rtgt, rptr, @Const(cnt), @Const(rscan), @Const(start), nl::Int32)
    e = @index(Global, Linear)
    @inbounds if cnt[e] > 0
        r = rscan[e] + Int32(1)
        rtgt[r] = Int32((e - 1) % nl + 1)
        rptr[r] = start[e] + Int32(1)
    end
end
@inline function _searchsortedlast(v, x::Int32, n::Int32)           # last i in 1:n with v[i] <= x
    lo, hi = Int32(0), n
    @inbounds while lo < hi
        mid = (lo + hi + Int32(1)) >> 1
        if v[mid] <= x
            lo = mid
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end
@kernel function _chunk_runs!(r1, r2, @Const(ca), @Const(cb), @Const(rptr), nruns::Int32)
    k = @index(Global, Linear)
    @inbounds begin
        r1[k] = _searchsortedlast(rptr, ca[k], nruns)
        r2[k] = _searchsortedlast(rptr, cb[k], nruns)
    end
end
@kernel function _set_entry!(a, i::Int32, v::Int32)
    g = @index(Global, Linear)
    @inbounds g == 1 && (a[i] = v)
end
# dst[j] = src[perm[j]]: the inputs into the tree's order
@kernel function _gather_perm!(dst, @Const(src), @Const(perm))
    j = @index(Global, Linear)
    @inbounds dst[j] = src[Int(perm[j])]
end

# Exclusive prefix sums as KernelAbstractions kernels (blocks of `_SCAN_BLOCK`
# entries: block totals, the recursive scan of the totals, then each block scanned
# with its offset). The backends' own `accumulate!` would do, but Metal's goes
# through MPSGraph, which compiles a graph per array size: 0.2–1.2 s of every plan
# build, since every tree has new level sizes.
const _SCAN_BLOCK = 1024
@kernel function _block_sums!(bsum, @Const(x), B::Int32, n::Int32)
    b = @index(Global, Linear)
    @inbounds begin
        s = Int32(0)
        for i in (Int32(b) - Int32(1)) * B + Int32(1):min(Int32(b) * B, n)
            s += x[i]
        end
        bsum[b] = s
    end
end
@kernel function _block_scan!(out, @Const(x), @Const(boff), B::Int32, n::Int32)
    b = @index(Global, Linear)
    @inbounds begin
        s = boff[b]
        for i in (Int32(b) - Int32(1)) * B + Int32(1):min(Int32(b) * B, n)
            out[i] = s
            s += x[i]
        end
    end
end
function _exclusive_scan!(out, x)
    n = length(x)
    n == 0 && return out
    backend = get_backend(x)
    B = _SCAN_BLOCK
    nb = cld(n, B)
    bsum = similar(x, nb)
    _block_sums!(backend, 256)(bsum, x, Int32(B), Int32(n); ndrange=nb)
    boff = similar(x, nb)
    if nb == 1
        _setindex_device!(boff, 1, 0)
    else
        _exclusive_scan!(boff, bsum)
    end
    _block_scan!(backend, 256)(out, x, boff, Int32(B), Int32(n); ndrange=nb)
    return out
end
_last_entry(v) = Int(Array(v[end:end])[1])
_setindex_device!(v, i::Int, x::Int) = (_set_entry!(get_backend(v), 1)(v, Int32(i), Int32(x); ndrange=1); v)

"""
    _device_level_lists(dt, col, lo, nl, far_index, class, nclass, chunk, Val(D)) -> (lists, chunks)

The class lists and chunk descriptors of the level whose nodes are `lo:lo+nl-1`.
"""
function _device_level_lists(dt, col, lo::Int, nl::Int, far_index, class, nclass::Int, chunk::Int,
                             ::Val{D}) where {D}
    backend = get_backend(dt.parent)
    cnt = similar(dt.parent, nl)
    _pair_count!(backend, 256)(cnt, dt.cell, dt.parent, dt.child_first, col, dt.ghost, dt.has_source, Int32(lo), Val(D); ndrange=nl)
    ptr0 = similar(cnt)
    _exclusive_scan!(ptr0, cnt)
    np = _last_entry(ptr0) + _last_entry(cnt)
    ptr0 .+= Int32(1)
    ptr = similar(cnt, nl + 1)                                  # with the sentinel np + 1
    copyto!(ptr, 1, ptr0, 1, nl)
    _setindex_device!(ptr, nl + 1, np + 1)
    src = similar(cnt, np)
    far = similar(cnt, np)
    np > 0 && _pair_fill!(backend, 256)(src, far, ptr, dt.cell, dt.parent, dt.child_first, col, dt.ghost, dt.has_source,
                                        far_index, Int32(lo), Val(D); ndrange=nl)
    # the class lists
    ccnt = similar(cnt, nl, nclass)
    _class_count!(backend, 256)(ccnt, far, ptr, class, Int32(nclass); ndrange=nl)
    cstart = similar(ccnt)
    _exclusive_scan!(vec(cstart), vec(ccnt))
    cursor = copy(cstart)
    csrc = similar(cnt, np)
    cfar = similar(cnt, np)
    np > 0 && _class_fill!(backend, 256)(csrc, cfar, cursor, src, far, ptr, class; ndrange=nl)
    # the runs
    flag = map(x -> Int32(x > 0), ccnt)
    rscan = similar(flag)
    _exclusive_scan!(vec(rscan), vec(flag))
    nruns = _last_entry(vec(rscan)) + _last_entry(vec(flag))
    rtgt = similar(cnt, nruns)
    rptr = similar(cnt, nruns + 1)
    nruns > 0 && _run_fill!(backend, 256)(rtgt, rptr, vec(ccnt), vec(rscan), vec(cstart), Int32(nl); ndrange=nl * nclass)
    _setindex_device!(rptr, nruns + 1, np + 1)
    # the chunks: per class, pieces of at most `chunk` pairs (the class totals from
    # the class-major scan: its entries at the start of each class, and np)
    cfirst = Int.(Array(vec(cstart)[1:nl:nl*nclass]))
    tot = [(c < nclass ? cfirst[c+1] : np) - cfirst[c] for c in 1:nclass]
    ca = Int32[]; cb = Int32[]; desc = NTuple{4,Int}[]
    pos = 1
    for c in 1:nclass
        for a in pos:chunk:pos+tot[c]-1
            b = min(a + chunk - 1, pos + tot[c] - 1)
            npk = b - a + 1
            push!(ca, a); push!(cb, b); push!(desc, (c, a, npk, max(64, nextpow(2, npk))))
        end
        pos += tot[c]
    end
    nch = length(desc)
    chunks = ClassChunk[]
    if nch > 0
        dca = similar(cnt, nch); copyto!(dca, ca)
        dcb = similar(cnt, nch); copyto!(dcb, cb)
        dr1 = similar(cnt, nch); dr2 = similar(cnt, nch)
        _chunk_runs!(backend, 256)(dr1, dr2, dca, dcb, rptr, Int32(nruns); ndrange=nch)
        r1 = Array(dr1); r2 = Array(dr2)
        chunks = [ClassChunk(desc[k][1], desc[k][2], desc[k][3], desc[k][4], Int(r1[k]), Int(r2[k])) for k in 1:nch]
    end
    return DeviceLevelLists(csrc, cfar, rtgt, rptr), chunks
end
