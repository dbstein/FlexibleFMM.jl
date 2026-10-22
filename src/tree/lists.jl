# Interaction (V) lists. For a non-ghost node `i` on level ≥ 3, the sources of its
# far field at the same level are the children of the colleagues of `parent(i)` that
# are not colleagues of `i` themselves, i.e. the children at a far offset
# (Chebyshev norm ≥ 2, at most 3). Only children with sources are listed. Ghosts are
# never targets (their local expansions are never read).
#
# Two passes per level, each threaded over the targets: a count of each target's
# pairs, a prefix sum, and a fill into exact-size arrays (the pairs of a target
# consecutive, targets in node order, the enumeration order of the colleagues and
# children as before); then the CSR by far offset (a counting sort) and the by-target
# view with the targets in Morton order. (The first version enumerated with `push!`
# and recovered the target groups by a scan: 1.2 of the 1.5 s build of 10⁷ uniform
# points at 200 per leaf, `docs/notes/tree_plan.md`.)

function build_interaction_lists(::Val{D}, level_range, cell, parent, child_first,
                                 colleagues, ghost, has_source, far_index) where {D}
    L = length(level_range)
    nfar = count(!=(0), far_index)
    ilists = Vector{InteractionList}(undef, L)
    for l in 1:L
        r = level_range[l]
        n = length(r)
        lo = first(r) - 1
        cnt = zeros(Int, n + 1)
        l >= 3 && _count_pairs!(cnt, lo, n, cell, parent, child_first, colleagues, ghost, has_source, Val(D))
        ptr = cumsum(cnt) .+ 1                       # the pairs of target q: ptr[q]:ptr[q+1]-1
        np = ptr[end] - 1
        fs = Vector{Int}(undef, np); ts = Vector{Int}(undef, np); ss = Vector{Int}(undef, np)
        l >= 3 && _fill_pairs!(fs, ts, ss, ptr, lo, n, cell, parent, child_first, colleagues, ghost, has_source, far_index, Val(D))
        # counting sort by far offset into CSR form
        fcount = zeros(Int, nfar)
        @inbounds for f in fs
            fcount[f] += 1
        end
        offset_ptr = Vector{Int}(undef, nfar + 1)
        offset_ptr[1] = 1
        for f in 1:nfar
            offset_ptr[f+1] = offset_ptr[f] + fcount[f]
        end
        wpos = offset_ptr[1:nfar]
        tgt = similar(ts)
        src = similar(ss)
        @inbounds for q in eachindex(fs)
            f = fs[q]
            tgt[wpos[f]] = ts[q]
            src[wpos[f]] = ss[q]
            wpos[f] += 1
        end
        # the same pairs by target, targets in Morton order
        withp = [q for q in 1:n if cnt[q+1] > 0]
        nbits = min(l - 1, 64 ÷ D)
        order = sortperm([morton_key(cell[lo+q], nbits) for q in withp])
        ng = length(withp)
        torder = Vector{Int}(undef, ng)
        tptr = Vector{Int}(undef, ng + 1)
        tsrc = similar(ss)
        tfar = similar(fs)
        tptr[1] = 1
        @inbounds for (g, oq) in enumerate(order)
            q = withp[oq]
            pr = ptr[q]:ptr[q+1]-1
            torder[g] = lo + q
            tptr[g+1] = tptr[g] + length(pr)
            tsrc[tptr[g]:tptr[g+1]-1] = view(ss, pr)
            tfar[tptr[g]:tptr[g+1]-1] = view(fs, pr)
        end
        ilists[l] = InteractionList(offset_ptr, tgt, src, torder, tptr, tsrc, tfar)
    end
    return ilists
end
# the far pairs of target `lo + q` (0 for a ghost): the sourced children of the
# parent's colleagues at a far offset
function _count_pairs!(cnt, lo, n, cell, parent, child_first, colleagues, ghost, has_source, ::Val{D}) where {D}
    nc = 1 << D
    Threads.@threads :dynamic for q in 1:n
        i = lo + q
        c = 0
        @inbounds if !ghost[i]
            p = parent[i]
            ci = cell[i]
            for k in 1:3^D
                cc = colleagues[k, p]
                cc == 0 && continue
                cf = child_first[cc]
                cf == 0 && continue
                for s in cf:cf+nc-1
                    has_source[s] || continue
                    is_far_offset(cell[s] - ci) && (c += 1)
                end
            end
        end
        @inbounds cnt[q+1] = c
    end
    return cnt
end
function _fill_pairs!(fs, ts, ss, ptr, lo, n, cell, parent, child_first, colleagues, ghost, has_source, far_index, ::Val{D}) where {D}
    nc = 1 << D
    Threads.@threads :dynamic for q in 1:n
        i = lo + q
        @inbounds begin
            pos = ptr[q]
            if !ghost[i]
                p = parent[i]
                ci = cell[i]
                for k in 1:3^D
                    cc = colleagues[k, p]
                    cc == 0 && continue
                    cf = child_first[cc]
                    cf == 0 && continue
                    for s in cf:cf+nc-1
                        has_source[s] || continue
                        off = cell[s] - ci
                        is_far_offset(off) || continue
                        fs[pos] = far_index[m2l_table_index(off)]
                        ts[pos] = i
                        ss[pos] = s
                        pos += 1
                    end
                end
            end
        end
    end
    return nothing
end
