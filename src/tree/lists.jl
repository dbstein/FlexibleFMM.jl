# Interaction (V) lists. For a non-ghost node `i` on level ≥ 3, the sources of its
# far field at the same level are the children of the colleagues of `parent(i)` that
# are not colleagues of `i` themselves, i.e. the children at a far offset
# (Chebyshev norm ≥ 2, at most 3). Only children with sources are listed. Ghosts are
# never targets (their local expansions are never read).

function build_interaction_lists(::Val{D}, level_range, cell, parent, child_first,
                                 colleagues, ghost, has_source, far_index) where {D}
    L = length(level_range)
    nfar = count(!=(0), far_index)
    nc = 1 << D
    ilists = Vector{InteractionList}(undef, L)
    for l in 1:L
        fs = Int[]; ts = Int[]; ss = Int[]
        if l >= 2
            for i in level_range[l]
                ghost[i] && continue
                p = parent[i]
                for k in 1:3^D
                    c = colleagues[k, p]
                    c == 0 && continue
                    cf = child_first[c]
                    cf == 0 && continue
                    for s in cf:cf+nc-1
                        has_source[s] || continue
                        off = cell[s] - cell[i]
                        is_far_offset(off) || continue
                        push!(fs, far_index[m2l_table_index(off)])
                        push!(ts, i)
                        push!(ss, s)
                    end
                end
            end
        end
        # counting sort by far offset into CSR form
        counts = zeros(Int, nfar)
        for f in fs
            counts[f] += 1
        end
        ptr = Vector{Int}(undef, nfar + 1)
        ptr[1] = 1
        for f in 1:nfar
            ptr[f+1] = ptr[f] + counts[f]
        end
        pos = ptr[1:nfar]
        tgt = similar(ts)
        src = similar(ss)
        for q in eachindex(fs)
            f = fs[q]
            tgt[pos[f]] = ts[q]
            src[pos[f]] = ss[q]
            pos[f] += 1
        end
        # the same pairs by target (`ts` is already grouped by target, in level order),
        # targets in Morton order
        starts = Int[]
        for q in eachindex(ts)
            (q == 1 || ts[q] != ts[q-1]) && push!(starts, q)
        end
        push!(starts, length(ts) + 1)
        ng = length(starts) - 1
        nbits = min(l - 1, 64 ÷ D)
        order = sortperm([morton_key(cell[ts[starts[g]]], nbits) for g in 1:ng])
        torder = Vector{Int}(undef, ng)
        tptr = Vector{Int}(undef, ng + 1)
        tsrc = similar(ss)
        tfar = similar(fs)
        tptr[1] = 1
        for (q, g) in enumerate(order)
            r = starts[g]:starts[g+1]-1
            torder[q] = ts[first(r)]
            tptr[q+1] = tptr[q] + length(r)
            tsrc[tptr[q]:tptr[q+1]-1] = ss[r]
            tfar[tptr[q]:tptr[q+1]-1] = fs[r]
        end
        ilists[l] = InteractionList(ptr, tgt, src, torder, tptr, tsrc, tfar)
    end
    return ilists
end
