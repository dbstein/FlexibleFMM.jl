# Symmetry classes of the far offsets under the hyperoctahedral group (the signed
# permutations of the axes: 8 elements in 2D, 48 in 3D). Every element maps the
# cube lattices to themselves, and a kernel that is covariant under orthogonal
# transformations (scalar, or `K(Rs, Rt) = R·K(s, t)·Rᵀ` for the Stokeslet, with
# the pressure and the point-source velocity transforming as vectors) satisfies
#   M2L_{R·o} = P_R · M2L_o · P_Rᵀ,
# with `P_R` the signed permutation of the component-major expansion entries
# (components permuted with the axis signs, surface points permuted by the lattice
# map, the scalar augmentation entry fixed). The 316 far offsets of 3D fall into
# 16 classes (7 of the 40 in 2D), so a dense translation path stores one operator
# per class and permutes. The relation is checked numerically on all offsets
# (`check_symmetry_classes`), not assumed.

struct SignedPermutation{D}
    perm::SVector{D,Int}      # (R·x)[k] = sign[k] · x[perm[k]]
    sign::SVector{D,Int}
end
(R::SignedPermutation{D})(x::SVector{D}) where {D} = SVector{D}(ntuple(k -> R.sign[k] * x[R.perm[k]], Val(D)))

function _permutations(v::Vector{Int})
    length(v) <= 1 && return [v]
    out = Vector{Int}[]
    for (k, x) in enumerate(v), rest in _permutations(v[[1:k-1; k+1:end]])
        push!(out, [x; rest])
    end
    return out
end
function hyperoctahedral_group(::Val{D}) where {D}
    els = SignedPermutation{D}[]
    for perm in _permutations(collect(1:D)), signs in Iterators.product(ntuple(_ -> (1, -1), Val(D))...)
        push!(els, SignedPermutation{D}(SVector{D,Int}(perm), SVector{D,Int}(signs)))
    end
    return els
end

# the class representative of an offset: its absolute entries in decreasing order
canonical_offset(o::SVector{D,Int}) where {D} = SVector{D,Int}(sort(abs.(o); rev=true))

"""
    SymmetryClasses{D}

For the far offsets of `far_offsets(Val(D))`: the class representatives, the class
of each offset, and for each offset the signed permutation `(idx, sgn)` of the
expansion entries with `(P·x)[idx[i]] = sgn[i]·x[i]`, so that
`M2L_o · m = P · (M2L_rep · (Pᵀ·m))` with `(Pᵀ·m)[i] = sgn[i]·m[idx[i]]`.
"""
struct SymmetryClasses{D}
    reps::Vector{SVector{D,Int}}
    class::Vector{Int}            # per far offset
    idx::Vector{Vector{Int}}      # per far offset
    sgn::Vector{Vector{Int8}}
end

"""
    symmetry_classes(::Val{D}, p, embed, nc, augmented) -> SymmetryClasses

The classes for `p`-per-edge cube lattices whose surface points have the padded
lattice indices `embed`, expansions with `nc` components and, if `augmented`, one
scalar entry after them.
"""
function symmetry_classes(::Val{D}, p::Int, embed::Vector{Int}, nc::Int, augmented::Bool) where {D}
    offs = far_offsets(Val(D))
    G = hyperoctahedral_group(Val(D))
    N = length(embed)
    lin = LinearIndices(ntuple(_ -> 2p, Val(D)))
    cart = CartesianIndices(ntuple(_ -> 2p, Val(D)))
    mi = [SVector{D,Int}(Tuple(cart[e]) .- 1) for e in embed]          # lattice indices in 0:p-1
    slot = Dict(m => i for (i, m) in enumerate(mi))
    reps = SVector{D,Int}[]
    class = Int[]; idxs = Vector{Int}[]; sgns = Vector{Int8}[]
    for o in offs
        r = canonical_offset(o)
        c = findfirst(==(r), reps)
        c === nothing && (push!(reps, r); c = length(reps))
        push!(class, c)
        g = G[findfirst(R -> R(r) == o, G)]
        # the point map: R acts on lattice indices as m'[k] = m[perm[k]] or p-1-m[perm[k]]
        ρ = [slot[SVector{D,Int}(ntuple(k -> g.sign[k] == 1 ? m[g.perm[k]] : p - 1 - m[g.perm[k]], Val(D)))] for m in mi]
        # the component map: (R·f)[k] = sign[k]·f[perm[k]], so entry (b, i) goes to (perm⁻¹(b), ρ(i))
        invperm = sortperm(Vector(g.perm))
        n = nc * N + (augmented ? 1 : 0)
        idx = Vector{Int}(undef, n); sgn = Vector{Int8}(undef, n)
        for b in 1:nc, i in 1:N
            k = nc == 1 ? 1 : invperm[b]
            idx[(b - 1) * N + i] = (k - 1) * N + ρ[i]
            sgn[(b - 1) * N + i] = nc == 1 ? Int8(1) : Int8(g.sign[k])
        end
        if augmented
            idx[n] = n; sgn[n] = Int8(1)
        end
        push!(idxs, idx); push!(sgns, sgn)
    end
    return SymmetryClasses{D}(reps, class, idxs, sgns)
end

# P·x and Pᵀ·x for the signed permutation (idx, sgn)
function permute(sc::SymmetryClasses, f::Int, x::AbstractVector)
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[sc.idx[f][i]] = sc.sgn[f][i] * x[i]
    end
    return y
end
function permute_t(sc::SymmetryClasses, f::Int, x::AbstractVector)
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = sc.sgn[f][i] * x[sc.idx[f][i]]
    end
    return y
end

"""
    class_operators(K, small, embed, w, p) -> (SymmetryClasses, Vector{Matrix})

The dense translation operators of the class representatives of the cube method
(multipole to check data, with the Stokes augmentations) for a box of width `w`.
"""
function class_operators(K, small::AbstractVector{SVector{D,R}}, embed::Vector{Int}, w::R, p::Int) where {D,R}
    nc = ncomp(K, Val(D))
    sc = symmetry_classes(Val(D), p, embed, nc, has_flux_source(K) || pins_pressure(K))
    origin = zero(SVector{D,R})
    ops = [_multipole_to_check_matrix(K, small .+ Ref(SVector{D,R}(r) * w), SVector{D,R}(r) * w, w, small, origin) for r in sc.reps]
    return sc, ops
end

"""
    check_symmetry_classes(K, p, d, w; rtol=1e-12) -> maximum relative error

Verifies `M2L_o = P·M2L_rep·Pᵀ` on every far offset against the dense matrices.
"""
function check_symmetry_classes(K, ::Val{D}, p::Int, d::Real, w::Real; rtol=1e-12) where {D}
    R = floattype(K)
    a_small = R(1 + d) * R(w) / 2
    small, embed = cube_boundary_points(Val(D), p, a_small)
    sc, ops = class_operators(K, small, embed, R(w), p)
    origin = zero(SVector{D,R})
    worst = 0.0
    for (f, o) in enumerate(far_offsets(Val(D)))
        A = _multipole_to_check_matrix(K, small .+ Ref(SVector{D,R}(o) * R(w)), SVector{D,R}(o) * R(w), R(w), small, origin)
        Aref = ops[sc.class[f]]
        # P·Aref·Pᵀ: permute rows and columns
        B = similar(A)
        idx = sc.idx[f]; sgn = sc.sgn[f]
        for j in axes(A, 2), i in axes(A, 1)
            B[idx[i], idx[j]] = sgn[i] * sgn[j] * Aref[i, j]
        end
        worst = max(worst, maximum(abs.(A - B)) / maximum(abs.(A)))
    end
    worst <= rtol || throw(ArgumentError("the symmetry classes do not reproduce the translation operators (error $worst)"))
    return worst
end
