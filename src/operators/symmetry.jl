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
of each offset, and for each offset the signed permutations of the expansion
entries, `(idx, sgn)` on the source side and `(idx_t, sgn_t)` on the target side,
with `(P·x)[idx[i]] = sgn[i]·x[i]` and `(Pᵀ·m)[i] = sgn[i]·m[idx[i]]`, so that
`M2L_o · m = P_t · (M2L_rep · (P_sᵀ·m))`. For the cube lattices the two sides are
the same permutation (the four-argument constructor); for the harmonic basis a
rotation acts on locals by the transpose of its action on multipoles. (The device
plan uses the source side for both and takes kernel-independent methods only.)
"""
struct SymmetryClasses{D}
    reps::Vector{SVector{D,Int}}
    class::Vector{Int}            # per far offset
    idx::Vector{Vector{Int}}      # per far offset, source side
    sgn::Vector{Vector{Int8}}
    idx_t::Vector{Vector{Int}}    # target side
    sgn_t::Vector{Vector{Int8}}
end
SymmetryClasses{D}(reps, class, idx, sgn) where {D} = SymmetryClasses{D}(reps, class, idx, sgn, idx, sgn)

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

# P_t·x (target side) and P_sᵀ·x (source side)
function permute(sc::SymmetryClasses, f::Int, x::AbstractVector)
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[sc.idx_t[f][i]] = sc.sgn_t[f][i] * x[i]
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

"""
    class_factors(K, small, embed, w, p, tol) -> (SymmetryClasses, A, Bt, ranks)

Low-rank factors `M2L_rep ≈ A·Bt` of the class representatives (truncated SVD
keeping the singular values above `tol` times the largest of the representative),
so that a translation costs `4·n·k` flops instead of `2·n²`. The far offsets of a
cube lattice see a smooth kernel: at `p = 8` the Stokes operators (n = 889) have
ranks about 40 at `1e-6` and 80 at `1e-8`, Laplace (n = 296) 13 and 26.
"""
function class_factors(K, small::AbstractVector{SVector{D,R}}, embed::Vector{Int}, w::R, p::Int, tol::Real) where {D,R}
    sc, ops = class_operators(K, small, embed, w, p)
    A = Matrix{eltype(ops[1])}[]; Bt = similar(A); ranks = Int[]
    for M in ops
        F = svd(M)
        k = max(1, count(σ -> σ > tol * F.S[1], F.S))
        push!(A, F.U[:, 1:k] * Diagonal(F.S[1:k])); push!(Bt, Matrix(F.Vt[1:k, :])); push!(ranks, k)
    end
    return sc, A, Bt, ranks
end

# ---- the harmonic basis: the subgroup fixing the z axis ----
#
# Of the 48 signed permutations only the 16 that keep the z axis (D4h) act on the
# real solid harmonics by signed permutations, each within one (n, m): writing
# R = Rot_z(kπ/2) ∘ Refl_y^εy ∘ Refl_z^εz, Υ_n^m(R x) = i^{mk} (−1)^{(n+m)εz} [conj]^{εy} Υ_n^m(x),
# so a multipole transforms by M_n^m → conj(c)·[conj]^{εy} M_n^m and a local by
# L_n^m → c·[conj]^{εy} L_n^m, c = i^{mk} (−1)^{(n+m)εz} (the local of Φ(R⁻¹x)). The
# other 32 elements mix all m of a degree (Wigner rotations). Under D4h the 316 far
# offsets fall into 34 classes, representative (max(|o₁|,|o₂|), min(|o₁|,|o₂|), |o₃|).

d4h_group() = [R for R in hyperoctahedral_group(Val(3)) if R.perm[3] == 3]
harmonic_canonical_offset(o::SVector{3,Int}) = SVector(max(abs(o[1]), abs(o[2])), min(abs(o[1]), abs(o[2])), abs(o[3]))

# R = Rot_z(kπ/2) ∘ Refl_y^εy ∘ Refl_z^εz
function _d4h_factors(R::SignedPermutation{3})
    R.perm[3] == 3 || throw(ArgumentError("not an element of D4h"))
    εz = R.sign[3] == -1
    A = zeros(Int, 2, 2)
    A[1, R.perm[1]] = R.sign[1]; A[2, R.perm[2]] = R.sign[2]
    εy = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1] == -1
    εy && (A[:, 2] .*= -1)                                    # A = Rot·diag(1, −1)
    rots = ([1 0; 0 1], [0 -1; 1 0], [-1 0; 0 -1], [0 1; -1 0])
    k = findfirst(==(A), rots)
    k === nothing && throw(ArgumentError("not a rotation"))
    return k - 1, εy, εz
end

# the signed permutation of the real storage for z → u·[conj]^εy z, u = i^{mk}(−1)^{(n+m)εz}
# on the target side and its conjugate on the source side
function _harmonic_permutation(p::Int, k::Int, εy::Bool, εz::Bool, target::Bool)
    N = (p + 1)^2
    idx = Vector{Int}(undef, N); sgn = Vector{Int8}(undef, N)
    for n in 0:p
        i0 = n * n + 1
        idx[i0] = i0; sgn[i0] = εz && isodd(n) ? -1 : 1
        for m in 1:n
            ir = n * n + 2m; ii = ir + 1
            q = mod(target ? m * k : -m * k, 4)
            s = εz && isodd(n + m) ? -1 : 1
            σ = εy ? -1 : 1                                     # the conjugation
            if q == 0
                idx[ir] = ir; sgn[ir] = s;  idx[ii] = ii; sgn[ii] = s * σ
            elseif q == 1                                       # ×i: (re, im) → (−im, re)
                idx[ir] = ii; sgn[ir] = s;  idx[ii] = ir; sgn[ii] = -s * σ
            elseif q == 2
                idx[ir] = ir; sgn[ir] = -s; idx[ii] = ii; sgn[ii] = -s * σ
            else                                                # ×(−i): (re, im) → (im, −re)
                idx[ir] = ii; sgn[ir] = -s; idx[ii] = ir; sgn[ii] = s * σ
            end
        end
    end
    return idx, sgn
end

"""
    harmonic_symmetry_classes(p) -> SymmetryClasses{3}

The 34 classes of the far offsets under D4h and, per offset, the source- and
target-side signed permutations of the real harmonic storage of degree `p`.
"""
function harmonic_symmetry_classes(p::Int)
    offs = far_offsets(Val(3))
    G = d4h_group()
    reps = SVector{3,Int}[]
    class = Int[]; idxs = Vector{Int}[]; sgns = Vector{Int8}[]; idxt = Vector{Int}[]; sgnt = Vector{Int8}[]
    for o in offs
        r = harmonic_canonical_offset(o)
        c = findfirst(==(r), reps)
        c === nothing && (push!(reps, r); c = length(reps))
        push!(class, c)
        g = G[findfirst(R -> R(r) == o, G)]
        k, εy, εz = _d4h_factors(g)
        is, ss = _harmonic_permutation(p, k, εy, εz, false)
        it, st = _harmonic_permutation(p, k, εy, εz, true)
        push!(idxs, is); push!(sgns, ss); push!(idxt, it); push!(sgnt, st)
    end
    return SymmetryClasses{3}(reps, class, idxs, sgns, idxt, sgnt)
end
