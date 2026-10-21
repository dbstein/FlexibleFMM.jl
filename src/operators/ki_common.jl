# Kernel-independent FMM machinery shared by the circle (2D) and square/cube (any
# D) surface families. A box carries an inner "small" surface (upward equivalent,
# downward check) and an outer "large" one (upward check, downward equivalent).
# Translation operators map densities to check data; the check→equivalent solves
# are truncated SVDs applied per level in factored form (see docs/kifmm.md for why
# a dense or folded pseudo-inverse would lose eps/σ_min in every direction).
#
# Vector kernels: densities have ncomp components per surface point, stored
# component major. Stokes needs two augmentations, both from the fact that a single-
# layer stokeslet density has zero flux: the multipole carries a point-source
# strength (the stresslet kernel carries a point mass source of strength -(f·ν)),
# and the downward check data carry the pressure at the box center (the single-layer
# null space has zero velocity but constant pressure inside). Both are
# nondimensionalised with the box width so that the augmented operators scale like
# the kernel.

abstract type AbstractKernelIndependent <: AbstractMethod end

"""
    FactoredPinv{T}

Truncated pseudo-inverse `V·Diagonal(invS)·Uᴴ` kept as factors; `apply!` uses them in
sequence so that the amplified rounding stays in the attenuated modes.
"""
struct FactoredPinv{T,R}
    Uh::Matrix{T}        # rank × m   (adjoint of the kept left singular vectors)
    invS::Vector{R}
    V::Matrix{T}         # n × rank
    sigma::Vector{R}
end
rank(P::FactoredPinv) = length(P.invS)

"""
    pinv_rtol(A, rtol) -> FactoredPinv

Truncated-SVD pseudo-inverse keeping singular values `≥ rtol·σ₁`.
"""
function pinv_rtol(A::AbstractMatrix, rtol::Real)
    F = svd(A)
    σ = F.S
    r = count(s -> s >= rtol * σ[1], σ)
    T = eltype(A)
    return FactoredPinv{T,real(T)}(Matrix{T}(F.U[:, 1:r]'), 1 ./ σ[1:r], Matrix{T}(F.V[:, 1:r]), σ)
end
Base.Matrix(P::FactoredPinv) = P.V * Diagonal(P.invS) * P.Uh

"""
    apply!(out, P::FactoredPinv, x, work, α=1)

`out = α·P·x` for a vector or matrix `x`, using `work` (at least `rank × size(x, 2)`).
"""
function apply!(out::AbstractVecOrMat, P::FactoredPinv{T}, x::AbstractVecOrMat, work::AbstractVecOrMat, α=true) where {T}
    r = rank(P)
    n = size(x, 2)
    tmp = x isa AbstractVector ? view(work, 1:r) : view(work, 1:r, 1:n)
    mul!(tmp, P.Uh, x)
    lmul!(Diagonal(P.invS), tmp)
    mul!(out, P.V, tmp, T(α), false)
    return out
end

"""
    KIExtras{T,R,D}

Surfaces (relative to the box center), the two pseudo-inverses, their shared-matrix
scale factor, and, for square/cube surfaces, the lattice data the spectral translation
needs (`p` points per edge, spacing `h` of the small surface, and the linear index
of each small-surface point in the padded `(2p)^D` lattice).
"""
struct KIExtras{T,R,D}
    small::Vector{SVector{D,R}}
    large::Vector{SVector{D,R}}
    P_up::FactoredPinv{T,R}
    P_down::FactoredPinv{T,R}
    pinv_scale::R
    p::Int
    h::R
    embed::Vector{Int}
end

"""
    pins_pressure(K), has_flux_source(K)

Whether the downward check data of `K` carry the pressure at the box center, and
whether its multipoles carry a point-source strength. Both hold for Stokes kernels.
"""
pins_pressure(K) = false
has_flux_source(K) = false

expansion_length(m::AbstractKernelIndependent) = ncomp(m.kernel, Val(dimension(m))) * nsurface(m)
multipole_length(m::AbstractKernelIndependent) = expansion_length(m) + (has_flux_source(m.kernel) ? 1 : 0)
partial_local_length(m::AbstractKernelIndependent) = expansion_length(m) + (pins_pressure(m.kernel) ? 1 : 0)
near_kernel(m::AbstractKernelIndependent) = m.kernel
function operator_eltype(m::AbstractKernelIndependent)
    R = floattype(m.kernel)
    D = dimension(m)
    return eltype(kernel_eltype(m.kernel, SVector{D,R}, SVector{D,R}))
end

# ---- augmentation helpers (component major) ----

# Row of `w·pressure` at `x` due to unit densities at `sources`.
function _pressure_row(K, sources::AbstractVector{SVector{D,R}}, x::SVector{D,R}, w::R) where {D,R}
    n = length(sources)
    nc = ncomp(K, Val(D))
    row = Matrix{R}(undef, 1, nc * n)
    for i in 1:n
        p = w * pressure_kernel(K, sources[i], x)
        for a in 1:nc
            row[1, (a - 1) * n + i] = p[a]
        end
    end
    return row
end
# Column of velocities at `targets` due to a point source of strength `w` at `c`
# (unit strength per unit width).
function _source_column(K, targets::AbstractVector{SVector{D,R}}, c::SVector{D,R}, w::R) where {D,R}
    n = length(targets)
    nc = ncomp(K, Val(D))
    col = Matrix{R}(undef, nc * n, 1)
    for j in 1:n
        v = w * source_velocity(K, c, targets[j])
        for a in 1:nc
            col[(a - 1) * n + j, 1] = v[a]
        end
    end
    return col
end
# Velocities on `targets` produced by the multipole representation of a box of width
# `w`: a density on `sources` and, if the kernel needs it, a point source at `center`.
function _rep_velocity_matrix(K, sources, center, w, targets)
    A = kernel_matrix(K, sources, targets)
    return has_flux_source(K) ? hcat(A, _source_column(K, targets, center, w)) : A
end
# Downward check data (velocities on `targets` and, if pinned, `w·pressure` at
# `tcenter`) produced by the multipole representation (`sources`, `scenter`, width `w`).
function _multipole_to_check_matrix(K, sources, scenter, w, targets, tcenter)
    A = _rep_velocity_matrix(K, sources, scenter, w, targets)
    pins_pressure(K) || return A
    row = _pressure_row(K, sources, tcenter, w)
    has_flux_source(K) && (row = hcat(row, zeros(eltype(row), 1, 1)))   # a point source has no pressure
    return vcat(A, row)
end
# Downward check data produced by a local representation (density on `sources` only)
# in a target box of width `w`.
function _local_to_check_matrix(K, sources, targets, tcenter, w)
    A = kernel_matrix(K, sources, targets)
    return pins_pressure(K) ? vcat(A, _pressure_row(K, sources, tcenter, w)) : A
end

# The dense parts common to both families: pseudo-inverses, M2M and L2L, given the
# surfaces of the box (relative to its center) and of its children.
function _ki_dense_operators(m::AbstractKernelIndependent, w::R, small, large, child_small) where {R}
    K = m.kernel
    D = dimension(m)
    origin = zero(SVector{D,R})
    P_up = pinv_rtol(_rep_velocity_matrix(K, small, origin, w, large), m.pinv_rtol)
    P_down = pinv_rtol(_local_to_check_matrix(K, large, small, origin, w), m.pinv_rtol)
    centers = [(SVector{D,R}(δ) .- R(0.5)) * (w / 2) for δ in child_offsets(Val(D))]
    # one block per child, so that the stacked children multipoles (each component
    # major within its child) match the operator's columns
    M2M = reduce(hcat, [_rep_velocity_matrix(K, child_small .+ Ref(c), c, w / 2, large) for c in centers])
    L2L = reduce(vcat, [_local_to_check_matrix(K, large, child_small .+ Ref(c), c, w / 2) for c in centers])
    return P_up, P_down, M2M, L2L
end

"""
    conditioning(ops) -> (σmin/σmax up, σmin/σmax down, dropped up, dropped down)

Diagnostics of the check maps of a level: smallest-to-largest singular value ratios
and how many modes the truncation dropped.
"""
function conditioning(ops::LevelOperators{T,R,<:KIExtras}) where {T,R}
    e = ops.extra
    su, sd = e.P_up.sigma, e.P_down.sigma
    return (su[end] / su[1], sd[end] / sd[1], length(su) - rank(e.P_up), length(sd) - rank(e.P_down))
end

function rescale(ref::LevelOperators{T,R,KIExtras{T,R,D}}, m::AbstractKernelIndependent, w::Real) where {T,R,D}
    a = homogeneity(m.kernel)
    a === nothing && throw(ArgumentError("rescale needs a homogeneous kernel"))
    λ = R(w) / ref.width
    f = λ^a
    e = ref.extra
    extra = KIExtras{T,R,D}(e.small .* λ, e.large .* λ, e.P_up, e.P_down, e.pinv_scale / f, e.p, e.h * λ, e.embed)
    return typeof(ref)(R(w), ref.M2M, ref.L2L, ref.M2L, ref.m2m_scale * f, ref.l2l_scale * f, ref.m2l_scale * f, extra)
end

# ---- component-major storage of a field with `nc` components at point `k` of `n` ----
@inline _store!(col, k::Int, n::Int, v::Number) = (@inbounds col[k] = v; nothing)
@inline function _store!(col, k::Int, n::Int, v::SVector{nc}) where {nc}
    @inbounds for a in 1:nc
        col[(a - 1) * n + k] = v[a]
    end
    return nothing
end
@inline _load(L, k::Int, n::Int, ::Val{1}) = @inbounds L[k]
@inline _load(L, k::Int, n::Int, ::Val{nc}) where {nc} = SVector(ntuple(a -> (@inbounds L[(a - 1) * n + k]), Val(nc)))
@inline _field_zero(::Type{S}, ::Val{1}) where {S} = zero(S)
@inline _field_zero(::Type{S}, ::Val{nc}) where {S,nc} = zero(SVector{nc,S})

"""
    s2m!(col, m, ops, xs, τs, c)

Partial multipole of a childless node: the check potentials its sources produce on
the large surface (relative to the center `c`).
"""
function s2m!(col::AbstractVector, m::AbstractKernelIndependent, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{D}) where {D}
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(D)))
    @inbounds for k in 1:N
        acc = _field_zero(eltype(col), nc)
        for j in eachindex(xs)
            acc += source_field(K, xs[j] - c, large[k], τs[j])
        end
        _store!(col, k, N, acc)
    end
    return col
end

"""
    partial_to_multipole!(M, m, ops, pM, work)
    partial_to_local!(L, m, ops, pL, work)

Check data of a whole level (one column per node) to equivalent densities.
"""
partial_to_multipole!(M::AbstractVecOrMat, m::AbstractKernelIndependent, ops::LevelOperators, pM::AbstractVecOrMat, work) =
    apply!(M, ops.extra.P_up, pM, work, ops.extra.pinv_scale)
partial_to_local!(L::AbstractVecOrMat, m::AbstractKernelIndependent, ops::LevelOperators, pL::AbstractVecOrMat, work) =
    apply!(L, ops.extra.P_down, pL, work, ops.extra.pinv_scale)

"""
    l2t(m, ops, L, xrel, out=Potential())

Output `out` of the local expansion `L` (density on the large surface) at `xrel`
relative to the box center.
"""
@inline function l2t(m::AbstractKernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{D}, ::Potential=Potential()) where {D}
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(D)))
    S = promote_type(eltype(L), eltype(kernel_eltype(K, eltype(large), typeof(xrel))))
    acc = _field_zero(S, nc)
    @inbounds for k in 1:N
        acc += K(large[k], xrel) * _load(L, k, N, nc)
    end
    return acc
end
@inline function l2t(m::AbstractKernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{D}, out::Gradient) where {D}
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(D)))
    acc = zero(Base.promote_op(target_field, typeof(K), eltype(large), typeof(xrel), typeof(_load(L, 1, N, nc)), typeof(out)))
    @inbounds for k in 1:N
        acc += target_field(K, large[k], xrel, _load(L, k, N, nc), out)
    end
    return acc
end
@inline function l2t(m::AbstractKernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{D}, ::Pressure) where {D}
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(D)))
    S = promote_type(eltype(L), eltype(kernel_eltype(K, eltype(large), typeof(xrel))))
    acc = zero(S)
    @inbounds for k in 1:N
        acc += sum(pressure_kernel(K, large[k], xrel) .* _load(L, k, N, nc))
    end
    return acc
end

"""
    multipole_field(m, ops, M, xrel)

Far field of the multipole `M` (density on the small surface, plus a point source
for Stokes) at `xrel`; used by tests.
"""
function multipole_field(m::AbstractKernelIndependent, ops::LevelOperators, M::AbstractVector, xrel::SVector{D}) where {D}
    K = m.kernel
    small = ops.extra.small
    N = length(small)
    nc = Val(ncomp(K, Val(D)))
    S = promote_type(eltype(M), eltype(kernel_eltype(K, eltype(small), typeof(xrel))))
    acc = _field_zero(S, nc)
    for k in 1:N
        acc += K(small[k], xrel) * _load(M, k, N, nc)
    end
    if has_flux_source(K)
        acc += (M[end] * ops.width) * source_velocity(K, zero(xrel), xrel)   # strength per unit width
    end
    return acc
end
