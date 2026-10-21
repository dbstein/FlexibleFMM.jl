# Kernel-independent FMM operators (Ying, Biros & Zorin), 2D.
#
# A box of width w carries two concentric circles with Nequiv equispaced points:
#   small, radius r1 = w√2/2         the box's circumcircle
#   large, radius r2 = w(4-√2)/2     tangent to the small circle of the nearest
#                                    interaction-list box (whose center is 2w away)
# Upward:   multipole = density on `small`, found from the check potentials the
#           sources (or the children's multipoles) produce on `large`.
# Downward: local = density on `large`, found from the check potentials that far
#           multipoles (or the parent's local) produce on `small`.
# The translation operators map densities to check potentials:
#   M2M = (children small → large),  M2L[f] = (shifted small → small),
#   L2L = (large → children small);
# the check→equivalent solves are done once per level on the whole level block with
# the truncated SVD of the check matrices, applied in factored form V·Σ⁻¹·Uᴴ.
#
# Accuracy: the worst-case translation error (a source at a box corner, a target at
# the facing corner of the nearest interaction-list box) is about (r1/r2)^N = 0.547^N
# with N points per circle: 7e-5, 1e-8, 2e-12 at N = 16, 32, 48, measured; global
# relative errors on uniform data are one to two orders smaller (6e-9 at N = 32).
# Rounding limits the floor to about 1e-14, reached near N = 64.
#
# Why factored and not as one matrix: mode m of the check map decays like
# (r1/r2)^|m| = 0.547^|m|, so the pseudo-inverse has norm 1/σ_min ≈ 0.547^(-N/2)
# (1e13 at N = 100). A dense pseudo-inverse, or one folded into M2M/M2L/L2L, is formed
# by products whose rounding errors are eps/σ_min in every direction, and the far
# field inherits them (6e-4 at N = 100, measured). In factored form the rounding
# amplified by 1/σ_j stays in mode j, which the evaluation attenuates again by σ_j.
#
# Vector kernels (Stokes): densities have ncomp components per circle point, stored
# component major (all first components, then all second). Two augmentations:
#  - upward, the multipole carries one extra entry, the strength of a point source
#    at the box center. A single-layer stokeslet density has zero flux through every
#    enclosing curve, but the stresslet (double-layer) kernel carries a point mass
#    source of strength -(f·ν), so without the extra term the far field of a box of
#    stresslets cannot be represented (0.4 relative error, measured). The source
#    velocity on the check circle is purely radial, exactly what the single layer
#    cannot produce, so fitting (density, strength) from the check velocities is
#    well posed; boxes of stokeslets simply get strength 0;
#  - downward, the check data carry one extra entry, the pressure at the box
#    center: the single-layer density on a closed curve has a one-dimensional null
#    space (the normal) whose velocity inside vanishes but whose pressure is a
#    constant, so without this row the pressure of a local expansion would be off
#    by a constant per box (the Python "pressure slot"). The field inside a box has
#    zero flux, so no source term is needed downward.
#
# Two known degeneracies (docs/kifmm.md): for Laplace the monopole check potential
# is -Q·log(r2)/2π, zero when w = 2/(4-√2) ≈ 0.7735; for Helmholtz mode m is singular
# when J_m(k·r1) = 0, first at k·w ≈ 3.40. The SVD truncation reports both.

struct KernelIndependent{K} <: AbstractMethod
    kernel::K
    nequiv::Int
    pinv_rtol::Float64
end

"""
    KernelIndependent(kernel; nequiv=50, pinv_rtol=50·eps(floattype(kernel)))

The kernel-independent method with `nequiv` points per equivalent/check circle.
Singular values of the check maps below `pinv_rtol` times the largest are dropped;
the default (1e-14 in Float64, 6e-6 in Float32) is where they become rounding noise.
"""
function KernelIndependent(kernel; nequiv::Integer=50, pinv_rtol::Real=50 * eps(floattype(kernel)))
    nequiv >= 4 || throw(ArgumentError("nequiv must be at least 4"))
    return KernelIndependent(kernel, Int(nequiv), Float64(pinv_rtol))
end

const KI_SMALL_RADIUS = sqrt(2) / 2
const KI_LARGE_RADIUS = (4 - sqrt(2)) / 2

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

struct KIExtras{T,R}
    small::Vector{SVector{2,R}}
    large::Vector{SVector{2,R}}
    P_up::FactoredPinv{T,R}     # check potentials on large → density on small
    P_down::FactoredPinv{T,R}   # check data on small (+ pinned pressure) → density on large
    pinv_scale::R               # factor on both pseudo-inverses when the matrices are shared
end

"""
    pins_pressure(K), has_flux_source(K)

Whether the downward check data of `K` carry the pressure at the box center, and
whether its multipoles carry a point-source strength. Both hold for the Stokes kernel.
"""
pins_pressure(K) = false
pins_pressure(::Stokeslet2D) = true
has_flux_source(K) = false
has_flux_source(::Stokeslet2D) = true

"""
    source_velocity(K::Stokeslet2D, s, t)

Velocity at `t` of a unit point mass source at `s`, `d/(2π r²)` (a potential flow,
zero pressure).
"""
@inline function source_velocity(K::Stokeslet2D, s::SVector{2}, t::SVector{2})
    d = t - s
    return (2 * K.c / _sqnorm(d)) * d
end

dimension(::KernelIndependent) = 2
expansion_length(m::KernelIndependent) = ncomp(m.kernel, Val(2)) * m.nequiv
multipole_length(m::KernelIndependent) = expansion_length(m) + (has_flux_source(m.kernel) ? 1 : 0)
partial_local_length(m::KernelIndependent) = expansion_length(m) + (pins_pressure(m.kernel) ? 1 : 0)
near_kernel(m::KernelIndependent) = m.kernel
function operator_eltype(m::KernelIndependent)
    R = floattype(m.kernel)
    return eltype(kernel_eltype(m.kernel, SVector{2,R}, SVector{2,R}))
end

function circle_points(n::Int, r::R) where {R<:AbstractFloat}
    return [r * SVector{2,R}(cos(θ), sin(θ)) for θ in (2 * R(π) / n) .* (0:n-1)]
end

# The augmentations are nondimensionalised with the box width: the pinned pressure
# row is `w·p` and the point-source strength is per unit width (its velocity column
# is `w·v`). Both then scale with the width like the velocity kernel itself, so for
# a homogeneous kernel the whole augmented operator is a scalar multiple of its
# reference copy, and the pin's least-squares weight is the same on every level.

# Row of `w·pressure` at `x` due to unit densities at `sources`, component major.
function _pressure_row(K, sources::AbstractVector{SVector{2,R}}, x::SVector{2,R}, w::R) where {R}
    n = length(sources)
    row = Matrix{R}(undef, 1, 2n)
    for i in 1:n
        p = w * pressure_kernel(K, sources[i], x)
        row[1, i] = p[1]
        row[1, n + i] = p[2]
    end
    return row
end
# Column of velocities at `targets` (component major) due to a point source of
# strength `w` at `c` (i.e. unit strength per unit width).
function _source_column(K, targets::AbstractVector{SVector{2,R}}, c::SVector{2,R}, w::R) where {R}
    n = length(targets)
    col = Matrix{R}(undef, 2n, 1)
    for j in 1:n
        v = w * source_velocity(K, c, targets[j])
        col[j, 1] = v[1]
        col[n + j, 1] = v[2]
    end
    return col
end
# Velocities on `targets` produced by the multipole representation of a box of width
# `w`: a density on `sources` and, for the Stokes kernel, a point source at `center`.
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

function level_operators(m::KernelIndependent, w::Real)
    K = m.kernel
    N = m.nequiv
    R = floattype(K)
    w = R(w)
    small = circle_points(N, R(KI_SMALL_RADIUS) * w)
    large = circle_points(N, R(KI_LARGE_RADIUS) * w)
    origin = zero(SVector{2,R})
    P_up = pinv_rtol(_rep_velocity_matrix(K, small, origin, w, large), m.pinv_rtol)
    P_down = pinv_rtol(_local_to_check_matrix(K, large, small, origin, w), m.pinv_rtol)
    # children: width w/2, centers at (δ - 1/2)·w/2 relative to the parent
    child_small = circle_points(N, R(KI_SMALL_RADIUS) * w / 2)
    centers = [(SVector{2,R}(δ) .- R(0.5)) * (w / 2) for δ in child_offsets(Val(2))]
    # one block per child, so that the stacked children multipoles (each component
    # major within its child) match the operator's columns
    M2M = reduce(hcat, [_rep_velocity_matrix(K, child_small .+ Ref(c), c, w / 2, large) for c in centers])
    L2L = reduce(vcat, [_local_to_check_matrix(K, large, child_small .+ Ref(c), c, w / 2) for c in centers])
    T = eltype(M2M)
    M2L = [M2LOperator(_multipole_to_check_matrix(K, small .+ Ref(SVector{2,R}(off) * w), SVector{2,R}(off) * w, w, small, origin))
           for off in far_offsets(Val(2))]
    extra = KIExtras{T,R}(small, large, P_up, P_down, one(R))
    return LevelOperators(w, M2M, L2L, M2L, extra)
end

function rescale(ref::LevelOperators{T,R,KIExtras{T,R}}, m::KernelIndependent, w::Real) where {T,R}
    a = homogeneity(m.kernel)
    a === nothing && throw(ArgumentError("rescale needs a homogeneous kernel"))
    λ = R(w) / ref.width
    f = λ^a
    e = ref.extra
    extra = KIExtras{T,R}(e.small .* λ, e.large .* λ, e.P_up, e.P_down, e.pinv_scale / f)
    return LevelOperators{T,R,KIExtras{T,R}}(R(w), ref.M2M, ref.L2L, ref.M2L,
                                            ref.m2m_scale * f, ref.l2l_scale * f, ref.m2l_scale * f, extra)
end

"""
    conditioning(ops) -> (σmin/σmax up, σmin/σmax down, dropped up, dropped down)

Diagnostics of the check maps of a level: smallest-to-largest singular value ratios
and how many modes the truncation dropped.
"""
function conditioning(ops::LevelOperators{T,R,KIExtras{T,R}}) where {T,R}
    e = ops.extra
    su, sd = e.P_up.sigma, e.P_down.sigma
    return (su[end] / su[1], sd[end] / sd[1], length(su) - rank(e.P_up), length(sd) - rank(e.P_down))
end

# component-major storage of a field with `nc` components at point `k` of `n`
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
the large circle (relative to the center `c`).
"""
function s2m!(col::AbstractVector, m::KernelIndependent, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{2})
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(2)))
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
partial_to_multipole!(M::AbstractVecOrMat, m::KernelIndependent, ops::LevelOperators, pM::AbstractVecOrMat, work) =
    apply!(M, ops.extra.P_up, pM, work, ops.extra.pinv_scale)
partial_to_local!(L::AbstractVecOrMat, m::KernelIndependent, ops::LevelOperators, pL::AbstractVecOrMat, work) =
    apply!(L, ops.extra.P_down, pL, work, ops.extra.pinv_scale)

"""
    l2t(m, ops, L, xrel, out=Potential())

Output `out` of the local expansion `L` (density on the large circle) at `xrel`
relative to the box center.
"""
@inline function l2t(m::KernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Potential=Potential())
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(2)))
    S = promote_type(eltype(L), eltype(kernel_eltype(K, eltype(large), typeof(xrel))))
    acc = _field_zero(S, nc)
    @inbounds for k in 1:N
        acc += K(large[k], xrel) * _load(L, k, N, nc)
    end
    return acc
end
@inline function l2t(m::KernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, out::Gradient)
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(2)))
    S = promote_type(eltype(L), eltype(kernel_eltype(K, eltype(large), typeof(xrel))))
    acc = zero(Base.promote_op(target_field, typeof(K), eltype(large), typeof(xrel), typeof(_load(L, 1, N, nc)), typeof(out)))
    @inbounds for k in 1:N
        acc += target_field(K, large[k], xrel, _load(L, k, N, nc), out)
    end
    return acc
end
@inline function l2t(m::KernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Pressure)
    K = m.kernel
    large = ops.extra.large
    N = length(large)
    nc = Val(ncomp(K, Val(2)))
    S = promote_type(eltype(L), eltype(kernel_eltype(K, eltype(large), typeof(xrel))))
    acc = zero(S)
    @inbounds for k in 1:N
        acc += sum(pressure_kernel(K, large[k], xrel) .* _load(L, k, N, nc))
    end
    return acc
end

"""
    multipole_field(m, ops, M, xrel)

Far field of the multipole `M` (density on the small circle) at `xrel`; used by
tests.
"""
function multipole_field(m::KernelIndependent, ops::LevelOperators, M::AbstractVector, xrel::SVector{2})
    K = m.kernel
    small = ops.extra.small
    N = length(small)
    nc = Val(ncomp(K, Val(2)))
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
