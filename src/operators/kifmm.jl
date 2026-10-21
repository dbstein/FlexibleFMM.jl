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
# Why factored and not as one matrix: mode m of the check map decays like
# (r1/r2)^|m| = 0.547^|m|, so the pseudo-inverse has norm 1/σ_min ≈ 0.547^(-N/2)
# (1e13 at N = 100). A dense pseudo-inverse, or one folded into M2M/M2L/L2L, is formed
# by products whose rounding errors are eps/σ_min in every direction, and the far
# field inherits them (6e-4 at N = 100, measured). In factored form the rounding
# amplified by 1/σ_j stays in mode j, which the evaluation attenuates again by σ_j.
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
    apply!(out, P::FactoredPinv, x, work)

`out = P x` for a vector or matrix `x`, using `work` (at least `rank × size(x, 2)`).
"""
function apply!(out::AbstractVecOrMat, P::FactoredPinv, x::AbstractVecOrMat, work::AbstractVecOrMat)
    r = rank(P)
    n = size(x, 2)
    tmp = x isa AbstractVector ? view(work, 1:r) : view(work, 1:r, 1:n)
    mul!(tmp, P.Uh, x)
    lmul!(Diagonal(P.invS), tmp)
    mul!(out, P.V, tmp)
    return out
end

struct KIExtras{T,R}
    small::Vector{SVector{2,R}}
    large::Vector{SVector{2,R}}
    P_up::FactoredPinv{T,R}     # check potentials on large → density on small
    P_down::FactoredPinv{T,R}   # check potentials on small → density on large
end

expansion_length(m::KernelIndependent) = m.nequiv
near_kernel(m::KernelIndependent) = m.kernel
function operator_eltype(m::KernelIndependent)
    R = floattype(m.kernel)
    return kernel_eltype(m.kernel, SVector{2,R}, SVector{2,R})
end

function circle_points(n::Int, r::R) where {R<:AbstractFloat}
    return [r * SVector{2,R}(cos(θ), sin(θ)) for θ in (2 * R(π) / n) .* (0:n-1)]
end

function level_operators(m::KernelIndependent, w::Real)
    K = m.kernel
    N = m.nequiv
    R = floattype(K)
    w = R(w)
    small = circle_points(N, R(KI_SMALL_RADIUS) * w)
    large = circle_points(N, R(KI_LARGE_RADIUS) * w)
    P_up = pinv_rtol(kernel_matrix(K, small, large), m.pinv_rtol)
    P_down = pinv_rtol(kernel_matrix(K, large, small), m.pinv_rtol)
    # children: width w/2, centers at (δ - 1/2)·w/2 relative to the parent
    child_small = circle_points(N, R(KI_SMALL_RADIUS) * w / 2)
    centers = [(SVector{2,R}(δ) .- R(0.5)) * (w / 2) for δ in child_offsets(Val(2))]
    all_child_small = reduce(vcat, [child_small .+ Ref(c) for c in centers])
    M2M = kernel_matrix(K, all_child_small, large)
    L2L = kernel_matrix(K, large, all_child_small)
    T = eltype(M2M)
    M2L = [M2LOperator(kernel_matrix(K, small .+ Ref(SVector{2,R}(off) * w), small)) for off in far_offsets(Val(2))]
    extra = KIExtras{T,R}(small, large, P_up, P_down)
    return LevelOperators{T,KIExtras{T,R}}(w, M2M, L2L, M2L, extra)
end

"""
    conditioning(ops) -> (σmin/σmax up, σmin/σmax down, dropped up, dropped down)

Diagnostics of the check maps of a level: smallest-to-largest singular value ratios
and how many modes the truncation dropped.
"""
function conditioning(ops::LevelOperators{T,KIExtras{T,R}}) where {T,R}
    e = ops.extra
    su, sd = e.P_up.sigma, e.P_down.sigma
    return (su[end] / su[1], sd[end] / sd[1], length(su) - rank(e.P_up), length(sd) - rank(e.P_down))
end

"""
    s2m!(col, m, ops, xs, τs, c)

Partial multipole of a childless node: the check potentials its sources produce on
the large circle (relative to the center `c`).
"""
function s2m!(col::AbstractVector, m::KernelIndependent, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{2})
    K = m.kernel
    large = ops.extra.large
    @inbounds for k in eachindex(large)
        acc = zero(eltype(col))
        for j in eachindex(xs)
            acc += source_field(K, xs[j] - c, large[k], τs[j])
        end
        col[k] = acc
    end
    return col
end

"""
    partial_to_multipole!(M, m, ops, pM, work)
    partial_to_local!(L, m, ops, pL, work)

Check potentials of a whole level (one column per node) to equivalent densities.
"""
partial_to_multipole!(M::AbstractVecOrMat, m::KernelIndependent, ops::LevelOperators, pM::AbstractVecOrMat, work) =
    apply!(M, ops.extra.P_up, pM, work)
partial_to_local!(L::AbstractVecOrMat, m::KernelIndependent, ops::LevelOperators, pL::AbstractVecOrMat, work) =
    apply!(L, ops.extra.P_down, pL, work)

"""
    l2t(m, ops, L, xrel, out=Potential())

Output `out` of the local expansion `L` (density on the large circle) at `xrel`
relative to the box center.
"""
@inline function l2t(m::KernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Potential=Potential())
    K = m.kernel
    large = ops.extra.large
    acc = zero(promote_type(eltype(L), kernel_eltype(K, eltype(large), typeof(xrel))))
    @inbounds for k in eachindex(large)
        acc += K(large[k], xrel) * L[k]
    end
    return acc
end
@inline function l2t(m::KernelIndependent, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Gradient)
    K = m.kernel
    large = ops.extra.large
    S = promote_type(eltype(L), kernel_eltype(K, eltype(large), typeof(xrel)))
    acc = zero(SVector{2,S})
    @inbounds for k in eachindex(large)
        acc += gradient_t(K, large[k], xrel) * L[k]
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
    acc = zero(promote_type(eltype(M), kernel_eltype(K, eltype(small), typeof(xrel))))
    for k in eachindex(small)
        acc += K(small[k], xrel) * M[k]
    end
    return acc
end
