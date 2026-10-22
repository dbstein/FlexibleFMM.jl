# Kernel-independent FMM with circle surfaces (Ying, Biros & Zorin), 2D.
#
# A box of width w carries two concentric circles with Nequiv equispaced points:
#   small, radius r1 = w√2/2         the box's circumcircle
#   large, radius r2 = w(4-√2)/2     tangent to the small circle of the nearest
#                                    interaction-list box (whose center is 2w away)
# Accuracy: the worst-case translation error (a source at a box corner, a target at
# the facing corner of the nearest interaction-list box) is about (r1/r2)^N = 0.547^N
# with N points per circle: 7e-5, 1e-8, 2e-12 at N = 16, 32, 48, measured; global
# relative errors on uniform data are one to two orders smaller (6e-9 at N = 32).
# Rounding limits the floor to about 1e-14, reached near N = 64.
#
# Two known degeneracies (docs/kifmm.md): for Laplace the monopole check potential
# is -Q·log(r2)/2π, zero when w = 2/(4-√2) ≈ 0.7735; for Helmholtz mode m is singular
# when J_m(k·r1) = 0, first at k·w ≈ 3.40. The SVD truncation reports both.

struct KernelIndependent{K} <: AbstractKernelIndependent
    kernel::K
    nequiv::Int
    pinv_rtol::Float64
end

"""
    KernelIndependent(kernel; nequiv=50, pinv_rtol=50·eps(floattype(kernel)))

The kernel-independent method with `nequiv` points per equivalent/check circle (2D).
Singular values of the check maps below `pinv_rtol` times the largest are dropped;
the default (1e-14 in Float64, 6e-6 in Float32) is where they become rounding noise.
"""
function KernelIndependent(kernel; nequiv::Integer=50, pinv_rtol::Real=50 * eps(floattype(kernel)))
    nequiv >= 4 || throw(ArgumentError("nequiv must be at least 4"))
    return KernelIndependent(kernel, Int(nequiv), Float64(pinv_rtol))
end

const KI_SMALL_RADIUS = sqrt(2) / 2
const KI_LARGE_RADIUS = (4 - sqrt(2)) / 2

dimension(::KernelIndependent) = 2
nsurface(m::KernelIndependent) = m.nequiv
function operators_type(m::KernelIndependent)
    T = operator_eltype(m)
    R = floattype(m.kernel)
    return LevelOperators{T,R,KIExtras{T,R,2},M2LOperator{T}}
end
pins_pressure(::Stokeslet2D) = true
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
# The same augmentations in 3D (a unit mass source has velocity d/(4π r³)); used by the
# cube method, which is the only kernel-independent method in 3D.
pins_pressure(::Stokeslet3D) = true
has_flux_source(::Stokeslet3D) = true
@inline function source_velocity(K::Stokeslet3D, s::SVector{3}, t::SVector{3})
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    return (2 * K.c * ir^3) * d
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
    child_small = circle_points(N, R(KI_SMALL_RADIUS) * w / 2)
    P_up, P_down, M2M, L2L = _ki_dense_operators(m, w, small, large, child_small)
    T = operator_eltype(m)
    origin = zero(SVector{2,R})
    M2L = [M2LOperator(Matrix{T}(_multipole_to_check_matrix(K, small .+ Ref(SVector{2,R}(off) * w), SVector{2,R}(off) * w, w, small, origin)))
           for off in far_offsets(Val(2))]
    extra = KIExtras{T,R,2}(small, large, P_up, P_down, one(R), 0, zero(R), Int[])
    return LevelOperators(w, Matrix{T}(M2M), Matrix{T}(L2L), M2L, extra)::operators_type(m)
end
