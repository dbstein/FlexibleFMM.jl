# Source and target types.
#
# A source is a point with a density; the density's type says what kind of
# singularity sits there. `source_field(K, s, t, τ)` is the field at `t` of the
# source `(s, τ)`, and is all that `s2m!`, the near field and the reference sums
# need. An output type selects what is evaluated at a target:
# `target_field(K, s, t, τ, out)` composes the source with the output operator.
# Kernel derivatives come from `gradient_t`/`hessian_t`, hand-written for the
# built-in kernels and taken by ForwardDiff on the kernel otherwise. All kernels are
# translation invariant, so `gradient_s = -gradient_t`.
#
# Adding a singularity type (a Stokes rotlet, say) means a struct, a `source_field`
# method, and, for the black-box method, an `accumulate_moments!` method giving its
# action on the Chebyshev basis.

"""
    Dipole(d)
    Dipole(μ, n)

A dipole source with moment vector `d` (strength `μ` times direction `n`). Its field
is `d·∇ₛK(s, t)`, the double-layer kernel for `n` a unit normal.
"""
struct Dipole{D,T<:Number}
    d::SVector{D,T}
end
Dipole(μ::Number, n::SVector{D}) where {D} = Dipole(μ * n)
Base.zero(::Type{Dipole{D,T}}) where {D,T} = Dipole(zero(SVector{D,T}))
Base.:+(a::Dipole, b::Dipole) = Dipole(a.d + b.d)
Base.:*(a::Number, b::Dipole) = Dipole(a * b.d)

"""
    Potential()  (alias `Velocity()`)
    Gradient()
    Pressure()

Output types: the potential (velocity, for a Stokes kernel), its gradient (an
`SVector` for scalar kernels, the velocity-gradient `SMatrix` for Stokes), or the
pressure (Stokes only).
"""
struct Potential end
struct Gradient end
struct Pressure end
const Velocity = Potential

# ---- Stokes singularities (2D formulas; d = t - s, r = |d|, c = 1/4π) ----
#   stokeslet   f (a plain SVector):   u = c(-log r·f + d(d·f)/r²),         p = 2c(d·f)/r²
#   Stresslet(f, ν):  u = -(1/π) d (d·f)(d·ν)/r⁴,  p = (1/π)[(f·ν)/r² - 2(d·f)(d·ν)/r⁴]
#   Doublet(g):  u = c(g/r² - 2d(d·g)/r⁴) = ½∇²G·g (potential dipole), p = 0
#   Rotlet(ω):   u = 2cω(-d₂, d₁)/r² = -ω ε_{jk}∂_{s_k}G_{ij},            p = 0
# The stresslet and doublet are the 2D analogues of JellySim's -3/(4π) and 1/(8π)
# kernels; μ is carried by the densities.

"""
    Stresslet(f, ν)

A Stokes stresslet (double-layer source) with strength `f` and direction `ν`.
"""
struct Stresslet{D,T<:Number}
    f::SVector{D,T}
    ν::SVector{D,T}
end
Stresslet(f::SVector{D}, ν::SVector{D}) where {D} = (T = promote_type(eltype(f), eltype(ν)); Stresslet{D,T}(f, ν))

"""
    Doublet(g)

A potential dipole (source doublet) with moment `g`, the `O(ε²)` term of slender
body theory.
"""
struct Doublet{D,T<:Number}
    g::SVector{D,T}
end

"""
    Rotlet(ω)

A point torque `ω` (2D: a scalar, 3D: an `SVector{3}`).
"""
struct Rotlet{W}
    ω::W
end

"""
    density_scalar(T)

The scalar type a density of type `T` carries (the eltype of its expansions).
"""
density_scalar(::Type{T}) where {T<:Number} = T
density_scalar(::Type{Dipole{D,T}}) where {D,T} = T
density_scalar(::Type{SVector{D,T}}) where {D,T} = T
density_scalar(::Type{Stresslet{D,T}}) where {D,T} = T
density_scalar(::Type{Doublet{D,T}}) where {D,T} = T
density_scalar(::Type{Rotlet{W}}) where {W} = density_scalar(W)
density_scalar(::Type{T}) where {T<:Tuple} = promote_type(map(density_scalar, fieldtypes(T))...)

# ---- kernel derivatives: generic fallbacks by ForwardDiff on the kernel ----

@inline _is_complex(K, s, t) = kernel_eltype(K, typeof(s), typeof(t)) <: Complex

"""
    gradient_t(K, s, t) -> SVector
    gradient_s(K, s, t) -> SVector
    hessian_t(K, s, t)  -> SMatrix

Derivatives of `K(s, t)` with respect to the target (or source) point. Default:
ForwardDiff on the kernel (real and imaginary parts separately for complex kernels);
built-in kernels override them.
"""
function gradient_t(K, s::SVector{D}, t::SVector{D}) where {D}
    if _is_complex(K, s, t)
        gr = ForwardDiff.gradient(tt -> real(K(s, tt)), t)
        gi = ForwardDiff.gradient(tt -> imag(K(s, tt)), t)
        return gr + im * gi
    else
        return ForwardDiff.gradient(tt -> K(s, tt), t)
    end
end
@inline gradient_s(K, s::SVector{D}, t::SVector{D}) where {D} = -gradient_t(K, s, t)
function hessian_t(K, s::SVector{D}, t::SVector{D}) where {D}
    if _is_complex(K, s, t)
        hr = ForwardDiff.jacobian(tt -> real.(gradient_t(K, s, tt)), t)
        hi = ForwardDiff.jacobian(tt -> imag.(gradient_t(K, s, tt)), t)
        return hr + im * hi
    else
        return ForwardDiff.jacobian(tt -> gradient_t(K, s, tt), t)
    end
end

# ---- fields ----

"""
    source_field(K, s, t, τ)

Field at `t` of the source at `s` with density `τ`: `K(s,t)·τ` for a charge,
`d·∇ₛK(s,t)` for a dipole, the sum for a tuple of densities.
"""
@inline source_field(K, s::SVector, t::SVector, τ::Number) = K(s, t) * τ
@inline source_field(K, s::SVector, t::SVector, d::Dipole) = sum(d.d .* gradient_s(K, s, t))
@inline source_field(K, s::SVector, t::SVector, τ::Tuple) = _sum_fields(K, s, t, τ...)
@inline _sum_fields(K, s, t, a) = source_field(K, s, t, a)
@inline _sum_fields(K, s, t, a, rest...) = source_field(K, s, t, a) + _sum_fields(K, s, t, rest...)

"""
    target_field(K, s, t, τ, out)

`out` applied at `t` to the field of the source `(s, τ)`.
"""
@inline target_field(K, s::SVector, t::SVector, τ, ::Potential) = source_field(K, s, t, τ)
@inline target_field(K, s::SVector, t::SVector, τ::Number, ::Gradient) = gradient_t(K, s, t) * τ
# ∇ₜ(d·∇ₛK) = -∇ₜ(d·∇ₜK) = -Hₜ d
@inline target_field(K, s::SVector, t::SVector, d::Dipole, ::Gradient) = -(hessian_t(K, s, t) * d.d)
@inline target_field(K, s::SVector, t::SVector, τ::Tuple, out::Potential) = _sum_targets(K, s, t, out, τ...)
@inline target_field(K, s::SVector, t::SVector, τ::Tuple, out::Gradient) = _sum_targets(K, s, t, out, τ...)
@inline _sum_targets(K, s, t, out, a) = target_field(K, s, t, a, out)
@inline _sum_targets(K, s, t, out, a, rest...) = target_field(K, s, t, a, out) + _sum_targets(K, s, t, out, rest...)

# ---- Stokes fields ----

@inline source_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, f::SVector{2}) = K(s, t) * f
@inline function source_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Stresslet{2})
    d = t - s
    r2 = _sqnorm(d)
    return (-(4 * K.c) * sum(d .* τ.f) * sum(d .* τ.ν) / (r2 * r2)) * d
end
@inline function source_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Doublet{2})
    d = t - s
    r2 = _sqnorm(d)
    g = τ.g
    return (K.c / r2) * (g - (2 * sum(d .* g) / r2) * d)
end
@inline function source_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Rotlet)
    d = t - s
    return (2 * K.c * τ.ω / _sqnorm(d)) * SVector(-d[2], d[1])
end

@inline target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, f::SVector{2}, ::Pressure) =
    sum(pressure_kernel(K, s, t) .* f)
@inline function target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Stresslet{2}, ::Pressure)
    d = t - s
    r2 = _sqnorm(d)
    return (4 * K.c) * (sum(τ.f .* τ.ν) / r2 - 2 * sum(d .* τ.f) * sum(d .* τ.ν) / (r2 * r2))
end
@inline target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Doublet{2}, ::Pressure) =
    zero(promote_type(eltype(s), eltype(t), density_scalar(typeof(τ)), typeof(K.c)))
@inline target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Rotlet, ::Pressure) =
    zero(promote_type(eltype(s), eltype(t), density_scalar(typeof(τ)), typeof(K.c)))
@inline target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Tuple, out::Pressure) =
    _sum_targets(K, s, t, out, τ...)

# velocity gradient ∂u_a/∂t_j: hand-written for the stokeslet, ForwardDiff otherwise
@inline function target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, f::SVector{2}, ::Gradient)
    d = t - s
    r2 = _sqnorm(d)
    df = sum(d .* f)
    b = K.c / r2
    I2 = one(SMatrix{2,2,typeof(b)})
    return b * (df * I2 + d * transpose(f) - f * transpose(d)) - (2b * df / r2) * (d * transpose(d))
end
@inline function target_field(K::Stokeslet2D, s::SVector{2}, t::SVector{2}, τ::Union{Stresslet{2},Doublet{2},Rotlet}, ::Gradient)
    return ForwardDiff.jacobian(tt -> source_field(K, s, tt, τ), t)
end

# ---- Stokes fields, 3D (JellySim's conventions; d = t - s, r = |d|, c = 1/8π) ----
#   stokeslet f:      u = c(f/r + d(d·f)/r³),                      p = 2c(d·f)/r³
#   Stresslet(f, ν):  u = -6c d (d·f)(d·ν)/r⁵  (JellySim's -3/4π),  p = 4c[(f·ν)/r³ - 3(d·f)(d·ν)/r⁵]
#   Doublet(g):       u = c(g/r³ - 3d(d·g)/r⁵) = ½∇²G·g,            p = 0
#   Rotlet(ω):        u = c ω×d/r³ = ½ ε_{jkl} ω_j ∂_{s_k} G_{il},   p = 0
const Rotlet3 = Rotlet{<:SVector{3}}

@policied @inline function source_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, f::SVector{3}) where {T}
    d = t - s                                   # c(f/r + d(d·f)/r³) directly: 12 products
    ir = _rinv(policy(K), _sqnorm(d))           # against 27 through the 3 × 3 matrix (1.15× on the near field)
    a = K.c * ir
    return a * f + (a * ir * ir * _dot(d, f)) * d
end
@policied @inline function source_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, τ::Stresslet{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    return (-(6 * K.c) * _dot(d, τ.f) * _dot(d, τ.ν) * ir^5) * d
end
@policied @inline function source_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, τ::Doublet{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    g = τ.g
    return (K.c * ir^3) * (g - (3 * _dot(d, g) * ir * ir) * d)
end
@policied @inline function source_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, τ::Rotlet3) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    return (K.c * ir^3) * cross(τ.ω, d)
end

@inline target_field(K::Stokeslet3D, s::SVector{3}, t::SVector{3}, f::SVector{3}, ::Pressure) =
    sum(pressure_kernel(K, s, t) .* f)
@policied @inline function target_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, τ::Stresslet{3}, ::Pressure) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    ir3 = ir^3
    return (4 * K.c) * (_dot(τ.f, τ.ν) * ir3 - 3 * _dot(d, τ.f) * _dot(d, τ.ν) * ir3 * ir * ir)
end
@inline target_field(K::Stokeslet3D, s::SVector{3}, t::SVector{3}, τ::Union{Doublet{3},Rotlet3}, ::Pressure) =
    zero(promote_type(eltype(s), eltype(t), density_scalar(typeof(τ)), typeof(K.c)))
@inline target_field(K::Stokeslet3D, s::SVector{3}, t::SVector{3}, τ::Tuple, out::Pressure) =
    _sum_targets(K, s, t, out, τ...)

@policied @inline function target_field(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}, f::SVector{3}, ::Gradient) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    df = _dot(d, f)
    b = K.c * ir^3
    I3 = one(SMatrix{3,3,typeof(b)})
    return b * (df * I3 + d * transpose(f) - f * transpose(d)) - (3b * df * ir * ir) * (d * transpose(d))
end
@inline function target_field(K::Stokeslet3D, s::SVector{3}, t::SVector{3}, τ::Union{Stresslet{3},Doublet{3},Rotlet3}, ::Gradient)
    return ForwardDiff.jacobian(tt -> source_field(K, s, tt, τ), t)
end
