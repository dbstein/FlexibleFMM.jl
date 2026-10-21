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
    Potential()
    Gradient()

Output types: the potential `u`, or its gradient `∇u` as an `SVector`.
"""
struct Potential end
struct Gradient end

"""
    density_scalar(T)

The scalar type a density of type `T` carries (the eltype of its expansions).
"""
density_scalar(::Type{T}) where {T<:Number} = T
density_scalar(::Type{Dipole{D,T}}) where {D,T} = T
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
