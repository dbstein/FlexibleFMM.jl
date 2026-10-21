# Kernels are callables `K(source, target)` on `SVector`s returning a scalar.
# `kernel_matrix(K, sources, targets)[j, i] = K(sources[i], targets[j])`, so rows are
# targets and `A * τ` is the potential. All kernels here are translation invariant,
# which the width-cached translation operators require.

abstract type AbstractKernel end

"""
    floattype(K)

The floating-point precision a kernel's constants are stored in, and hence the
precision of the operators built from it. `Float64` unless a kernel says otherwise.
"""
floattype(::AbstractKernel) = Float64

@inline _sqnorm(d::SVector) = sum(abs2, d)

"""
    Laplace2D(T=Float64)

Free-space Laplace Green's function `-log|t - s| / 2π`, with constants in precision `T`.
"""
struct Laplace2D{T<:AbstractFloat} <: AbstractKernel
    c::T       # -1/(4π): the kernel is c·log(r²)
end
Laplace2D(::Type{T}=Float64) where {T<:AbstractFloat} = Laplace2D{T}(-1 / (4 * T(π)))
floattype(::Laplace2D{T}) where {T} = T
@inline (K::Laplace2D)(s::SVector{2}, t::SVector{2}) = K.c * log(_sqnorm(t - s))

"""
    Helmholtz2D(k)

Helmholtz Green's function `(i/4) H₀⁽¹⁾(k |t - s|)` (complex valued), written as
`(i/4)(J₀ + i·Y₀)` so that it keeps the precision of `k` (`Float32` included).
"""
struct Helmholtz2D{T<:AbstractFloat} <: AbstractKernel
    k::T
    c::Complex{T}
end
Helmholtz2D(k::Real) = (T = typeof(float(k)); Helmholtz2D{T}(T(k), Complex{T}(0, 1 // 4)))
floattype(::Helmholtz2D{T}) where {T} = T
@inline function (K::Helmholtz2D)(s::SVector{2}, t::SVector{2})
    kr = K.k * sqrt(_sqnorm(t - s))
    return K.c * (besselj0(kr) + im * bessely0(kr))
end

"""
    Yukawa2D(k)

Modified Helmholtz Green's function `K₀(k |t - s|) / 2π` in the precision of `k`
(`besselk` computes in Float64; the result is converted back).
"""
struct Yukawa2D{T<:AbstractFloat} <: AbstractKernel
    k::T
    c::T       # 1/(2π)
end
Yukawa2D(k::Real) = (T = typeof(float(k)); Yukawa2D{T}(T(k), 1 / (2 * T(π))))
floattype(::Yukawa2D{T}) where {T} = T
@inline function (K::Yukawa2D)(s::SVector{2}, t::SVector{2})
    kr = K.k * sqrt(_sqnorm(t - s))
    return K.c * oftype(kr, besselk(0, kr))
end

"""
    LaplaceDipoleTest(c)

`-log r / 2π + c (t₁ - s₁) / (2π r²)`, `r = |t - s|`: the field of a unit charge plus an
`x₁`-dipole of strength `c` placed at the source. It is harmonic in both arguments and
translation invariant, so every expansion scheme here is exact for it, but it is not
symmetric under `s ↔ t`. Radial kernels hide any transposed operator; this one does
not, which is its only purpose.
"""
struct LaplaceDipoleTest{T<:AbstractFloat} <: AbstractKernel
    c::T
    a::T       # -1/(4π)
    b::T       # 1/(2π)
end
LaplaceDipoleTest(c::Real) = (T = typeof(float(c)); LaplaceDipoleTest{T}(T(c), -1 / (4 * T(π)), 1 / (2 * T(π))))
floattype(::LaplaceDipoleTest{T}) where {T} = T
@inline function (K::LaplaceDipoleTest)(s::SVector{2}, t::SVector{2})
    d = t - s
    r2 = _sqnorm(d)
    return K.a * log(r2) + K.c * K.b * d[1] / r2
end

"""
    kernel_eltype(K, S, T)

Type returned by `K(::S, ::T)`.
"""
kernel_eltype(K, ::Type{S}, ::Type{T}) where {S,T} = Base.promote_op(K, S, T)

"""
    kernel_matrix(K, sources, targets) -> Matrix

`A[j, i] = K(sources[i], targets[j])`: rows are targets, columns are sources.
"""
function kernel_matrix(K, sources::AbstractVector, targets::AbstractVector)
    S = kernel_eltype(K, eltype(sources), eltype(targets))
    A = Matrix{S}(undef, length(targets), length(sources))
    for i in eachindex(sources)
        s = sources[i]
        for j in eachindex(targets)
            A[j, i] = K(s, targets[j])
        end
    end
    return A
end

# Accumulates Σ_{i ≠ skip} K(sources[i], t) τ[i] onto `acc`.
@inline function _direct_target(K, sources, τ, t, skip::Int, acc)
    @inbounds for i in eachindex(sources)
        i == skip && continue
        acc += K(sources[i], t) * τ[i]
    end
    return acc
end

"""
    direct!(out, K, sources, τ, targets)

`out[j] = Σ_i K(sources[i], targets[j]) τ[i]`: the O(N²) reference sum, threaded over
targets.
"""
function direct!(out::AbstractVector, K, sources::AbstractVector, τ::AbstractVector,
                 targets::AbstractVector)
    length(sources) == length(τ) || throw(DimensionMismatch("sources and τ differ in length"))
    length(out) == length(targets) || throw(DimensionMismatch("out and targets differ in length"))
    z = zero(eltype(out))
    Threads.@threads :dynamic for j in eachindex(targets)
        @inbounds out[j] = _direct_target(K, sources, τ, targets[j], 0, z)
    end
    return out
end

"""
    direct_self!(out, K, sources, τ)

`out[j] = Σ_{i ≠ j} K(sources[i], sources[j]) τ[i]`.
"""
function direct_self!(out::AbstractVector, K, sources::AbstractVector, τ::AbstractVector)
    length(sources) == length(τ) || throw(DimensionMismatch("sources and τ differ in length"))
    length(out) == length(sources) || throw(DimensionMismatch("out and sources differ in length"))
    z = zero(eltype(out))
    Threads.@threads :dynamic for j in eachindex(sources)
        @inbounds out[j] = _direct_target(K, sources, τ, sources[j], j, z)
    end
    return out
end

function _potential_eltype(K, sources, τ, targets)
    return promote_type(kernel_eltype(K, eltype(sources), eltype(targets)), eltype(τ))
end

"""
    direct(K, sources, τ, targets)
    direct_self(K, sources, τ)

Allocating versions of `direct!` and `direct_self!`.
"""
function direct(K, sources::AbstractVector, τ::AbstractVector, targets::AbstractVector)
    out = Vector{_potential_eltype(K, sources, τ, targets)}(undef, length(targets))
    return direct!(out, K, sources, τ, targets)
end
function direct_self(K, sources::AbstractVector, τ::AbstractVector)
    out = Vector{_potential_eltype(K, sources, τ, sources)}(undef, length(sources))
    return direct_self!(out, K, sources, τ)
end
