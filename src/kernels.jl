# Kernels are callables `K(source, target)` on `SVector`s returning a scalar.
# `kernel_matrix(K, sources, targets)[j, i] = K(sources[i], targets[j])`, so rows are
# targets and `A * τ` is the potential. All kernels here are translation invariant,
# which the width-cached translation operators require.

abstract type AbstractKernel end

@inline _sqnorm(d::SVector) = sum(abs2, d)

"""
    Laplace2D()

Free-space Laplace Green's function `-log|t - s| / 2π`.
"""
struct Laplace2D <: AbstractKernel end
@inline function (::Laplace2D)(s::SVector{2}, t::SVector{2})
    return -log(_sqnorm(t - s)) / (4π)
end

"""
    Helmholtz2D(k)

Helmholtz Green's function `(i/4) H₀⁽¹⁾(k |t - s|)` (complex valued).
"""
struct Helmholtz2D{T<:Real} <: AbstractKernel
    k::T
end
@inline function (K::Helmholtz2D)(s::SVector{2}, t::SVector{2})
    return (im / 4) * hankelh1(0, K.k * sqrt(_sqnorm(t - s)))
end

"""
    Yukawa2D(k)

Modified Helmholtz Green's function `K₀(k |t - s|) / 2π`.
"""
struct Yukawa2D{T<:Real} <: AbstractKernel
    k::T
end
@inline function (K::Yukawa2D)(s::SVector{2}, t::SVector{2})
    return besselk(0, K.k * sqrt(_sqnorm(t - s))) / (2π)
end

"""
    LaplaceDipoleTest(c)

`-log r / 2π + c (t₁ - s₁) / (2π r²)`, `r = |t - s|`: the field of a unit charge plus an
`x₁`-dipole of strength `c` placed at the source. It is harmonic in both arguments and
translation invariant, so every expansion scheme here is exact for it, but it is not
symmetric under `s ↔ t`. Radial kernels hide any transposed operator; this one does
not, which is its only purpose.
"""
struct LaplaceDipoleTest{T<:Real} <: AbstractKernel
    c::T
end
@inline function (K::LaplaceDipoleTest)(s::SVector{2}, t::SVector{2})
    d = t - s
    r2 = _sqnorm(d)
    return -log(r2) / (4π) + K.c * d[1] / (2π * r2)
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
