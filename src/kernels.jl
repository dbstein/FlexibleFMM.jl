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
floattype(K) = Float64                               # any callable `K(source, target)`

@inline _sqnorm(d::SVector) = sum(abs2, d)
# fused for 2- and 3-vectors (one rounding per term; the compiler does not fuse on
# its own, and `fma` rather than `muladd` so that every context, vectorised or not,
# rounds alike: `kernel_matrix` and the near field agree bitwise), as is the dot
# product of the kernel bodies
@inline _sqnorm(d::SVector{2}) = fma(d[2], d[2], d[1] * d[1])
@inline _sqnorm(d::SVector{3}) = fma(d[3], d[3], fma(d[2], d[2], d[1] * d[1]))
@inline _dot(a::SVector, b::SVector) = sum(a .* b)
@inline _dot(a::SVector{3}, b::SVector{3}) = fma(a[3], b[3], fma(a[2], b[2], a[1] * b[1]))

"""
    AbstractRadialKernel

A kernel `K(s, t) = scale(K)·f(|t - s|)`. Subtypes define `scale(K)`, `radial_value(K, r)`
and `radial_derivatives(K, r) -> (f, f′, f″)`; evaluation, gradient and Hessian follow.
"""
abstract type AbstractRadialKernel <: AbstractKernel end
@inline function (K::AbstractRadialKernel)(s::SVector{D}, t::SVector{D}) where {D}
    return scale(K) * radial_value(K, sqrt(_sqnorm(t - s)))
end
@inline function gradient_t(K::AbstractRadialKernel, s::SVector{D}, t::SVector{D}) where {D}
    d = t - s
    r = sqrt(_sqnorm(d))
    _, f′, _ = radial_derivatives(K, r)
    return (scale(K) * f′ / r) * d
end
@inline function hessian_t(K::AbstractRadialKernel, s::SVector{D}, t::SVector{D}) where {D}
    d = t - s
    r2 = _sqnorm(d)
    r = sqrt(r2)
    _, f′, f″ = radial_derivatives(K, r)
    c = scale(K)
    a = c * (f″ / r2 - f′ / (r2 * r))
    b = c * f′ / r
    return a * (d * transpose(d)) + b * one(SMatrix{D,D,typeof(b)})
end

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
@inline function gradient_t(K::Laplace2D, s::SVector{2}, t::SVector{2})
    d = t - s
    return (2 * K.c / _sqnorm(d)) * d
end
@inline function hessian_t(K::Laplace2D, s::SVector{2}, t::SVector{2})
    d = t - s
    r2 = _sqnorm(d)
    a = 2 * K.c / r2
    return a * one(SMatrix{2,2,typeof(a)}) - (2a / r2) * (d * transpose(d))
end

"""
    Helmholtz2D(k)

Helmholtz Green's function `(i/4) H₀⁽¹⁾(k |t - s|)` (complex valued), written through
`J₀, Y₀` (and `J₁, Y₁` for derivatives) so that it keeps the precision of `k`.
"""
struct Helmholtz2D{T<:AbstractFloat} <: AbstractRadialKernel
    k::T
    c::Complex{T}
end
Helmholtz2D(k::Real) = (T = typeof(float(k)); Helmholtz2D{T}(T(k), Complex{T}(0, 1 // 4)))
floattype(::Helmholtz2D{T}) where {T} = T
@inline scale(K::Helmholtz2D) = K.c
@inline function radial_value(K::Helmholtz2D, r)
    kr = K.k * r
    return besselj0(kr) + im * bessely0(kr)
end
@inline function radial_derivatives(K::Helmholtz2D, r)
    kr = K.k * r
    h0 = besselj0(kr) + im * bessely0(kr)
    h1 = besselj1(kr) + im * bessely1(kr)
    return h0, -K.k * h1, -K.k^2 * (h0 - h1 / kr)
end

"""
    Yukawa2D(k)

Modified Helmholtz Green's function `K₀(k |t - s|) / 2π` in the precision of `k`
(`besselk` computes in Float64; results are converted back).
"""
struct Yukawa2D{T<:AbstractFloat} <: AbstractRadialKernel
    k::T
    c::T       # 1/(2π)
end
Yukawa2D(k::Real) = (T = typeof(float(k)); Yukawa2D{T}(T(k), 1 / (2 * T(π))))
floattype(::Yukawa2D{T}) where {T} = T
@inline scale(K::Yukawa2D) = K.c
@inline function radial_value(K::Yukawa2D, r)
    kr = K.k * r
    return oftype(kr, besselk(0, kr))
end
@inline function radial_derivatives(K::Yukawa2D, r)
    kr = K.k * r
    k0 = oftype(kr, besselk(0, kr))
    k1 = oftype(kr, besselk(1, kr))
    return k0, -K.k * k1, K.k^2 * (k0 + k1 / kr)
end

# ---- 3D kernels: 1/(4π r) scalar potentials and JellySim's 1/(8π) Stokeslet ----

"""
    PreciseMath, FastMath, fastmath(K)

The arithmetic policy of the Laplace and Stokes 3D kernels' radial part, a type
parameter: `PreciseMath` (`1/sqrt(r²)`, the default) or `FastMath`, the kernel
bodies under `@fastmath` (`@policied`: contraction, reciprocals, the approximate
root; 1.2× on the CPU near field and 1.22–1.25× on the Metal evaluation measured
in Float32, deviations of 2–4e-7). `fastmath(K)` switches a kernel's
policy: `FMM(m, …)` with a method built on `fastmath(K)` relaxes everything the
kernel enters (the near field, the kernel-independent operators and check
potentials, the harmonic S2M and L2T); `DeviceEvaluator`/`DevicePlan` take
`fast=true` to relax the device evaluation alone.
"""
abstract type MathPolicy end
struct PreciseMath <: MathPolicy end
struct FastMath <: MathPolicy end
@inline _rinv(::PreciseMath, r2) = 1 / _sqrt(r2)
@inline _rinv(::FastMath, r2) = @fastmath 1 / sqrt(r2)
# the square root without Julia's domain check: `r2` is a sum of squares, and the
# check's throw inside the near-field loop keeps LLVM from vectorising the loop
# over the sources on x86 (the same instruction, `llvm.sqrt`, otherwise)
@inline _sqrt(x::Union{Float32,Float64}) = Base.sqrt_llvm(x)
@inline _sqrt(x) = sqrt(x)
fastmath(K) = K

"""
    @policied function f(K::Kernel{T,P}, args...) where {T} body end

Defines the method for both math policies from one body: as written for
`K::Kernel{T,PreciseMath}`, and under `@fastmath` for `K::Kernel{T,FastMath}`
(the relaxed arithmetic of the opt-in policy: contraction, reciprocals, the
approximate root; the calls inside the body, `_rinv` among them, keep their own
definitions). The signature names the policy `P` wherever the kernel type appears.
"""
macro policied(def)
    inner = def
    while inner isa Expr && inner.head === :macrocall           # through @inline
        inner = inner.args[end]
    end
    inner isa Expr && inner.head === :function || error("@policied expects a function definition")
    swap(ex, P) = ex === :P ? P : ex isa Expr ? Expr(ex.head, (swap(a, P) for a in ex.args)...) : ex
    rebuild(outer, newinner) = outer === inner ? newinner :
        Expr(outer.head, outer.args[1:end-1]..., rebuild(outer.args[end], newinner))
    sig, body = inner.args[1], inner.args[2]
    precise = rebuild(def, Expr(:function, swap(sig, :PreciseMath), body))
    fast = rebuild(def, Expr(:function, swap(sig, :FastMath), Expr(:block, :(@fastmath $body))))
    return esc(Expr(:block, precise, fast))
end

"""
    Laplace3D(T=Float64)

Free-space Laplace Green's function `1/(4π|t - s|)`, with constants in precision `T`.
"""
struct Laplace3D{T<:AbstractFloat,P<:MathPolicy} <: AbstractKernel
    c::T       # 1/(4π)
end
Laplace3D(::Type{T}=Float64) where {T<:AbstractFloat} = Laplace3D{T,PreciseMath}(1 / (4 * T(π)))
fastmath(K::Laplace3D{T}) where {T} = Laplace3D{T,FastMath}(K.c)
floattype(::Laplace3D{T}) where {T} = T
@inline policy(::Laplace3D{T,P}) where {T,P} = P()
@policied @inline function (K::Laplace3D{T,P})(s::SVector{3}, t::SVector{3}) where {T}
    return K.c * _rinv(policy(K), _sqnorm(t - s))
end
@policied @inline function gradient_t(K::Laplace3D{T,P}, s::SVector{3}, t::SVector{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    return (-K.c * ir * ir * ir) * d
end
@policied @inline function hessian_t(K::Laplace3D{T,P}, s::SVector{3}, t::SVector{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    a = K.c * ir * ir * ir
    return (3 * a * ir * ir) * (d * transpose(d)) - a * one(SMatrix{3,3,typeof(a)})
end
homogeneity(::Laplace3D) = -1

"""
    Helmholtz3D(k)

Helmholtz Green's function `e^{ik|t - s|}/(4π|t - s|)` (complex valued).
"""
struct Helmholtz3D{T<:AbstractFloat} <: AbstractRadialKernel
    k::T
    c::T       # 1/(4π)
end
Helmholtz3D(k::Real) = (T = typeof(float(k)); Helmholtz3D{T}(T(k), 1 / (4 * T(π))))
floattype(::Helmholtz3D{T}) where {T} = T
@inline scale(K::Helmholtz3D) = K.c
@inline radial_value(K::Helmholtz3D, r) = cis(K.k * r) / r
@inline function radial_derivatives(K::Helmholtz3D, r)
    k = K.k
    ir = 1 / r
    ir2 = ir * ir
    e = cis(k * r)
    return e * ir, e * (im * k * ir - ir2), e * (-k * k * ir - 2im * k * ir2 + 2 * ir2 * ir)
end

"""
    Yukawa3D(k)

Modified Helmholtz (Yukawa) Green's function `e^{-k|t - s|}/(4π|t - s|)`.
"""
struct Yukawa3D{T<:AbstractFloat} <: AbstractRadialKernel
    k::T
    c::T       # 1/(4π)
end
Yukawa3D(k::Real) = (T = typeof(float(k)); Yukawa3D{T}(T(k), 1 / (4 * T(π))))
floattype(::Yukawa3D{T}) where {T} = T
@inline scale(K::Yukawa3D) = K.c
@inline radial_value(K::Yukawa3D, r) = exp(-K.k * r) / r
@inline function radial_derivatives(K::Yukawa3D, r)
    k = K.k
    ir = 1 / r
    ir2 = ir * ir
    e = exp(-k * r)
    return e * ir, -e * (k * ir + ir2), e * (k * k * ir + 2 * k * ir2 + 2 * ir2 * ir)
end

"""
    Stokeslet3D(T=Float64)

The 3D Stokeslet `(1/8π)(I + r̂r̂ᵀ)/r`, an `SMatrix{3,3}`, in JellySim's convention:
`K(s,t)·f` is the velocity at `t` of a point force `f` at `s` with unit viscosity (μ
is carried by the densities). Its pressure is `pressure_kernel(K, s, t)·f =
(r̂·f)/(4π r²)`. The singularities built on it (`Stresslet`, `Doublet`, `Rotlet`)
follow JellySim's `-3/(4π)` and `1/(8π)` kernels; see docs/stokes.md.
"""
struct Stokeslet3D{T<:AbstractFloat,P<:MathPolicy} <: AbstractKernel
    c::T       # 1/(8π)
end
Stokeslet3D(::Type{T}=Float64) where {T<:AbstractFloat} = Stokeslet3D{T,PreciseMath}(1 / (8 * T(π)))
fastmath(K::Stokeslet3D{T}) where {T} = Stokeslet3D{T,FastMath}(K.c)
floattype(::Stokeslet3D{T}) where {T} = T
@inline policy(::Stokeslet3D{T,P}) where {T,P} = P()
@policied @inline function (K::Stokeslet3D{T,P})(s::SVector{3}, t::SVector{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    a = K.c * ir
    b = a * ir * ir
    return a * one(SMatrix{3,3,typeof(a)}) + b * (d * transpose(d))
end
@policied @inline function pressure_kernel(K::Stokeslet3D{T,P}, s::SVector{3}, t::SVector{3}) where {T}
    d = t - s
    ir = _rinv(policy(K), _sqnorm(d))
    return (2 * K.c * ir * ir * ir) * d
end
homogeneity(::Stokeslet3D) = -1

"""
    LaplaceDipoleTest(c)

`-log r / 2π + c (t₁ - s₁) / (2π r²)`, `r = |t - s|`: the field of a unit charge plus an
`x₁`-dipole of strength `c` placed at the source. It is harmonic in both arguments and
translation invariant, so every expansion scheme here is exact for it, but it is not
symmetric under `s ↔ t`. Radial kernels hide any transposed operator; this one does
not, which is its only purpose. Its derivatives use the generic fallback.
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
    Stokeslet2D(T=Float64)

The 2D Stokeslet `(1/4π)(-log r·I + r̂r̂ᵀ)`, an `SMatrix{2,2}`, so that `K(s,t)·f` is
the velocity at `t` of a point force `f` at `s` with unit viscosity (as in JellySim,
μ is carried by the density convention). The pressure of that flow is
`pressure_kernel(K, s, t)·f = (r̂·f)/(2π r)`. See docs/stokes.md for the other
singularities (`Stresslet`, `Doublet`, `Rotlet`) built on this kernel.
"""
struct Stokeslet2D{T<:AbstractFloat} <: AbstractKernel
    c::T       # 1/(4π)
end
Stokeslet2D(::Type{T}=Float64) where {T<:AbstractFloat} = Stokeslet2D{T}(1 / (4 * T(π)))
floattype(::Stokeslet2D{T}) where {T} = T
@inline function (K::Stokeslet2D)(s::SVector{2}, t::SVector{2})
    d = t - s
    r2 = _sqnorm(d)
    a = -K.c * log(r2) / 2
    b = K.c / r2
    return SMatrix{2,2}(a + b * d[1] * d[1], b * d[2] * d[1], b * d[1] * d[2], a + b * d[2] * d[2])
end
"""
    pressure_kernel(K::Stokeslet2D, s, t) -> SVector{2}

`p = pressure_kernel(K, s, t)·f` is the pressure at `t` of a point force `f` at `s`.
"""
@inline function pressure_kernel(K::Stokeslet2D, s::SVector{2}, t::SVector{2})
    d = t - s
    return (2 * K.c / _sqnorm(d)) * d
end

"""
    kernel_eltype(K, S, T)

Type returned by `K(::S, ::T)`: a number, or an `SMatrix` for vector kernels.
"""
kernel_eltype(K, ::Type{S}, ::Type{T}) where {S,T} = Base.promote_op(K, S, T)

"""
    ncomp(K, Val(D))

Number of components of the fields of `K` in `D` dimensions: 1 for a scalar kernel,
`m` for an `SMatrix{m,n}`-valued one. The dimension is a `Val` so that the result
constant-folds in the per-point bodies.
"""
ncomp(K, ::Val{D}) where {D} = _ncomp(kernel_eltype(K, SVector{D,floattype(K)}, SVector{D,floattype(K)}))
_ncomp(::Type{<:Number}) = 1
_ncomp(::Type{<:SMatrix{m}}) where {m} = m
# stated outright for the kernels with a math policy (no need to call them)
ncomp(::Laplace3D, ::Val{3}) = 1
ncomp(::Stokeslet3D, ::Val{3}) = 3

"""
    homogeneity(K)

The degree `a` such that `K(λs, λt) = λ^a K(s, t)` for all `λ > 0`, or `nothing` if
the kernel is not homogeneous. For a homogeneous kernel the translation operators of
all box widths are scalar multiples of one reference set, which the operator cache
then shares between levels (no built-in 2D kernel qualifies: they contain logarithms
or Bessel functions; `Laplace3D` and `Stokeslet3D` have degree -1).
"""
homogeneity(K) = nothing

"""
    covariant(K)

Whether the kernel is covariant under the signed permutations of the axes
(`K(Rs, Rt) = R·K(s, t)·Rᵀ`, which for a scalar kernel means invariance), the
property the symmetry classes of the far offsets rest on. True for the radial
kernels and the Stokeslets, false by default: a merely translation-invariant kernel
such as `LaplaceDipoleTest` takes one translation operator per offset on the device.
"""
covariant(K) = false
covariant(::AbstractRadialKernel) = true
covariant(::Laplace2D) = true
covariant(::Laplace3D) = true
covariant(::Stokeslet2D) = true
covariant(::Stokeslet3D) = true

"""
    masks_self(K)

Whether the direct sums may evaluate the kernel at coincident points and discard
the value (the branch-free masked self term), which needs the evaluation to return
rather than throw. True for the built-in kernels; a callable kernel that guards
against `r = 0` takes the skipping loop instead.
"""
masks_self(K) = false
masks_self(::AbstractKernel) = true

"""
    symmetric_kernel(K)

Whether a matrix-valued kernel has `K(s, t)[a, b] == K(s, t)[b, a]` (the Stokeslet),
which lets the spectral translation share the tables of the component pairs.
"""
symmetric_kernel(K) = false
symmetric_kernel(::Stokeslet2D) = true
symmetric_kernel(::Stokeslet3D) = true

"""
    kernel_matrix(K, sources, targets) -> Matrix

`A[j, i] = K(sources[i], targets[j])`: rows are targets, columns are sources. For an
`SMatrix{m,n}`-valued kernel the matrix is `m·nt × n·ns`, component major:
`A[(a-1)nt + j, (b-1)ns + i] = K(sources[i], targets[j])[a, b]`, so a density vector
stores all first components, then all second components, and so on.
"""
function kernel_matrix(K, sources::AbstractVector, targets::AbstractVector)
    S = kernel_eltype(K, eltype(sources), eltype(targets))
    return _kernel_matrix(S, K, sources, targets)
end
function _kernel_matrix(::Type{S}, K, sources, targets) where {S<:Number}
    A = Matrix{S}(undef, length(targets), length(sources))
    for i in eachindex(sources)
        s = sources[i]
        for j in eachindex(targets)
            A[j, i] = K(s, targets[j])
        end
    end
    return A
end
function _kernel_matrix(::Type{S}, K, sources, targets) where {m,n,S<:SMatrix{m,n}}
    ns, nt = length(sources), length(targets)
    A = Matrix{eltype(S)}(undef, m * nt, n * ns)
    for i in eachindex(sources)
        s = sources[i]
        for j in eachindex(targets)
            k = K(s, targets[j])
            for b in 1:n, a in 1:m
                A[(a - 1) * nt + j, (b - 1) * ns + i] = k[a, b]
            end
        end
    end
    return A
end

# The sums over sources accumulate with `@fastmath`: a floating-point reduction can
# only be vectorised across the sources if its additions may be reassociated (x86;
# AArch64 vectorises strict reductions in order, and gains 1.6× from the
# reassociation as well). Nothing else in the kernels is relaxed.
@inline _accumulate(acc, v) = @fastmath acc + v

# Accumulates Σ_{i ≠ skip} target_field(K, sources[i], t, τ[i], out) onto `acc`: the
# excluded term masked (branch-free) when the kernel tolerates coincident points,
# skipped otherwise.
@inline function _direct_target(K, sources, τ, t, skip::Int, acc, out)
    if masks_self(K)
        @inbounds @simd for i in eachindex(sources)
            v = target_field(K, sources[i], t, τ[i], out)
            acc = _accumulate(acc, ifelse(i == skip, zero(v), v))
        end
    else
        @inbounds for i in eachindex(sources)
            i == skip && continue
            acc = _accumulate(acc, target_field(K, sources[i], t, τ[i], out))
        end
    end
    return acc
end

"""
    direct!(out, K, sources, τ, targets; output=Potential())

`out[j] = Σ_i target_field(K, sources[i], targets[j], τ[i], output)`: the O(N²)
reference sum, threaded over targets.
"""
function direct!(out::AbstractVector, K, sources::AbstractVector, τ::AbstractVector,
                 targets::AbstractVector; output=Potential())
    length(sources) == length(τ) || throw(DimensionMismatch("sources and τ differ in length"))
    length(out) == length(targets) || throw(DimensionMismatch("out and targets differ in length"))
    z = zero(eltype(out))
    Threads.@threads :dynamic for j in eachindex(targets)
        @inbounds out[j] = _direct_target(K, sources, τ, targets[j], 0, z, output)
    end
    return out
end

"""
    direct_self!(out, K, sources, τ; output=Potential())

`out[j] = Σ_{i ≠ j} target_field(K, sources[i], sources[j], τ[i], output)`.
"""
function direct_self!(out::AbstractVector, K, sources::AbstractVector, τ::AbstractVector; output=Potential())
    length(sources) == length(τ) || throw(DimensionMismatch("sources and τ differ in length"))
    length(out) == length(sources) || throw(DimensionMismatch("out and sources differ in length"))
    z = zero(eltype(out))
    Threads.@threads :dynamic for j in eachindex(sources)
        @inbounds out[j] = _direct_target(K, sources, τ, sources[j], j, z, output)
    end
    return out
end

function _potential_eltype(K, sources, τ, targets, output)
    return Base.promote_op(target_field, typeof(K), eltype(sources), eltype(targets), eltype(τ), typeof(output))
end

"""
    direct(K, sources, τ, targets; output=Potential())
    direct_self(K, sources, τ; output=Potential())

Allocating versions of `direct!` and `direct_self!`.
"""
function direct(K, sources::AbstractVector, τ::AbstractVector, targets::AbstractVector; output=Potential())
    out = Vector{_potential_eltype(K, sources, τ, targets, output)}(undef, length(targets))
    return direct!(out, K, sources, τ, targets; output=output)
end
function direct_self(K, sources::AbstractVector, τ::AbstractVector; output=Potential())
    out = Vector{_potential_eltype(K, sources, τ, sources, output)}(undef, length(sources))
    return direct_self!(out, K, sources, τ; output=output)
end
