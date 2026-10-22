# Shared by bench.jl and compare/compare.jl: key=value arguments, point
# distributions, and the kernel and method specifications.
#
# Kernels: laplace, helmholtz:k, yukawa:k, stokes (stokeslets with random vector
# densities), stresslet (stresslets with random strengths and unit directions),
# each in the dimension given and in the precision asked for. Methods: ki:nequiv
# (kernel-independent on circles, 2D), square:p or cube:p, optionally :d
# (kernel-independent on squares/cubes, p points per edge, margin d, FFT
# translations, pc = 2p), bb:p (black-box, 2D), harmonics:p, optionally :tol (analytic
# solid-harmonic expansions, Laplace, Yukawa and Stokes 3D; tol > 0 for low-rank translations). Distributions: uniform, circle
# (sphere in 3D; both names work), clustered, helix (3D, a jittered helix: the fibre case).
using FlexibleFMM, StaticArrays, Random, LinearAlgebra

parse_kv(args) = Dict{String,String}(String(a[1]) => String(a[2]) for a in split.(args, "="))

uniform(rng, N, ::Val{D}) where {D} = [rand(rng, SVector{D,Float64}) for _ in 1:N]
circle(rng, N, ::Val{2}) = [0.5 .+ 0.4 * SVector(cos(t), sin(t)) for t in 2π .* rand(rng, N)]
circle(rng, N, ::Val{3}) = [SVector(0.5, 0.5, 0.5) + 0.4 * normalize(randn(rng, SVector{3,Float64})) for _ in 1:N]
function clustered(rng, N, ::Val{D}) where {D}
    centers = [0.1 .+ 0.8 .* rand(rng, SVector{D,Float64}) for _ in 1:10]
    return [centers[1 + (i - 1) % 10] + 0.01 * randn(rng, SVector{D,Float64}) for i in 1:N]
end
# a jittered helix through the cube (the fibre geometry), 3D only
function helix(rng, N, ::Val{3})
    return [SVector(0.5 + 0.3cos(θ), 0.5 + 0.3sin(θ), 0.05 + 0.9θ / (6π)) + 1e-3 * randn(rng, SVector{3,Float64}) for θ in 6π .* rand(rng, N)]
end
helix(rng, N, ::Val{2}) = error("the helix distribution is 3D")
const GENERATORS = Dict("uniform" => uniform, "circle" => circle, "sphere" => circle, "clustered" => clustered, "helix" => helix)

# the kernel of a specification, in precision T; `stresslet` is the Stokes kernel
# with stresslet densities (see `density`)
function parse_kernel(s, ::Val{D}, ::Type{T}) where {D,T}
    name, rest = occursin(":", s) ? split(s, ":") : (s, "")
    k = isempty(rest) ? zero(T) : T(parse(Float64, rest))
    if D == 2
        name == "laplace" && return ("Laplace", Laplace2D(T))
        name == "helmholtz" && return ("Helmholtz k=$rest", Helmholtz2D(k))
        name == "yukawa" && return ("Yukawa k=$rest", Yukawa2D(k))
        name == "stokes" && return ("Stokes", Stokeslet2D(T))
        name == "stresslet" && return ("Stresslets", Stokeslet2D(T))
    else
        name == "laplace" && return ("Laplace", Laplace3D(T))
        name == "helmholtz" && return ("Helmholtz k=$rest", Helmholtz3D(k))
        name == "yukawa" && return ("Yukawa k=$rest", Yukawa3D(k))
        name == "stokes" && return ("Stokes", Stokeslet3D(T))
        name == "stresslet" && return ("Stresslets", Stokeslet3D(T))
    end
    error("unknown kernel $s (laplace, helmholtz:k, yukawa:k, stokes, stresslet)")
end
kernel_kind(s) = Symbol(first(split(s, ":")))

function parse_method(s, K, ::Val{D}) where {D}
    parts = split(s, ":")
    name, rest = parts[1], parts[2]
    if name == "square" || name == "cube"
        d = length(parts) > 2 ? parse(Float64, parts[3]) : 0.1
        return ("$(D == 2 ? "Square" : "Cube")$rest d=$d", CubeKernelIndependent{D}(K; p=parse(Int, rest), d=d))
    end
    if name == "harmonics"
        D == 3 || error("method $s is 3D")
        tol = length(parts) > 2 ? parse(Float64, parts[3]) : 0.0
        label = "Harmonics$rest" * (tol > 0 ? " tol=$tol" : "")
        K isa Union{Laplace3D,Yukawa3D} && return (label, SphericalHarmonics(K; p=parse(Int, rest), tol=tol))
        K isa Stokeslet3D && return (label, StokesHarmonics(K; p=parse(Int, rest), tol=tol))
        return (label, nothing)                                  # not for this kernel: the scripts skip it
    end
    D == 2 || error("method $s is 2D; use cube:p[:d] or harmonics:p in 3D")
    name == "ki" && return ("KI$rest", KernelIndependent(K; nequiv=parse(Int, rest)))
    name == "bb" && return ("BB$rest", BlackBox(K; p=parse(Int, rest)))
    error("unknown method $s (ki:nequiv, square:p[:d] / cube:p[:d], bb:p, harmonics:p[:tol])")
end

# random densities of the kind a kernel specification takes, in the kernel's precision
function density(rng, kind::Symbol, K, N, ::Val{D}) where {D}
    T = FlexibleFMM.floattype(K)
    kind === :stokes && return [randn(rng, SVector{D,T}) for _ in 1:N]
    kind === :stresslet && return [Stresslet(randn(rng, SVector{D,T}), normalize(randn(rng, SVector{D,T}))) for _ in 1:N]
    return randn(rng, T, N)
end
