# Timing harness. Run from the package root (the package environment stacks on the
# default one) with key=value arguments, all optional:
#
#     julia --project=. -t auto bench/bench.jl N=1000000 ppl=25,50,100 \
#           dist=uniform,circle kernel=laplace,helmholtz:10,yukawa:5 method=ki:60,square:12,bb:12
#     julia --project=. -t auto bench/bench.jl dim=3 N=100000 kernel=laplace,stokes method=cube:8
#
# Defaults: dim=2, N=100000, ppl=50, dist=uniform,circle,clustered (sphere for the circle in
# 3D), kernel=laplace,helmholtz:10, method=ki:60,bb:12 (cube:8 in 3D). Kernels: laplace,
# helmholtz:k, yukawa:k, stokes (stokeslets with random vector densities), each in the
# dimension given. Methods: ki:nequiv (kernel-independent on circles, 2D), square:p or
# cube:p, optionally :d (kernel-independent on squares/cubes, p points per edge, margin d,
# FFT translations, pc = 2p), bb:p (black-box, 2D).
#
# `ppl` is the maximum number of points per leaf: a node with more than `ppl` points
# is split. It trades near-field work (∝ ppl per target) against the number of boxes
# (∝ N/ppl) and hence the translation work; the optimum depends on the machine, the
# method and the expansion order.
#
# Phases, in milliseconds: tree (FMM construction: tree and operator lookup; built
# once), up and down (the expansion passes), self (evaluation at the sources). The
# "matvec" total is up + down + self, the cost of one iteration of a solver that
# applies the operator to new densities on the same points; "with tree" adds the
# construction. Target evaluation at 10⁶ uniform targets is timed separately.
# Rates are thousands of points per second per core, with cores = half the hardware
# threads as in the Python README.
using FlexibleFMM
using FlexibleFMM: upward_pass!, downward_pass!, FMMState, expansion_eltype, nlevels
using StaticArrays
using Random
using Printf
using LinearAlgebra

const CORES = max(1, Sys.CPU_THREADS ÷ 2)

# point distributions in the unit square / cube
uniform(rng, N, ::Val{D}) where {D} = [rand(rng, SVector{D,Float64}) for _ in 1:N]
circle(rng, N, ::Val{2}) = [0.5 .+ 0.4 * SVector(cos(t), sin(t)) for t in 2π .* rand(rng, N)]
circle(rng, N, ::Val{3}) = [SVector(0.5, 0.5, 0.5) + 0.4 * normalize(randn(rng, SVector{3,Float64})) for _ in 1:N]
function clustered(rng, N, ::Val{D}) where {D}
    centers = [0.1 .+ 0.8 .* rand(rng, SVector{D,Float64}) for _ in 1:10]
    return [centers[1 + (i - 1) % 10] + 0.01 * randn(rng, SVector{D,Float64}) for i in 1:N]
end
const GENERATORS = Dict("uniform" => uniform, "circle" => circle, "sphere" => circle, "clustered" => clustered)

function parse_kernel(s, ::Val{D}) where {D}
    name, rest = occursin(":", s) ? split(s, ":") : (s, "")
    k = isempty(rest) ? 0.0 : parse(Float64, rest)
    if D == 2
        name == "laplace" && return ("Laplace", Laplace2D())
        name == "helmholtz" && return ("Helmholtz k=$rest", Helmholtz2D(k))
        name == "yukawa" && return ("Yukawa k=$rest", Yukawa2D(k))
        name == "stokes" && return ("Stokes", Stokeslet2D())
    else
        name == "laplace" && return ("Laplace", Laplace3D())
        name == "helmholtz" && return ("Helmholtz k=$rest", Helmholtz3D(k))
        name == "yukawa" && return ("Yukawa k=$rest", Yukawa3D(k))
        name == "stokes" && return ("Stokes", Stokeslet3D())
    end
    error("unknown kernel $s (laplace, helmholtz:k, yukawa:k, stokes)")
end
function parse_method(s, K, ::Val{D}) where {D}
    parts = split(s, ":")
    name, rest = parts[1], parts[2]
    if name == "square" || name == "cube"
        d = length(parts) > 2 ? parse(Float64, parts[3]) : 0.1
        return ("$(D == 2 ? "Square" : "Cube")$rest d=$d", CubeKernelIndependent{D}(K; p=parse(Int, rest), d=d))
    end
    D == 2 || error("method $s is 2D; use cube:p[:d] in 3D")
    name == "ki" && return ("KI$rest", KernelIndependent(K; nequiv=parse(Int, rest)))
    name == "bb" && return ("BB$rest", BlackBox(K; p=parse(Int, rest)))
    error("unknown method $s (ki:nequiv, square:p[:d] / cube:p[:d], bb:p)")
end
density(rng, K, N) = randn(rng, N)
density(rng, ::Stokeslet2D, N) = [randn(rng, SVector{2,Float64}) for _ in 1:N]
density(rng, ::Stokeslet3D, N) = [randn(rng, SVector{3,Float64}) for _ in 1:N]

kpsc(n, t) = n / t / CORES / 1000

function run_case(label, m, pts, τ, targets; ppl)
    N = length(pts); Nt = length(targets)
    nw = min(N, 4000)
    box = (zero(eltype(pts)), 1.0)
    fw = FMM(m, pts[1:nw]; ppl=ppl, bbox=box)                    # warm-up (compilation)
    sw = build_expansions(fw, pts[1:nw], τ[1:nw])
    evaluate_at_sources(fw, sw); evaluate(fw, sw, targets[1:1000])
    cache = fw.cache
    t_tree = @elapsed f = FMM(m, pts; ppl=ppl, bbox=box, cache=cache)
    st = FMMState(f, expansion_eltype(m, Float64, eltype(τ)), Float64, eltype(τ))
    perm = f.tree.perm
    st.sources .= pts[perm]; st.τ .= τ[perm]
    t_up = @elapsed upward_pass!(st, f)
    t_down = @elapsed downward_pass!(st, f)
    t_self = @elapsed evaluate_at_sources(f, st)
    t_targ = @elapsed evaluate(f, st, targets)
    t_mv = t_up + t_down + t_self
    @printf "%-36s ppl=%-4d N=%-8d %2d lev | tree %7.1f | up %7.1f | down %7.1f | self %7.1f | matvec %7.1f ms (%5.0f kpts/s/core) | with tree %7.1f ms (%5.0f) | targets %7.1f ms (%5.0f)\n" label ppl N nlevels(f) 1e3t_tree 1e3t_up 1e3t_down 1e3t_self 1e3t_mv kpsc(N, t_mv) 1e3(t_tree + t_mv) kpsc(N, t_tree + t_mv) 1e3t_targ kpsc(Nt, t_targ)
    return nothing
end

function main(dim, N, ppls, dists, kernels, methods)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES, ", dimension: ", dim)
    targets = uniform(rng, 1_000_000, Val(dim))
    for dname in dists
        pts = GENERATORS[dname](rng, N, Val(dim))
        for ppl in ppls, kspec in kernels, mspec in methods
            kname, K = parse_kernel(kspec, Val(dim))
            mname, m = parse_method(mspec, K, Val(dim))
            τ = density(rng, K, N)
            run_case("$kname $mname / $dname", m, pts, τ, targets; ppl=ppl)
        end
    end
end

args = Dict{String,String}()
for a in ARGS
    k, v = split(a, "=")
    args[k] = v
end
dim = parse(Int, get(args, "dim", "2"))
dim in (2, 3) || error("dim must be 2 or 3")
N = parse(Int, get(args, "N", "100000"))
ppls = parse.(Int, split(get(args, "ppl", "50"), ","))
dists = String.(split(get(args, "dist", "uniform,circle,clustered"), ","))
kernels = String.(split(get(args, "kernel", "laplace,helmholtz:10"), ","))
methods = String.(split(get(args, "method", dim == 2 ? "ki:60,bb:12" : "cube:8"), ","))
main(dim, N, ppls, dists, kernels, methods)
