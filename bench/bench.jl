# Timing harness. Run from the package root (the package environment stacks on the
# default one) with key=value arguments, all optional:
#
#     julia --project=. -t auto bench/bench.jl N=1000000 ppl=25,50,100 \
#           dist=uniform,circle kernel=laplace,helmholtz:10,yukawa:5 method=ki:60,square:12,bb:12
#
# Defaults: N=100000, ppl=50, dist=uniform,circle,clustered, kernel=laplace,helmholtz:10,
# method=ki:60,bb:12. Kernels: laplace, helmholtz:k, yukawa:k, stokes (stokeslets with
# random vector densities). Methods: ki:nequiv (kernel-independent on circles), square:p or
# square:p:d (kernel-independent on squares, p points per edge, margin d, FFT translations),
# bb:p (black-box).
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

const P2 = SVector{2,Float64}
const CORES = max(1, Sys.CPU_THREADS ÷ 2)

uniform(rng, N) = [P2(rand(rng), rand(rng)) for _ in 1:N]
circle(rng, N) = [0.5 .+ 0.4 * P2(cos(t), sin(t)) for t in 2π .* rand(rng, N)]
function clustered(rng, N)
    centers = [P2(0.1 + 0.8rand(rng), 0.1 + 0.8rand(rng)) for _ in 1:10]
    return [centers[1 + (i - 1) % 10] + 0.01 * P2(randn(rng), randn(rng)) for i in 1:N]
end
const GENERATORS = Dict("uniform" => uniform, "circle" => circle, "clustered" => clustered)

function parse_kernel(s)
    name, rest = occursin(":", s) ? split(s, ":") : (s, "")
    name == "laplace" && return ("Laplace", Laplace2D())
    name == "helmholtz" && return ("Helmholtz k=$rest", Helmholtz2D(parse(Float64, rest)))
    name == "yukawa" && return ("Yukawa k=$rest", Yukawa2D(parse(Float64, rest)))
    name == "stokes" && return ("Stokes", Stokeslet2D())
    error("unknown kernel $s (laplace, helmholtz:k, yukawa:k, stokes)")
end
function parse_method(s, K)
    parts = split(s, ":")
    name, rest = parts[1], parts[2]
    name == "ki" && return ("KI$rest", KernelIndependent(K; nequiv=parse(Int, rest)))
    if name == "square"
        d = length(parts) > 2 ? parse(Float64, parts[3]) : 0.1
        return ("Square$rest d=$d", SquareKernelIndependent(K; p=parse(Int, rest), d=d))
    end
    name == "bb" && return ("BB$rest", BlackBox(K; p=parse(Int, rest)))
    error("unknown method $s (ki:nequiv, square:p[:d], bb:p)")
end
density(rng, K, N) = randn(rng, N)
density(rng, ::Stokeslet2D, N) = [P2(randn(rng), randn(rng)) for _ in 1:N]

kpsc(n, t) = n / t / CORES / 1000

function run_case(label, m, pts, τ, targets; ppl)
    N = length(pts); Nt = length(targets)
    nw = min(N, 4000)
    fw = FMM(m, pts[1:nw]; ppl=ppl, bbox=(P2(0, 0), 1.0))        # warm-up (compilation)
    sw = build_expansions(fw, pts[1:nw], τ[1:nw])
    evaluate_at_sources(fw, sw); evaluate(fw, sw, targets[1:1000])
    cache = fw.cache
    t_tree = @elapsed f = FMM(m, pts; ppl=ppl, bbox=(P2(0, 0), 1.0), cache=cache)
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

function main(N, ppls, dists, kernels, methods)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES)
    targets = uniform(rng, 1_000_000)
    for dname in dists
        pts = GENERATORS[dname](rng, N)
        for ppl in ppls, kspec in kernels, mspec in methods
            kname, K = parse_kernel(kspec)
            mname, m = parse_method(mspec, K)
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
N = parse(Int, get(args, "N", "100000"))
ppls = parse.(Int, split(get(args, "ppl", "50"), ","))
dists = String.(split(get(args, "dist", "uniform,circle,clustered"), ","))
kernels = String.(split(get(args, "kernel", "laplace,helmholtz:10"), ","))
methods = String.(split(get(args, "method", "ki:60,bb:12"), ","))
main(N, ppls, dists, kernels, methods)
