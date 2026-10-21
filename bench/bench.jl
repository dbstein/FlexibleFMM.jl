# Timing harness. Run from the package root with
#     julia --project=. -t auto bench/bench.jl [N] [ppl,ppl,...] [dist,dist,...]
# e.g.  julia --project=. -t auto bench/bench.jl 1000000 25,50,100,200 uniform
# (the package environment stacks on the default one, which has Chairmarks), or from
# a dedicated environment after `Pkg.develop(path=".")`. Defaults: N = 100000,
# ppl = 50, all of uniform, circle, clustered.
#
# `ppl` is the maximum number of points per leaf: a node with more than `ppl` points
# is split. It trades near-field work (∝ ppl per target) against the number of boxes
# (∝ N/ppl) and hence the translation work; the optimum depends on the machine, the
# method and the expansion order.
#
# Reports, per phase, milliseconds and thousands of points per second per core
# (cores = half the hardware threads, as in the Python README), for Laplace with the
# kernel-independent (nequiv = 60) and black-box (p = 12) methods and for Helmholtz
# with the kernel-independent method.
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
    @printf "%-30s ppl=%-4d N=%-8d %2d levels | tree %7.1f | up %7.1f | down %7.1f | expansions %7.1f ms (%5.0f kpts/s/core) | self eval %7.1f ms (%5.0f) | target eval %7.1f ms (%5.0f)\n" label ppl N nlevels(f) 1e3t_tree 1e3t_up 1e3t_down 1e3(t_up + t_down) kpsc(N, t_up + t_down) 1e3t_self kpsc(N, t_self) 1e3t_targ kpsc(Nt, t_targ)
    return nothing
end

function main(N, ppls, dists)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES)
    targets = uniform(rng, 1_000_000)
    for dname in dists
        pts = GENERATORS[dname](rng, N)
        τ = randn(rng, N)
        for ppl in ppls
            run_case("Laplace KI60 / $dname", KernelIndependent(Laplace2D(); nequiv=60), pts, τ, targets; ppl=ppl)
            run_case("Laplace BB12 / $dname", BlackBox(Laplace2D(); p=12), pts, τ, targets; ppl=ppl)
            run_case("Helmholtz k=10 KI60 / $dname", KernelIndependent(Helmholtz2D(10.0); nequiv=60), pts, τ, targets; ppl=ppl)
        end
    end
end

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
ppls = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [50]
dists = length(ARGS) >= 3 ? String.(split(ARGS[3], ",")) : ["uniform", "circle", "clustered"]
main(N, ppls, dists)
