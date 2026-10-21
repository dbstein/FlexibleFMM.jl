# Timing harness. Run from the package root with
#     julia --project=. -t auto bench/bench.jl [N]
# (the package environment stacks on the default one, which has Chairmarks), or from
# a dedicated environment after `Pkg.develop(path=".")`.
#
# Reports, per phase, milliseconds and thousands of points per second per core
# (cores = half the hardware threads, as in the Python README), for Laplace with the
# kernel-independent (nequiv = 60) and black-box (p = 12) methods and for Helmholtz
# with the kernel-independent method, on uniform, circle and clustered sources.
using FlexibleFMM
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

kpsc(n, t) = n / t / CORES / 1000

function run_case(label, m, pts, τ, targets; ppl)
    N = length(pts); Nt = length(targets)
    # warm-up on a small problem (compilation)
    fw = FMM(m, pts[1:min(N, 4000)]; ppl=ppl, bbox=(P2(0, 0), 1.0))
    sw = build_expansions(fw, pts[1:min(N, 4000)], τ[1:min(N, 4000)])
    evaluate_at_sources(fw, sw); evaluate(fw, sw, targets[1:1000])
    cache = fw.cache
    t_tree = @elapsed f = FMM(m, pts; ppl=ppl, bbox=(P2(0, 0), 1.0), cache=cache)
    st = FlexibleFMM.FMMState(f, FlexibleFMM.expansion_eltype(m, Float64, eltype(τ)), Float64, eltype(τ))
    t_build = @elapsed build_expansions!(st, f, pts, τ)
    t_self = @elapsed us = evaluate_at_sources(f, st)
    t_targ = @elapsed ut = evaluate(f, st, targets)
    @printf "%-28s N=%-8d tree %7.1f ms | expansions %7.1f ms (%6.0f kpts/s/core) | self eval %7.1f ms (%6.0f) | target eval %7.1f ms (%6.0f), %d levels\n" label N 1e3t_tree 1e3t_build kpsc(N, t_build) 1e3t_self kpsc(N, t_self) 1e3t_targ kpsc(Nt, t_targ) FlexibleFMM.nlevels(f)
    return nothing
end

function main(N)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES)
    targets = uniform(rng, 1_000_000)
    for (dname, gen) in (("uniform", uniform), ("circle", circle), ("clustered", clustered))
        pts = gen(rng, N)
        τ = randn(rng, N)
        run_case("Laplace KI60 / $dname", KernelIndependent(Laplace2D(); nequiv=60), pts, τ, targets; ppl=50)
        run_case("Laplace BB12 / $dname", BlackBox(Laplace2D(); p=12), pts, τ, targets; ppl=50)
        run_case("Helmholtz k=10 KI60 / $dname", KernelIndependent(Helmholtz2D(10.0); nequiv=60), pts, τ, targets; ppl=50)
    end
end

main(isempty(ARGS) ? 100_000 : parse(Int, ARGS[1]))
