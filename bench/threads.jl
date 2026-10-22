# Threading diagnostics for a machine: the cost of the primitives the passes are
# made of (thread regions, compute and allocating tasks, the small GEMMs of the
# class translations), then the phases of one case, all at the thread count Julia
# was started with. Run from the package root:
#
#     julia --project=. -t auto bench/threads.jl            # N=100000 ppl=600 by default
#     JULIA_THREAD_SLEEP_THRESHOLD=infinite julia --project=. -t auto bench/threads.jl
#
# The passes consist of hundreds of short thread regions per matvec (the class
# translations: a gather, a GEMM and a scatter for each of the 34 classes of each
# level), so a machine where waking the threads of a region costs milliseconds
# instead of microseconds (deep idle states, a slow scheduler) is many times slower
# than its cores allow; JULIA_THREAD_SLEEP_THRESHOLD=infinite keeps idle threads
# spinning and removes the wake-ups from the picture.
using FlexibleFMM, StaticArrays, LinearAlgebra, Random, Printf
using FlexibleFMM: upward_pass!, downward_pass!, FMMState
include(joinpath(@__DIR__, "common.jl"))

args = parse_kv(ARGS)
N = parse(Int, get(args, "N", "100000"))
ppl = parse(Int, get(args, "ppl", "600"))
println(config_line())
nt = Threads.nthreads()

# 1. thread regions with no work: spawning the tasks and waking the threads
region() = Threads.@threads :dynamic for i in 1:nt; end
region(); t = @elapsed for _ in 1:200; region(); end
@printf "empty thread region: %8.1f µs each (200 regions)\n" 1e6t / 200

# 2. equal compute per thread
f(n) = (s = 0.0; for i in 1:n; s += sin(i); end; s)
f(10); t1 = @elapsed f(10^7)
tn = @elapsed Threads.@threads :dynamic for k in 1:nt; f(10^7); end
@printf "compute: %.3f s on one thread, %.3f s for %d threads each doing the same (ideal: equal)\n" t1 tn nt

# 3. tasks that allocate: the garbage collector with many threads
g(n) = (s = 0.0; for i in 1:n; v = [Float64(i)]; s += v[1]; end; s)
g(10); t1 = @elapsed g(10^6)
tn = @elapsed Threads.@threads :dynamic for k in 1:nt; g(10^6); end
@printf "allocating: %.3f s on one thread, %.3f s for %d threads each doing the same\n" t1 tn nt

# 4. the GEMM of a class translation at p = 6 (49 × 49 applied to 40000 columns)
A = rand(Float32, 49, 49); X = rand(Float32, 49, 40000); Y = similar(X)
nb0 = BLAS.get_num_threads()
mul!(Y, A, X); t = @elapsed for _ in 1:100; mul!(Y, A, X); end
@printf "GEMM 49×49 by 49×40000: %8.2f ms each with %d BLAS threads\n" 1e3t / 100 nb0
BLAS.set_num_threads(1); mul!(Y, A, X); t = @elapsed for _ in 1:100; mul!(Y, A, X); end
@printf "GEMM 49×49 by 49×40000: %8.2f ms each with 1 BLAS thread\n" 1e3t / 100
BLAS.set_num_threads(nb0)

# 5. the phases of the Stokes harmonic method at p = 6 on sphere points, Float32
rng = MersenneTwister(1)
pts = SVector{3,Float32}.(GENERATORS["sphere"](rng, N, Val(3)))
τ = [randn(rng, SVector{3,Float32}) for _ in 1:N]
m = StokesHarmonics(Stokeslet3D(Float32); p=6)
fm = FMM(m, pts; ppl=ppl, bbox=(zero(eltype(pts)), 1f0))
st = FMMState(fm, Float32, Float32, SVector{3,Float32})
st.sources .= pts[fm.tree.perm]; st.τ .= τ[fm.tree.perm]
upward_pass!(st, fm); downward_pass!(st, fm); evaluate_at_sources(fm, st)
t_up = minimum(@elapsed(upward_pass!(st, fm)) for _ in 1:3)
t_down = minimum(@elapsed(downward_pass!(st, fm)) for _ in 1:3)
t_self = minimum(@elapsed(evaluate_at_sources(fm, st)) for _ in 1:3)
nreg = sum(2 * length(fm.ops[l].extra.reps) for l in 3:FlexibleFMM.nlevels(fm))   # regions of the class translations
@printf "Stokes harmonics p=6, N=%d, ppl=%d, %d levels, %d threads: up %.3f s, down %.3f s (about %d thread regions), self %.3f s\n" N ppl FlexibleFMM.nlevels(fm) nt t_up t_down nreg t_self
