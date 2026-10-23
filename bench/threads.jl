# Threading diagnostics for a machine: the near-field loop on one core (rate and
# vectorisation), the cost of the primitives the passes are made of (thread regions, compute and allocating tasks, the small GEMMs of the
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
# spinning and removes the wake-ups from the picture (then OPENBLAS_NUM_THREADS=1 as
# well: spinning Julia threads starve OpenBLAS's own, and with one BLAS thread the
# passes split the large products over Julia threads instead).
using FlexibleFMM, StaticArrays, LinearAlgebra, Random, Printf, InteractiveUtils
using FlexibleFMM: upward_pass!, downward_pass!, FMMState
include(joinpath(@__DIR__, "common.jl"))

args = parse_kv(ARGS)
N = parse(Int, get(args, "N", "100000"))
ppl = parse(Int, get(args, "ppl", "600"))
println(config_line())
nt = Threads.nthreads()

# 0. the near-field loop on one core: its rate, and whether the compiler vectorised
#    it for this CPU (vector float types in its LLVM IR; the M3 Max: 110 lines,
#    720 M Stokeslet evaluations per second)
let rng = MersenneTwister(2), n = 600, K = Stokeslet3D(Float32)
    src = [rand(rng, SVector{3,Float32}) for _ in 1:n]
    τ = [randn(rng, SVector{3,Float32}) for _ in 1:n]
    x = rand(rng, SVector{3,Float32})
    near(K, src, τ, x) = (acc = zero(SVector{3,Float32}); @inbounds for j in eachindex(src); acc += FlexibleFMM.source_field(K, src[j], x, τ[j]); end; acc)
    near(K, src, τ, x)
    ir = sprint(code_llvm, near, Tuple{typeof(K),typeof(src),typeof(τ),typeof(x)})
    lines = split(ir, '\n')
    nvec = count(l -> occursin(r"<\d+ x float>", l), lines)
    # a loop vectorised across sources has a vector.body block and vector square roots
    # (a vectorised body with only 3- or 4-wide vectors is the per-point arithmetic)
    vbody = count(l -> occursin("vector.body", l), lines)
    vsqrt = [m.match for m in eachmatch(r"llvm\.sqrt\.v\d+f32|fdiv (fast )?<\d+ x float>|sqrt <\d+ x float>", ir)]
    reps = 2000
    tt = minimum(@elapsed(for _ in 1:reps; near(K, src, τ, x); end) for _ in 1:5)
    @printf "near field, one core, Float32 Stokeslet: %.0f M evaluations/s, %d vector float lines in the loop's IR, vector.body %d, vector sqrt/div: %s\n" reps * n / tt / 1e6 nvec vbody join(unique(vsqrt), " ")
    # the same loop on structure-of-arrays storage (one vector per coordinate and
    # density component): unit-stride loads instead of the stride-3 loads of a vector
    # of points, which ARM loads with one instruction (ld3) and x86 with shuffles
    X, Y, Z = [getindex.(src, k) for k in 1:3]
    FX, FY, FZ = [getindex.(τ, k) for k in 1:3]
    near_soa(K, X, Y, Z, FX, FY, FZ, x) = (acc = zero(SVector{3,Float32}); @inbounds @simd for j in eachindex(X)
        acc += FlexibleFMM.source_field(K, SVector(X[j], Y[j], Z[j]), x, SVector(FX[j], FY[j], FZ[j])); end; acc)
    near_soa(K, X, Y, Z, FX, FY, FZ, x) ≈ near(K, src, τ, x) || error("the two loops disagree")
    ts = minimum(@elapsed(for _ in 1:reps; near_soa(K, X, Y, Z, FX, FY, FZ, x); end) for _ in 1:5)
    @printf "near field, one core, the same on structure-of-arrays storage: %.0f M evaluations/s\n" reps * n / ts / 1e6
end

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
st.sources[1] .= pts[fm.tree.perm]; st.τ[1] .= τ[fm.tree.perm]
upward_pass!(st, fm); downward_pass!(st, fm); evaluate_at_sources(fm, st)
t_up = minimum(@elapsed(upward_pass!(st, fm)) for _ in 1:3)
t_down = minimum(@elapsed(downward_pass!(st, fm)) for _ in 1:3)
t_self = minimum(@elapsed(evaluate_at_sources(fm, st)) for _ in 1:3)
nreg = sum(2 * length(fm.ops[l].extra.reps) for l in 3:FlexibleFMM.nlevels(fm))   # regions of the class translations
@printf "Stokes harmonics p=6, N=%d, ppl=%d, %d levels, %d threads: up %.3f s, down %.3f s (about %d thread regions), self %.3f s\n" N ppl FlexibleFMM.nlevels(fm) nt t_up t_down nreg t_self
