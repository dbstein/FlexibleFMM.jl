# Timing harness. Run from the package root (the package environment stacks on the
# default one) with key=value arguments, all optional:
#
#     julia --project=. -t auto bench/bench.jl N=1000000 ppl=25,50,100 \
#           dist=uniform,circle kernel=laplace,helmholtz:10,yukawa:5 method=ki:60,square:12,bb:12
#     julia --project=. -t auto bench/bench.jl dim=3 N=100000 kernel=laplace,stokes method=cube:8
#     julia --project=bench/gpu -t auto bench/bench.jl dim=3 N=100000 ppl=200 \
#           kernel=laplace,stokes method=cube:8 device=metal
#     julia --project=bench/cuda -t auto bench/bench.jl dim=3 N=100000 ppl=200 \
#           kernel=laplace,stokes method=cube:8,harmonics:16:1e-6 device=cuda precision=float32
#
# Defaults: dim=2, N=100000, ppl=50, dist=uniform,circle,clustered (sphere for the circle in
# 3D), kernel=laplace,helmholtz:10, method=ki:60,bb:12 (cube:8 in 3D). The kernel, method
# and distribution specifications are those of common.jl. device=metal or device=cuda adds a
# row per case with the device matvec (DevicePlan + device_matvec!) and the device
# evaluation at the targets (their location on the CPU excluded, as it is reusable),
# in the precision of precision=float32|float64 (Float32 on Metal by default, Float64
# elsewhere: a card with weak Float64 wants precision=float32), for the
# kernel-independent and harmonic methods. Needs Metal.jl or CUDA.jl in the
# environment: bench/gpu has Metal, bench/cuda has CUDA (instantiate it on the
# machine), or add them to the default environment, on which the package
# environment stacks. device=ka_cpu runs
# the same device passes through KernelAbstractions' CPU backend (Float64, Julia
# threads, no package needed): the device code path without a device.
#
# `ppl` is the maximum number of points per leaf: a node with more than `ppl` points
# is split. It trades near-field work (∝ ppl per target) against the number of boxes
# (∝ N/ppl) and hence the translation work; the optimum depends on the machine, the
# method and the expansion order.
#
# Phases, in milliseconds: tree (FMM construction: tree and operator lookup; built
# once), first (the first application of a fresh state: it creates the method's
# workspaces, so it is reported apart), then up and down (the expansion passes) and
# self (evaluation at the sources), each the best of three warm repetitions. The
# "matvec" total is up + down + self, the cost of one iteration of a solver that
# applies the operator to new densities on the same points; "with tree" adds the
# construction. Target evaluation at 10⁶ uniform targets is timed separately.
# Rates are thousands of points per second per core, with cores = half the hardware
# threads as in the Python README.
using FlexibleFMM: upward_pass!, downward_pass!, FMMState, expansion_eltype, nlevels
using Printf
include(joinpath(@__DIR__, "common.jl"))

const CORES = max(1, Sys.CPU_THREADS ÷ 2)

kpsc(n, t) = n / t / CORES / 1000

function run_case(label, m, pts, τ, targets; ppl, to=nothing)
    N = length(pts); Nt = length(targets)
    T = eltype(eltype(pts))
    nw = min(N, 4000)
    box = (zero(eltype(pts)), one(T))
    fw = FMM(m, pts[1:nw]; ppl=ppl, bbox=box)                    # warm-up (compilation)
    sw = build_expansions(fw, pts[1:nw], τ[1:nw])
    evaluate_at_sources(fw, sw); evaluate(fw, sw, targets[1:1000])
    cache = fw.cache
    t_tree = @elapsed f = FMM(m, pts; ppl=ppl, bbox=box, cache=cache)
    st = FMMState(f, expansion_eltype(m, T, eltype(τ)), T, eltype(τ))
    perm = f.tree.perm
    st.sources .= pts[perm]; st.τ .= τ[perm]
    # the first application of a fresh state (it creates the method's workspaces, the
    # spectral one of the cube method in particular), then the warm passes, best of three
    t_first = @elapsed (upward_pass!(st, f); downward_pass!(st, f); evaluate_at_sources(f, st))
    t_up = minimum(@elapsed(upward_pass!(st, f)) for _ in 1:3)
    t_down = minimum(@elapsed(downward_pass!(st, f)) for _ in 1:3)
    local us, ut
    t_self = minimum(@elapsed(us = evaluate_at_sources(f, st)) for _ in 1:3)
    t_targ = minimum(@elapsed(ut = evaluate(f, st, targets)) for _ in 1:3)
    t_mv = t_up + t_down + t_self
    @printf "%-36s ppl=%-4d N=%-8d %2d lev | tree %7.1f | first %7.1f | up %7.1f | down %7.1f | self %7.1f | matvec %7.1f ms (%5.0f kpts/s/core) | with tree %7.1f ms (%5.0f) | targets %7.1f ms (%5.0f)\n" label ppl N nlevels(f) 1e3t_tree 1e3t_first 1e3t_up 1e3t_down 1e3t_self 1e3t_mv kpsc(N, t_mv) 1e3(t_tree + t_mv) kpsc(N, t_tree + t_mv) 1e3t_targ kpsc(Nt, t_targ)
    if to !== nothing && m isa Union{FlexibleFMM.AbstractKernelIndependent,FlexibleFMM.HarmonicMethod}   # the device passes
        plan = DevicePlan(f, to)
        ds = (sources=to(st.sources), τ=to(st.τ))
        outd = to(similar(us))
        device_matvec!(outd, plan, ds)
        t_dmv = minimum(@elapsed(device_matvec!(outd, plan, ds)) for _ in 1:3)
        tgd = to(targets); nodes = to(Int32.(locate(f.tree, targets))); outt = to(similar(ut))
        dsl = (locals=plan.locals, sources=ds.sources, τ=ds.τ)
        evaluate!(outt, plan.dev, dsl, tgd, nodes)
        t_dtg = minimum(@elapsed(evaluate!(outt, plan.dev, dsl, tgd, nodes)) for _ in 1:3)
        err = maximum(norm.(Array(outd) .- us)) / maximum(norm.(us))
        @printf "%-36s %s device: matvec %7.1f ms (%6.0f kpts/s) | targets %7.1f ms (%6.0f kpts/s) | vs CPU %.1e\n" label repeat(" ", 25) 1e3t_dmv N / t_dmv / 1000 1e3t_dtg Nt / t_dtg / 1000 err
    end
    return nothing
end

function main(dim, N, ppls, dists, kernels, methods, device, to, T)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES,
            ", dimension: ", dim, ", precision: ", T, device == "cpu" ? "" : ", device: $device")
    targets = SVector{dim,T}.(uniform(rng, 1_000_000, Val(dim)))
    for dname in dists
        pts = SVector{dim,T}.(GENERATORS[dname](rng, N, Val(dim)))
        for ppl in ppls, kspec in kernels, mspec in methods
            kname, K = parse_kernel(kspec, Val(dim), T)
            mname, m = parse_method(mspec, K, Val(dim))
            m === nothing && continue                            # a method not for this kernel
            τ = density(rng, kernel_kind(kspec), K, N, Val(dim))
            run_case("$kname $mname / $dname", m, pts, τ, targets; ppl=ppl, to=to)
        end
    end
end

args = parse_kv(ARGS)
dim = parse(Int, get(args, "dim", "2"))
dim in (2, 3) || error("dim must be 2 or 3")
N = parse(Int, get(args, "N", "100000"))
ppls = parse.(Int, split(get(args, "ppl", "50"), ","))
dists = String.(split(get(args, "dist", "uniform,circle,clustered"), ","))
kernels = String.(split(get(args, "kernel", "laplace,helmholtz:10"), ","))
methods = String.(split(get(args, "method", dim == 2 ? "ki:60,bb:12" : "cube:8"), ","))
device = get(args, "device", "cpu")
device in ("cpu", "ka_cpu", "metal", "cuda") || error("device must be cpu, ka_cpu, metal or cuda")
# the device package is loaded at top level, in its own expression, so that the
# bindings are visible to what follows
if device == "metal"
    using Metal
elseif device == "cuda"
    using CUDA
end
to = device == "metal" ? MtlArray : device == "cuda" ? CuArray : device == "ka_cpu" ? identity : nothing
precision = get(args, "precision", device == "metal" ? "float32" : "float64")
precision in ("float32", "float64") || error("precision must be float32 or float64")
T = precision == "float32" ? Float32 : Float64
main(dim, N, ppls, dists, kernels, methods, device, to, T)
