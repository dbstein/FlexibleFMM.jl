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
# and distribution specifications are those of common.jl. device=metal or device=cuda runs
# the device matvec (DevicePlan + device_matvec!) instead of the CPU passes,
# in the precision of precision=float32|float64 (Float32 on Metal by default, Float64
# elsewhere: a card with weak Float64 wants precision=float32), for the
# kernel-independent and harmonic methods; with a device the CPU passes do not run,
# and the device matvec is checked against Float64 direct sums at nerr=500 of the
# sources (nerr=0 skips the check); the plan build's stages are reported. Needs Metal.jl or CUDA.jl in the
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
# construction. Evaluation at separate targets is not timed: it is not the usual
# use of an FMM, and its cost is the near field of the leaves the targets land in.
# Rates are thousands of points per second per core, with cores = half the hardware
# threads as in the Python README.
using FlexibleFMM: upward_pass!, downward_pass!, FMMState, expansion_eltype, nlevels, near_kernel
using Printf
include(joinpath(@__DIR__, "common.jl"))

const CORES = max(1, Sys.CPU_THREADS ÷ 2)

kpsc(n, t) = n / t / CORES / 1000

# Σ_{j ≠ i} over all sources at the sources `sub`, threaded (the device rows' reference)
function direct_subset(K, pts, τ, sub)
    out = Vector{typeof(FlexibleFMM.source_field(K, pts[1], pts[2], τ[1]))}(undef, length(sub))
    Threads.@threads for q in eachindex(sub)
        i = sub[q]
        acc = zero(eltype(out))
        @inbounds for j in eachindex(pts)
            j == i && continue
            acc += FlexibleFMM.source_field(K, pts[j], pts[i], τ[j])
        end
        out[q] = acc
    end
    return out
end

# densities in the run's precision from the Float64 ones (the reference keeps those)
cvt(::Type{T}, x::Number) where {T} = T(x)
cvt(::Type{T}, x::Complex) where {T} = Complex{T}(x)
cvt(::Type{T}, x::SVector{D}) where {T,D} = SVector{D,T}(x)
cvt(::Type{T}, x::Stresslet) where {T} = Stresslet(cvt(T, x.f), cvt(T, x.ν))
cvt(::Type{T}, x::Doublet) where {T} = Doublet(cvt(T, x.g))
cvt(::Type{T}, x::Tuple) where {T} = map(y -> cvt(T, y), x)

function run_case(label, m, pts, τ; ppl, to=nothing, nerr=500, K64=nothing, pts64=nothing, τ64=nothing)
    N = length(pts)
    T = eltype(eltype(pts))
    nw = min(N, 4000)
    box = (zero(eltype(pts)), one(T))
    fw = FMM(m, pts[1:nw]; ppl=ppl, bbox=box)                    # warm-up (compilation)
    sw = build_expansions(fw, pts[1:nw], τ[1:nw])
    usw = evaluate_at_sources(fw, sw)
    if to !== nothing
        pw = DevicePlan(fw, to)
        device_matvec!(to(similar(usw)), pw, (sources=to(sw.sources), τ=to(sw.τ)))
    end
    cache = fw.cache
    t_tree = @elapsed f = FMM(m, pts; ppl=ppl, bbox=box, cache=cache)
    perm = f.tree.perm
    if to === nothing
        st = FMMState(f, expansion_eltype(m, T, eltype(τ)), T, eltype(τ))
        st.sources .= pts[perm]; st.τ .= τ[perm]
        # the first application of a fresh state (it creates the method's workspaces, the
        # spectral one of the cube method in particular), then the warm passes, best of three
        t_first = @elapsed (upward_pass!(st, f); downward_pass!(st, f); evaluate_at_sources(f, st))
        t_up = minimum(@elapsed(upward_pass!(st, f)) for _ in 1:3)
        t_down = minimum(@elapsed(downward_pass!(st, f)) for _ in 1:3)
        t_self = minimum(@elapsed(evaluate_at_sources(f, st)) for _ in 1:3)
        t_mv = t_up + t_down + t_self
        @printf "%-36s ppl=%-4d N=%-8d %2d lev | tree %7.1f | first %7.1f | up %7.1f | down %7.1f | self %7.1f | matvec %7.1f ms (%5.0f kpts/s/core) | with tree %7.1f ms (%5.0f)\n" label ppl N nlevels(f) 1e3t_tree 1e3t_first 1e3t_up 1e3t_down 1e3t_self 1e3t_mv kpsc(N, t_mv) 1e3(t_tree + t_mv) kpsc(N, t_tree + t_mv)
    else
        # the device only: the plan from the tree, no CPU pass; the error against direct
        # sums over all sources at nerr of them
        ds = (sources=to(pts[perm]), τ=to(τ[perm]))
        outd = to(similar(τ, eltype(usw)))
        local plan
        t_plan = @elapsed plan = DevicePlan(f, to; verbose=true)
        t_first = @elapsed device_matvec!(outd, plan, ds)
        t_mv = minimum(@elapsed(device_matvec!(outd, plan, ds)) for _ in 1:3)
        err = NaN
        if nerr > 0                     # against Float64 direct sums, whatever the run's precision
            sub = 1:min(N, nerr)
            uex = direct_subset(K64, pts64, τ64, sub)
            out = Array(outd)[sub]
            err = maximum(norm.(out .- uex)) / maximum(norm.(uex))
        end
        @printf "%-36s ppl=%-4d N=%-8d %2d lev | tree %7.1f | plan %7.1f | first %7.1f | matvec %7.1f ms (%6.0f kpts/s) | with tree %7.1f ms | vs direct %.1e\n" label ppl N nlevels(f) 1e3t_tree 1e3t_plan 1e3t_first 1e3t_mv N / t_mv / 1000 1e3(t_tree + t_mv) err
    end
    return nothing
end

function main(dim, N, ppls, dists, kernels, methods, device, to, T, nerr)
    rng = MersenneTwister(1)
    println("Julia threads: ", Threads.nthreads(), ", BLAS threads: ", BLAS.get_num_threads(), ", cores assumed: ", CORES,
            ", dimension: ", dim, ", precision: ", T, device == "cpu" ? "" : ", device: $device")
    for dname in dists
        pts64 = GENERATORS[dname](rng, N, Val(dim))
        pts = SVector{dim,T}.(pts64)
        for ppl in ppls, kspec in kernels, mspec in methods
            kname, K = parse_kernel(kspec, Val(dim), T)
            _, K64 = parse_kernel(kspec, Val(dim), Float64)
            mname, m = parse_method(mspec, K, Val(dim))
            m === nothing && continue                            # a method not for this kernel
            τ64 = density(rng, kernel_kind(kspec), K64, N, Val(dim))
            τ = T === Float64 ? τ64 : map(x -> cvt(T, x), τ64)
            run_case("$kname $mname / $dname", m, pts, τ; ppl=ppl, to=to, nerr=nerr, K64=K64, pts64=pts64, τ64=τ64)
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
nerr = parse(Int, get(args, "nerr", "500"))
main(dim, N, ppls, dists, kernels, methods, device, to, T, nerr)
