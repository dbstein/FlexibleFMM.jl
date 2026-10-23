# Comparison with the Flatiron FMM libraries through FMM2D.jl and FMM3D.jl. This
# directory is its own environment (the libraries are not dependencies of the
# package; FlexibleFMM is a development dependency pointing at the repository):
#
#     julia --project=. -t auto compare.jl dim=3 N=100000 ppl=200 kernel=laplace,stokes \
#           method=cube:6,cube:8 eps=1e-3,1e-6 device=metal
#     julia --project=. -t auto compare.jl dim=2 N=1000000 kernel=laplace,helmholtz:10,stokes
#
# Defaults: dim=3, N=100000, ppl=200 (50 in 2D), dist=uniform, kernel=laplace,
# helmholtz:3,stokes,stresslet (helmholtz:10 in 2D), method=cube:6,cube:8,cube:10
# (ki:30,ki:60,square:12 in 2D), eps=1e-3,1e-6,1e-9 (the library's requested
# precisions), device=cpu, fast=false (true: ours with the FastMath kernel policy,
# relaxed arithmetic; the reference and the libraries are unaffected). The kernel,
# method and distribution specifications are
# those of ../common.jl; the black-box method takes no stresslets. Each case runs
# our method at every `method` and the library at every `eps` on the same points,
# evaluation at the sources only (the usual FMM matvec; separate targets are not
# timed). Ours is split as tree / first (the first application of a fresh state,
# which creates the method's workspaces; on a device the plan build and the first
# device matvec) / matvec (expansions + self evaluation, the best of three warm
# repetitions), with the operators built once and cached as in a solver loop;
# "total" is tree + matvec. The library's single call builds its tree and evaluates
# at the sources. With a device our rows run on the device only (no CPU passes). Errors are relative max errors at the targets against our direct sum,
# which is exact for both codes. The libraries' kernel normalisations differ from
# ours; each case first calibrates the constant factor on a small problem and
# reports it (a non-constant ratio would mean a different kernel). With
# device=metal (Float32 by default) or device=cuda (Float64 by default; the
# bench/cuda environment has CUDA.jl and the libraries), precision=float32|float64
# sets ours (a card with weak Float64 wants float32); our rows are then the device
# matvec (DevicePlan + device_matvec!) and the library rows stay on the CPU. Errors
# are relative max errors at nerr=2000 of the sources against the Float64 direct
# sum over all sources, on the points and densities rounded to ours' precision,
# which the libraries get as well: every row is measured on the same problem, and
# Float32 rows show the Float32 FMM's error, not the rounding of the inputs
# (docs/framework.md). Helmholtz stays on the CPU on Metal (no complex GEMM).
# device=ka_cpu runs the device passes on KernelAbstractions' CPU backend (Float64).
using FlexibleFMM, FMM2D, FMM3D, StaticArrays, Random, Printf, LinearAlgebra
using FlexibleFMM: nlevels
include(joinpath(@__DIR__, "..", "common.jl"))

args = parse_kv(ARGS)
dim = parse(Int, get(args, "dim", "3"))
N = parse(Int, get(args, "N", "100000"))
ppl = parse(Int, get(args, "ppl", dim == 2 ? "50" : "200"))
dists = String.(split(get(args, "dist", "uniform"), ","))
kernels = String.(split(get(args, "kernel", dim == 2 ? "laplace,helmholtz:10,stokes,stresslet" : "laplace,helmholtz:3,stokes,stresslet"), ","))
methods = String.(split(get(args, "method", dim == 2 ? "ki:30,ki:60,square:12" : "cube:6,cube:8,cube:10"), ","))
epss = parse.(Float64, split(get(args, "eps", "1e-3,1e-6,1e-9"), ","))
device = get(args, "device", "cpu")
device in ("cpu", "ka_cpu", "metal", "cuda") || error("device must be cpu, ka_cpu, metal or cuda")
if device == "metal"
    using Metal
elseif device == "cuda"
    using CUDA
end
to = device == "metal" ? MtlArray : device == "cuda" ? CuArray : device == "ka_cpu" ? identity : nothing
precision = get(args, "precision", device == "metal" ? "float32" : "float64")
precision in ("float32", "float64") || error("precision must be float32 or float64")
TP = precision == "float32" ? Float32 : Float64         # our precision (the libraries run in Float64)
nerr = parse(Int, get(args, "nerr", "2000"))
fast = get(args, "fast", "false") == "true"                 # ours with the FastMath kernel policy
rng = MersenneTwister(1)
println(config_line())
println("dim ", dim, ", N ", N, ", ppl ", ppl, device == "cpu" ? "" : ", ours on $device in $TP")

mat(v) = reduce(hcat, v)                                 # D × n matrix for the libraries
unmat(M) = [SVector{size(M, 1)}(c) for c in eachcol(M)]
relerr(u, uex) = maximum(norm.(u .- uex)) / maximum(norm.(uex))
cvt(x::Number) = TP(x)
cvt(x::SVector) = SVector{dim,TP}(x)
cvt(x::Stresslet) = Stresslet(cvt(x.f), cvt(x.ν))
cvt(τ::AbstractVector) = TP === Float64 ? τ : map(cvt, τ)
cvt64(x::Number) = Float64(x)                            # back to Float64: the rounded inputs for the reference and the libraries
cvt64(x::SVector) = SVector{dim,Float64}(x)
cvt64(x::Stresslet) = Stresslet(cvt64(x.f), cvt64(x.ν))
round_trip(τ::AbstractVector) = TP === Float64 ? τ : map(cvt64, map(cvt, τ))

# ---- ours: tree / first application / warm matvec ----
# A small warm-up compiles everything (the device path too); the tree is built
# twice (the first builds the operators into the cache), the first application of
# a fresh state is timed on its own (on the CPU it creates the method's workspaces;
# on a device it is the plan build and the first device matvec), and the matvec is
# the best of three warm repetitions. With a device no CPU pass runs.
function run_ours(m, τ, ptsT; dev=to)
    nw = min(length(ptsT), max(2000, 100 * ppl))   # a warm-up tree with far-field interactions (compiles the whole matvec)
    fw = FMM(m, ptsT[1:nw]; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)))
    sw = build_expansions(fw, ptsT[1:nw], τ[1:nw]); evaluate_at_sources(fw, sw)
    if dev !== nothing
        pw = DevicePlan(fw, dev)
        device_matvec!(dev(similar(evaluate_at_sources(fw, sw))), pw, (sources=dev(ptsT[1:nw]), τ=dev(τ[1:nw])))
    end
    cache = OperatorCache(m)            # operators are built once per kernel and order and reused
    f = FMM(m, ptsT; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)), cache=cache)
    t_tree = @elapsed f = FMM(m, ptsT; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)), cache=cache)
    local out
    if dev === nothing
        local st
        t_first = @elapsed (st = build_expansions(f, ptsT, τ); out = evaluate_at_sources(f, st))
        t_mv = minimum(@elapsed((build_expansions!(st, f, ptsT, τ); out = evaluate_at_sources(f, st))) for _ in 1:3)
    else
        ds = (sources=dev(ptsT), τ=dev(τ))                  # original order: the plan gathers
        outd = dev(similar(τ, typeof(evaluate_at_sources(fw, sw)[1])))
        local plan
        t_first = @elapsed (plan = DevicePlan(f, dev); device_matvec!(outd, plan, ds))
        t_mv = minimum(@elapsed(device_matvec!(outd, plan, ds)) for _ in 1:3)
        out = Array(outd)
    end
    return t_tree, t_first, t_mv, out, nlevels(f)
end
# ---- library: one call evaluating at sources and targets, best of two ----
function run_lib(call)
    local out
    t = Inf
    for _ in 1:2
        t = min(t, @elapsed out = call())
    end
    return t, out
end
# constant factor between the library's kernel and ours, from a small problem
function calibrate(call_small, uex_small)
    ratios = vec(mat(call_small())) ./ vec(mat(uex_small))
    r = ratios[1]
    spread = maximum(abs.(ratios .- r)) / abs(r)
    spread < 1e-6 || @warn "the library's kernel is not a constant multiple of ours (spread $spread)"
    return r
end
mat(v::AbstractVector{<:Number}) = reshape(v, 1, :)

function report(name, rows)
    println("\n=== $name ===")
    @printf "%-38s %10s %10s %10s %10s %9s\n" "code / parameter" "tree" "first" "matvec" "total" "error"
    fmt(x) = x === nothing ? "" : @sprintf("%.3f", x)
    for r in rows
        @printf "%-38s %10s %10s %10s %10.3f %9.1e\n" r[1] fmt(r[2]) fmt(r[3]) fmt(r[4]) r[5] r[6]
    end
end

# the library call for a kernel kind: densities as the library takes them, output
# at the sources as a vector of scalars or SVectors
function lib_call(kind, k, eps, S, τ)
    if dim == 2
        kind === :laplace && return () -> rfmm2d(; eps=eps, sources=S, charges=τ, pg=1).pot
        kind === :helmholtz && return () -> hfmm2d(; eps=eps, zk=k, sources=S, charges=complex.(τ), pg=1).pot
        kind === :yukawa && return () -> real.(hfmm2d(; eps=eps, zk=im * k, sources=S, charges=complex.(τ), pg=1).pot)
        kind === :stokes && return () -> unmat(stfmm2d(; eps=eps, sources=S, stoklet=mat(τ), ppreg=1).pot)
        kind === :stresslet && return () -> unmat(stfmm2d(; eps=eps, sources=S, strslet=mat([s.f for s in τ]), strsvec=mat([s.ν for s in τ]), ppreg=1).pot)
    else
        kind === :laplace && return () -> lfmm3d(eps, S; charges=τ, pg=1).pot
        kind === :helmholtz && return () -> hfmm3d(eps, k, S; charges=complex.(τ), pg=1).pot
        kind === :yukawa && return () -> real.(hfmm3d(eps, im * k, S; charges=complex.(τ), pg=1).pot)
        kind === :stokes && return () -> unmat(stfmm3d(eps, S; stoklet=mat(τ), ppreg=1).pot)
        kind === :stresslet && return () -> unmat(stfmm3d(eps, S; strslet=mat([s.f for s in τ]), strsvec=mat([s.ν for s in τ]), ppreg=1).pot)
    end
    error("no library call for $kind")
end
libname(kind) = (dim == 2 ? "FMM2D " : "FMM3D ") * (kind === :laplace ? (dim == 2 ? "rfmm2d" : "lfmm3d") :
                 kind in (:helmholtz, :yukawa) ? (dim == 2 ? "hfmm2d" : "hfmm3d") : (dim == 2 ? "stfmm2d" : "stfmm3d"))

# Σ_{j ≠ i} over all sources at the sources `sub`, threaded (the reference)
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

for dname in dists
    pts = GENERATORS[dname](rng, N, Val(dim))
    ptsT = SVector{dim,TP}.(pts)
    pts = SVector{dim,Float64}.(ptsT)                    # what ours sees, for the reference and the libraries
    S = mat(pts)
    nsm = 300; Ssm = mat(pts[1:nsm])
    sub = 1:min(N, nerr)
    for kspec in kernels
        kind = kernel_kind(kspec)
        kname, K64 = parse_kernel(kspec, Val(dim), Float64)          # the reference kernel
        _, KT = parse_kernel(kspec, Val(dim), TP)                    # ours, in our precision
        fast && (KT = fastmath(KT))
        k = occursin(":", kspec) ? parse(Float64, split(kspec, ":")[2]) : 0.0
        τ = round_trip(density(rng, kind, K64, N, Val(dim)))
        uex = direct_subset(K64, pts, τ, sub)
        c = calibrate(lib_call(kind, k, 1e-12, Ssm, τ[1:nsm]), direct_self(K64, pts[1:nsm], τ[1:nsm]))
        println("$(libname(kind)) kernel / ours = $c")
        dev = (device == "metal" && kind === :helmholtz) ? nothing : to   # no complex GEMM on Metal
        rows = Any[]
        for mspec in methods
            mname, m = parse_method(mspec, KT, Val(dim))
            m === nothing && (println("  ($mname: not for this kernel)"); continue)
            devm = m isa Union{FlexibleFMM.AbstractKernelIndependent,FlexibleFMM.HarmonicMethod} ? dev : nothing   # the device passes
            r = run_ours(m, cvt(τ), ptsT; dev=devm)
            push!(rows, ("ours $mname$(fast ? " fast" : "") ($(r[5]) levels$(devm === nothing ? "" : ", $device"))", r[1], r[2], r[3], r[1] + r[3], relerr(r[4][sub], uex)))
        end
        for e in epss
            t, out = run_lib(lib_call(kind, k, e, S, τ))
            push!(rows, ("$(libname(kind)) eps=$e", nothing, nothing, nothing, t, relerr(out[sub] ./ c, uex)))
        end
        report("$dim" * "D $kname, N = $N $dname", rows)
    end
end
