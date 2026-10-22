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
# precisions), device=cpu. The kernel, method and distribution specifications are
# those of ../common.jl; the black-box method takes no stresslets. Each case runs
# our method at every `method` and the library at every `eps` on the same points,
# with 2000 separate targets. Ours is split as tree / first (the first application
# of a fresh state, which creates the method's workspaces) / matvec (expansions +
# self evaluation, the best of three warm repetitions) / targets, with the
# operators built once and cached as in a solver loop; "total" is tree + matvec +
# targets. The library's single call builds its tree and evaluates at sources and
# targets. Errors are relative max errors at the targets against our direct sum,
# which is exact for both codes. The libraries' kernel normalisations differ from
# ours; each case first calibrates the constant factor on a small problem and
# reports it (a non-constant ratio would mean a different kernel). With
# device=metal (Float32) or device=cuda (Float64; add CUDA.jl to this environment)
# our rows are the device matvec (DevicePlan + device_matvec!) and the device
# evaluation at the targets (their location on the CPU excluded), the library rows
# stay on the CPU; errors are still against the Float64 direct sum, so Float32 rows
# show Float32 rounding. Helmholtz stays on the CPU on Metal (no complex GEMM).
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
TP = device == "metal" ? Float32 : Float64              # our precision
rng = MersenneTwister(1)
const NT = 2000
println("Julia threads ", Threads.nthreads(), ", OMP_NUM_THREADS ", get(ENV, "OMP_NUM_THREADS", "unset"), ", dim ", dim,
        ", N ", N, ", ppl ", ppl, device == "cpu" ? "" : ", ours on $device in $TP")

mat(v) = reduce(hcat, v)                                 # D × n matrix for the libraries
unmat(M) = [SVector{size(M, 1)}(c) for c in eachcol(M)]
relerr(u, uex) = maximum(norm.(u .- uex)) / maximum(norm.(uex))
cvt(x::Number) = TP(x)
cvt(x::SVector) = SVector{dim,TP}(x)
cvt(x::Stresslet) = Stresslet(cvt(x.f), cvt(x.ν))
cvt(τ::AbstractVector) = TP === Float64 ? τ : map(cvt, τ)

# ---- ours: tree / first application / warm matvec / targets ----
# The tree is built twice (the first builds the operators into the cache; the
# second is timed), the first application of a fresh state is timed on its own (it
# creates the method's workspaces, the cube method's spectral one in particular),
# and the matvec is the best of three warm repetitions on that state.
function run_ours(m, τ, ptsT, tgT; dev=to)
    local f, st, ut
    nw = min(length(ptsT), 2000)        # a small warm-up compiles everything, so "first" is the workspaces
    fw = FMM(m, ptsT[1:nw]; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)))
    sw = build_expansions(fw, ptsT[1:nw], τ[1:nw]); evaluate_at_sources(fw, sw); evaluate(fw, sw, tgT[1:10])
    cache = OperatorCache(m)            # operators are built once per kernel and order and reused
    f = FMM(m, ptsT; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)), cache=cache)
    t_tree = @elapsed f = FMM(m, ptsT; ppl=ppl, bbox=(zero(eltype(ptsT)), one(TP)), cache=cache)
    t_first = @elapsed (st = build_expansions(f, ptsT, τ); evaluate_at_sources(f, st))
    t_mv = minimum(@elapsed((build_expansions!(st, f, ptsT, τ); evaluate_at_sources(f, st))) for _ in 1:3)
    t_tg = minimum(@elapsed(ut = evaluate(f, st, tgT)) for _ in 1:3)
    if dev !== nothing                  # the device matvec and target evaluation (locations on the CPU excluded)
        plan = DevicePlan(f, dev)
        ds = (sources=dev(st.sources), τ=dev(st.τ))
        outd = dev(similar(evaluate_at_sources(f, st)))
        t_first = @elapsed device_matvec!(outd, plan, ds)
        t_mv = minimum(@elapsed(device_matvec!(outd, plan, ds)) for _ in 1:3)
        tgd = dev(tgT); nodes = dev(Int32.(locate(f.tree, tgT))); outt = dev(similar(ut))
        dsl = (locals=plan.locals, sources=ds.sources, τ=ds.τ)
        evaluate!(outt, plan.dev, dsl, tgd, nodes)
        t_tg = minimum(@elapsed(evaluate!(outt, plan.dev, dsl, tgd, nodes)) for _ in 1:3)
        ut = Array(outt)
    end
    return t_tree, t_first, t_mv, t_tg, ut, nlevels(f)
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
    @printf "%-38s %10s %10s %10s %10s %10s %9s\n" "code / parameter" "tree" "first" "matvec" "targets" "total" "error"
    fmt(x) = x === nothing ? "" : @sprintf("%.3f", x)
    for r in rows
        @printf "%-38s %10s %10s %10s %10s %10.3f %9.1e\n" r[1] fmt(r[2]) fmt(r[3]) fmt(r[4]) fmt(r[5]) r[6] r[7]
    end
end

# the library call for a kernel kind: densities as the library takes them, output
# at the targets as a vector of scalars or SVectors
function lib_call(kind, k, eps, S, T, τ)
    if dim == 2
        kind === :laplace && return () -> rfmm2d(; eps=eps, sources=S, charges=τ, targets=T, pg=1, pgt=1).pottarg
        kind === :helmholtz && return () -> hfmm2d(; eps=eps, zk=k, sources=S, charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg
        kind === :yukawa && return () -> real.(hfmm2d(; eps=eps, zk=im * k, sources=S, charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg)
        kind === :stokes && return () -> unmat(stfmm2d(; eps=eps, sources=S, stoklet=mat(τ), targets=T, ppreg=1, ppregt=1).pottarg)
        kind === :stresslet && return () -> unmat(stfmm2d(; eps=eps, sources=S, strslet=mat([s.f for s in τ]), strsvec=mat([s.ν for s in τ]), targets=T, ppreg=1, ppregt=1).pottarg)
    else
        kind === :laplace && return () -> lfmm3d(eps, S; charges=τ, targets=T, pg=1, pgt=1).pottarg
        kind === :helmholtz && return () -> hfmm3d(eps, k, S; charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg
        kind === :yukawa && return () -> real.(hfmm3d(eps, im * k, S; charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg)
        kind === :stokes && return () -> unmat(stfmm3d(eps, S; stoklet=mat(τ), targets=T, ppreg=1, ppregt=1).pottarg)
        kind === :stresslet && return () -> unmat(stfmm3d(eps, S; strslet=mat([s.f for s in τ]), strsvec=mat([s.ν for s in τ]), targets=T, ppreg=1, ppregt=1).pottarg)
    end
    error("no library call for $kind")
end
libname(kind) = (dim == 2 ? "FMM2D " : "FMM3D ") * (kind === :laplace ? (dim == 2 ? "rfmm2d" : "lfmm3d") :
                 kind in (:helmholtz, :yukawa) ? (dim == 2 ? "hfmm2d" : "hfmm3d") : (dim == 2 ? "stfmm2d" : "stfmm3d"))

for dname in dists
    pts = GENERATORS[dname](rng, N, Val(dim))
    tg = uniform(rng, NT, Val(dim))
    ptsT = SVector{dim,TP}.(pts); tgT = SVector{dim,TP}.(tg)
    S = mat(pts); T = mat(tg)
    nsm = 300; Ssm = mat(pts[1:nsm]); Tsm = mat(tg[1:100])
    for kspec in kernels
        kind = kernel_kind(kspec)
        kname, K64 = parse_kernel(kspec, Val(dim), Float64)          # the reference kernel
        _, KT = parse_kernel(kspec, Val(dim), TP)                    # ours, in our precision
        k = occursin(":", kspec) ? parse(Float64, split(kspec, ":")[2]) : 0.0
        τ = density(rng, kind, K64, N, Val(dim))
        uex = direct(K64, pts, τ, tg)
        c = calibrate(lib_call(kind, k, 1e-12, Ssm, Tsm, τ[1:nsm]), direct(K64, pts[1:nsm], τ[1:nsm], tg[1:100]))
        println("$(libname(kind)) kernel / ours = $c")
        dev = (device == "metal" && kind === :helmholtz) ? nothing : to   # no complex GEMM on Metal
        rows = Any[]
        for mspec in methods
            mname, m = parse_method(mspec, KT, Val(dim))
            m === nothing && (println("  ($mname: not for this kernel)"); continue)
            devm = m isa Union{FlexibleFMM.AbstractKernelIndependent,FlexibleFMM.HarmonicMethod} ? dev : nothing   # the device passes
            r = run_ours(m, cvt(τ), ptsT, tgT; dev=devm)
            push!(rows, ("ours $mname ($(r[6]) levels$(devm === nothing ? "" : ", $device"))", r[1], r[2], r[3], r[4], r[1] + r[3] + r[4], relerr(r[5], uex)))
        end
        for e in epss
            t, out = run_lib(lib_call(kind, k, e, S, T, τ))
            push!(rows, ("$(libname(kind)) eps=$e", nothing, nothing, nothing, nothing, t, relerr(out ./ c, uex)))
        end
        report("$dim" * "D $kname, N = $N $dname", rows)
    end
end
