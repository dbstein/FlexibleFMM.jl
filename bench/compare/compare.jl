# Comparison with the Flatiron FMM libraries through FMM2D.jl and FMM3D.jl. This
# directory is its own environment (the libraries are not dependencies of the
# package; FlexibleFMM is a development dependency pointing at the repository):
#
#     julia --project=. -t auto compare.jl dim=3 N=100000 ppl=200
#     julia --project=. -t auto compare.jl dim=2 N=1000000
#
# Each case runs our method at several orders and the library at several requested
# precisions on the same uniform points, with 2000 separate targets. Ours is split
# as tree / matvec (expansions + self evaluation) / targets, with the operators
# built once and cached as in a solver loop; the library's single call builds its
# tree and evaluates at sources and targets. Errors are relative max
# errors at the targets against our direct sum, which is exact for both codes. The
# libraries' kernel normalisations differ from ours; each case first calibrates the
# constant factor on a small problem and reports it (a non-constant ratio would
# mean a different kernel).
using FlexibleFMM, FMM2D, FMM3D, StaticArrays, Random, Printf, LinearAlgebra
using FlexibleFMM: nlevels

args = Dict{String,String}(String(a[1]) => String(a[2]) for a in split.(ARGS, "="))
dim = parse(Int, get(args, "dim", "3"))
N = parse(Int, get(args, "N", "100000"))
ppl = parse(Int, get(args, "ppl", dim == 2 ? "50" : "200"))
rng = MersenneTwister(1)
const NT = 2000
P = SVector{dim,Float64}
pts = [rand(rng, P) for _ in 1:N]
tg = [rand(rng, P) for _ in 1:NT]
mat(v) = reduce(hcat, v)                                 # D × n matrix for the libraries
unmat(M) = [SVector{size(M, 1)}(c) for c in eachcol(M)]
relerr(u, uex) = maximum(norm.(u .- uex)) / maximum(norm.(uex))
println("Julia threads ", Threads.nthreads(), ", OMP_NUM_THREADS ", get(ENV, "OMP_NUM_THREADS", "unset"), ", dim ", dim, ", N ", N, ", ppl ", ppl)

# ---- ours: tree / matvec / targets, best of two runs (the first compiles) ----
function run_ours(m, K, τ)
    local f, st, ut
    t_tree = t_mv = t_tg = Inf
    cache = OperatorCache(m)            # operators are built once per kernel and order and reused
    for _ in 1:2
        t1 = @elapsed f = FMM(m, pts; ppl=ppl, bbox=(zero(P), 1.0), cache=cache)
        t2 = @elapsed (st = build_expansions(f, pts, τ); evaluate_at_sources(f, st))
        t3 = @elapsed ut = evaluate(f, st, tg)
        t_tree, t_mv, t_tg = min(t_tree, t1), min(t_mv, t2), min(t_tg, t3)
    end
    return t_tree, t_mv, t_tg, ut, nlevels(f)
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
    ratios = call_small() ./ uex_small
    r = ratios[1]
    spread = maximum(abs.(ratios .- r)) / abs(r)
    spread < 1e-6 || @warn "the library's kernel is not a constant multiple of ours (spread $spread)"
    return r
end
vecratio(u, uex) = (r = [dot(a, b) / dot(b, b) for (a, b) in zip(u, uex)]; r[1])

function report(name, rows)
    println("\n=== $name ===")
    @printf "%-34s %10s %10s %10s %10s %9s\n" "code / parameter" "tree" "matvec" "targets" "total" "error"
    for r in rows
        @printf "%-34s %10s %10s %10s %10.3f %9.1e\n" r[1] (r[2] === nothing ? "" : @sprintf("%.3f", r[2])) (r[3] === nothing ? "" : @sprintf("%.3f", r[3])) (r[4] === nothing ? "" : @sprintf("%.3f", r[4])) r[5] r[6]
    end
end
ours_row(label, m, K, τ, uex) = (t1, t2, t3, ut, nl) = run_ours(m, K, τ) |> r -> ("ours $label ($(r[5]) levels)", r[1], r[2], r[3], r[1] + r[2] + r[3], relerr(r[4], uex))
lib_row(label, call, uex, scale) = (t, out) = run_lib(call) |> r -> ("$label", nothing, nothing, nothing, r[1], relerr(r[2] ./ scale, uex))

S = mat(pts); T = mat(tg)
nsm = 300; Ssm = mat(pts[1:nsm]); Tsm = mat(tg[1:100])

if dim == 3
    # Laplace, charges
    K = Laplace3D(); τ = randn(rng, N); uex = direct(K, pts, τ, tg)
    c = calibrate(() -> lfmm3d(1e-12, Ssm; charges=τ[1:nsm], targets=Tsm, pgt=1).pottarg, direct(K, pts[1:nsm], τ[1:nsm], tg[1:100]))
    println("FMM3D Laplace kernel / ours = ", c)
    rows = Any[ours_row("cube p=$p", CubeKernelIndependent{3}(K; p=p), K, τ, uex) for p in (6, 8, 10)]
    append!(rows, [lib_row("FMM3D lfmm3d eps=$e", () -> lfmm3d(e, S; charges=τ, targets=T, pg=1, pgt=1).pottarg, uex, c) for e in (1e-3, 1e-6, 1e-9)])
    report("3D Laplace, N = $N uniform charges", rows)
    # Helmholtz, charges
    k = 3.0; H = Helmholtz3D(k); uexh = direct(H, pts, τ, tg)
    ch = calibrate(() -> hfmm3d(1e-12, k, Ssm; charges=complex.(τ[1:nsm]), targets=Tsm, pgt=1).pottarg, direct(H, pts[1:nsm], τ[1:nsm], tg[1:100]))
    println("FMM3D Helmholtz kernel / ours = ", ch)
    rows = Any[ours_row("cube p=$p", CubeKernelIndependent{3}(H; p=p), H, τ, uexh) for p in (8,)]
    append!(rows, [lib_row("FMM3D hfmm3d eps=$e", () -> hfmm3d(e, k, S; charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg, uexh, ch) for e in (1e-6, 1e-9)])
    report("3D Helmholtz k = $k, N = $N uniform charges", rows)
    # Stokes, stokeslets
    KS = Stokeslet3D(); fs = [randn(rng, P) for _ in 1:N]; uexs = direct(KS, pts, fs, tg)
    cs = calibrate(() -> vec(stfmm3d(1e-12, Ssm; stoklet=mat(fs[1:nsm]), targets=Tsm, ppregt=1).pottarg), vec(mat(direct(KS, pts[1:nsm], fs[1:nsm], tg[1:100]))))
    println("FMM3D stokeslet / ours = ", cs)
    rows = Any[ours_row("cube p=$p", CubeKernelIndependent{3}(KS; p=p), KS, fs, uexs) for p in (6, 8)]
    append!(rows, [lib_row("FMM3D stfmm3d eps=$e", () -> unmat(stfmm3d(e, S; stoklet=mat(fs), targets=T, ppreg=1, ppregt=1).pottarg), uexs, cs) for e in (1e-3, 1e-6)])
    report("3D Stokes, N = $N uniform stokeslets", rows)
    # Stokes, stresslets
    nu = [normalize(randn(rng, P)) for _ in 1:N]; τt = [Stresslet(fs[i], nu[i]) for i in 1:N]; uext = direct(KS, pts, τt, tg)
    ct = calibrate(() -> vec(stfmm3d(1e-12, Ssm; strslet=mat(fs[1:nsm]), strsvec=mat(nu[1:nsm]), targets=Tsm, ppregt=1).pottarg), vec(mat(direct(KS, pts[1:nsm], τt[1:nsm], tg[1:100]))))
    println("FMM3D stresslet / ours = ", ct)
    rows = Any[ours_row("cube p=8", CubeKernelIndependent{3}(KS; p=8), KS, τt, uext)]
    append!(rows, [lib_row("FMM3D stfmm3d eps=$e", () -> unmat(stfmm3d(e, S; strslet=mat(fs), strsvec=mat(nu), targets=T, ppreg=1, ppregt=1).pottarg), uext, ct) for e in (1e-6,)])
    report("3D Stokes, N = $N uniform stresslets", rows)
else
    K = Laplace2D(); τ = randn(rng, N); uex = direct(K, pts, τ, tg)
    c = calibrate(() -> rfmm2d(; eps=1e-12, sources=Ssm, charges=τ[1:nsm], targets=Tsm, pgt=1).pottarg, direct(K, pts[1:nsm], τ[1:nsm], tg[1:100]))
    println("FMM2D Laplace kernel / ours = ", c)
    rows = Any[ours_row("circles N=$n", KernelIndependent(K; nequiv=n), K, τ, uex) for n in (30, 60)]
    push!(rows, ours_row("square p=12", SquareKernelIndependent(K; p=12), K, τ, uex))
    append!(rows, [lib_row("FMM2D rfmm2d eps=$e", () -> rfmm2d(; eps=e, sources=S, charges=τ, targets=T, pg=1, pgt=1).pottarg, uex, c) for e in (1e-6, 1e-9, 1e-12)])
    report("2D Laplace, N = $N uniform charges", rows)
    k = 10.0; H = Helmholtz2D(k); uexh = direct(H, pts, τ, tg)
    ch = calibrate(() -> hfmm2d(; eps=1e-12, zk=k, sources=Ssm, charges=complex.(τ[1:nsm]), targets=Tsm, pgt=1).pottarg, direct(H, pts[1:nsm], τ[1:nsm], tg[1:100]))
    println("FMM2D Helmholtz kernel / ours = ", ch)
    rows = Any[ours_row("circles N=60", KernelIndependent(H; nequiv=60), H, τ, uexh)]
    append!(rows, [lib_row("FMM2D hfmm2d eps=$e", () -> hfmm2d(; eps=e, zk=k, sources=S, charges=complex.(τ), targets=T, pg=1, pgt=1).pottarg, uexh, ch) for e in (1e-6, 1e-9)])
    report("2D Helmholtz k = $k, N = $N uniform charges", rows)
    KS = Stokeslet2D(); fs = [randn(rng, P) for _ in 1:N]; uexs = direct(KS, pts, fs, tg)
    cs = calibrate(() -> vec(stfmm2d(; eps=1e-12, sources=Ssm, stoklet=mat(fs[1:nsm]), targets=Tsm, ppregt=1).pottarg), vec(mat(direct(KS, pts[1:nsm], fs[1:nsm], tg[1:100]))))
    println("FMM2D stokeslet / ours = ", cs)
    rows = Any[ours_row("circles N=60", KernelIndependent(KS; nequiv=60), KS, fs, uexs)]
    append!(rows, [lib_row("FMM2D stfmm2d eps=$e", () -> unmat(stfmm2d(; eps=e, sources=S, stoklet=mat(fs), targets=T, ppreg=1, ppregt=1).pottarg), uexs, cs) for e in (1e-6, 1e-9)])
    report("2D Stokes, N = $N uniform stokeslets", rows)
end
