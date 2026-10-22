# Device evaluation on Apple Silicon through Metal.jl (own environment; Metal is not
# a dependency of the package), in Float32: the near field and the local expansions
# of 10⁶ targets and the self evaluation at the sources, against the threaded CPU
# evaluation.
#     julia --project=. -t auto metal_eval.jl N=100000 NT=1000000 ppl=200
using FlexibleFMM, Metal, StaticArrays, Random, Printf, LinearAlgebra
using FlexibleFMM: locate
args = Dict{String,String}(String(a[1]) => String(a[2]) for a in split.(ARGS, "="))
N = parse(Int, get(args, "N", "100000")); NT = parse(Int, get(args, "NT", "1000000")); ppl = parse(Int, get(args, "ppl", "200"))
fast = parse(Bool, get(args, "fast", "false"))      # the FastMath policy (relaxed arithmetic)
rng = MersenneTwister(1)
const P3f = SVector{3,Float32}
pts = [rand(rng, P3f) for _ in 1:N]; tg = [rand(rng, P3f) for _ in 1:NT]
relerr(u, uex) = maximum(norm.(u .- uex)) / maximum(norm.(uex))
println("Metal device: ", Metal.current_device().name, ", Julia threads ", Threads.nthreads(), ", fast ", fast)
for (name, K, τ) in (("Laplace3D", Laplace3D(Float32), randn(rng, Float32, N)),
                     ("Stokeslet3D", Stokeslet3D(Float32), [(randn(rng, P3f), Doublet(randn(rng, P3f))) for _ in 1:N]))
    f = FMM(CubeKernelIndependent{3}(K; p=8), pts; ppl=ppl, bbox=(zero(P3f), 1f0))
    st = build_expansions(f, pts, τ)
    nodes = Int32.(locate(f.tree, tg))
    # CPU reference and timing
    ref = evaluate(f, st, tg); t_cpu = minimum(@elapsed(evaluate(f, st, tg)) for _ in 1:2)
    refs = evaluate_at_sources(f, st); ts_cpu = minimum(@elapsed(evaluate_at_sources(f, st)) for _ in 1:2)
    # device
    dev = DeviceEvaluator(f, MtlArray; fast=fast); ds = device_state(st, MtlArray)
    tgd = MtlArray(tg); nd = MtlArray(nodes)
    out = MtlArray{eltype(ref)}(undef, NT); outs = MtlArray{eltype(refs)}(undef, N)
    evaluate!(out, dev, ds, tgd, nd); t_dev = minimum(@elapsed(evaluate!(out, dev, ds, tgd, nd)) for _ in 1:3)
    evaluate_at_sources!(outs, dev, ds); ts_dev = minimum(@elapsed(evaluate_at_sources!(outs, dev, ds)) for _ in 1:3)
    t_up = @elapsed (device_state(st, MtlArray); MtlArray(tg))
    @printf "%-12s targets %d: CPU %.3f s, Metal %.3f s (%.1fx), err %.1e | self %d: CPU %.3f s, Metal %.3f s (%.1fx), err %.1e | upload per matvec %.3f s\n" name NT t_cpu t_dev t_cpu/t_dev relerr(Array(out), ref) N ts_cpu ts_dev ts_cpu/ts_dev relerr(Array(outs), refs) t_up
end
