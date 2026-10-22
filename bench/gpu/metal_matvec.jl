# The full device matvec on Apple Silicon through Metal.jl in Float32: expansions
# (S2M kernel, GEMMs, dense translations by symmetry class) and the self
# evaluation, against the threaded CPU matvec and a direct sum on a subset.
#     julia --project=. -t auto metal_matvec.jl N=100000 ppl=200
using FlexibleFMM, Metal, StaticArrays, Random, Printf, LinearAlgebra
args = Dict{String,String}(String(a[1]) => String(a[2]) for a in split.(ARGS, "="))
N = parse(Int, get(args, "N", "100000")); ppl = parse(Int, get(args, "ppl", "200"))
fast = parse(Bool, get(args, "fast", "false"))      # the FastMath policy (relaxed arithmetic)
tol = parse(Float64, get(args, "tol", "0"))        # > 0: low-rank class operators
rng = MersenneTwister(1)
const P3f = SVector{3,Float32}
pts = [rand(rng, P3f) for _ in 1:N]
relerr(u, uex) = maximum(norm.(u .- uex)) / maximum(norm.(uex))
println("Metal device: ", Metal.current_device().name, ", Julia threads ", Threads.nthreads(), ", N ", N, ", ppl ", ppl, ", tol ", tol, ", fast ", fast)
sub = 1:1000
for (name, K, τ) in (("Laplace3D", Laplace3D(Float32), randn(rng, Float32, N)),
                     ("Stokeslet3D", Stokeslet3D(Float32), [(randn(rng, P3f), Doublet(randn(rng, P3f))) for _ in 1:N]))
    f = FMM(CubeKernelIndependent{3}(K; p=8), pts; ppl=ppl, bbox=(zero(P3f), 1f0))
    st = build_expansions(f, pts, τ)
    ref = evaluate_at_sources(f, st)
    t_cpu = minimum(@elapsed(begin build_expansions!(st, f, pts, τ); evaluate_at_sources(f, st) end) for _ in 1:2)
    t_plan = @elapsed plan = DevicePlan(f, MtlArray; tol=tol, fast=fast)
    ds = (sources=MtlArray(st.sources), τ=MtlArray(st.τ))
    out = MtlArray{eltype(ref)}(undef, N)
    device_matvec!(out, plan, ds; sorted=true)
    t_dev = minimum(@elapsed(device_matvec!(out, plan, ds; sorted=true)) for _ in 1:3)
    t_exp = minimum(@elapsed(device_build_expansions!(plan, ds; sorted=true)) for _ in 1:3)
    u = Array(out)
    uex = [sum(FlexibleFMM.source_field(K, pts[j], pts[i], τ[j]) for j in 1:N if j != i) for i in sub]
    @printf "%-12s matvec: CPU %.3f s, Metal %.3f s (%.1fx; expansions %.3f s) | plan build %.1f s | vs CPU %.1e, vs direct %.1e (CPU vs direct %.1e)\n" name t_cpu t_dev t_cpu/t_dev t_exp t_plan relerr(u, ref) relerr(u[sub], uex) relerr(ref[sub], uex)
end
