# Checks of the device plan on Metal that the test suite (KernelAbstractions' CPU
# backend) cannot make, each against a behaviour of Metal.jl 1.11:
#   1. a non-finite density does not poison the plan: Metal's default matrix
#      multiply (MPSGraph) reads C at β = 0, so the plan clears its GEMM outputs;
#   2. a plan built from device points at a point count not seen before compiles
#      nothing that depends on the count: Metal compiles its reductions per length;
#   3. plans rebuilt for moving points add few GEMM graphs to Metal's cache, which
#      keeps one per GEMM shape and never evicts;
#   4. plans built with one operator cache reuse its device operators.
# Two groups: sources on short random segments with stokeslets, targets on a sphere.
#     julia --project=bench/gpu -t auto bench/gpu/metal_checks.jl
using FlexibleFMM, Metal, StaticArrays, LinearAlgebra, Random, Printf
const P = SVector{3,Float32}
rng = MersenneTwister(1)
function segments(rng, nseg; len=0.4f0, per=32)
    pts = P[]
    for _ in 1:nseg
        c = 2 .* rand(rng, P) .- 1
        while norm(c) > 0.8f0
            c = 2 .* rand(rng, P) .- 1
        end
        d = normalize(randn(rng, P))
        append!(pts, [c + d * len * (k / (per - 1) - 0.5f0) for k in 0:per-1])
    end
    return pts
end
sphere(n, r) = [(z = 1 - (2k + 1) / n; ρ = sqrt(1 - z^2); φ = k * π * (3 - √5); P(r * ρ * cos(φ), r * ρ * sin(φ), r * z)) for k in 0:n-1]
box = (P(-1.6, -1.6, -1.6), 3.2f0)
m = CubeKernelIndependent{3}(Stokeslet3D(Float32); p=6)
cache = OperatorCache(m)
ppl = 400
Xt = MtlArray(sphere(4000, 1.5f0))
function apply!(outs, plan, Xs, τ)
    device_matvec!(outs, plan, (sources=(Xs, Xt), τ=(τ, nothing)))
    Metal.synchronize()
    return (Array(outs[1]), Array(outs[2]))
end
println("Metal device: ", Metal.current_device().name)
ok = true

# 1. recovery from a non-finite density
Xs = MtlArray(segments(rng, 300))
n = length(Xs)
τh = [randn(rng, P) for _ in 1:n]
τ = MtlArray(τh)
τh[5] = P(NaN32, 0, 0)
τbad = MtlArray(τh)
plan = DevicePlan(m, (Xs, Xt); ppl=ppl, bbox=box, cache=cache)
outs = (MtlArray{P}(undef, n), MtlArray{P}(undef, length(Xt)))
clean = apply!(outs, plan, Xs, τ)
apply!(outs, plan, Xs, τbad)
after = apply!(outs, plan, Xs, τ)
r1 = after == clean
global ok &= r1
@printf "1. after a NaN density the next application equals a clean one, bitwise: %s\n" r1

# 2. plan builds at point counts not seen before
for _ in 1:2
    DevicePlan(m, (MtlArray(segments(rng, 400)), Xt); ppl=ppl, bbox=box, cache=cache)
end
ct = Float64[]
for nseg in 401:3:434
    Xn = MtlArray(segments(rng, nseg))
    r = @timed DevicePlan(m, (Xn, Xt); ppl=ppl, bbox=box, cache=cache)
    push!(ct, r.compile_time)
end
# every new count compiled before (0.5-1 s); now a build compiles only when one of
# its kernels first meets a range that divides evenly by the workgroup size
# (KernelAbstractions then compiles a variant without the bounds check), once per
# kernel and run
r2 = count(>(0.01), ct) <= length(ct) ÷ 3
global ok &= r2
@printf "2. builds at %d new point counts that compiled anything: %d (compile times %s s): %s\n" length(ct) count(>(0.01), ct) join(round.(ct, digits=2), " ") r2

# 3. GEMM graphs added by plans rebuilt for moving points, against the new GEMM
# shapes of the per-level buffers with their column counts padded and unpadded
graphs() = length(Metal.MPSGraphs._matmul_graph_cache)
families(p) = [(f, c) for l in 1:p.nlevels for (f, c) in ((:up, length(p.s2m_nodes[l])), (:m2m, l < p.nlevels ? p.nl[l+1] ÷ 8 : 1),
                                                         (:l2l, p.nl[l] ÷ 8), (:down, length(p.leaf_nodes[l])))]
gemms(s) = sum(f in (:up, :down) ? 2 : 1 for (f, _) in s; init=0)          # the pseudo-inverse stages are two GEMMs
Xh = segments(rng, 400)
τm = MtlArray([randn(rng, P) for _ in 1:length(Xh)])
outs = (MtlArray{P}(undef, length(Xh)), MtlArray{P}(undef, length(Xt)))
added, padded, unpadded = Int[], Int[], Int[]
seen_p, seen_u = Set{Any}(), Set{Any}()
for step in 1:20
    global Xh = [x + 2f-3 * normalize(randn(rng, P)) for x in Xh]
    Xd = MtlArray(Xh)
    g0 = graphs()
    p = DevicePlan(m, (Xd, Xt); ppl=ppl, bbox=box, cache=cache)
    apply!(outs, p, Xd, τm)
    push!(added, graphs() - g0)
    F = families(p)
    Fp = [(f, FlexibleFMM._pad_cols(max(1, c))) for (f, c) in F]
    push!(padded, gemms(setdiff(Set(Fp), seen_p))); union!(seen_p, Fp)
    push!(unpadded, gemms(setdiff(Set(F), seen_u))); union!(seen_u, F)
end
# graphs follow the padded shapes (a class chunk of a new length may add a few)
r3 = sum(added[2:end]) <= sum(padded[2:end]) + 4
global ok &= r3
@printf "3. GEMM graphs added over 19 rebuilds: %d; new padded shapes %d, new shapes without padding %d: %s\n" sum(added[2:end]) sum(padded[2:end]) sum(unpadded[2:end]) r3

# 4. the operators' device copies come from the cache
n0 = length(cache.device)
t = @elapsed DevicePlan(m, (MtlArray(Xh), Xt); ppl=ppl, bbox=box, cache=cache)
r4 = length(cache.device) == n0
global ok &= r4
@printf "4. a rebuild adds nothing to the cache's device operators (%d entries), plan %.3f s: %s\n" n0 t r4
println(ok ? "all checks passed" : "SOME CHECKS FAILED")
