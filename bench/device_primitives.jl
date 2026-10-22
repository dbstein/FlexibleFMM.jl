# The device primitives a device-resident tree build rests on, timed on the backend
# given (docs/notes/gpu_everything_plan.md): the Morton keys of the points by the
# builder's own child-slot rule, the backend's `sortperm!` and `sort!` of 10⁷ 64-bit
# keys, a gather by the permutation, the package's prefix-sum kernels, and the
# launch, synchronisation and one-element readback latencies that bound the level
# loop's host round trips.
#
#     julia --project=bench/cuda -t auto bench/device_primitives.jl device=cuda [N=10000000]
#     julia --project=bench/gpu  -t auto bench/device_primitives.jl device=metal
using FlexibleFMM, StaticArrays, Random, Printf, LinearAlgebra
using FlexibleFMM: KernelAbstractions, get_backend, _exclusive_scan!
using KernelAbstractions: @kernel, @index, @Const, synchronize
include(joinpath(@__DIR__, "common.jl"))

args = parse_kv(ARGS)
device = get(args, "device", "metal")
N = parse(Int, get(args, "N", "10000000"))
if device == "metal"
    using Metal
elseif device == "cuda"
    using CUDA
else
    error("device must be metal or cuda")
end
to = device == "metal" ? MtlArray : CuArray
println(config_line())
const P3f = SVector{3,Float32}
rng = MersenneTwister(1)
pts = [P3f(0.5f0, 0.5f0, 0.5f0) + 0.4f0 * normalize(randn(rng, P3f)) for _ in 1:N]
best(f, n=5) = (f(); minimum(@elapsed(f()) for _ in 1:n))

@inline function spread21(v::UInt64)          # 21 bits to every third bit
    v &= 0x1fffff
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8)) & 0x100f00f00f00f00f
    v = (v | (v << 4)) & 0x10c30c30c30c30c3
    v = (v | (v << 2)) & 0x1249249249249249
    return v
end
# the key of a point: 21 child slots by the exact comparison against the box centers
# that the host builder and `locate` use (no rounding of a coordinate)
@kernel function morton_keys!(keys, @Const(xs), origin, W0, ::Val{nb}) where {nb}
    j = @index(Global, Linear)
    @inbounds begin
        x = xs[j]
        cx = Int32(0); cy = Int32(0); cz = Int32(0)
        w = W0
        for b in 1:nb
            c = origin + (SVector(Float32(cx), Float32(cy), Float32(cz)) .+ 0.5f0) * w
            cx = 2cx + (x[1] > c[1]); cy = 2cy + (x[2] > c[2]); cz = 2cz + (x[3] > c[3])
            w *= 0.5f0
        end
        keys[j] = spread21(UInt64(cx)) | (spread21(UInt64(cy)) << 1) | (spread21(UInt64(cz)) << 2)
    end
end
@kernel function gather!(dst, @Const(src), @Const(perm))
    j = @index(Global, Linear)
    @inbounds dst[j] = src[Int(perm[j])]
end
@kernel function touch!(x)
    j = @index(Global, Linear)
    @inbounds x[j] += one(eltype(x))
end

xs = to(pts)
backend = get_backend(xs)
keys = to(zeros(UInt64, N))
kk = morton_keys!(backend, 256)
t = best(() -> (kk(keys, xs, zero(P3f), 1f0, Val(21); ndrange=N); synchronize(backend)))
@printf "%-52s %8.4f s\n" "Morton keys, 21 levels, builder's rule" t
perm = to(zeros(Int32, N))
t = best(() -> (sortperm!(perm, keys); synchronize(backend)), 3)
ok = issorted(Array(keys)[Array(perm)])
@printf "%-52s %8.4f s  (sorted: %s)\n" "sortperm! of the UInt64 keys -> Int32" t ok
k2 = copy(keys)
t = best(() -> (copyto!(k2, keys); sort!(k2); synchronize(backend)), 3)
@printf "%-52s %8.4f s  (includes the copy)\n" "sort! of the UInt64 keys" t
ys = similar(xs)
kg = gather!(backend, 256)
t = best(() -> (kg(ys, xs, perm; ndrange=N); synchronize(backend)))
@printf "%-52s %8.4f s\n" "gather of the points by the permutation" t
x32 = to(ones(Int32, N)); s32 = similar(x32)
t = best(() -> (_exclusive_scan!(s32, x32); synchronize(backend)))
@printf "%-52s %8.4f s  (correct: %s)\n" "exclusive scan of 10^7 Int32 (package kernels)" t Array(s32[end:end])[1] == N - 1
one1 = to(zeros(Int32, 1))
kt = touch!(backend, 1)
t = best(() -> kt(one1; ndrange=1), 200)
@printf "%-52s %8.1f µs\n" "kernel launch, no synchronisation" 1e6t
t = best(() -> (kt(one1; ndrange=1); synchronize(backend)), 200)
@printf "%-52s %8.1f µs\n" "launch + synchronize" 1e6t
t = best(() -> (kt(one1; ndrange=1); Array(one1)[1]), 200)
@printf "%-52s %8.1f µs\n" "launch + one-element readback" 1e6t
