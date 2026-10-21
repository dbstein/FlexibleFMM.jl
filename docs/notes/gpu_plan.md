# FlexibleFMM on the GPU with KernelAbstractions: a concrete plan

Scope: 3D `CubeKernelIndependent{3}` with `Laplace3D` and `Stokeslet3D` (stokeslets,
stresslets, doublets; `Velocity`, `Pressure`, `Gradient` outputs), Float32 on Apple
Silicon (Metal.jl, unified memory) and Float64 on CUDA, with `ForwardDiff.Dual`
positions and densities for Newton–Krylov. The tree is built once and reused across
hundreds of matvecs. Everything below names the functions in `src/` as they are today.

All timings marked "est." are estimates from flop and byte counts, not measurements;
I give the counts so they can be re-derived. Things I am unsure of are marked (?).

---------------------------------------------------------------------------------------

## 0. Recommendation in one paragraph

Make the whole matvec device-resident (expansions, pair lists, operators, workspaces
uploaded once per tree; per matvec only `τ`, dual positions and the output move), keep
the tree build, list construction, operator construction (SVDs) and target location
on the CPU, and run the passes as: three KernelAbstractions kernels for the per-item
bodies (`s2m_kernel!`, `evaluate_kernel!` = L2T + near field, plus small gather/scatter
kernels), library GEMMs for M2M/L2L/`FactoredPinv` (MPS on Metal, cuBLAS on CUDA), and
for M2L **a dense batched-GEMM formulation first** (the existing default `m2l_level!`
gather → GEMM → scatter, with the cube method building dense `M2LOperator`s), reduced
from 316 operators to the 16 symmetry classes as the second step. The spectral (FFT)
M2L goes on the GPU last, and only matters for `p ≥ 10` or for memory: at `p = 8` on a
GPU the dense GEMM path is as fast as the Hadamard path and far simpler (§3.3, §4).
On Metal there is no FFT library and the MPSGraph FFT op is not wrapped in Metal.jl; if
the spectral path is wanted there, a 100-line KA "DFT-16 along one axis" kernel is the
right tool (the transforms are < 5 % of the stage's cost). Expected end state for
10⁵ uniform Stokes points at `p = 8`: 4.5 s (12 cores) → ≈ 0.25–0.35 s on an M3 Max
(est., Float32) and ≈ 0.1–0.15 s on an A100 (est., Float64); for surface
distributions, where the dense M2M/L2L dominate on the CPU, the GEMM speedup is larger.

---------------------------------------------------------------------------------------

## 1. Sizes and counts the plan is built on (3D, `p = 8`, `pc = 16`, `d = 0.1`)

| quantity | Laplace | Stokes |
|---|---|---|
| `nsurface` (small, `6p²−12p+8`) | 296 | 296 |
| `nlarge` (large, `pc = 2p`) | 1352 | 1352 |
| `multipole_length` (`nc·nsurface [+1 source]`) | 296 | 889 |
| `partial_multipole_length = local_length` (`nc·nlarge`) | 1352 | 4056 |
| `partial_local_length` (`nc·nsurface [+1 pin]`) | 296 | 889 |
| `M2M` (`nlarge·nc × 8·multipole_length`) | 1352 × 2368, 25.6 MB (F64) | 4056 × 7112, 231 MB (F64) / 115 MB (F32) |
| `L2L` (transpose shape) | same | same |
| `P_up` (`Uh`: rank × 1352·nc, `V`: multipole_length × rank) | 296 × 1352, 296 × 296 | ≤889 × 4056, 889 × ≤889 |
| `P_down` (`Uh`: rank × partial_local_length, `V`: local_length × rank) | 296 × 296, 1352 × 296 | ≤889 × 889, 4056 × ≤889 |
| dense M2L per offset (`partial_local_length × multipole_length`) | 296², 350 KB (F32) | 889², 3.2 MB (F32) / 6.3 MB (F64) |
| 316 dense M2L | 110 MB (F32) | 1.0 GB (F32) / 2.0 GB (F64) |
| 16 symmetry-class M2L (§3.3) | 5.6 MB | 51 MB (F32) / 101 MB (F64) |
| spectral tables (`nc²` × 4096 complex per offset, 316 offsets) | 20 MB (F64) | 186 MB (F64) / 93 MB (F32) |
| spectra per box per component (`16³` complex) | 64 KB (F64) / 32 KB (F32) | ×3 |

Tree for 10⁵ uniform points, `ppl = 50` (complete octree, leaves on level 5):
4096 leaves (24 points each), 4681 nodes; same-level far pairs (exact for a complete
octree, computed combinatorially): level 3: 3 096, level 4: 53 352, level 5: 584 136;
total **640 584** (189 per interior box, 143 on average). For 10⁶ points: leaves on
level 6, **6.04 M** pairs. Near field: ≈ 24 × 27 × 24 ≈ 16k source-target
interactions per leaf, ≈ 55–65 M in total.

Work per matvec (10⁵ Stokes, `p = 8`), consistent with the measured 4.5 s given in
the brief. Note: commit `1c29900` (cache-blocked Hadamard, after the brief's numbers)
and the new FMM3D comparison in `docs/kifmm.md` put the current figures at **2.6 s**
for 10⁵ stokeslets and **0.45 s** for Laplace at `p = 8` (FMM3D: 1.15 s and 0.36 s at
comparable accuracy). The GPU estimates below do not change; the speedup ratios
against the CPU are therefore ≈ 0.6× those quoted when measured against 2.6 s.

| stage | flops | measured / est. CPU |
|---|---|---|
| Hadamard: 640k pairs × 9 × 4096 × 8 | 189 GFlop | 4 s (47 GFlop/s) |
| dense M2L instead: 640k × 2 × 889² | 1.01 TFlop | — |
| M2M + P_up at level 5 (512 groups, 4096 nodes) | 30 + 30 + 6.5 GFlop | ≈ 0.1 s |
| L2L + P_down at level 5 | 30 + 6.5 + 30 GFlop | ≈ 0.1 s |
| all levels, GEMMs | ≈ 150 GFlop | 0.2–0.3 s |
| S2M: 10⁵ × 1352 × ~40 flops | 5 GFlop | ≈ 0.05 s |
| L2T: 10⁵ × 1352 × ~40 | 5 GFlop | ≈ 0.05 s |
| near field: 6 × 10⁷ × ~40 (more with 3 singularities) | 2.5–7 GFlop | ≈ 0.05–0.1 s |
| spectra workspace (Σ̂ + Û, level 5) | — | 1.6 GB (F64), 0.8 GB (F32) |

Laplace: Hadamard 21 GFlop (0.3 s measured), dense M2L 112 GFlop, GEMMs ≈ 20 GFlop.

Sphere, 10⁵ points: 87 000 nodes at `ppl = 50` (33 s), ≈ 22 000 at `ppl = 200`
(5.4 s). The GEMM work scales with the node count (≈ 18× the uniform case at
`ppl = 50`), which is why those trees are GEMM-bound on the CPU and why they gain the
most from library GEMMs on a GPU.

Device peak numbers used below: M3 Max 40-core GPU ≈ 14 TFlop/s FP32 peak, MPS
SGEMM ≈ 7–9 TFlop/s on large shapes (?), unified memory ≈ 400 GB/s; A100 ≈ 19.5
TFlop/s FP64 (tensor core DGEMM ≈ 15 effective), 1.55 TB/s HBM, 40 MB L2; H100 ≈ 50–60
TFlop/s FP64 effective, 3.35 TB/s, 50 MB L2.

---------------------------------------------------------------------------------------

## 2. Device-side data layout and the backend abstraction

### 2.1 One structure, adapted with `Adapt`

Add `Adapt`, `KernelAbstractions` (and weak deps / extensions for `Metal`, `CUDA`,
`cuFFT`) to `Project.toml`. Define device mirrors of the CPU structures with array-type
parameters and `Adapt.@adapt_structure`, so that `adapt(MtlArray, x)` /
`adapt(CuArray, x)` converts every `Array` field recursively:

```julia
struct DeviceTree{D,R,VI<:AbstractVector{Int32},VP<:AbstractVector,MI<:AbstractMatrix{Int32}}
    origin::SVector{D,R}; W0::R
    level_range::Vector{UnitRange{Int}}      # host side, used for launch sizes only
    level::VI; parent::VI; child_first::VI
    plo::VI; phi::VI                         # tree.points split into two Int32 vectors
    leaf::VI                                 # Bool arrays as UInt8/Int32
    centers::VP                              # SVector{D,R} per node, precomputed from cell
    colleagues::MI                           # 27 × nnodes, Int32
    group_parent::Vector{VI}; s2m_nodes::Vector{VI}   # per level
    point_node::VI
end
struct DeviceInteractionList{VI}             # per level
    offset_ptr::VI; tgt::VI; src::VI         # CSR by offset (dense M2L gather/scatter)
    torder::VI; tptr::VI; tsrc::VI; tfar::VI # CSR by target in Morton order (Hadamard, corrections)
    class_ptr::VI; class_perm::VI            # §3.3: pairs grouped by symmetry class, plus
    ctorder::Vector{VI}; ctptr::Vector{VI}; cpos::Vector{VI}   # per-class CSR-by-target scatter lists
end
struct DeviceLevelOperators{T,R,MT<:AbstractMatrix{T},VR,VP}
    width::R; m2m_scale::R; l2l_scale::R; m2l_scale::R; pinv_scale::R
    M2M::MT; L2L::MT
    Uh_up::MT; invS_up::VR; V_up::MT; Uh_down::MT; invS_down::VR; V_down::MT
    small::VP; large::VP                     # SVector{D,R} device arrays
    M2L_dense::Union{Nothing,<:AbstractArray{T,3}}   # 889 × 889 × (316 or 16)
    tables::Union{Nothing,<:AbstractArray{Complex{R},3}}   # nfreq × nc² × 316 (spectral)
    pin_rows::MT; src_cols::MT; embed::VI    # Stokes corrections, lattice embedding
end
```

Rules: index arrays are `Int32` (Apple GPUs do 64-bit integer arithmetic slowly;
halves index traffic everywhere), `Bool` flags become `UInt8`, `UnitRange` vectors
become two vectors, centers are precomputed once per tree (nnodes × 12 B) instead of
recomputed from `cell` (keeps `ldexp`/integer→float conversions out of the kernels;
`center(tree, i)` is exact and so is a stored copy). Nothing holding a `Float64`
field (`CubeKernelIndependent.d`, `.pinv_rtol`) is ever passed to a kernel; kernels
receive the kernel struct `K` (whose constants are already in `floattype(K)`), the
surfaces, `Val(nc)`, `Val(np)` and arrays.

For a homogeneous kernel the matrices are shared between levels (`rescale`): upload
the reference `LevelOperators` once and let the per-level `DeviceLevelOperators`
alias the same device matrices with their own scale factors (exactly what
`rescale` does on the host; `adapt` on a struct that shares a matrix would upload it
once per level, so build the device levels from one uploaded reference explicitly).
The surfaces differ per level (`e.large .* λ`); upload them as a `nlarge × L` matrix
of `SVector{3,R}` (`large_all[:, level[i]]`) so that the evaluation kernel, which
mixes levels, can index them.

### 2.2 Expansions: real arrays with a "parts" dimension (duals and complex)

Library GEMMs do not take `Dual` matrices. The spectral code already splits
expansions into parts (`_nparts`, `_cpart`, `_cassemble` in `spectral_m2l.jl`); make
that the device representation of *all* expansions:

```
multipoles :: AbstractArray{R,3}   (multipole_length, nnodes, np)      np = 1 + npartials
locals     :: AbstractArray{R,3}   (local_length,     nnodes, np)
partial_l  :: AbstractArray{R,3}   (rows_l, nnodes_l, np)   one per level, exact sizes
```

Part-major blocks, so that the sibling-group reshape `(8·Nm, ngroups)` of a level's
block is still free within each part, and the M2M/L2L/pinv GEMMs run once per part
(np = 2 for the one-partial Newton–Krylov duals; a loop over parts, or one GEMM over
the flattened `(rows, nodes·np)` block when the level's block is contiguous in parts,
which it is if the per-level partial arrays are separate). The per-item kernels do
their arithmetic on `Dual`s (as JellySim's kernels do) and split on store /
reassemble on load:

```julia
@inline store_parts!(A, row, col, x::Real, ::Val{1}) = (A[row, col, 1] = x)
@inline function store_parts!(A, row, col, x::ForwardDiff.Dual{T,V,N}, ::Val{np}) where {T,V,N,np}
    A[row, col, 1] = ForwardDiff.value(x)
    ntuple(i -> A[row, col, 1 + i] = ForwardDiff.partials(x, i), Val(N)); nothing
end
@inline load_parts(::Type{S}, A, row, col, ::Val{np})   # inverse, for l2t and the gathers
```

Complex kernels (`Helmholtz3D`) are `Complex{R}` arrays on CUDA (cuBLAS handles them);
on Metal MPS/MPSGraph GEMM is real `Float32`/`Float16` only (its valid-type tables in
Metal 1.9 list no complex types), so complex would need a real/imag split into the
parts dimension (4 real GEMMs, or 3 with the Gauss trick). Not needed for JellySim;
leave it for later.

### 2.3 Workspaces: exact sizes, no views

Metal's `mul!` dispatches to MPS only for `MtlMatrix` = `MtlArray{T,2}`; a `SubArray`
view goes to `GPUArrays.generic_matmatmul!`, a slow generic kernel. (CUDA.jl accepts
`StridedCuMatrix`, so views are fine there.) The present driver uses views
(`view(state.work1, 1:pml, 1:ng)`, `reshape(view(state.partial, 1:pml*nl), pml, nl)`)
to share one workspace between levels; on the device allocate per-level buffers of
exact size instead (sizes are fixed once the tree is): `partial[l]`, `tmp_m2m[l]`,
`pinv_work[l]`, `gather[l]`/`scatter[l]` for the M2L. Memory is small
(level 5 Stokes: 4056 × 4096 × 4 B = 66 MB per buffer in Float32). `reshape` of a
whole device array is free and allowed.

### 2.4 Which arrays live where (per tree vs per matvec)

Uploaded once per tree (`DeviceFMM(fmm; backend)`):
tree arrays (≈ 60 B/node), pair lists (≈ 40 B/pair: 25 MB at 10⁵, 250 MB at 10⁶
uniform), operators (§1 table; one set per width for homogeneous kernels, one per level
otherwise), surfaces, the per-level workspaces and expansions, the target location
and bucketing of the plan's targets (`nodes`, `order`; §3.6).

Per matvec: `τ` in tree order (`N × sizeof(density)`; a `(SVector{3}, Stresslet, Doublet)`
tuple is 12 floats = 48 B in Float32 → 4.8 MB at 10⁵), the positions only when they
are duals (24 B/point in `Dual{Float32,1}`), and the output (12 B/point) back. On
CUDA these are explicit `copyto!`s (< 1 ms over PCIe at 10⁵; use pinned host buffers
`CUDA.pin` for async copies if it ever matters). On Metal allocate the per-matvec
input/output arrays with `storage = Metal.SharedStorage` and keep host views of them
(`unsafe_wrap(Array, ...)`): `stokes_densities` then writes `τ` straight into the
device buffer and the host reads the velocities in place after
`Metal.synchronize()`; no copies at all. Keep the big operator and workspace arrays in
the default `PrivateStorage` (GPU-only pages).

### 2.5 Backend selection

`DeviceFMM{B}` carries `backend::B` (`KernelAbstractions.get_backend(array)`); kernels
are launched as `s2m_kernel!(backend, 256)(args...; ndrange = ...)`. A `KA.CPU()`
backend runs the same kernels on CPU threads, which is how the device code path is
tested without a GPU (and with `Float64`), but the existing threaded CPU code stays
the production CPU path: KA's CPU backend does not vectorise the `@simd ivdep` inner
loops of `_hadamard_block!` and JellySim found the same for its Stokes kernel.

---------------------------------------------------------------------------------------

## 3. Phase by phase

Summary table (device = GPU; "CPU" = stays on the host):

| phase | where | how | data |
|---|---|---|---|
| tree, colleagues, lists, Morton order, group/class lists | CPU, once per tree | `src/tree/*` as is + list post-processing | uploaded as Int32 CSR |
| operator construction (surfaces, kernel tables, SVDs, M2M/L2L, symmetry classes) | CPU, once per width | `level_operators`, `_ki_dense_operators`, `pinv_rtol` | uploaded as device matrices |
| `τ` assembly (`stokes_densities`), permutation to tree order | CPU, per matvec | unchanged | written into a shared/pinned buffer |
| S2M | GPU | `s2m_kernel!`, thread = (check point, listed node) | parts layout |
| M2M | GPU | GEMM per part + `scatter_parent_kernel!` | — |
| `partial_to_multipole!` | GPU | `Uh` GEMM, `.*= invS` broadcast, `V` GEMM | — |
| L2L | GPU | `gather_parent_kernel!` + GEMM | — |
| M2L dense | GPU | per offset or per class: `gather_kernel!` → GEMM → `scatter_by_target_kernel!` | CSR lists |
| M2L spectral (later) | GPU | `embed_kernel!`, `dft_axis_kernel!` ×3 (or cuFFT), `hadamard_kernel!`, `corrections_kernel!`, `dft_axis_kernel!` ×3, `extract_kernel!` | freq-fastest spectra |
| `partial_to_local!` | GPU | as above with `P_down` | — |
| target location | CPU once per plan (GPU kernel optional) | `locate` + `bucket_by_node`, cached in the plan | `nodes`, `order` |
| L2T + near field | GPU | `evaluate_kernel!`, thread = target (bucketed order) | output array |
| result to caller | per matvec | in place (Metal shared) / `copyto!` (CUDA) | — |

### 3.1 S2M — `s2m_kernel!`

The CPU body `s2m!(col, m, ops, xs, τs, c)` loops `k` over the 1352 large-surface
points and `j` over the node's sources. The natural GPU mapping is one thread per
(check point `k`, listed node `q`); consecutive threads are consecutive `k`, so the
component-major stores `pM[(a-1)·N + k, col]` coalesce and the node's sources
(≤ `ppl`) are broadcast reads shared by the 1352 threads of the node.

```julia
@kernel function s2m_kernel!(pM, @Const(nodes), @Const(plo), @Const(phi), @Const(centers),
                             @Const(xs), @Const(τs), @Const(large), K, first::Int32,
                             ::Val{nc}, ::Val{np}) where {nc,np}
    k, q = @index(Global, NTuple)            # ndrange = (nlarge, length(nodes))
    @inbounds begin
        node = nodes[q]; c = centers[node]; y = large[k]
        acc = zero(SVector{nc, S})            # S = promote_op(source_field, K, eltype(xs), eltype(large), eltype(τs)) scalar
        for j in plo[node]:phi[node]
            acc += source_field(K, xs[j] - c, y, τs[j])
        end
        col = node - first + 1
        for a in 1:nc
            store_parts!(pM, (a - 1) * N + k, col, acc[a], Val(np))
        end
    end
end
```

`nc = 1` uses the scalar form (`_field_zero`/`_store!` already dispatch on `Val(nc)`).
Cost: 5 GFlop → ≈ 1–3 ms on either GPU (est.). The level's `s2m_nodes[l]` list is
the launch range; nodes with no sources are skipped by the list as today. Ghost
nodes are ordinary entries of the list.

### 3.2 M2M + `partial_to_multipole!` (and L2L + `partial_to_local!`)

Per level, per part `k`:

```julia
B = reshape(view_whole(multipoles, :, cr, k), 8·Nm, ng)          # whole-array reshape of the part block: no SubArray
mul!(tmp_m2m[l], ops.M2M, B, R(ops.m2m_scale), false)               # library GEMM
scatter_parent_kernel!(backend, 256)(partial[l], tmp_m2m[l], group_parent[l+1], off; ndrange=(pml, ng))
mul!(pinv_work[l], Uh_up, partial[l]); pinv_work[l] .*= invS_up     # broadcast = KA kernel under the hood
mul!(view_whole(multipoles, :, lr, k), V_up, pinv_work[l], R(pinv_scale), false)
```

To avoid `SubArray`s on Metal, keep `multipoles` as one array per level per part
instead of one global matrix (the global column indexing `node - first + 1` already
exists in the driver), i.e. `multipoles[l] :: (Nm, nl, np)`. The gather of the
children block for M2M is then a whole-array reshape of `multipoles[l+1][:, :, k]`
— make the part the slowest index so that `multipoles[l][:, :, k]` is contiguous, and
store them as `np` separate matrices per level to avoid even that slice. (This is the
one place where the "one column per global node" layout has to bend; everything else
addresses level-local columns already.)

The factored solve stays factored (`Uh` → diagonal → `V`), as `docs/kifmm.md`
requires; the diagonal is a broadcast. Folding `invS` into `V` (column scaling) would
also be stable — it is not the folded-pinv problem the doc describes, since a column
scaling has no cancellation — but keeping three operations costs nothing and keeps
the GPU path operation-for-operation comparable with the CPU path.

GEMM cost per level 5 (Stokes): ≈ 130 GFlop total up + down → M3 Max ≈ 15–20 ms
(est., MPS at ≈ 8 TFlop/s), A100 FP64 ≈ 9 ms. All levels ≈ 1.2× that. On the sphere
trees (18× the nodes at `ppl = 50`) ≈ 0.3–0.4 s on M3 Max vs tens of seconds on the CPU.

### 3.3 M2L

**Option A (first): dense GEMM, 316 operators.** The cube method's `level_operators`
gets a switch producing `M2LOperator`s built exactly like the circle method's,
`_multipole_to_check_matrix(K, small .+ Ref(o·w), o·w, w, small, origin)` (889 × 889
for Stokes with the pin row and source column included; `rescale` already works for
`M2LOperator`). Then the *default* `m2l_level!` structure (gather → GEMM →
scatter-add per offset) runs on the device with two small kernels:

```julia
@kernel function gather_kernel!(G, @Const(Mp), @Const(src), lo::Int32)       # ndrange = (Nm, n_f)
    r, q = @index(Global, NTuple); G[r, q] = Mp[r, src[lo + q - 1]]
end
@kernel function scatter_kernel!(pL, @Const(Rw), @Const(tgt), lo::Int32, off::Int32)   # targets unique per offset
    r, q = @index(Global, NTuple); pL[r, tgt[lo + q - 1] - off] += Rw[r, q]   # conflict-free, no atomics
end
```

Per level: 316 × (2 kernels + 1 GEMM of `889² × ~2000`, 3.2 GFlop). Total 1.0 TFlop
at 10⁵ Stokes → ≈ 0.13 s on M3 Max (est.), ≈ 0.07 s A100 FP64, ≈ 0.02 s H100; Laplace
112 GFlop ≈ 0.02–0.04 s (the 296-wide K dimension runs MPS below peak). Launch
overhead ≈ 950 launches per level × 10–50 µs (Metal command buffers are the slow
end (?)) ≈ 10–50 ms per level: noticeable, which is one reason for option A'.
Memory: 1 GB of operators for Stokes in Float32, 2 GB in Float64 — fine on a 64 GB
Mac or an 80 GB A100, acceptable on 40 GB; Laplace 110 MB.

**Option A' (second): 16 symmetry classes.** The 316 far offsets fall into 16
orbits under the 48-element octahedral group (axis permutations and reflections, the
PVFMM/exafmm reduction). The small surface (boundary points of a centred cube
lattice) is invariant, and for the built-in kernels the operators transform
covariantly: `K(Rs, Rt) = K(s, t)` for Laplace, `= R·K(s,t)·Rᵀ` for the Stokeslet,
`pressure_kernel(Rs, Rt) = R·p(s,t)`, `source_velocity(Rs, Rt) = R·v(s,t)`. So
`M2L[R·o] = P_R · M2L[o] · P_Rᵀ` with `P_R` the signed permutation "point permutation
⊗ R on components" extended by the identity on the pin/source entries. Per class `c`
with representative `o_c`: gather with the permutation (`G[r, q] = sgn_f[r] ·
Mp[perm_f[r], s]`, `f` the pair's offset), one GEMM `(889 × 889) × (889 × n_c)` with
`n_c ≈ 40 000` at level 5, and scatter with the inverse permutation. Within a class a
target appears up to ~20 times, so the scatter must not be per pair: use a per-class
CSR-by-target list (`ctorder`, `ctptr`, `cpos`: for each target, the positions of its
pairs in the class's gathered column order) and one thread per (row, target) that
loops over those positions — deterministic, no atomics, result buffer
`889 × n_c × 4 B ≈ 140 MB` per class at 10⁵ (1.3 GB at 10⁶: process classes in
chunks of targets there). This cuts the operator memory 20× (51 MB Stokes Float32)
and the launches to 16 × 3 per level, at the same flop count. The permutations and
signs are built on the CPU once (`far_offsets` orbit enumeration + a test that
`P_R·M2L[o]·P_Rᵀ == M2L[R·o]` to rounding for every offset, which also validates the
covariance claims for each kernel; a user kernel without the symmetry falls back to
option A).

**Option B (later): the spectral path on the device.** See §4 for the Hadamard
kernel. Pieces:

- `embed_kernel!`: thread per (surface point `i`, box, column) writes
  `Σ̂[embed[i], col, box] = part_k(Mp[(b−1)N + i, box])` into a zeroed
  `(nfreq, nc·np, nboxes)` complex array.
- Forward transform of all boxes: on CUDA `plan_fft!(reshape(Σ̂, 16, 16, 16, ncols), 1:3)`
  (cuFFT batches over the trailing dimension; the plan is built once per level size
  and stored). On Metal there is no FFT library: Metal.jl 1.9 wraps the
  `MPSGraphFFTDescriptor` but not the `fastFourierTransform` graph op, and adding that
  via ObjectiveC.jl is possible but untested here (?). Use instead a KA kernel that
  applies the `n × n` DFT matrix along one axis, three launches per transform:

  ```julia
  @kernel function dft_axis_kernel!(Y, @Const(X), @Const(F), ::Val{n}, ::Val{axis}) where {n,axis}
      line = @index(Global)                   # one thread per line of n entries; ndrange = nfreq/n · ncols
      # decode (i, j, col) for the fixed axis; load the n complex entries into an NTuple{n}
      # (32 Float32 registers at n = 16), multiply by F (n×n complex, @Const, L1-resident), store
  end
  ```

  Cost: 3 passes × read+write of the spectra = 6 × 400 MB (Stokes, F32, level 5) ≈
  2.4 GB of traffic ≈ 6 ms on M3 Max (est.), and `n²` MACs per line = 6.4 GFlop
  for 10⁵ Laplace — negligible against the Hadamard stage. The same kernel is the
  portable fallback on CUDA. The embed can be fused into the first pass and the
  extract into the last. Real-input (Hermitian) transforms halve the frequencies for
  real kernels: with cuFFT `plan_rfft` is immediate; in the KA DFT it is a later
  optimisation.
- `hadamard_kernel!` (§4).
- `corrections_kernel!` for the Stokes pin row and source column: thread per target
  looping its pairs (CSR by target) for the 889-long `pin_row · Mp[:, s]` dot
  product, and thread per (row, target) looping pairs for `src_col[f][row] · Mp[end, s]`
  — no atomics (the present CPU code threads over pairs within an offset, which is
  conflict-free only per offset).
- Inverse transform and `extract_kernel!` (thread per (surface point, box, column),
  multiplies by `α/nfreq` and assembles the parts into `pL`).

Memory: `Σ̂` and `Û` = 2 × nboxes × 4096 × nc·np × 8 B (F32): 0.8 GB for 10⁵ Stokes,
6.4 GB for 10⁶; Float64 doubles it. Mitigations are the same as planned for the CPU:
Hermitian symmetry (halves), chunking the targets of a level (only `Û` shrinks), and
Float32 spectra on CUDA (§4.4).

**Why dense first and spectral later.** At `p = 8` the Hadamard stage is
memory-bound on a GPU (§4.2: ≈ 0.75 flop/byte in Float32) while the dense GEMMs run
near peak; the estimates for 10⁵ Stokes are 0.13 s dense vs 0.15–0.6 s spectral on
M3 Max and 0.07 s vs 0.05–0.4 s on A100 — a tie at best for the spectral path, with
6 kernels and an FFT where the dense path has 2 trivial kernels and a library call.
With the 16 classes the dense operators are also *smaller* than the spectral tables
(51 MB vs 93 MB). The spectral path wins when `p ≥ 10` (dense `(488·3+1)² × 16
classes` = 137 MB and 2.3× the flops of `p = 8`, spectral tables 8000 frequencies;
at `p = 12` dense is 7× `p = 8`) or when one wants the 20× flop reduction on a
compute-bound device. So: dense A → A′ delivers the speedup with the least new
code; B is the optimisation for high order.

### 3.4 L2L + `partial_to_local!`

Mirror of §3.2: `gather_parent_kernel!` (thread per (row, group): `G[r, g] =
Lc[r, group_parent[g]]`), one GEMM `L2L · G` into `reshape(partial[l], 8·pll, ng)`,
then the M2L accumulation, then `P_down` as `Uh` GEMM → broadcast → `V` GEMM into
`locals[l]`.

### 3.5 L2T + near field — `evaluate_kernel!`

One thread per target in the bucketed order (`order[q]`), exactly JellySim's
`KernelAbstractionsStokesKernel!` pattern (`@index(Global)`, loop over sources,
`@Const` inputs, `Val(SELF)` for the skip):

```julia
@kernel function evaluate_kernel!(out, @Const(order), @Const(tnodes), @Const(targets), @Const(skip),
                                  @Const(locals_all), @Const(large_all), @Const(level), @Const(centers),
                                  @Const(colleagues), @Const(plo), @Const(phi), @Const(xs), @Const(τs),
                                  K, output, ::Val{nc}, ::Val{np}, ::Val{SELF}) where {nc,np,SELF}
    q = @index(Global); j = order[q]
    @inbounds begin
        i = tnodes[j]; x = targets[j]; l = level[i]
        u = l2t_device(K, large_all, l, locals_all, i, x - centers[i], output, Val(nc), Val(np))   # 1352-term loop
        for k in 1:27
            c = colleagues[k, i]; c == 0 && continue
            for jj in plo[c]:phi[c]
                (SELF && jj == skip[j]) && continue
                u += target_field(K, xs[jj], x, τs[jj], output)
            end
        end
        out[j] = u
    end
end
```

`locals_all` is the per-level locals stacked, or indexed `(level, column)`; with the
per-level arrays of §3.2 pass a tuple of device arrays and index with `l` through a
`Val`-unrolled `ntuple` lookup, or keep one global locals matrix for the evaluation
only (4056 × 4681 × 4 B = 76 MB) filled by a copy per level — simplest. Targets
bucketed by leaf mean the 32 threads of a warp mostly share `i`: the local column,
the colleague ranges and the source ranges are broadcast reads. For
`evaluate_at_sources` the targets are `xs` in tree order, `tnodes = point_node`,
`order = 1:N` and `skip[j] = j`, so the same kernel serves both entry points of
`stokes_fmm_eval_src_trg!`. Cost ≈ 10 GFlop at 10⁵ Stokes → ≈ 3–5 ms (est.).
Output `out` is a device array of `SVector{3,R}` (or of `SVector{3,Dual}`), written
once per target: no atomics, deterministic.

### 3.6 Target location and bucketing

For the solver pattern the targets are fixed per plan: do `locate` and
`bucket_by_node` **once** in `StokesFMMPlan` (today `evaluate!` repeats them on every
call, which is also a free win for the CPU path) and upload `nodes`, `order`. If
moving targets are ever needed: `locate` is a per-target tree walk over
`child_first`/`leaf`/`centers` → a trivial KA kernel; the counting sort needs either
a histogram with atomics + scan or `AcceleratedKernels.sortperm` (installed, works
on Metal and CUDA). Not needed for JellySim.

---------------------------------------------------------------------------------------

## 4. The Hadamard stage on the GPU (option B of §3.3)

### 4.1 Thread mapping and layout

Drop the frequency-blocked real/imag-split layout (`_to_blocked!`/`_from_blocked!`,
`SpectralWorkspace.Sb/Ub`): it exists to give the CPU cache-resident blocks and a
vectorisable real inner loop. On a GPU the natural layout is the FFT's own,
**frequency fastest**: `Σ̂, Û :: (nfreq, nc·np, nboxes)` interleaved complex, tables
`:: (nfreq, nc², nfar)`. A thread owns one (frequency `ω`, target `t`) pair and
loops over the target's pairs from the CSR-by-target list, accumulating `nc·np`
complex values in registers and writing them once:

```julia
@kernel function hadamard_kernel!(Û, @Const(Σ̂), @Const(tables), @Const(torder), @Const(tptr),
                                  @Const(tsrc), @Const(tfar), off::Int32, ::Val{nc}, ::Val{np}) where {nc,np}
    ω, q = @index(Global, NTuple)              # workgroup = 256 consecutive ω of one target; ndrange = (nfreq, ntargets)
    @inbounds begin
        t = torder[q] - off
        acc = ntuple(_ -> zero(eltype(Û)), Val(nc * np))
        for kp in tptr[q]:tptr[q+1]-1
            s = tsrc[kp] - off; f = tfar[kp]
            src = ntuple(c -> Σ̂[ω, c, s], Val(nc * np))                  # coalesced across ω
            tab = ntuple(ab -> tables[ω, ab, f], Val(nc * nc))            # coalesced across ω, shared across targets
            acc = ntuple(i -> (a = (i - 1) ÷ np + 1; k = (i - 1) % np + 1;
                          acc[i] + sum(b -> tab[(a-1)*nc + b] * src[(b-1)*np + k], 1:nc)), Val(nc * np))
        end
        for i in 1:nc*np; Û[ω, i, t] = acc[i]; end
    end
end
```

(`ntuple` with `Val` unrolls at compile time; nothing dynamic.) The generated
`_hadamard_block!` is the same arithmetic with `B` frequencies per call and `nc`
unrolled; here `B` becomes the hardware's SIMD width. No atomics: every `(ω, i, t)`
is written by exactly one thread. The target-major pair walk in Morton order
(`il.torder`) is kept as the launch order of the workgroups (target index fastest),
so that consecutive workgroups use the same 256-frequency slice of the tables and
neighbouring targets share sources — the reuse the CPU design gets from cache, the
GPU gets from L2. Threads with the same `t` and consecutive `ω` read consecutive
addresses in both `Σ̂` and `tables`: fully coalesced. The "one block per target" and
"one thread per frequency looping over all targets" alternatives are worse: the
latter has only 4096 threads (a GPU wants 10⁵–10⁶), the former wastes the
frequency-coalescing.

Sibling-group blocking (one thread per (ω, target *group* of 8), `8·nc·np`
accumulators, pairs enumerated per (target group, source group, direction) with an
8×8 far-mask table per direction): 8× fewer source reads and 8× fewer target writes,
same table reads; worth ≈ 20–40 % once tables come from L2, so a later refinement.

### 4.2 Throughput estimate

Per pair per frequency: `nc²` complex MACs (`8·nc²` flops) for `nc·np` source reads
and `nc²` table reads: Stokes Float32: 72 flops / 96 B = 0.75 flop/B; Float64: 0.37.
Both GPUs are memory-bound here (A100 needs ≈ 12 flop/B FP64 to be compute-bound,
M3 Max ≈ 35 flop/B FP32).

10⁵ Stokes, 640k pairs × 4096 frequencies:

| | bytes from DRAM | M3 Max (400 GB/s) | A100 (1.55 TB/s) | H100 (3.35 TB/s) |
|---|---|---|---|---|
| everything streamed, F32 | 252 GB | 0.6–0.8 s | 0.19 s | 0.08 s |
| everything streamed, F64 | 503 GB | — | 0.38 s | 0.17 s |
| tables L2-resident (one 256-ω slice = 5.8 MB F32 / 11.6 MB F64), sources from DRAM | 63 GB F32 / 126 GB F64 | 0.15–0.25 s | 0.05–0.1 s | 0.03–0.05 s |

The L2-resident case needs the workgroup order above and an L2 of a few tens of MB
(A100 40 MB, H100 50 MB; Apple's GPU L2 plus the 48 MB (?) system cache on M3 Max).
L2 bandwidth then bounds it (≈ 5 TB/s on A100; unknown but likely 1–2 TB/s on Apple
(?)). For Laplace divide by ≈ 9 (table bytes) and 3 (sources): ≈ 0.02–0.05 s on
M3 Max. Compare the CPU: 4 s Stokes, 0.3 s Laplace. So the Hadamard stage alone gains
≈ 15–25× on M3 Max and 40–80× on A100 — but the dense GEMM path (§3.3) reaches the
same 0.1–0.15 s with none of this machinery, which is the basis of the recommendation.

### 4.3 Memory

Level-5 spectra: 4096 boxes × 4096 × 3 × 8 B = 403 MB each for `Σ̂` and `Û` in
Float32 (806 MB in Float64), × `np` for duals; 10⁶ points: 3.2 GB each (F32). Tables
93 MB (F32) / 186 MB (F64). Chunk the targets of a level when `Û` does not fit (the
sources `Σ̂` must be whole); Hermitian symmetry halves everything for real kernels.

### 4.4 Float32 spectra with Float64 elsewhere (CUDA)

Rounding of the convolution in Float32 is ≈ 1e-7·√(pairs) ≈ 1e-6 relative in the
check data; the method's own translation error at `p = 8` is 1e-6 (Laplace) to 1e-3
(doublets), so it is at the noise floor of the method at `p = 8` and not acceptable at
`p ≥ 10` (3e-8). The amplification by `P_down` (`1/σ` up to 1e12 at the Float64
truncation) is the question: by the argument of `docs/kifmm.md` the amplified error
stays in the weakly-radiating modes and is attenuated again by `l2t`, so the output
error should stay ≈ 1e-6 — but that is the same argument the folded-pinv experiment
once contradicted in a different setting, so **measure it** (`test_3d.jl` layouts,
Float32 spectra vs Float64, against direct sums) before adopting it. It halves the
Hadamard traffic and memory. The safe default is one precision end to end: Float64
on CUDA, Float32 on Metal (where there is no choice).

---------------------------------------------------------------------------------------

## 5. Reusing the per-item bodies under KernelAbstractions

What already works: `source_field`/`target_field` (all Stokes singularities, tuples
via `_sum_fields`/`_sum_targets`, which inline), the kernels as callables on
`SVector`s with typed constants (`K.c::T`), `Val`-dimensioned `ncomp`, `_store!`/
`_load` on `Val(nc)`, the allocation-free bodies (tested by `test_threads.jl`), and
`Dual` arithmetic (ForwardDiff's rules are plain Julia; JellySim runs
`MtlArray{SVector{3,Dual{Tag,Float32,1}}}` through the same kind of kernel, adding only
`Metal.rsqrt_fast(::Dual)` because it uses the hardware `rsqrt`; FlexibleFMM's kernels
use `sqrt` and `/`, which need nothing). Density tuples such as
`Tuple{SVector{3,F}, Stresslet{3,F}, Doublet{3,F}}` are `isbits`, so a device vector
of them works directly; this is cleaner than JellySim's separate arrays with
`nothing` flags and keeps `stokes_densities` unchanged. The weights are folded in on
the host as today.

Changes needed:

1. **Accumulator types from the arguments, not from the output array.** `s2m!` uses
   `_field_zero(eltype(col), nc)`; on the device `col` is real while the accumulator
   may be `Dual` → use `Base.promote_op(source_field, typeof(K), eltype(xs),
   eltype(large), eltype(τs))`. `l2t` builds `S` from `eltype(L)` likewise →
   `load_parts(S, …)`.
2. **No `AbstractMethod`/`LevelOperators` in kernels.** `s2m!(col, m, ops, …)` and
   `l2t(m, ops, L, …)` take the method and operator structs; the kernels take `K`,
   the surface array(s), `Val(nc)`, `Val(np)`. Refactor the bodies into
   `_s2m_point(K, xs, τs, lo, hi, c, y, ::Val{nc})` (one check point) and
   `_l2t_point(K, large, L, xrel, output, ::Val{nc})` and call them from both the CPU
   functions and the kernels — one implementation, two wrappers.
3. **Index types.** Loops over `plo[node]:phi[node]` on `Int32`; `@index` gives
   `Int`; mixing is fine but keep arrays `Int32`.
4. **Float64 literals.** The existing rule (constants in `floattype(K)`) holds; audit
   the fallbacks: `density_scalar`-based `zero(promote_type(…, typeof(K.c)))` is
   fine; `_pressure_row` etc. are host-only. Metal has no Float64 at all — a stray
   `Float64` in a kernel is a compile error, not a slowdown, so it is caught early.
5. **`Gradient()` for stresslets/doublets** uses `ForwardDiff.jacobian(tt ->
   source_field(K, s, tt, τ), t)` inside the body; `ForwardDiff.jacobian` on an
   `SVector` is allocation-free and should compile (nested tags), but on Metal this
   is the most likely body to fail (?); JellySim does not need `Gradient`, so hand-
   write those two `target_field` methods if and when needed.
6. **`Rotlet` and `Dipole`** go through `gradient_s`/`cross` — fine.
7. **Dynamic dispatch**: none in the bodies once `K`, the density tuple type, the
   output type and `nc`/`np` are type parameters; each combination is a separate
   GPU compilation (see risks).
8. **The near-field skip**: `skip[j]` as an `Int32` array or `Val(SELF)` with
   `skip = j`, as JellySim does.

---------------------------------------------------------------------------------------

## 6. Data movement and the solver loop

`StokesFMMPlan(sources; targets, backend = :CPU | :Metal | :CUDA, …)`:
builds the CPU `FMM` as now, then `DeviceFMM(fmm, backend)` uploads everything of
§2.4 and allocates the device state (`DeviceFMMState{R,np}` for the expansion eltype
and parts count the plan will see — rebuilt if the next call brings duals when the
previous did not, as `_state!` does today). Per `stokes_fmm_eval!`:

1. host: `stokes_densities` → `τ` (tree order via `perm`), into the shared/pinned
   buffer; positions likewise only if dual (`Dual` arrays are `isbits`, upload as is).
2. device: `upward_pass!`, `downward_pass!`, `evaluate_kernel!` (self and/or targets).
3. host: read the output buffer (Metal: in place after `synchronize`; CUDA:
   `copyto!`), permute from tree order for the self case (`tree.perm`), as today.

Duals and device arrays: positions/densities/outputs stay as arrays of `Dual`
(`SVector{3,Dual{Tag,F,N}}`), exactly JellySim's `to_device` convention — no
StructArrays needed; `Adapt` converts `Array{Dual}` to `MtlArray{Dual}` unchanged.
Only the expansions use the split parts layout (§2.2), because they go through
library GEMMs. The CPU `FMMState{S}` with `S = Dual` is unchanged.

KA launches are asynchronous; a matvec is a sequence of ≈ 100–1000 launches
(dense-316: ≈ 4000; classes: ≈ 300) on one queue/stream, synchronised once at the
end. On Metal, keep the whole pass inside one `Metal.@sync`-free sequence and call
`synchronize()` only before the host reads the result.

---------------------------------------------------------------------------------------

## 7. Staged implementation, milestones, expected speedups, testing, risks

Baseline: 10⁵ uniform Stokes, `p = 8`, `ppl = 50`: CPU 4.5 s (Hadamard 4.0, GEMMs
≈ 0.3, S2M/L2T/near ≈ 0.2) as given in the brief; the current code measures 2.6 s
(stokeslets) and 0.45 s (Laplace) per the new table in `docs/kifmm.md`, with the
Hadamard stage presumably ≈ 2 s of the 2.6 s. The GPU estimates are absolute, so read
the ratios below against whichever baseline applies (≈ 8–10× on M3 Max and ≈ 20×
on A100 after stage 3 against 2.6 s; FMM3D on 12 cores is 1.15 s, so stage 3 is also
the point where the GPU path overtakes FMM3D by ≈ 3–4× on an M3 Max).

| stage | content | est. matvec, M3 Max F32 | est. A100 F64 | notes |
|---|---|---|---|---|
| 0 | CPU-side prep: Int32 flat arrays, `plo/phi`, centers, parts layout for duals, per-level buffers, cached target location in the plan, `Adapt` mirrors, dense cube `M2LOperator`s + symmetry-class lists, `KA.CPU()` path | 4.5 s (unchanged) | — | pure refactor, tested against the current outputs bitwise where the arithmetic order is unchanged |
| 1 | `evaluate_kernel!` (L2T + near field) with the state built on the CPU and the locals uploaded | ≈ 4.4 s | — | first device code: exercises Duals, tuples, Float32, `@Const`, outputs; validates the KA plumbing with the least risk |
| 2 | upward pass on device: `s2m_kernel!`, M2M/P_up GEMMs | ≈ 4.3 s | — | multipoles resident; the pinv in Float32 on Metal is now measurable |
| 3 | downward pass with **dense M2L (316 ops)** + L2L/P_down → fully device-resident matvec | ≈ 0.3–0.45 s (M2L 0.13 + GEMM 0.05 + launches 0.05–0.2 + rest 0.02) | ≈ 0.12–0.2 s | **the speedup milestone**, ≈ 10–15× / 25–35× |
| 4 | 16 symmetry classes | ≈ 0.25–0.3 s | ≈ 0.1 s | 20× less operator memory, 10× fewer launches; needed for `p = 10` dense |
| 5 | spectral M2L on device (KA DFT or cuFFT, `hadamard_kernel!`) | ≈ 0.2–0.4 s | ≈ 0.08–0.15 s | only for `p ≥ 10` / memory; the Hermitian and Float32-spectra options sit here |
| 6 | refinements: sibling-group blocking, real-input transforms, Float32 spectra on CUDA, target chunking for 10⁶+ | — | — | each measured individually |

Sphere 10⁵ at `ppl = 200` (CPU 5.4 s): after stage 3 ≈ 0.6–1 s on M3 Max (est.; ≈ 2 M
pairs → 3 TFlop dense M2L dominates); at `ppl = 50` (33 s) ≈ 3–5 s — the GEMM stages
scale with nodes and pairs, so `ppl ≈ 200` remains the right setting on the GPU too.

**First milestone.** Stage 1 (the evaluation kernel), not the Hadamard stage: it is
the JellySim pattern verbatim, it is the piece that must work with duals, Float32 and
density tuples, it is testable in isolation against `evaluate_target`, and it costs a
day. The speed comes from stage 3, whose dense M2L needs no new numerics; the
Hadamard stage on the GPU (stage 5) is the most work for the least gain at `p = 8`.

**Testing** (`test/test_gpu.jl`, backend chosen at runtime: `Metal.functional()`,
`CUDA.functional()`, else `KA.CPU()` so the device code path always runs in CI):

- per phase against the CPU arrays (download and compare): S2M and the evaluation
  kernel have the same summation order as the CPU bodies, so expect ULP-level
  agreement (GPU compilers contract `a*b+c` to FMA by default, CPU Julia does not,
  so bitwise equality is not achievable; `≤ 4 ulp` elementwise in the same precision
  is the target, with an explicit check of the maximum ulp distance); GEMM phases
  against `mul!` on the host at `1e-13` (F64) / `1e-5` (F32) relative in the
  Frobenius norm; M2L dense vs spectral CPU at `1e-12`/`1e-5`; the symmetry classes
  by `P_R·M2L[o]·P_Rᵀ == M2L[R·o]` for all 316 offsets and all kernels;
- end to end: the whole `test_3d.jl` and `test_stokes.jl` matrix (uniform, sphere,
  helix, sphere + cluster; stokeslet/stresslet/doublet; `Velocity`, `Pressure`,
  `Gradient`; dual positions; self-evaluation) through `StokesFMMPlan(…; backend)`
  against direct sums with the same tolerances as the CPU path (Float64) and the
  Float32 tolerances measured on the CPU first;
- determinism: two runs on the same device agree bitwise (no atomics anywhere; cuBLAS
  and MPS are deterministic for a fixed configuration (?)); the output does not depend
  on the workgroup size;
- allocation: no host allocations inside a matvec after the first call (the solver
  loop); `CUDA.@allocated`/`Metal.@allocated` for device-side allocations in the
  per-level loops.

**Risks, with mitigations:**

1. *Float32 pseudo-inverses on Metal.* 2D Stokes Float32 reaches only 5e-3 (`docs/
   stokes.md`); 3D at `p = 8` is unmeasured. This is a numerics question independent
   of the GPU: measure it now on the CPU path (`precision = Float32`, the `test_3d`
   layouts, `conditioning(ops)`). If 5e-3 is too coarse for the solver, the hybrid
   "GEMMs on the host in Float64" is available on Apple Silicon at zero copy cost
   (unified memory; `P_up`/`P_down` at level 5 are ≈ 40 GFlop ≈ 40–80 ms on 12 cores),
   or try `pinv_rtol` and `pc` variations. No Float64 exists on Metal.
2. *FFT on Metal.* None in Metal.jl beyond the MPSGraph descriptor; the KA DFT kernel
   is the answer, and the dense path removes the need at `p = 8`.
3. *Compile times.* Each (kernel, density tuple type, output type, dual-or-not,
   backend) combination compiles separately, tens of seconds each on Metal (?);
   enable `GPUCompiler.enable_disk_cache!()` (a preference; supported by CUDA.jl and
   Metal.jl through GPUCompiler) and keep the number of combinations small (one
   `evaluate_kernel!` per output type; the density tuple is fixed per plan).
4. *Dense operator sizes.* 1 GB (F32) / 2 GB (F64) at `p = 8` Stokes for 316
   operators until stage 4; unusable at `p ≥ 10` without the classes. M2M/L2L at
   115/231 MB per width are fine; non-homogeneous kernels multiply them by the level
   count.
5. *Metal GEMM on views.* Silent 10–100× slowdown through `GPUArrays.generic_matmatmul!`;
   the exact-size per-level buffers avoid it, and a test should assert
   `typeof(A) <: MtlMatrix` for every GEMM operand on Metal.
6. *Launch overhead.* 316 offsets × 3 launches × levels; measure on Metal, which is
   the slower side (?); the classes reduce it 20×.
7. *Spectra memory at 10⁶.* 6–13 GB; chunk targets per level; Hermitian symmetry.
8. *`ForwardDiff.jacobian` inside device code* for `Gradient()` of stresslets/doublets:
   hand-write if it fails to compile.
9. *Int64 on Apple GPUs.* Keep index arrays `Int32`; `@index` arithmetic in `Int` is
   unavoidable but cheap.
10. *Complex kernels on Metal.* No complex GEMM; split into parts if `Helmholtz3D`
    is ever wanted there.

**What I could not check here** (a test suite was running, so no Julia was run
beyond reading): the actual MPS GEMM rate on this M3 Max for the 889-wide shapes,
Metal launch overhead per command buffer, the Apple GPU L2 size, and the Float32
accuracy of the 3D Stokes pinv; each is a 5-minute measurement once the machine is
free and each moves an estimate above by at most ≈ 2×.
