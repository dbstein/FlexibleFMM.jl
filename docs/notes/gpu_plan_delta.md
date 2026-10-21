# GPU plan: delta after commits 5aae32f (folded pseudo-inverses) and 4a1288e (real-input transforms, symmetric tables)

Read against `docs/notes/gpu_plan.md` (= the scratchpad `gpu_plan.md`). Sections of
that plan that this delta supersedes are listed at the end. Sources read: the two
commit diffs, the current `src/fmm.jl`, `src/operators/{ki_common,spectral_m2l,cubeki}.jl`
(including the uncommitted `ysoa` change), `docs/kifmm.md`, `docs/notes/pvfmm_report.md`,
and CUDA.jl's `cuFFT/src/fft.jl`. No Julia was run; the orbit count below was computed
with a 20-line Python script (`scratchpad/orbits.py`).

## 0. What changed and what it does to the plan

1. **Folded pseudo-inverses.** `M2M = V·(Σ⁻¹·(Uᴴ·M2Mᶜ))` is now `889 × 7112` (Stokes,
   `p = 8`) and maps children multipoles to the parent multipole; `L2L = ((L2Lᶜ·V)·Σ⁻¹)·Uᴴ`
   is `7112 × 889` and maps the parent's *partial local* (check data, kept per node in
   `FMMState.plocals`) to the children's partial locals. `P_up` is applied only to the
   check data of the childless nodes (`partial_to_multipole!` before the M2M at each
   level), `P_down` only to produce the densities the evaluation reads. Both folded
   operators are width-invariant for a homogeneous kernel (`rescale` keeps
   `m2m_scale = l2l_scale = 1`). Per matvec the GEMM budget drops from ≈ 150 to ≈ 70–100
   GFlop and the two operators from 231 MB to 50 MB each (Float64).
2. **Half spectra and 6 tables.** Real expansions of a real kernel use `plan_rfft` /
   `plan_brfft` along dimension 1 of the `(2p)³` cube: `nfreq = (p+1)(2p)² = 2304`
   (0.5625× of 4096), tables `trh/tih` are the half slabs, and `symmetric_kernel(K)`
   (Stokeslet) makes `_hadamard_block!` load 6 of the 9 tables (still 9 complex products
   per entry: 12 table + 6 source reals for 36 real FMAs). Complex kernels/densities keep
   the full complex path. Spectra are 72 KB per box per component (Laplace, Float64),
   ×3 for Stokes.
3. **Measured now** (12 cores, 10⁵ uniform, `p = 8`, **ppl = 200**): Laplace 0.31 s
   (translations 0.21), Stokes 1.41 s (translations 1.05, upward 0.16, near field 0.20);
   FMM3D 0.36 / 1.15 s. The brief's 4.5 s was at ppl = 50 before both commits.
4. **S2M/L2T** run sources-outer, check-points-inner over `KIExtras.ysoa` (`nlarge × D`
   SoA); only 1.25× because the bodies are `sqrt`/`div`-bound on 2-wide NEON. This is
   a signal for the GPU: those bodies are throughput-bound on `rsqrt`, which GPUs do
   cheaply (JellySim's `select_kernel_basics` → `Metal.rsqrt_fast` / `CUDA.rsqrt`), so
   their GPU gain is larger than the flop ratio alone suggests.

Consequence for the tree shape in the cost model: at ppl = 200 a complete level-4 leaf
layer has 195 points per box, so ≈ 36 % of the 512 boxes exceed 200 and split, their
neighbours get ghost children (≈ 2 600 ghosts), and the level-5 pair count is driven by
≈ 1 500 real targets: **≈ 320 k pairs (est. ± 50 k)** against 640 k for the ppl = 50
complete tree — consistent with the measured 1.05 s (≈ 320 k × 2304 × 72 flops ≈ 53
GFlop at the stated 50 GFlop/s). Node count ≈ 4 700 either way (10⁶: ≈ 37 k nodes,
≈ 3–6 M pairs). I give every estimate for both pair counts.

## 1. Revised per-pair cost model (Stokes, `p = 8`)

| quantity | spectral (half spectrum) | dense M2L (`889²`) |
|---|---|---|
| flops per pair | 2304 × 72 = 166 kFlop | 2 × 889² = 1.58 MFlop (9.5×) |
| bytes per pair, Float32, all streamed | 2304 × (3 src + 6 tab) × 8 B = 166 KB | GEMM operands: `G`, `R` 2 × 889 × 4 B = 7 KB + operator amortised |
| bytes per pair, Float32, tables L2-resident | 2304 × 24 B = 55 KB | — |
| Float64 | ×2 | ×2 |
| arithmetic intensity (streamed) | 1.0 flop/B (F32), 0.5 (F64) | ≈ 200+ flop/B (GEMM) |
| table slice per 256 frequencies | 316 × 6 × 256 × 8 B = 3.9 MB (F32) / 7.8 MB (F64) | class operator 3.2 MB (F32) |
| Laplace | 18 kFlop, 37 KB (F32) per pair | 175 kFlop (9.5×) |

Devices as in the plan: M3 Max ≈ 300–400 GB/s DRAM, MPS SGEMM ≈ 7–9 TFlop/s (?); A100
1.55 TB/s, DGEMM ≈ 15 TFlop/s, 40 MB L2; H100 3.35 TB/s, DGEMM ≈ 50 TFlop/s, 50 MB L2.

## 2. (a) Dense-GEMM-first versus the spectral path, revised

M2L stage only (the rest is common), Stokes 10⁵:

| pairs | path | M3 Max F32 | A100 F64 | H100 F64 |
|---|---|---|---|---|
| 640 k (ppl 50) | spectral, all streamed: 106 GB F32 / 212 GB F64 | 0.30–0.35 s | 0.14–0.16 s | 0.065 s |
| 640 k | spectral, tables in L2: 35 GB F32 / 71 GB F64 (+ L2-bandwidth bound) | 0.10–0.12 s | 0.05 s | 0.02 s |
| 640 k | dense: 1.01 TFlop (A: 316 ops, A′: 16 classes) | 0.11–0.14 s | 0.067 s | 0.02 s |
| 320 k (ppl 200) | spectral, all streamed | 0.15–0.18 s | 0.07–0.08 s | 0.03 s |
| 320 k | spectral, tables in L2 | 0.05–0.06 s | 0.025 s | 0.01 s |
| 320 k | dense: 0.51 TFlop | 0.06–0.07 s | 0.034 s | 0.01 s |

plus, for the spectral path, the transforms and the dense Stokes corrections: R2C/C2R
of 4 096 boxes × 3 components ≈ 1 GB of traffic ≈ 3–5 ms on M3 Max (KA DFT, 3 passes
each way: ≈ 8 ms), ≈ 1–2 ms with cuFFT; the pin/source corrections read the 889-vector
of the source once per pair (2.3 GB at 640 k pairs, F32) ≈ 6 ms.

Reading: the half spectrum and the 6 tables cut the spectral bytes per pair by ≈ 0.56 ×
0.78 ≈ 0.44, so on a GPU the spectral M2L is now **at parity with the dense GEMM path
when its table slices stay L2-resident and ≈ 2–2.5× slower when they stream**; the
dense path's estimate is the more certain of the two (GEMM rates are known, L2
residency on Apple GPUs is not). The dense path therefore still goes first: same or
better time at `p = 8`, no FFT on Metal, two trivial kernels plus a library call, and
exactly the permute–GEMM–permute shape PVFMM's own GPU offload uses (`EvalListGPU`).
The spectral GPU path remains the choice for `p ≥ 10` (dense flops grow as `(nc·N)²`,
spectral as `(2p)³`), and it is no longer the memory-saving option at `p = 8`: its
tables are 35 MB (F32) but its spectra are 110 KB per box (Stokes, F32), i.e. 3.6 GB
at 10⁶, whereas the dense-class path's working set is bounded by chunking (§4).

Whole matvec, Stokes 10⁵ at ppl = 200 (CPU 1.41 s), dense-class M2L:

| stage | M3 Max F32 (est.) | A100 F64 (est.) |
|---|---|---|
| S2M + P_up (29.5 + 6.5 GFlop GEMM at level 5 over all 4 096 childless columns) | 6–8 ms | 4–5 ms |
| M2M (6.5 GFlop) + L2L (6.5) + P_down on leaves (≤ 36 GFlop) | 6–8 ms | 4–5 ms |
| M2L dense, 16 classes | 60–70 ms | 34 ms |
| near field (≈ 5 × 10⁸ interactions ≈ 30 GFlop) + L2T (5 GFlop) | 8–12 ms | 8–12 ms |
| ≈ 250 launches | 5–25 ms (?) | 2–5 ms |
| total | **0.09–0.13 s (≈ 11–15×)** | **≈ 0.055–0.065 s (≈ 22–25×)** |

Laplace (0.31 s): dense M2L 56 GFlop with a 296-wide inner dimension ≈ 10–20 ms on MPS;
total ≈ 0.03–0.05 s on M3 Max (≈ 6–10×), ≈ 0.02 s on A100 (≈ 15×). With the spectral
M2L instead, the Stokes total is 0.08–0.22 s on M3 Max and 0.05–0.11 s on A100. For
ppl = 50 trees double the M2L row and quarter the near-field row. Against FMM3D on 12
cores (1.15 s) the stage-3 GPU path is ≈ 10× faster on the M3 Max.

## 3. (b) The 16 symmetry classes of the dense M2L operators, exactly

**Geometry.** The small surface `x_1..x_N` (`N = 296`, `cube_boundary_points`) is the
boundary of a centred `p³` lattice with `coords[i] = -a + 2a(i-1)/(p-1)`, so it is
invariant under the 48 signed axis permutations `R_g` of `O_h`: `R_g x_i = x_{ρ_g(i)}`
with `ρ_g` an index permutation obtained from the lattice index tuple (a sign flip on
axis `d` sends index `i_d ↦ p+1-i_d`, an axis permutation permutes the tuple) — build
`ρ_g` by index arithmetic, never by comparing floats.

**Orbits.** The 316 far offsets `o ∈ {-3..3}³`, `‖o‖_∞ ≥ 2`, fall into **16 orbits**
under `O_h` (computed by enumeration: `scratchpad/orbits.py`): representatives with
`|o|` sorted descending and orbit sizes `(3,2,1): 48`; `(2,1,0), (2,1,1), (2,2,1),
(3,1,0), (3,1,1), (3,2,0), (3,2,2), (3,3,1), (3,3,2): 24` each; `(2,2,0), (3,3,0):
12`; `(2,2,2), (3,3,3): 8`; `(2,0,0), (3,0,0): 6`; total `48 + 216 + 24 + 16 + 12 =
316`. This is PVFMM's `class_hash` (sort `|c|`), which also gives 16 for `V_Type`.
All 316 offsets occur in the sibling-group geometry (`2d + c_s − c_t` over the 26
parent directions and 8 × 8 child pairs), so no class is empty.

**Vectors.** A multipole `m ∈ R^889` is component-major: `m[(b-1)N + i] = σ_b(x_i)`,
`m[889] = q` (point-source strength, a scalar flux). A partial local `c ∈ R^889`:
`c[(a-1)N + j] = u_a(y_j)` at the target's small-surface point `y_j = x_j`,
`c[889] = w·p(0)` (pinned pressure at the target centre, a scalar). `M2L_o` is the
`889 × 889` matrix of `_multipole_to_check_matrix(K, small .+ Ref(o·w), o·w, w, small,
origin)`: the velocity block `K(o·w + x_i, y_j)` (`3 × 3` per point pair), the source
column `w·v(o·w, y_j)` (`source_velocity`), the pressure row
`w·pressure_kernel(o·w + x_i, 0)`, and a zero at `(889, 889)` (a point source has no
pressure).

**Covariance.** For every orthogonal `R`, in particular every signed permutation:
`K(Rs, Rt) = R·K(s,t)·Rᵀ` (Stokeslet: a tensor built from `I` and `d dᵀ/r²`),
`pressure_kernel(Rs, Rt) = R·pressure_kernel(s,t)` (`∝ d/r³`), `source_velocity(Rs,
Rt) = R·source_velocity(s,t)` (`∝ d/r³`); Laplace: `K(Rs, Rt) = K(s,t)`. Reflections
(`det R = −1`) are included — nothing here is a pseudo-vector (a rotlet would be, but
rotlets never enter an M2L: the multipole is a stokeslet density plus a scalar source).
This is what PVFMM finds numerically in `Kernel::Initialize` as `K(Rx) = P1·K(x)·P2`.

**The signed permutation `P_g` on `R^889`.** Write `R_g` as `(R_g)_{ab} = s_b·δ_{a,π(b)}`
(axis `b` goes to axis `π(b)` with sign `s_b`). Define

    (P_g m)[(π(b)-1)·N + ρ_g(i)] = s_b · m[(b-1)·N + i]      for b = 1..3, i = 1..N
    (P_g m)[889]                 = m[889]

i.e. the index map `(b, i) ↦ (π(b), ρ_g(i))` with sign `s_b`, and the augmented entry
fixed with sign `+1`. `P_g` is orthogonal (`P_g⁻¹ = P_gᵀ`), and the same `P_g` acts on
check vectors `c` (velocity components transform like density components; the pinned
pressure is the invariant scalar `p'(0) = Σ_i pk(R(o w + x_i), 0)·(Rσ_i) = Σ_i
pk(o w + x_i, 0)·σ_i`). Substituting `x_{ρ(i)} = R x_i`, `y_{ρ(j)} = R y_j` into the
definition of `M2L_{Ro}` gives, entry by entry,

    M2L_{R_g o} = P_g · M2L_o · P_gᵀ            (all 889 rows and columns, augmentations included).

**Application for class `C` with representative `o_C`.** For each member offset
`f` choose one `g_f` with `R_{g_f} o_C = o_f` (any element of the coset works; take
the first found) and precompute two `Int32` index tables and two `Int8` sign tables
of length 889, `perm_in[f], sgn_in[f]` for `P_gᵀ` and `perm_out[f], sgn_out[f]` for
`P_g` (for a signed permutation `(P y)[r] = sgn[r]·y[idx[r]]`). Then for every pair
`(s, t)` at offset `f ∈ C`:

    c_t += P_g · ( M2L_{o_C} · ( P_gᵀ · m_s ) )

as three device steps per class and level: `gather_kernel!`: `G[r, q] = sgn_in[f_q][r]
· Mp[perm_in[f_q][r], s_q]` (thread per `(r, q)`); one GEMM `R = M2L_{o_C} · G`
(`889 × 889` by `889 × n_C`, `n_C ≈ 20–90 k` at level 5 for 10⁵ points, the orbit
sizes 6..48 making the classes unequal); `scatter_by_target_kernel!`: thread per
`(r, t)` looping over the target's pairs in class `C` (a per-class CSR-by-target list
`ctptr/cpos`), `pL[r, t] += Σ sgn_out[f][r] · R[perm_out[f][r], q]` — a target has up
to ≈ 20 pairs in one class (orbit size 48 → up to 48), so this must be the
target-owned form; no atomics, deterministic. Dual/complex parts are extra columns;
the permutation is the same for every part. `m2l_scale` multiplies the class
operator as today. Operator memory `16 × 889² × 4 B = 51 MB` (F32) / 101 MB (F64)
against 1.0 / 2.0 GB for 316; GEMM flops unchanged; launches 48 instead of 948 per
level.

**Validation and fallback.** At operator-build time assert
`‖P_g M2L_{o_C} P_gᵀ − M2L_{o_f}‖ ≤ 100·eps·‖M2L_{o_f}‖` for every one of the 316
offsets (a one-time `316 × 889²` check, ≈ 0.25 GFlop); a kernel for which it fails (a
user kernel that is not isotropic, e.g. `LaplaceDipoleTest`-like anisotropy) falls
back to the 316-operator path automatically. A `test/test_symmetry.jl` should run the
assertion for `Laplace3D` and `Stokeslet3D` in Float64 and Float32, plus an end-to-end
equality of the class path against the 316-operator path to rounding.

**PVFMM.** As `docs/notes/pvfmm_report.md` §1.2/§2.3 records, PVFMM computes the 16
classes for every list type but builds the permutations (`PrecompPerm`,
`equiv_surf_perm`, `Perm_R/Perm_C`) only for M2M/L2L (`assert(type != V_Type)`): its
V-list is spectral, and a spatial symmetry acting on spectra re-strides the frequency
runs. Our dense M2L operators are ordinary matrices, so the M2M/L2L mechanism
transfers to them unchanged; this is a use PVFMM does not make, hence the assertion
above rather than trust.

## 4. (c) Revised staging; the half spectrum on cuFFT and on a KA DFT

**Device pass with the folded operators** (per level, Stokes sizes):

- upward `L..3`: `s2m_kernel!` into `checkup[l]` (`4056 × n_childless_l`, gathered by
  `s2m_nodes[l]` so the pinv GEMMs do not run over the parents' zero columns — the CPU
  runs them over the whole level; the saving is up to ≈ 2× at mixed levels); `P_up`:
  `Uh_up` GEMM (`889 × 4056`), broadcast `invS`, `V_up` GEMM into the childless
  columns of `multipoles[l]` (`889 × n_l`); then `M2M` on `reshape(multipoles[l+1],
  7112, ng)` → `889 × ng` → `scatter_parent_kernel!`.
- downward `3..L`: `gather_parent_kernel!` of `plocals[l-1]` by `group_parent` →
  `L2L` GEMM (`7112 × 889`) into `partial[l]` (`889 × n_l`); M2L (dense-class, §3)
  accumulates into `partial[l]`; keep it as `plocals[l]` (`889 × n_l`, 17 MB for 10⁵
  points in F32 over all levels); `P_down` only on the real leaves of the level
  (`Uh_down` `889 × 889`, `V_down` `4056 × 889`, columns gathered by a leaf list) into
  `locals[l]` (`4056 × n_leaves_l`; at ppl = 200 ≈ 64 % of level 5 are ghosts, so this
  saves ≈ 2/3 of the largest downward GEMM there and of the `locals` memory).

GEMM budget per matvec ≈ 70–100 GFlop (level 5 dominates: 29.5 + 6.5 up, 6.5 + 6.5 +
≤ 36 down) ≈ 10–12 ms on M3 Max, ≈ 6 ms on A100. The device-side `FMMState` becomes
per level: `checkup[l]`, `multipoles[l]`, `partial[l]`, `plocals[l]`, `locals[l]`,
`pinv_work[l]`, `m2m_tmp[l]`, `l2l_gather[l]`, all exact sizes (no views, §2.3 of the
plan).

**Staging** (order unchanged; content of stages 2–3 shrinks):

| stage | content | note |
|---|---|---|
| 0 | CPU prep as before, plus: `s2m_nodes` and leaf lists as Int32 device arrays, class lists (`class_ptr`, per-class `ctorder/ctptr/cpos`), `perm_in/out`, `sgn_in/out` per offset, dense cube `M2LOperator`s, the symmetry assertion | — |
| 1 | `evaluate_kernel!` (L2T + near field) on CPU-built `locals` | first milestone, unchanged |
| 2 | upward: `s2m_kernel!`, `P_up` GEMMs, folded `M2M` GEMM | now 3 GEMM calls per level, 4.6× smaller M2M |
| 3 | downward: `L2L` on `plocals`, dense-class M2L (go straight to A′; A as the test oracle only), `P_down` on leaves; full device matvec | **speedup milestone** (≈ 0.1 s M3 Max, ≈ 0.06 s A100 for Stokes 10⁵) |
| 4 | spectral M2L on device with the half spectrum (below) | for `p ≥ 10`; measure L2 residency first |
| 5 | refinements: Float32 spectra on CUDA, sibling blocking, chunking at 10⁶+ | — |

**Half spectrum on cuFFT.** `plan_rfft(X, 1:3)` on a `CuArray{R,4}` of size
`(16, 16, 16, ncols)` returns `(9, 16, 16, ncols)` complex: cuFFT halves `region[1]`
exactly as FFTW does (CUDA.jl's `fft.jl` enforces an increasing region for that
reason), so the device layout equals the CPU's `hdims = (n÷2+1, n, n, ncp, nl)` and
the `trh/tih` half tables and the embed indices apply verbatim; `plan_brfft(Û, 16,
1:3)` returns unnormalised like FFTW's `brfft`, so the CPU's `β = α/n³` carries over.
Plans are built once per level size and stored (they own device scratch). Two
caveats from CUDA.jl's source: R2C/C2R need aligned base addresses per batch (it
rotates into scratch when they are not — pass whole arrays, not views), and the
transform is out-of-place: the real lattice array `X` can alias `Û`'s buffer
(`reinterpret`/`unsafe_wrap` of the device buffer, as the CPU workspace does with
`X`/`Y`) since `Û` is dead during the forward transform and `Σ̂` during the backward.

**Half spectrum with a KA DFT on Metal.** Forward: pass 1 along axis 1 is a
real-input 16-point DFT keeping outputs `k = 0..8` (thread per line: 16 real loads,
9 complex stores, `16 × 9` MACs); passes 2 and 3 are complex 16-point DFTs along
axes 2 and 3 of the `(9, 16, 16)` slab. Backward: complex inverse DFTs along axes 3
and 2, then C2R along axis 1 from the 9 stored entries,
`x_n = X_0 + (−1)^n X_8 + 2·Re Σ_{k=1}^{7} X_k e^{2πikn/16}` (unnormalised, matching
`brfft`). Same layout as FFTW/cuFFT, so one `hadamard_kernel!` and one table array
serve both backends and the device spectra are bitwise-comparable with the CPU's
up to FMA contraction. Traffic ≈ 1.3 GB per direction at 10⁵ Stokes (F32) ≈ 4 ms
(est.). The `(p+1)(2p)²` count factors as `9 × 256`, so workgroups of 256
consecutive frequencies tile a target's spectrum exactly (9 per target); the CPU's
`B = 16` blocking has no device counterpart.

## 5. (d) Memory budget: dense-class versus spectral, Stokes `p = 8`

Sizes per item (F32 on the 32 GB M3 Max / F64 on the 40 GB A100); node counts
≈ 4.7 k (10⁵) and ≈ 37 k (10⁶) at either ppl; pairs 320–640 k and 3–6 M.

| item | 10⁵ | 10⁶ |
|---|---|---|
| operators common: folded M2M + L2L (2 × 889 × 7112), pinv factors (`Uh_up` 889 × 4056, `V_up` 889², `Uh_down` 889², `V_down` 4056 × 889), surfaces | 86 / 172 MB | same |
| expansions: `multipoles` + `plocals` (2 × 889/node) + `locals` (4056/node, leaves) | 110 / 220 MB | 0.88 / 1.75 GB |
| level work: `checkup` (4056 × childless), `partial`, `pinv_work`, M2M/L2L temps | 110 / 220 MB | 0.9 / 1.8 GB |
| pair lists (≈ 10 Int32 per pair incl. class lists) | 13–26 MB | 120–240 MB |
| points, densities (48 B), output, perm | 8 MB | 80 MB |
| **dense-class extra**: 16 class operators | 51 / 101 MB | same |
| dense-class extra: gather + result buffers, chunked at ≤ 65 536 columns | 0.47 / 0.93 GB | same (chunked) |
| (dense-316 instead of classes) | 1.0 / 2.0 GB | same |
| **spectral extra**: half tables, 6 distinct (+ pin/src rows) | 37 / 72 MB | same |
| spectral extra: `Σ̂` + `Û`, 2 × 2304 × 3 × boxes of the largest level | 0.45 / 0.91 GB | 3.6 / 7.2 GB |
| spectral extra: real staging `X` if not aliased to `Û` | 0.2 / 0.4 GB | 1.6 / 3.2 GB |
| **total dense-class** | ≈ 0.9 / 1.8 GB | ≈ 2.7 / 5.4 GB |
| **total spectral** (X aliased) | ≈ 0.85 / 1.7 GB | ≈ 5.8 / 11.6 GB |
| with one-partial duals (`np = 2`: expansions, work, spectra ×2) | ≈ 1.5 / 3 GB either path | dense ≈ 4.6 / 9.2 GB; spectral ≈ 10.5 / 21 GB |

Both paths fit both devices at both sizes. The only case near a limit is the spectral
path at 10⁶ with duals in Float64 on the 40 GB A100 (≈ 21 GB, plus cuFFT scratch) and
in Float32 on the 32 GB Mac (≈ 10.5 GB of unified memory next to the OS and the CPU
state, which holds the same expansions again, ≈ 2 GB); chunking the targets of a level
halves the spectra term. The dense-class path's working set is bounded by the chunk
size, ≈ 3 GB at 10⁶ in Float32, and the 316-operator variant costs 1–2 GB more.
Laplace: divide the expansion, work and spectra rows by ≈ 3 and the operators by ≈ 9.

## 6. What this supersedes in `gpu_plan.md`

- §1 table: `M2M`/`L2L` rows → `889 × 7112` / `7112 × 889`, 25 MB (F32) / 50 MB (F64)
  each (Laplace `296 × 2368`); the pinv rows unchanged but applied to childless nodes
  and leaves only; the "work per matvec" GEMM row 150 → 70–100 GFlop; spectra 197 KB →
  110 KB per box (Stokes, F32); tables 93 MB → 35 MB (F32).
- §3.2/§3.4: `partial_to_multipole!` precedes the M2M on each level; `L2L` consumes
  `plocals`; `P_down` only feeds the evaluation. The remark that fragmented trees are
  GEMM-bound stands but with a smaller constant (sphere, ppl = 200: 4.6 → 3.7 s).
- §3.3/§4.2: the Hadamard bytes per pair-frequency 96 → 72 B (F32); the dense/spectral
  comparison is the table in §2 above; the conclusion (dense first) stands, the margin
  is smaller and depends on L2 residency of the 3.9 MB table slices.
- §4.1: workgroups of 256 frequencies, 9 per target; the kernel reads 6 tables for the
  Stokeslet and forms 9 products; the CPU's real/imag-split blocked layout is still
  not needed on the device.
- §4.3: spectra 0.45 GB (10⁵) / 3.6 GB (10⁶) in F32 for Stokes.
- §7: baseline 1.41 s Stokes / 0.31 s Laplace at ppl = 200; stage-3 estimates 0.09–0.13
  s M3 Max (≈ 11–15×) and ≈ 0.06 s A100 (≈ 22–25×) for Stokes; ≈ 0.03–0.05 s / 0.02 s
  for Laplace (≈ 6–10× / 15×). Stage 4 becomes A′ (classes) only as a test oracle
  ordering: build the class path directly in stage 3.
- §5 item 1: the device `s2m_kernel!` keeps thread = (check point, node) with the
  node's sources as the inner loop; `ysoa` is the right device layout for the surface
  (coalesced across consecutive `k`), so upload `ysoa` rather than the `SVector` array.
