# What FlexibleFMM can take from kifmm (bempp/kifmm)

Sources studied (all read, nothing built or run):

- `bempp/kifmm` at commit `d4ca4b5` (clone in
  `/private/tmp/claude-504/-Users-dstein-Documents-Repos-flexmm/56594ad4-d0ed-4daa-b0c3-59387940395c/scratchpad/kifmm_src`;
  paths below are relative to `kifmm/src/` unless stated):
  `fmm/field_translation/source_to_target/{fft/single_node.rs, hadamard.rs,
  blas/svd/single_node.rs, blas/aca/single_node.rs, transfer_vector.rs}`,
  `fmm/field_translation/metadata/single_node/{metadata.rs, source/laplace.rs,
  target/laplace.rs, source_to_target/laplace.rs, source_to_target/helmholtz.rs,
  source/helmholtz.rs}`, `fmm/field_translation/{source,target}/single_node.rs`,
  `fmm/{eval/single_node.rs, builder/single_node.rs, helpers/single_node.rs,
  types.rs, constants.rs, kernel.rs, isa.rs}`, `linalg/{pinv.rs, rsvd.rs, aca.rs}`,
  `tree/{single_node.rs, morton.rs, constants.rs, types.rs, helpers.rs}`,
  `traits/{fftw.rs, general/single_node.rs}`, `fftw/{dft.rs, r2c.rs}`,
  `examples/*.rs`, `benches/laplace.rs`, `bench-conf-fmm.yaml`,
  `scripts/src/bin/grid_search_single_node.rs`, `python/experiments/*.ipynb`,
  `paper/paper.md` (JOSS).
- `skailasa/green-kernels` (`src/laplace_3d.rs`, `helmholtz_3d.rs`, `traits.rs`),
  `skailasa/rlst` branch `tmp` (`src/dense/simd.rs`, the `approx_recip_sqrt`),
  `sarah-quinones/pulp` (`src/aarch64.rs`, `Arch::new`).
- Kailasa, Betcke, El Kazdadi, "M2L Translation Operators for Kernel Independent
  Fast Multipole Methods on Modern Architectures", arXiv:2408.07436v4 (May 2025),
  32 pp.; text extracted to `scratchpad/paper_txt/m2l_paper.txt`. All performance
  numbers quoted below are from its Tables 1–8.

Our side (for the mapping): `src/operators/{cubeki.jl, ki_common.jl,
spectral_m2l.jl, symmetry.jl, interface.jl}`, `src/{fmm.jl, evaluate.jl,
kernels.jl, device.jl, device_passes.jl, stokes_api.jl}`, `src/tree/lists.jl`,
`docs/{kifmm,framework,stokes}.md`, `test/test_3d.jl`, `test/test_cubeki.jl`.
Throughout, "p = 8 Stokes" is our default: `nsurface = 296`, `nlarge = 1352`
(pc = 16), `multipole_length = partial_local_length = 889`, `local_length = 4056`.

Already done on our side of the PVFMM list (so not repeated here): the folded
`M2M = V·(Σ⁻¹·(Uᴴ·M2Mᶜ))` and `L2L = ((L2Lᶜ·V)·Σ⁻¹)·Uᴴ` (`ki_common.jl` 188–189,
PVFMM #1/#8), the real-input transform path (`SpectralWorkspace` `RP`, PVFMM #3),
FFTW threads (`num_threads=nt`, PVFMM #5).

---

## 0. Ranked recommendations

| # | Recommendation | New vs PVFMM? | kifmm evidence | Our files | Expected gain | Effort |
|---|---|---|---|---|---|---|
| 1 | **Compressed BLAS-M2L** (Fong–Darve single-approximation + Messner per-direction recompression, rSVD precompute): `Kₜ ≈ U·(Ūₜ·V̄'ₜ)·Sᵀ`, applied as one `Sᵀ` GEMM per level, two small GEMMs per far offset, one `U` GEMM per level | **New** (PVFMM has no BLAS M2L) | `blas/svd/single_node.rs` `m2l`; `metadata/.../source_to_target/laplace.rs` `source_to_target` (SaRcmp); paper Tables 2–5: f64 M2L time 1.6–3.7× below their FFT-M2L at 1e-6…1e-10, operator storage 2.3–34 MB vs 51–223 MB; 194 GFlop/s on M1 Pro (AMX) | CPU: `interface.jl` default `m2l_level!` (gather/GEMM/scatter per offset over `InteractionList` CSR, already "uncompressed BLAS-M2L"), new operator type beside `SpectralM2L`; GPU: `device_passes.jl` `chunks`/`_class_gather!`/`_class_scatter!` grouped by offset instead of class | Stokes CPU M2L (the 1.05 s): per pair `4·k·kₜ` ≈ 270 kFlop at k≈450, kₜ≈150 — same flops as the Hadamard but at GEMM rate (300–700 GFlop/s vs 50): est. 3–5×; Laplace M2L ≈ 3×; GPU: 2–3× at ≥10⁶ points, none at 10⁵ (316 groups × ~500 columns). **Ranks must be measured first** (½ day) | Medium: precompute (SVD of `296×93 536` / `888×280 608` + augmentation rows) 1 day; CPU apply 1 day; GPU 1 day |
| 2 | **Kernel-dependent `pc`**: `pc = p` (or `p+2`) for single-layer kernels, `2p` only for the augmented (stresslet/doublet) Stokes operators | New as evidence (PVFMM report left it open) | FFT path: `Pc = Pe` (`FieldTranslation::surface_diff() = 0`), reaches L2-rel 1e-8 at P = 8, 1e-10 at P = 10 (Table 6); BLAS path `Pc = Pe + 1…2` with `Pe` one less (Table 7); `ALPHA 1.05/2.95` | `cubeki.jl` `CubeKernelIndependent{D}(kernel; pc=2p)`, `stokes_api.jl` `pc === nothing && (pc = 2p)`; `ki_common.jl` `nlarge`, `local_length` | S2M/L2T kernel evaluations per point 1352 → 296 (4.6×) for Laplace/Yukawa/Helmholtz and single-layer Stokes plans; `P_down` 889×4056 → 889×889; `locals` memory /4.6 (CPU and `device.jl` `Ldn`) | Low: default + corner tests (`test_3d.jl`, `test_cubeki.jl`) at pc = p, p+2 |
| 3 | **Reciprocal square root by estimate + Newton, branch-free self term** in the kernel bodies | Partly new (PVFMM/SCTL uses `1/sqrt` in f64 on AArch64; kifmm uses `vrsqrte`+3 Newton steps in f64) | green-kernels `evaluate_laplace_one_target` + rlst `simd_approx_recip_sqrt`; Table 1: 0.57 G f64 / 1.5 G f32 Laplace interactions per s per M1 Pro core | `kernels.jl` `radial_value(::Laplace3D)`, `Stokeslet3D` body; `evaluate.jl` `_direct_range`; `ki_common.jl` `s2m!`, `l2t` | 1.2–2× on the three kernel-evaluation stages in f64 if they are sqrt/div-bound (measure: a microbenchmark of `_direct_range` against 0.5 G/s/core) | Low (pure-Julia bit-trick seed + Newton; `ifelse` mask) |
| 4 | **Apple Accelerate as BLAS** on Apple Silicon (`using AppleAccelerate`) | New | Table 4 vs 5: BLAS-M2L 194 GFlop/s f64 on 8-core M1 Pro vs 77 on 32-core AMD/OpenBLAS; precompute 9.6 s vs 28.9 s | zero code; affects `fmm.jl` `mul!` (M2M/L2L), `ki_common.jl` `apply!` (pinv), any BLAS-M2L | 2–3× on the dense f64 GEMMs (the "dense M2M/L2L on fragmented trees" cost); essential for #1 on the CPU | Trivial; check f32 accuracy (their footnote on Accelerate f32) |
| 5 | **Calibration of the sibling-group Hadamard layout** (PVFMM #2): kifmm implements exactly PVFMM's 26×(8×8) frequency-major scheme and measures it | Same idea as PVFMM; new numbers | `fft/single_node.rs` step 2, `hadamard.rs`; Table 4/5: whole M2L stage 53–83 GFlop/s f64 on 8 M1 Pro cores = 2.1–3.2 flop/cycle/core, vs our Hadamard-only 70 GFlop/s on 12 M3 Max cores = 1.4 flop/cycle/core (Stokes 1.0) | `spectral_m2l.jl` `_spectral_m2l_level!`, `_hadamard_block!` | Confirms 1.5–2.2× per core-cycle for the PVFMM layout, consistent with the PVFMM report's 1.5–2.5× estimate; but #1 is the larger lever for Stokes | Medium (as in PVFMM report) |
| 6 | Zero-sentinel source group (8 zero columns appended; missing halo neighbours index them) | New detail | `fft/single_node.rs` `nzeros = 8`, `zero_displacement` in `displacements` | only if #5 is built: `tree/lists.jl` group list | removes the per-slot branch in the sibling-group walk | Trivial, with #5 |
| 7 | Variable expansion order per level (`expansion_order: &[usize]`, rectangular M2M/L2L) | New | `builder/single_node.rs` `parameters`, `source/laplace.rs` level loop, `metadata.rs` `MetadataAccess` | `cubeki.jl` `level_operators`, `OperatorCache` | Helmholtz only (more points on coarse levels); nothing for Laplace/Stokes | Medium; not now |

Not applicable / not recommended (§8): W/X lists (kifmm has none either: uniform tree
with empty-branch pruning), adaptive refinement, MPI ghost exchange, ACA+
pseudo-inverse (our check matrices are 1352×296: the SVD is milliseconds),
per-leaf stored surfaces, per-call buffer allocation, the 2× table copy, the
per-target `Mutex` scatter, the `is_adjacent`/`HashSet` list construction
(44–6164 ms "map creation"; our counting sort is the better design), multiple
right-hand-side evaluation (`FmmEvalType::Matrix`), C/Python bindings. There is no
GPU code in kifmm at all; our `device_passes.jl` is ahead of it.

---

## 1. kifmm in one page, and where it equals PVFMM

- **Tree** (`tree/single_node.rs`): uniform octree to depth
  `minimum_depth(N, n_crit)` (divide N by 8 until ≤ n_crit; 10⁶/150 → 5), Morton
  keys from 16-bit anchors with 256-entry encode/decode tables
  (`tree/constants.rs`, `morton.rs`). `prune_empty` drops empty leaves but keeps
  every sibling of a non-empty leaf and of every ancestor
  (`uniform_tree_pruned`), so every level is a sequence of complete sibling
  octets in Morton order — the same storage order as our `child_first` groups.
  No adaptivity, no W/X/U0/U2 lists; the near field is the 27 same-level leaf
  neighbours (`target/single_node.rs` `p2p`). The paper argues (§2.1) that a fast
  P2P makes adaptivity unnecessary; their battleship test (1.44 M points, depth 5,
  2184 non-empty leaves holding 1…10⁵ points) costs 47 % (FFT) / 8 % (BLAS) more
  than uniform data.
- **Surfaces** (`morton.rs` `surface_grid`, `scale_surface`): the boundary of a
  `P`-per-edge lattice, `6(P−1)²+2` points, on cubes of side `α·w` with
  `ALPHA_INNER = 1.05` (upward equivalent, downward check) and
  `ALPHA_OUTER = 2.95` (upward check, downward equivalent) — PVFMM's radii, i.e.
  `d = 0.05 / 0.025` in our parametrisation (we use 0.1). The check surface has
  order `Pc = Pe + surface_diff` where `surface_diff` is 0 for the FFT path
  (`FieldTranslation::overdetermined() == false`) and a user parameter (1–2 in
  practice) for the BLAS path (`builder/single_node.rs` `parameters`). Locals and
  multipoles both have `n_equiv = 6(Pe−1)²+2` entries; only the check data have
  `n_check`.
- **Check-to-equivalent solves** (`linalg/pinv.rs` `pinv`): SVD, singular values
  below `(atol + rtol)·σ₁` zeroed, `rtol = max(M,N)·ε_mach` by default (SciPy's
  rule; 602·2.2e-16 = 1.3e-13 for a 602×296 map; ours 50 ε = 1.1e-14; PVFMM 4 ε),
  returned as two factors `(V·Σ⁺, Uᵀ)` ("store in two parts to avoid propagating
  error", `types.rs` field docs, paper §2.1.1 footnote 3) — the same safe
  factorisation as ours.
- **M2M / L2L** (`metadata/single_node/source/laplace.rs`,
  `target/laplace.rs`): `M2M = uc2e_inv_1·(uc2e_inv_2·ce2pc)` folded in the safe
  order, stored once as `n_equiv × 8·n_equiv` (`source[level]`) and applied as one
  GEMM over chunks of 16 parents (`M2M_MAX_BLOCK_SIZE`);
  `L2L_child = dc2e_inv_1·(dc2e_inv_2·pe2cc)·2^{−level}` stored per child
  (`target_vec`), applied as 8 GEMMs per chunk. Identical to what we now have.
- **P2M / L2P** (`source/single_node.rs` `p2m`, `target/single_node.rs` `l2p`):
  kernel evaluation to the leaf's upward check surface (`Pc` points), then the
  two pinv factors as GEMMs over chunks of 32 leaves; L2P evaluates the `Pe`-point
  outer surface density at the targets. The leaf surfaces are precomputed and
  stored for every leaf (`leaf_upward_check_surfaces_sources`, …): memory for
  nothing, not worth copying.
- **Homogeneous scaling** (`helpers/single_node.rs` `m2l_scale`,
  `homogenous_kernel_scale`): one operator set computed at level 3 and scaled per
  level, as ours (`rescale`). Helmholtz gets one operator set per level
  (`source/helmholtz.rs`: `curr = curr.first_child()` loop) and complex (c2c)
  transforms.
- **Threading**: rayon `par_iter`/`par_chunks` everywhere; no pinning; BLAS calls
  single-threaded inside parallel loops on high-core-count machines, threaded
  inside a serial loop on laptops (paper §3.2).
- **Precision**: `f32`/`f64` (and `c32`/`c64`) end to end; nothing precision-
  specific beyond the pinv `rtol` and the SIMD kernels.

Where it differs from PVFMM: no adaptive tree, no W/X lists, no MPI tree in the
single-node path (the MPI build keeps separate ghost FMMs, `ghost_exchange.rs`),
no CUDA, and — the point of the paper — a second M2L family based on compressed
dense operators.

---

## 2. The FFT-based M2L (`fft/single_node.rs`, `hadamard.rs`)

### 2.1 Layout

Identical to PVFMM's `V1_Type` scheme (PVFMM report §1), so only the deltas are
listed.

- **Tables** (`metadata/.../source_to_target/laplace.rs` `source_to_target`): for
  each of the 26 halo positions of a parent (`parent.neighbors()`, Morton order
  of `DIRECTIONS`) and each of the 64 (sibling, halo-child) pairs, the kernel is
  evaluated on the `(2P−1)³` convolution grid around the source box relative to
  the first check point of the target (`evaluate_greens_fct_convolution_grid`),
  zero-padded to `(2P)³`, flipped (`flip3`), R2C-transformed
  (`size_out = (2P)²(P+1)`: 1008 / 2304 / 4400 at P = 6/8/10), and zero when the
  child pair is adjacent. Stored twice: `kernel_data` (Morton/child order) and
  `kernel_data_f` (frequency-major, 64-complex block per (frequency, halo)).
  The paper's "FFT-M2L storage" (Table 4: 51 / 117 / 223 MiB at P = 6/8/10, f64)
  is exactly `2 × 26·64·size_out·16 B`, i.e. both copies are kept. Only the
  second is used at run time; do not copy this.
- **Signal spectra**: per level, a frequency-major buffer
  `signals_hat_f[freq·(n_sources+8) + box]` (one R2C spectrum per box, the 8
  siblings of a parent contiguous), **plus 8 zero columns**; a parent whose halo
  neighbour is absent (domain boundary or pruned) indexes those zeros
  (`displacements` → `zero_displacement = nsources_parents·8`), so the Hadamard
  loop has no branch. ~10 % of the 8×8 products are wasted on zero tables or zero
  sources (paper A.2).
- **Hadamard** (step 2): `(0..size_out).into_par_iter()` — one rayon task per
  frequency (2304 tasks at P = 8), each walking all target parents in chunks of
  `chunk_size_kernel ≤ block_size` (default 128; best 32–128 on M1 Pro, 16–32 on
  the 32 KB-L1 AMD, Table 6), inner loop over the 26 halo positions with the
  1 KB table block hoisted, then over the parents of the chunk:
  `hadamard8x8(table, 8 source coefficients, 8 target accumulators, scale)`.
  Working set per task ≈ 26 KB tables + 256 B per parent in the chunk
  (≈ 58 KB at 128 parents): L1-resident on Apple (128 KB L1d), not on x86 (hence
  the smaller chunks there).
- **8×8 kernel** (`hadamard.rs`): AArch64 uses `pulp::aarch64::NeonFcma`
  (`Isa::new()`), i.e. the ARMv8.3 complex multiply-accumulate instructions
  (`c64s_mul_add_e`); 8 accumulators, fully unrolled, `scale` applied in the
  epilogue. `c32` packs two complexes per register. x86 uses AVX2 with
  `shuffle_ps` + `addsub` (no FMA for the complex product). A generic
  autovectorised fallback exists. **For us FCMA brings nothing**: our split
  real/imaginary layout (`_hadamard_block!`) already does a complex MAC in 4 full-
  width FMAs per lane pair, the same instruction count as `fcmla` rot0+rot90 on
  interleaved data, without the interleaving.
- **Post-processing** (step 3): transpose frequency-major → box-major
  (parallel over targets), batched C2R (`c2r_batch_par`, one plan executed per
  box in rayon tasks — PVFMM report #5), `conv_to_surf` extraction, then
  `dc2e_inv_1·(dc2e_inv_2·check)` as two GEMMs with 8 columns per sibling group,
  added into the locals. The pseudo-inverse is therefore applied at M2L time on
  every box (not only leaves), like our `partial_to_local!`.
- **Memory**: four per-level buffers (`signals_hat_f`, `check_potentials_hat_f`,
  `check_potential_hat_c`: `size_out × n_boxes` complex; `check_potential`:
  `size_in × n_boxes` real) are **allocated on every call** (`AlignedVec::new`)
  and sized for the whole level. No chunking by a byte budget; our 10⁶-point
  spectra memory problem is not addressed by kifmm (PVFMM's `n_blk0` blocking,
  PVFMM report #6, remains the reference).
- **Scaling for homogeneous kernels**: `m2l_scale(l)·homogenous_kernel_scale(l)`
  folded into the 8×8 epilogue.

### 2.2 Measured performance (paper Tables 4–6; Laplace, uniform points)

| machine | prec. | ε (L2-rel) | P | tree depth | block | T_M2L (ms) | throughput (GFlop/s) | table storage |
|---|---|---|---|---|---|---|---|---|
| M1 Pro, 8 P-cores, 200 GB/s | f32 | 1e-3 | 3 | 5 | 128 | 64 | 113 | 3.65 MiB |
| | f32 | 1e-4 | 4 | 5 | 64 | 163 | 101 | 8.1 |
| | f64 | 1e-6 | 6 | 5 | 128 | 813 | 67 | 51 |
| | f64 | 1e-8 | 8 | 4 | 32 | 257 | 53 | 117 |
| | f64 | 1e-10 | 10 | 4 | 64 | 482 | 83 | 223 |
| AMD 3790X, 32 cores, 102 GB/s | f64 | 1e-6 | 6 | 4 | 64 | 240 | 28 | 51 |
| | f64 | 1e-8 | 8 | 4 | 32 | 449 | 36 | 117 |
| | f64 | 1e-10 | 10 | 4 | 16 | 893 | 36 | 223 |

(N = 10⁶ sources = targets; depth-4 trees have 4096 leaves of ≈ 244 points,
depth-5 ≈ 30 points. The "throughput" counts the 8×8 Hadamard flops over the
whole `m2l(level)` time, which includes the FFTs, transposes and the pinv GEMMs.
Checking: depth 4, P = 8: 4672 boxes × 2304 freq × 26·64·8 flops/8 boxes =
17.9 GFlop in 257 ms = 70 GFlop/s, consistent with their 53 once the zero blocks
are discounted.) The paper's reading: the FFT-M2L saturates at ≈ 80 GFlop/s f64 /
120 f32 on the M1 Pro and ≈ 40 / 80 on the AMD, i.e. it is bandwidth-bound once
the tables exceed L2.

**Calibration against ours.** Per core and per clock, kifmm's *whole* M2L stage
runs at 53–83 GFlop/s / 8 cores / 3.2 GHz = 2.1–3.2 flop/cycle; our Hadamard
stage alone runs at 70 GFlop/s / 12 cores / 4.05 GHz = 1.4 flop/cycle (Laplace)
and 1.0 (Stokes). So the PVFMM/kifmm sibling-block layout is at least 1.5–2.2×
ours per core-cycle (more, since their figure also pays for FFTs and transposes),
which matches the 1.5–2.5× the PVFMM report estimated for #2 — now with a
measurement behind it. Their per-box cost at 1e-8 (P = 8, depth 4) is 257 ms /
4672 boxes ≈ 55 µs per box ≈ 0.3 µs per pair. If our "translations 0.21 s" for the
10⁵-point case is mostly M2L, that is ≈ 350 µs per box for ≈ 600 boxes, far from
the Hadamard's share (≈ 40–60 k pairs × 0.5 µs ≈ 0.03 s at 70 GFlop/s): profile
the 0.21 s (S2M/L2T on 1352 points, M2M/L2L, FFTs, blocking transposes) before
spending effort on the Hadamard itself. Recommendation #2 attacks the S2M/L2T
part directly.

---

## 3. The BLAS-based M2L (new relative to PVFMM)

### 3.1 Precomputation (`metadata/single_node/source_to_target/laplace.rs`, impl for `BlasFieldTranslationSaRcmp`)

For the 316 transfer vectors `t` (`transfer_vector.rs`
`compute_transfer_vectors_at_level(3)`: all non-adjacent child pairs of a sibling
octet and its parent's neighbours, deduplicated by hash):

1. `K_fat = [K_1 … K_316]` (`n_check × 316·n_equiv`) and `K_thin = [K_1; …; K_316]`
   (`316·n_check × n_equiv`), `K_t = kernel(check_t, equiv_t)` assembled by
   `assemble_st`.
2. SVD of `K_fat` → `U` (`n_check × k`), of `K_thin` → `S` (`n_equiv × k`)
   (skipped when `Pc = Pe`: then `S = U`). Either deterministic (`gesvd`) or
   **randomised** (`linalg/rsvd.rs` `rsvd_fixed_rank`: Gaussian test matrix,
   target rank `max(n_equiv, n_check)/2` + `n_oversamples` (5–20), **no power
   iteration** by default, optional QR power iterations `Normaliser::Qr(n)`).
   `find_cutoff_rank(σ, threshold)` cuts at the first `σ_i ≤ threshold`
   (an **absolute** threshold on the singular values of `K_fat`; 1e-3…1e-7 in the
   benchmarks).
3. Per direction: `C_t = Σ·V_tᵀ·S` (`k × k`), SVD → `Ū_t` (`k × k_t`),
   `V̄'_t = Σ̄_t V̄_tᵀ` (`k_t × k`), `k_t` from the same threshold, computed in a
   parallel loop over the 316 directions (`into_par_iter`).

Stored (`BlasMetadataSaRcmp`): `u`, `st`, `c_u[316]`, `c_vt[316]`. For a
homogeneous kernel one set, scaled per level by the same scalar as the FFT path.
The "M2L compression" column of Table 7 is `1 − Σ_t k·k_t / (316 k²)` (82–97 %).

Precompute cost (Table 7, "M2L precalculation"): f64 Laplace, Pe/Pc = 5/6:
326 ms (M1 Pro, Accelerate) / 966 ms (AMD, OpenBLAS); 8/10: 5.0 s / 5.1 s;
9/11: 9.6 s / 28.9 s. Versus the FFT tables: 0.8–1.8 s. (Our case: `K_fat` for
Laplace p = 8 is `296 × 93 536`; a deterministic SVD costs ≈ 296²·93 536 ≈ 8 GFlop,
well under a second; the Stokes `888 × 280 608` matrix (+ the pin row / point-
source column for the augmented operators) is ≈ 2×10¹¹ flop deterministic, a few
seconds with rSVD.)

### 3.2 Run time (`blas/svd/single_node.rs` `m2l`, `FmmEvalType::Vector`)

Per level:

1. `q̃ = Sᵀ·[q_1 … q_{N_σ}]` — one GEMM (`k × n_equiv × N_σ`), scaled by the
   homogeneity factor.
2. `(0..316).into_par_iter()`: for direction `t`, gather the `N_t` compressed
   multipoles of the sources having a target at offset `t`
   (`displacements[t][source] = target index or −1`, a `316 × N_σ` `i32` table
   built at setup), then `ξ = V̄'_t·q̃_sub` (`k_t × k × N_t`), `φ̃ = Ū_t·ξ`
   (`k × k_t × N_t`), scatter-add into the compressed check potentials of the
   targets under a **per-target `Mutex`** (`compressed_level_check_potentials`).
3. `Φ = U·Φ̃` — one GEMM (`n_check × k × N_τ`), then
   `dc2e_inv_1·(dc2e_inv_2·Φ)` and add into the locals.

Flops per pair `4·k·k_t`; per box `4·k·(n_equiv + n_check)/2`-ish plus the pinv.
The gather/scatter (`k`-vectors) is memory-bound (OI ½, paper Appendix B) but
small: `2·N_pairs·k·8 B`. The paper's operational-intensity argument for step 2
is `4·8ˡ·k/(3·8ˡ+k)`: it grows with `k`, which is why the method wins at high
accuracy.

### 3.3 Measured results (paper Tables 2, 3, 4, 5, 7; Laplace, N = 10⁶ unless noted)

| machine | prec. | ε | Pe/Pc | thr. | depth | T_M2L BLAS (ms) | T_M2L FFT (ms) | BLAS GFlop/s | BLAS storage |
|---|---|---|---|---|---|---|---|---|---|
| M1 Pro | f32 | 1e-3 | 3/3 | 1e-3 | 5 | 107 | 64 | 27 | 0.15 MiB |
| | f32 | 1e-4 | 3/4 | – | 5 | 147 | 163 | 100 | 0.79 |
| | f64 | 1e-6 | 5/6 | 1e-3 | 5 | 499 | 813 | 53 | 2.3 |
| | f64 | 1e-8 | 8/10 | 1e-3 | 5 (FFT: 4) | 875 | 257 (depth 4) | 83 | 4.8 |
| | f64 | 1e-10 | 9/11 | 1e-7 | 4 | 237 | 482 | 194 | 33.9 |
| AMD 3790X | f64 | 1e-6 | 5/6 | 1e-5 | 4 | 64 | 240 | 85 | 4.4 |
| | f64 | 1e-8 | 7/8 | 1e-5 | 4 | 179 | 449 | 71 | 9.5 |
| | f64 | 1e-10 | 9/11 | 1e-7 | 4 | 526 | 893 | 77 | 33.9 |
| M1 Pro, N = 8·10⁶ | f64 | 1e-10 | 9/11 | 1e-7 | 5 | 2205 | 4495 | 191 | 33.9 |

Same-depth comparisons: BLAS-M2L is 1.6× (M1, 1e-6), 2.0× (M1, 1e-10), 3.7× /
2.5× / 1.7× (AMD, 1e-6/1e-8/1e-10) faster than FFT-M2L in double precision; in
single precision at 1e-3 the FFT wins (tables fit in cache, 113 GFlop/s). End to
end (Table 2/3) the BLAS variant is the fastest f64 configuration on both
machines at every ε ≥ 1e-6; at ε = 1e-3/1e-4 f32 they are within 10 %.

Ranks are not printed in the paper. From the storage column and the compression
ratio (`U`: `n_check·k`, `Sᵀ`: `k·n_equiv`, directions: `2 k k_t` each, f64):
at Pe/Pc = 9/11, 33.9 MiB and 84 % compression give `k ≈ 205` of 386 and a mean
`k_t ≈ 33`; at 8/10, 4.8 MiB and 97 % give `k ≈ 180` and `k_t ≈ 4` (this last
number says the 1e-8 run is dominated by the P2P-level error, not that 5 modes
per direction carry 8 digits; treat all of these as order-of-magnitude). The
threshold is absolute, so the achieved accuracy is set by grid search
(`scripts/src/bin/grid_search_single_node.rs`), not predicted.

Also worth knowing: the `BlasFieldTranslationIa` variant (one SVD per direction
per level, no shared basis) is for Helmholtz, where the shared basis is poor; the
`BlasFieldTranslationAca` variant builds `K_t ≈ U_t V_tᵀ` by ACA+ (`linalg/aca.rs`)
without assembling `K_t`. Neither is relevant at our sizes.

### 3.4 What it would do for us

**CPU, Stokes (the remaining 1.05 s).** Our ghost-cell tree gives exactly 316
same-width offsets and `InteractionList` already stores the pairs CSR *by offset*
(`tree/lists.jl`, "counting sort by far offset"), with a target appearing at most
once per offset (`docs/framework.md`), so the per-direction GEMM is lock-free for
us where kifmm needs a mutex. The default `m2l_level!` of `interface.jl`
(gather columns, `mul!` with `M2L[f]`, scatter-add) *is* the uncompressed
BLAS-M2L; the compressed one replaces `M2L[f]` (`Nexp × Nexp`, 889² for Stokes)
by `(Ū_f, V̄'_f)` and adds the per-level `Sᵀ`/`U` GEMMs on the whole level block.

Per-pair flops: now `2304 × 9 × 8 = 166 kFlop` (R2C, the fused 6-table kernel) or
`4096 × 9 × 8 = 295 kFlop` (if the complex path is in use), at the measured
50 GFlop/s → 3.3–5.9 µs per pair. Compressed, with the Stokeslet ranks assumed
3× Laplace's (`k ≈ 450` of 888, `k_t ≈ 150`): `4·450·150 = 270 kFlop` per pair —
the same flop count — but as GEMMs at 300 GFlop/s (OpenBLAS, NEON) or
≈ 700 (Accelerate, #4): 0.4–0.9 µs per pair, plus `4·888·450 ≈ 1.6 MFlop` per box
for `Sᵀ`/`U` (≈ 3 % of the pair work at 189 pairs/box). Estimated Stokes M2L
3–5× faster; the gather/scatter traffic (`2·N_pairs·k·8 B` ≈ 7 KB per pair) is
≈ 10 ms per level at memory bandwidth. **The ranks are the whole question**: build
`K_fat` with `_multipole_to_check_matrix` (`ki_common.jl` 152) for `Laplace3D` and
`Stokeslet3D` at p = 8 (with the `pin_row`/`src_col` augmentation rows for the
stresslet/doublet operators of `cubeki.jl` 74–76), take the SVD, and read
`k(ε)` and `k_t(ε)` against the corner-source worst-case error of `test_3d.jl` —
half a day, and it decides whether #1 is a 5× or a 1.5×.

**CPU, Laplace.** `k ≈ 100–150`, `k_t ≈ 30–70` (the Fong–Darve `k ≈ n/2` rule and
Messner's directional ranks) → `12–40 kFlop` per pair against the Hadamard's
`18 kFlop` (R2C) / `33 kFlop` (C2C); at GEMM rate the M2L becomes a few tens of
ms at 10⁵ points. Not the bottleneck there, but it removes the FFT buffers.

**GPU** (`device_passes.jl`). Now: 16 symmetry classes, pairs by class in chunks
of 4096, `_class_gather!` with signed permutations, one `Nm × Nm × 4096` GEMM,
`_class_scatter!`. Compressed: 316 groups (no permutations needed; the symmetry
cannot reduce them further without extra `k × k` rotations, and 34 MB of
operators do not need reducing), gather `k`-vectors, two skinny GEMMs
(`k_t × k × N_f`, `k × k_t × N_f`), scatter `k`-vectors; `Sᵀ` and `U` once per
level. Flops per pair drop 6× (Laplace, `2·296² = 175 kFlop` → ≈ 30 k) and ≈ 6×
(Stokes, 1.58 MFlop → ≈ 270 k), but at 10⁵ points each group has only ≈ 150–500
columns and the GEMMs are `k_t = 50–150` rows tall, so Metal GEMM efficiency
collapses and there is **no gain at 10⁵ points** (the measured 0.036/0.129 s are
not M2L-bound anyway). At ≥ 10⁶ points (≈ 5 000 columns per group) expect 2–3×
on the M2L stage. The paper itself recommends this layout for GPUs (batched
GEMM per transfer vector, Takahashi et al. 2012) and has no GPU code.

Implementation notes: precompute on the CPU in `level_operators` (new
`CompressedM2L{T}` with `St`, `U`, `cu::Vector{Matrix}`, `cvt::Vector{Matrix}`,
`rank`, `dir_rank`); `rescale` scales the whole operator by the one `m2l_scale`
(all 316 blocks share it); `apply!` for a level = three stages inside the
existing `m2l_level!` hook; threads: either `Threads.@threads` over offsets with
`BLAS.set_num_threads(1)` (kifmm's choice on 32 cores) or serial over offsets with
threaded BLAS (their choice on 8 cores) — try both on 12 cores. Dual numbers:
the compressed operators are real matrices, so `ForwardDiff.Dual` expansions go
through `mul!` as they do for `M2M`/`L2L`. Accuracy control: use a *relative*
threshold `σ_i ≤ τ·σ₁` and verify with the corner tests at `τ ∈ {1e-8, 1e-10, 1e-12}`.

---

## 4. Surfaces, conditioning, orders

- **Oversampled check surface** (`surface_diff`): the BLAS path's optimal
  settings use `Pc = Pe + 1` or `+ 2` *with a lower `Pe`* than the FFT path needs
  for the same ε (5/6 vs 6 at 1e-6; 7/8 vs 8 at 1e-8 on AMD; 9/11 vs 10 at
  1e-10), citing Barnett–Betcke 2008 on MFS stability with more check than
  equivalent points (paper §5.1). The FFT path uses `Pc = Pe` and still reaches
  1e-8 at P = 8, 1e-10 at P = 10. PVFMM likewise uses equal counts. **Our
  `pc = 2p` on the large surface is far outside this range** (1352 vs 296 points
  at p = 8); it was forced by the stresslet/doublet translations
  (`cubeki.jl` header comment: "three orders of magnitude at pc = p"), not by the
  single-layer kernels. kifmm/PVFMM's data say the truncated-SVD solve copes with
  the coarser outer lattice for Laplace, and PVFMM runs its single-layer Stokes
  the same way. Hence #2: a per-kernel default — `pc = p` for `Laplace3D`,
  `Yukawa3D`, `Helmholtz3D` (test `p` and `p + 2` with the corner tests of
  `test_3d.jl`), `2p` kept whenever the operator is augmented
  (`has_flux_source(K) || pins_pressure(K)`, the flag `symmetry_classes` already
  uses), and a plan option for single-layer-only Stokes evaluations
  (`StokesFMMPlan(...; pc=p)`), since the plan cannot know at build time which
  singularities will be supplied. Effects at p = 8: `nlarge` 1352 → 296;
  `s2m!` and `l2t` 4.6× fewer kernel evaluations per point (and per `large` row
  in the `@simd` loops); `local_length` 4056 → 889 for Stokes, so `P_down`
  (`889 × 4056` → `889 × 889`), `partial_to_local!`, the `locals` array and the
  device `Ldn`/`V_dn` buffers shrink 4.6×; `M2M`/`L2L` are unchanged in size
  (they are folded onto the 889-long check data) but cheaper to build. Whether
  an intermediate `pc` (12 → 728 points) is enough for the augmented operators is
  not answered by kifmm (no Stokes there, as with PVFMM).
- **Margins**: `d = 0.05 / 0.025` (theirs) vs our 0.1 on both surfaces. Our
  `test_cubeki.jl` sweep already covers `d ∈ {0.05, 0.1, 0.2}`; nothing new.
- **Pseudo-inverse**: §1; the only difference from ours is the default
  `rtol = max(M,N)·ε` (≈ 1e-13) against our `50 ε` (1e-14) — immaterial.
  `PinvMode::AcaPlus` (`linalg/pinv.rs` `pinv_aca_plus`: ACA+ of the check map,
  QR of both factors, SVD of the small core) exists to avoid an SVD of a large
  map; ours are `1352 × 296` (or `296 × 296` after #2): not applicable.
- **Variable order per level** (#7): `expansion_order` may be a vector of length
  `depth + 1`; all operators are indexed by level (`MetadataAccess`
  `expansion_index`, `m2m_operator_index`, …), M2M/L2L are rectangular between
  levels, the FFT tables and `surf_to_conv` maps are per level. The examples
  (`laplace_variable_expansion.rs`: `[6, 5, 4]`) exist for the API; the use case
  is Helmholtz (`README`: "variable expansion orders by tree level useful for
  oscillatory problems"; the JOSS paper claims `kD ≈ 100` in single precision).
  No accuracy or timing data for Helmholtz are given anywhere in the repository
  or the paper; the tests use `k = 2.5` on a unit box at 1e-5. For our
  `Helmholtz3D` this is the natural extension if large `kW` is ever needed; not
  for Laplace/Stokes.
- **Conditioning diagnostics**: none beyond the pinv truncation; the Laplace scale
  degeneracy and the Helmholtz resonances we document are not discussed.

---

## 5. Tree and lists

- Construction (`tree/single_node.rs` `new`, `uniform_tree_pruned`): points
  Morton-sorted (`Point` carries the key at `DEEPEST_LEVEL = 16` and at `depth`),
  grouped by leaf key, siblings completed, ancestors completed; `keys` sorted by
  level then Morton (`levels_to_keys`), `HashMap` from key to index per level
  (`level_index_pointer_single_node`). Cost (Table 6/7 "Tree"): 0.36–0.56 s for
  10⁶ points, 3–5 s for 8·10⁶ — dominated by hashing; our `builder.jl` is not
  the issue.
- Interaction data: FFT path: per level, 26 vectors (one per halo direction) of
  per-target-parent displacements (`displacements[i][parent] = index of the
  neighbour's first child, or the zero sentinel`), built from
  `parent.all_neighbors()` (anchor arithmetic, `find_key_in_direction`); 1–51 ms.
  BLAS path: `316 × N_σ` table built by a parallel loop over sources with
  `parent().neighbors().children()` filtered by `is_adjacent` (ancestor
  `HashSet`s) and a `HashSet` of target keys; 25–6164 ms (Table 7, "Map
  creation"), the one setup cost they flag as growing with depth. Our
  `build_interaction_lists` (CSR by offset through `far_index`, plus the Morton-
  ordered by-target view) is already the cheap version of both; nothing to adopt.
- Ghost/halo handling: none in shared memory. The MPI build keeps separate
  single-node FMM objects for the V-list and U-list ghosts (`ghost_fmm_v`,
  `ghost_fmm_u` in `KiFmmMulti`) and a global FMM at rank 0; nothing that informs a
  shared-memory layout beyond "levels are whole octets in Morton order", which we
  have.
- Threading (rayon): P2M over leaves; M2M/L2L over chunks of 16 parents; FFT
  pre/post over chunks of parents; Hadamard one task per frequency; BLAS-M2L one
  task per transfer vector; P2P one task per target leaf, inner loop over the
  ≤ 27 neighbour leaves with `evaluate_st` on the whole (source leaf, target
  leaf) block. Equivalent to our `Threads.@threads :dynamic` structure; the only
  idea worth a measurement is #1's "parallel over offsets, single-threaded BLAS".

---

## 6. Kernel evaluation (green-kernels + rlst)

`laplace_3d.rs` `evaluate_laplace_one_target` (used by `evaluate_st` per target,
`evaluate_mt` with `par_chunks` over targets): the target is broadcast, sources
are read as AoS triples (`pulp::as_arrays::<3>`) and `deinterleave`d into x/y/z
lanes, `r² = fma(dx,dx,fma(dy,dy,dz·dz))`, `is_zero = (r² == 0)`,
`rinv = select(is_zero, 0, approx_recip_sqrt(r²))`, one accumulator (potential)
or four (potential + gradient, with `rinv³ = rinv·rinv·rinv`), `reduce_add` at
the end, scalar tail, the `1/4π` applied once. `assemble_st` (operator build)
has the same body writing a row. Helmholtz (`helmholtz_3d.rs`) uses
`sqrt` + `div` + `sin_cos` (SLEEF when the feature is on), two accumulators
(re/im).

`approx_recip_sqrt` (rlst `simd.rs` `simd_approx_recip_sqrt`): NEON f32
`vrsqrteq_f32` + 2 × (`vrsqrtsq_f32`, `mul`); NEON f64 `vrsqrteq_f64` + **3**
Newton steps; AVX2 f32 `rsqrt_ps` + 1 step; AVX2 f64: convert to f32, `rsqrt_ps`,
convert back, 2 steps; AVX-512 `rsqrt14`. `pulp::Arch::new()` on AArch64 selects
`Neon` (`aarch64.rs` 3419), so this path is the one that runs on Apple Silicon
(the FCMA ISA is used only by the Hadamard). This differs from PVFMM/SCTL, which
in f64 on AArch64 falls back to `1/sqrt` (PVFMM report §3.1). Measured direct
sums (Table 1, Laplace, sources = targets): M1 Pro 8 cores, f64 10⁵ points 2.2 s
→ 0.57 G interactions/s/core (5.6 cycles each at 3.2 GHz, 2 lanes); f32 0.82 s →
1.5 G/s/core; AMD 32 cores f64 0.31 s → 1.0 G/s/core, f32 0.15 s → 2.1 G/s/core.

Mapping: `kernels.jl` `(K::AbstractRadialKernel)(s, t) = scale(K)·radial_value(K,
sqrt(_sqnorm(t−s)))` and `Stokeslet3D` (`a = K.c/sqrt(r2)`), `evaluate.jl`
`_direct_range` (`j == skip && continue`, scalar reduction over sources),
`ki_common.jl` `s2m!` (`@simd ivdep` over the SoA `ysoa` rows — the lanes are
check points, PVFMM's orientation) and `l2t` (`@simd` over `ysoa` rows). Two
things transfer: (i) the self term as a mask (`ifelse(j == skip, zero(acc),
field)` — keep the index semantics rather than `r² > 0`, which would also drop
coincident distinct sources), removing the branch that stops LLVM from
vectorising `_direct_range`; (ii) `rinv` from an estimate plus Newton steps
instead of `sqrt` + `/`. Julia has no `rsqrt` intrinsic, but the SCTL bit-trick
seed (`reinterpret(Float64, 0x5FE6EB50C7B537A9 - (reinterpret(UInt64, x) >> 1))`)
with 3–4 Newton steps (`y *= 1.5 - 0.5·x·y²`) is scalar code that vectorises and
needs no `llvmcall`; whether it beats `fsqrt`+`fdiv` on the M3 depends on their
throughput and must be measured on `_direct_range` (target: ≥ 0.5 G f64
interactions/s/core; the PVFMM report's lane layout, #4 there, is the companion
change). For `Dual`/complex positions keep the current generic bodies (dispatch on
`eltype`).

---

## 7. Precision, memory, caching

- Single precision is first-class (`f32` operators, `c32` tables, `f32x4` 8×8
  kernel); Table 2: f32 reaches 1e-4 at N = 10⁶ and saturates at 1e-3 at
  8·10⁶ on the M1 Pro ("aggressive single-precision optimizations applied by
  Apple Accelerate, accumulated roundoff, SIMD reduction ordering") — relevant
  to #4: validate `Float32` with `test_float32.jl` after switching BLAS.
- Per-level buffers are allocated per call; tables are stored twice; leaf
  surfaces are stored per leaf; the BLAS path copies the gathered multipoles
  into a fresh matrix per direction per call. None of this is worth copying;
  our `state.scratch` reuse is the better practice.
- Caching: operators keyed by level only when the kernel is inhomogeneous or the
  order varies; otherwise one set + scalar scaling (as ours). Setup breakdowns in
  Tables 6–7 are the only "memory/precompute" data: FFT tables 0.7–1.8 s,
  BLAS compression 0.3–30 s (see §3.1).

---

## 8. Not applicable to the ghost-cell tree / not recommended

- W/X/U0/U2 lists: absent in kifmm too (uniform tree); the near field is the 27
  same-level neighbours, exactly our colleague sum. Their non-uniform experiment
  (§1) shows the price of no adaptivity; our ghost construction is strictly
  better here.
- Adaptive `n_crit` refinement: not implemented (only `prune_empty`).
- MPI (`ghost_exchange.rs`, `neighbour_comm.rs`, `sorting/hyksort.rs`,
  `multi_node.rs` variants): distributed only.
- ACA+ pseudo-inverse and ACA M2L: for large or implicit check matrices; ours are
  small and dense.
- `FmmEvalType::Matrix` (several charge vectors through the BLAS-M2L at once):
  per-vector gains of 1.2–1.3× on the M1 Pro, none on the AMD (Table 2/3); our
  solves have one right-hand side per evaluation.
- Per-target mutex scatter, `HashSet`-based list construction, per-call
  allocation, duplicate tables, per-leaf stored surfaces: inferior to what we
  have.
- GPU: nothing in kifmm; the paper's GPU remarks (batched BLAS per transfer
  vector, P2P on the GPU with shallow trees) describe what we already do
  (dense) and what #1 would add (compressed).

---

## 9. Parameters, accuracy and speed side by side

| | kifmm (Laplace, uniform, sources = targets) | FlexibleFMM (3D cube) |
|---|---|---|
| order | FFT: `P = Pe = Pc`, 3/4 (f32), 6/8/10 (f64) for ε = 1e-3/4, 1e-6/8/10; BLAS: `Pe/Pc` = 3/3, 3/4, 5/6, 7/8 or 8/10, 9/11 | `p = 8`, `pc = 2p = 16` |
| surface points | `6(P−1)²+2`: 26/56/98/152/218/296/386/488/602 at P = 3…11; locals have `n_equiv` entries | small 296, large 1352 |
| radii | 1.05 / 2.95 (d = 0.05 / 0.025) | 1.1 / 2.8 (d = 0.1) |
| points per leaf | depth from `N/8^d ≤ n_crit` (default 150); optimal benchmarks: ≈ 30 (M1 Pro, ε ≤ 1e-6) or ≈ 244 (depth 4) | `ppl = 200` (3D), adaptive |
| pseudo-inverse | SVD, `rtol = max(M,N)·ε_mach`, two factors; M2M/L2L folded in the safe order | SVD, `50·ε`, two factors; folded the same way |
| M2L | FFT (R2C, 26 × 8×8 blocks, FCMA/AVX2) or BLAS (Fong–Darve + Messner, rSVD) | FFT (R2C, per child pair, fused components), dense per class on the GPU |
| accuracy metric | L2-relative error of the potential over all targets (eq. 16) | ℓ∞ worst case (corner source) and global ℓ∞ |
| Laplace accuracy | FFT: ε ≈ 10^{−P}: 1e-6 (6), 1e-8 (8), 1e-10 (10) in f64; BLAS reaches the same with `Pe` one lower and `Pc = Pe+1…2` | worst case 1e-6 at p = 8 (ℓ∞), global 1e-7; consistent with 1e-8 in L2 and with PVFMM's 3e-7 max-relative at m = 8 |
| Stokes | not implemented, no data | stokeslet 3e-6, stresslet 4e-5, doublet 1e-3 worst case at p = 8 |
| Helmholtz | implemented (FFT c2c and BLAS-IA), no accuracy/timing data published | `Helmholtz3D` through the complex path |
| end-to-end, 10⁶ points, f64 | M1 Pro (8 cores): 1e-6 741 ms (BLAS) / 1061 (FFT); 1e-8 1293 / 1477; 1e-10 1563 / 1799. AMD (32 cores): 299 / 481; 433 / 730; 816 / 1191 | 10⁵ points, 12 M3 Max cores: Laplace 0.31 s, Stokes 1.41 s (FMM3D 0.35 / 1.15 s) |
| M2L throughput | FFT 53–83 GFlop/s f64 (M1 Pro), 28–36 (AMD); BLAS 53–194 (M1 Pro, Accelerate), 71–85 (AMD) | Hadamard 70 (Laplace) / 50 (Stokes) GFlop/s |
| direct sum | 0.57 G f64 interactions/s/core (M1 Pro), 1.0 (AMD) | not measured here |

---

## 10. Detailed mapping of the top recommendations

### 10.1 (#1) Compressed BLAS-M2L

Precompute, in `level_operators` for the cube method (`cubeki.jl` 55–80), next
to `_kernel_tables`:

```
K_f  = _multipole_to_check_matrix(K, small, o·w, w, small, 0) for f in far offsets   # Nexp × Nexp
       (+ the pin row / point-source column of lines 74–76 for augmented kernels)
Kfat = hcat(K_f...)            # Nexp × 316·Nexp
Kthin = vcat(K_f...)           # 316·Nexp × Nexp
U, σ = svd(Kfat); k = count(σ .≥ τ σ[1]); U = U[:, 1:k]
S  = svd(Kthin).V[:, 1:k]      # kifmm takes S = U when Pc = Pe; that is exact only when
                               #  K_{-f} = K_fᵀ (symmetric kernel, same lattice both sides),
                               #  true for Laplace/stokeslet, false for the augmented operators
                               #  (pin row / point-source column) — do both SVDs, they are cheap
for f: C_f = U' * K_f * S; Ū, σ̄, V̄ = svd(C_f); k_f = count(σ̄ .≥ τ σ[1]);
       cu[f] = Ū[:, 1:k_f]; cvt[f] = Diagonal(σ̄[1:k_f]) * V̄[:, 1:k_f]'
```

Use a relative threshold `τ` (kifmm's absolute one needs a grid search per
scale) and verify against `conditioning`-style corner tests. Apply, inside
`m2l_level!` for a level with multipole block `Mp[:, lr]` and partial locals `pL`:

```
Q̃ = St * Mp[:, sources of the level]          # k × nl        (one GEMM)
for f with pairs:  G = Q̃[:, src range of f]   # gather k-vectors (il.ptr/il.src)
                   Y = cu[f] * (cvt[f] * G)    # k × n_f
                   scatter-add Y into Φ̃[:, il.tgt]           # conflict-free per offset
pL .+= m2l_scale * (U * Φ̃)                    # one GEMM
```

Threads: offsets in `Threads.@threads` with `BLAS.set_num_threads(1)` inside, or
serial with threaded BLAS — measure both. GPU (`device_passes.jl`): `chunks`
built per offset `f` (no `class`), `_class_gather!` without `idx/sgn`
(plain column gather of `Q̃`), `mul!` with `cvt[f]` then `cu[f]`, scatter as now;
`Q̃`/`Φ̃` as two extra level buffers (`k × nl`). Keep the dense class path for
`N ≤ 10⁵`.

### 10.2 (#2) Kernel-dependent `pc`

`cubeki.jl`: `CubeKernelIndependent{D}(kernel; p=8, pc=default_pc(kernel, p), …)`
with `default_pc(K, p) = (has_flux_source(K) || pins_pressure(K)) ? 2p : p`;
`stokes_api.jl` 45: same default, plus a documented `pc=p` for single-layer
plans. Tests: `test_3d.jl` corner cases for `Laplace3D` and `Stokeslet3D` at
`pc ∈ {p, p+2, 2p}` (expect the single-layer numbers unchanged within a factor
~2; the stresslet must stay at `2p` — that is the existing finding). Then
re-measure the 10⁵-point Laplace matvec: S2M + L2T drop from 2·1352·10⁵ to
2·296·10⁵ kernel evaluations.

### 10.3 (#3) `rsqrt` and mask

`kernels.jl`: add `@inline _rinv(r2::Float64)` / `(::Float32)` (bit-trick seed +
3 or 4 / 1 or 2 Newton steps; fall back to `1/sqrt` for other types) and use it
in `Laplace3D`, `Stokeslet3D`, `Yukawa3D` (`exp(-k r)/r` needs `r` too: `r =
r2·rinv`). `evaluate.jl` `_direct_range`: `acc += ifelse(j == skip, zero(acc),
target_field(...))` (no `continue`); if `acc` is an `SVector`, LLVM still needs
the source loop to be a plain reduction — check `@code_llvm` for vector
instructions. Benchmark against 0.5 G/s/core before keeping it.

### 10.4 (#4) Accelerate

`using AppleAccelerate` before any BLAS call (libblastrampoline forwarding);
no source change. Compare `mul!` timings of the Stokes `M2M` (`889 × 7112 × ng`)
and of #1's GEMMs; run `test_float32.jl` and the f32 device tests, since
Accelerate's f32 kernels are what kifmm blames for its 1e-3 floor at 8·10⁶
points.
