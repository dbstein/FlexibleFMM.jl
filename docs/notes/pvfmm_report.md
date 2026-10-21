# What FlexibleFMM can take from PVFMM

Source studied: `dmalhotra/pvfmm` (master, cloned 2026-10-21) with its SCTL
submodule. Files read in full or in the relevant parts: `include/fmm_pts.txx`
(surfaces, `Precomp`, `PrecompPerm`, `SetupInterac`/`EvalList`, `EvalListGPU`,
`FFT_UpEquiv`, `FFT_Check2Equiv`, `matmult_8x8x2`, `VListHadamard`,
`V_ListSetup`, `V_List`, `Source2UpSetup`, `U_ListSetup`, `EvalListPts`,
`Down2TargetSetup`), `interac_list.txx`, `precomp_mat.txx`, `kernel.txx`,
`fmm_tree.txx`, `src/fmm_pts_gpu.cu`, `device_wrapper.*`, `pvfmm_common.hpp`,
`docs/concepts/*.md`, `docs/tutorials/particle-fmm-cpp.md`, `examples/src/*.cpp`,
and SCTL's `generic-kernel.*`, `kernel_functions.hpp`, `intrin-wrapper.txx`
(rsqrt), `fft_wrapper.hpp`. Nothing was built or run.

Our side, for the mapping: `src/operators/spectral_m2l.jl`,
`src/operators/cubeki.jl`, `src/operators/ki_common.jl`, `src/fmm.jl`,
`src/evaluate.jl`, `src/tree/{tree,lists,builder}.jl`, `src/kernels.jl`,
`src/sources.jl`, `docs/{kifmm,framework,tree,stokes}.md`.

Throughout, "p = 8 Stokes" means our default 3D Stokes configuration:
`nsurface = 296`, `nlarge = 1352` (pc = 16), `multipole_length =
partial_local_length = 889`, `local_length = partial_multipole_length = 4056`,
`(2p)^3 = 4096` frequencies, B = 16 (Stokes) / 64 (Laplace) frequencies per
block, 316 far offsets, 189 list entries per interior box of a uniform tree.

---

## 0. Ranked recommendations (summary table)

| # | Recommendation | PVFMM evidence | Our files | Expected gain | Effort |
|---|---|---|---|---|---|
| 1 | Fold the upward pseudo-inverse into M2M in the per-mode-safe order `V·(S⁺·(Uᴴ·M2M))` | `Precomp` case `U2U_Type`: `M=(M_ce2c*M_c2e0)*M_c2e1` with `M_c2e0 = Vᵀ·S⁺`, `M_c2e1 = Uᵀ` | `ki_common.jl` `_ki_dense_operators`, `partial_to_multipole!`; `fmm.jl` `upward_pass!` | Upward dense work per sibling group 128 → 12.6 MFlop (10×); M2M matrix 230 MB → 50 MB | Low (operator build + one driver branch) |
| 2 | Reorganise the Hadamard stage by sibling-group pairs: walk (target group, source group, parent offset) with the fixed 26×64 far pattern, hoist each table slice over the ≤ 8 child pairs that share it, reuse the source group's 8 columns across the 8 targets | `V1_Type` lists (26 parent offsets), `VListHadamard` loop order (table slice hoisted over all pairs of a target block), `FFT_UpEquiv` child interleave | `spectral_m2l.jl` `_spectral_m2l_level!`, `_hadamard_block!`; `tree/lists.jl` `build_interaction_lists` | L2 traffic per pair ≈ 3.0 KB → ≈ 1.1 KB; if the stage is L2-bound (consistent with 6 GFlop/s/core measured) 1.5–2.5× on Stokes M2L | Medium |
| 3 | Real-input transforms (R2C/C2R) for real kernels | `FFT_UpEquiv`: `FFT_Type::R2C`, `n3_ = n1²(n1/2+1)`; tables FFT'd once with the same plan (`V_Type`) | `spectral_m2l.jl` `SpectralWorkspace`, `_kernel_tables` | Hadamard flops and spectra memory 4096 → 2304 per box/component (×0.56); tables ×0.56 | Low–medium (index bookkeeping, `plan_rfft!`) |
| 4 | SIMD-shaped per-point bodies for `s2m!`, `l2t`, near field: SoA scratch per (leaf, source range), targets in lanes, sources broadcast, masked self term, FMA, no branch in the inner loop | `GenericKernel::Eval` (kernel.txx 421–530), `stokes_vel_::uKerEval`, SCTL `Stokes3D_FxU::uKerApply`, `approx_rsqrt` | `ki_common.jl` `s2m!`, `l2t`; `evaluate.jl` `_direct_range`, `evaluate_target`; `sources.jl` | 2–4× on the three kernel-evaluation stages; at p = 8 Stokes with pc = 16, l2t alone is 1352 kernel evaluations per target, more than the near field at ppl = 50 | Medium (Julia SIMD: `SIMD.jl` or hand-unrolled `ntuple` lanes) |
| 5 | Thread the FFTs (per-box plan execution over Julia threads) or enable FFTW threads; verify they are not single-threaded now | `FFT_UpEquiv`/`FFT_Check2Equiv`: `omp parallel for` over node ranges, per-thread buffer, one plan executed per node | `spectral_m2l.jl` `SpectralWorkspace` (`plan_fft!` on the whole level) | For Laplace at 100k points the FFTs are ≈ 2.4 GFlop; single-threaded that is 0.25–0.5 s of the 0.5 s matvec | Very low (one check, then a loop) |
| 6 | Bound the spectral working set: process a level in target-group chunks with a byte budget, recomputing the chunk's source transforms | `V_ListSetup` `n_blk0` blocks of `PVFMM_DEVICE_BUFFER_SIZE` (1 GB), `std::set` of sources per block, FFTs redone per block | `spectral_m2l.jl` `_spectral_m2l_level!`, `cubeki.jl` `m2l_level!` scratch key | 6 GB at 10⁶ Laplace points → fixed budget (e.g. 1 GB) at ~10–25 % extra FFTs | Medium |
| 7 | Stokes table symmetry (6 of 9 tables) and even-kernel offset parity (`κ̂₋ₒ = conj κ̂ₒ`) | Not in PVFMM (it stores all 316 offsets × 9 components and runs 9 separate passes); listed because it compounds with #2 | `spectral_m2l.jl` `SpectralM2L`, `_hadamard_block!` | Table memory 373 MB → ≈ 35–70 MB together with #3 and dropping the duplicate `tables` copy; 3 of 9 table streams removed (no multiply saved: a symmetric 3×3 matvec is still 9 products) | Low |
| 8 | Downward: keep locals as check data on the small surface, fold `P_down` into L2L in the safe order, apply `P_down` only on real leaves before `l2t` | `D2D_Type` folding; PVFMM's local lives on a surface with the same count as the multipole | `ki_common.jl`, `fmm.jl` `downward_pass!`, `evaluate.jl` | Downward dense per group 128 → ≈ 74 MFlop (1.7×); not 10× because pc = 2p makes the large-surface density 4.6× longer than the check data | Medium |
| 9 | Store one M2M/L2L block per child class and apply the other 7 by signed permutations | `PrecompPerm`, `equiv_surf_perm`, `InteracList::InitList` (class_hash), `EvalList` input/output permutations | `ki_common.jl` | Memory only (÷8 on M2M/L2L storage); no flops saved. After #1 the M2M is 50 MB, so low value | Medium; not recommended now |

Not applicable (documented in §7): W/X lists, U0/U2 lists, periodic-BC operator
(`BC_Type`), MPI/LET, the volume FMM, the CUDA offload (it covers only the
dense permute–GEMM–permute path; the V-list is CPU-only in PVFMM).

---

## 1. The V-list (M2L) in PVFMM

### 1.1 Layout: real-input FFT, 8 children interleaved per frequency

`FFT_UpEquiv` (fmm_pts.txx 2630–2698). Per *parent* node of level l−1 (the
"nodes_in" of `V_ListSetup` are the non-leaf nodes of level−1; the actual
sources and targets are their 8 children):

- the 8 children's upward equivalent densities (n = 6(m−1)²+2 points each,
  `ker_dim0` components) are scattered into 8 zero-padded real grids of
  `n1³ = (2m)³` points via `vlist_fft_map` (the surface lattice occupies
  indices 0..m−1 of each axis; the downward check lattice in
  `FFT_Check2Equiv` is embedded at m..2m−1 so the circular convolution never
  aliases; same idea as our `embed`/`mod(m, 2p)` tables);
- per-level scaling for scale-invariant kernels is applied here
  (`fft_scal = 2^(src_scal·depth)`), i.e. one table set serves all levels, as
  our `m2l_scale` does;
- one batched real-to-complex plan (`vlist_fft.Setup(FFT_Type::R2C,
  ker_dim0*8, {n1,n1,n1})`) transforms the 8 children at once →
  `n3_ = n1²(n1/2+1)` complex frequencies each;
- the result is transposed to `[component][frequency][child][re,im]`: for
  each frequency the 8 children are 16 contiguous reals (two cache lines).

So the spectrum of a box costs `n3_` complex numbers: 2304 for m = 8, 4400
for m = 10, against our `(2p)³ = 4096` at p = 8. Per box and component that is
36.9 KB (m = 8) versus our 64 KB, and we hold two such arrays (`Σ̂`, `Û`).

### 1.2 Tables: 316 offsets, pre-expanded into 26 parent-offset 8×8 blocks

`Precomp` case `V_Type` (316 = 7³−3³ offsets, `InitList(3,2,1,V_Type)`):
the kernel is evaluated on the `conv_grid` of `(2m)³` points at
`(i−m)·a/(m−1)` (a = RAD0·width, i.e. the equivalent-lattice spacing), one
grid per kernel component pair, transposed to component-major, and R2C-FFT'd
once (`vprecomp_fft`). This is exactly our `_kernel_tables`, half-spectrum.

`Precomp` case `V1_Type` (26 = 3³−1 parent offsets, `InitList(1,1,1,V1_Type)`):
for each parent offset `O` it assembles, per frequency and component pair, an
8×8 complex block `M[j][(j2·8+j1)·2 + {0,1}]` holding the `V_Type` spectrum of
the child offset `2·O − j1 + j2` (zero if that offset is near, i.e. Chebyshev
norm < 2), divided by `n3` (the inverse-FFT normalisation is folded into the
table). Storage: `ker_dim0·ker_dim1·n3_` rows × 128 reals.

Consequences worth knowing:

- the 8×8 blocks are 5.3× redundant (26·64 = 1664 slots for 316 distinct
  tables, each distinct child offset appearing in the slots of several parent
  offsets); for Stokes at m = 8 this is 26·9·2304·128·8 B = 552 MB (m = 10:
  1.05 GB) per precision, versus our 316·9·4096·16 B = 186 MB of complex
  tables (373 MB with the separate `tr`/`ti` copies we keep; 105 MB if we went
  half-spectrum and dropped `tables`);
- 9.1 % of the block entries are zero (face-adjacent parent offsets have 16/64
  near child pairs, edge 4/64, corner 1/64) and are multiplied anyway;
- **the 48-element symmetry group is not used for V-list tables.**
  `InteracList::InitList` computes a symmetry class for every list type
  (`class_hash` sorts |c|; the 316 far offsets fall into 16 classes), and
  `Precomp` tries to derive a matrix from its class matrix through
  `Perm_R`/`Perm_C`, but `PrecompPerm` only builds permutations for
  `U2U_Type` and `D2D_Type` (default branch leaves `P` empty), and
  `Perm_R`/`Perm_C` contain `assert(type!=V_Type)`. So all 316 V tables are
  computed and stored directly. The symmetry machinery is an M2M/L2L feature.

### 1.3 The Hadamard loop (`VListHadamard`, fmm_pts.txx 3410–3505)

Per level (per `n_blk0` block, §1.5):

```
fft_out = 0                                        # parallel over targets
build IN_/OUT_ pointer tables per (target block blk1, parent offset mat_indx)
parallel over threads pid:  k in [pid·M_dim/omp_p, (pid+1)·M_dim/omp_p)   # frequency range
  for in_dim, ot_dim (kernel component pair)
    for blk1 (target-parent block, ~ 25 KB · 64 / (2·sizeof(Real)) parents)
      for k in the thread's frequency range
        for mat_indx in 1..26 (parent offset)
          M = table block for (in_dim, ot_dim, k, mat_indx)       # 8×8 complex = 1 KB
          for pairs j, j+2 in the (blk1, mat_indx) list:
            matmult_8x8x2(M, IN[j], IN[j+1], OUT[j], OUT[j+1])   # 2 pairs share M
          odd tail: matmult_8x8x1
```

Points to carry over:

1. **Threads own disjoint frequency ranges** → no write conflicts, no atomics
   (we do the same with blocks of B frequencies).
2. **The table slice is hoisted over all pairs with the same parent offset in
   a target block.** At fixed (k, mat_indx) the 1 KB block stays in L1 across
   ~100 parent pairs; per pair only the 128-B input slice and the 128-B output
   slice (RMW) are streamed. Per parent pair and frequency: 64 complex MACs =
   512 flops for 384 B of box traffic (1.3 flop/B) with the table amortised.
   Our `_hadamard_block!` streams, per child pair and block of B = 16
   frequencies, 2304 B of table (9 × 2 × 16 reals), 768 B of source and 768 B
   of target for 1152 flops (0.3 flop/B), and the table slices of all 316
   offsets for one block (729 KB for Stokes) cannot stay in L1 while pairs are
   walked target-major with changing offsets, so the table stream comes from
   L2 on every pair. The measured 6 GFlop/s per core corresponds to ≈ 20 GB/s
   of L2 traffic, i.e. the stage is bandwidth-bound, not FMA-bound (NEON FP64
   peak on an Apple P-core is ≈ 50 GFlop/s).
3. **Register blocking over two pairs** (`matmult_8x8x2`): AVX version keeps 8
   accumulators (2 pairs × 8 outputs × complex) in `__m256d` registers;
   SSE/NEON (via `sse2neon.h`) version likewise with `__m128d`. The generic
   fallback is a 2×2-unrolled scalar loop. Explicit prefetch of the next two
   pairs' slices.
4. Component pairs are the **outermost** loop: 9 full passes over the data for
   Stokes, each with its own 26 × 1 KB of tables resident. PVFMM does not fuse
   the 9 products the way our generated kernel does; nor does it use the
   Stokeslet's `T_ab = T_ba`.
5. Target blocks (`blk1`) are formed from the Morton-ordered node list
   (`nodes_out` come from the postorder node list, which is Morton order), so
   a block's sources are spatially compact. We already order targets by Morton
   key (`InteractionList.torder`).

### 1.4 What the pair list looks like

`V1_Type` interaction lists are per non-leaf node: 26 slots, slot `i` = the
colleague at parent offset `i` if it is a non-leaf (`BuildList` case
`V1_Type`). The 64 child pairs and their tables follow from the fixed pattern
`2·O − j1 + j2`. There is no per-child-pair list at all. The pair list given to
`VListHadamard` (`interac_vec`) is built in `V_ListSetup` grouped by
(target block, parent offset) with CSR displacements (`interac_dsp`).

For our ghost-cell tree this structure is exact: every colleague of a node with
children has children (docs/tree.md, "Why it works"), so every M2L pair of
level l+1 is one of the 64 child pairs of a (node, non-leaf colleague) pair at
level l. Target groups are the children of nodes with `has_children && !xlist`
(ghost groups are never targets); source groups are the children of any node
with `has_children` (xlist leaves included), filtered by `has_source`. Levels
consist of whole sibling groups stored contiguously (`child_first`), so the 8
columns of a group are adjacent in `Mp`/`pL` — the interleave PVFMM builds by
hand we get from storage order.

### 1.5 Memory blocking of a level (`V_ListSetup`, 3509–3760)

`buff_size = PVFMM_DEVICE_BUFFER_SIZE = 1 GB`;
`n_blk0 = 2·fftsize·(ker_dim0·n_in + ker_dim1·n_out)·sizeof(Real)/buff_size`.
The target parents of the level are cut into `n_blk0` contiguous (Morton)
ranges; for each block the set of source parents is collected (`std::set`),
their 8-child FFTs are computed into `fft_in`, the Hadamard runs, and the
block's targets are inverse-transformed. Sources shared between blocks are
transformed once per block. Per-level working set is therefore bounded by the
budget regardless of N. The equivalent for us is a chunk loop inside
`_spectral_m2l_level!` over Morton-contiguous ranges of target groups with a
compact source index map (the halo of a chunk of G groups is roughly
`((G^{1/3}+2)³ − G)/G`, i.e. ~25 % for G ≈ 20 000, ~70 % for G ≈ 1000).

### 1.6 GPU

`V_List` begins with `assert(!device); //Can not run on accelerator yet.` and
`FMM_Tree::SetupFMM`/`DownwardPass` pass `device=false` for V. There is no
V-list GPU implementation in PVFMM; see §5.

---

## 2. M2M / L2L and the check-to-equivalent solves

### 2.1 Surfaces and sizes

`surface(p, c, alpha, depth)`: boundary points of a p-per-edge lattice,
`6(p−1)²+2` points — identical to our `cube_boundary_points` count
(`p³ − (p−2)³`). Radii (`pvfmm_common.hpp`): upward equivalent / downward check
at `RAD0 = 1.05` box half-widths, upward check / downward equivalent at
`RAD1 = 2.95`. In our parametrisation that is `d = 0.05` on the small surface
and `d = 0.025` on the large one (we use 0.1 for both). **Equal point counts on
both surfaces (`pc = p`)**: the upward check matrix `M_e2c` is square,
`(n·kd0) × (n·kd1)`.

Matrix sizes at the defaults, Stokes (kd = 3): m = 10 → n = 488, M2M/L2L
1464 × 1464 (17 MB); m = 8 → n = 296, 888 × 888 (6.3 MB). Laplace m = 10:
488 × 488 (1.9 MB). One matrix per type is stored (see 2.3).

### 2.2 Pseudo-inverse regularisation and the folding order

`Precomp` cases `UC2UE0_Type`/`UC2UE1_Type` (and `DC2DE*`):

```
M_e2c.SVD(U,S,V);                                  // M_e2c = U·S·V (sctl convention)
S[i][i] = (S[i][i] > eps·max_S·4 ? 1/S[i][i] : 0); // cutoff 4·eps·σ_max
M_c2e0 = Vᵀ·S⁺;   M_c2e1 = Uᵀ;                     // two factors, S⁺ folded into one
```

Cutoff `4·eps·σmax` (eps = machine epsilon of `Real_t`): essentially only the
numerical null space is dropped; ours is `50·eps`.

The translation matrices are **folded with the pseudo-inverse, in a specific
order**:

```
U2U_Type:  M = (M_ce2c · M_c2e0) · M_c2e1      // child equiv → parent equiv
D2D_Type:  M =  M_c2e0 · (M_c2e1 · M_pe2c)     // parent equiv → child equiv
D2T_Type:  M =  M_c2e0 · (M_c2e1 · M_e2t)      // (volume FMM only)
```

i.e. always `(kernel matrix · orthogonal factor) · diagonal · orthogonal`, never
`kernel matrix · (dense pseudo-inverse)`. This matters, and it is the one
place where PVFMM's practice contradicts our docs/kifmm.md ("Forming it as one
matrix, or folding it into M2M/M2L/L2L ... the rounding error is eps/σ_min in
every direction"). The two statements are reconciled by the order:

- *Unsafe order.* Form `P = V S⁺ Uᴴ` (entries up to 1/σ_min ≈ 10¹³) and then
  `P·T`. Each entry of `P·T` is a sum of terms of size 1/σ_min cancelling to
  O(1): rounding ≈ eps/σ_min in **every** output direction, including the
  well-conditioned ones whose far field is O(1). With σ_min ≈ 10⁻¹³ this
  predicts ≈ 10⁻³, matching the 6e-4 we measured for the folded design.
- *Safe order* (PVFMM's). `Uᴴ·T` first: orthogonal × O(1) kernel values,
  absolute rounding ≈ eps per entry. Then `S⁺` scales row j by 1/σ_j: the
  error in row j is eps/σ_j, and row j is the coefficient of **mode j**. Then
  `V` maps row j to the density `v_j`, whose field on the check surface is
  σ_j by definition and smaller beyond it. The rounding amplified by 1/σ_j
  therefore sits only in the direction the evaluation damps by σ_j — exactly
  the argument docs/kifmm.md makes for applying the factors at run time. No
  intermediate matrix with large entries is ever formed: rows of `S⁺UᴴT` are
  O(1) because `u_jᴴT` is O(σ_j) for fields that lie in the range of the check
  map (the ε_YBZ-sized remainder is amplified to ε_YBZ/σ_j, but that component
  is present in the run-time factored path too — it is inherent in truncated
  KIFMM and harmless for the far field).

So folding in the safe order is numerically equivalent to our factored run-time
application and must be verified only by the existing worst-case corner tests
(`test/test_cubeki.jl`, `test/test_3d.jl`), not re-derived.

**Cost for us (p = 8 Stokes, per sibling group).** Now: M2M GEMM
`4056 × 7112` = 57.7 MFlop, plus `P_up` on 8 nodes: `Uᴴ` (r ≤ 889 × 4056) and
`V` (889 × r) = 8.8 MFlop per node = 70 MFlop → 128 MFlop. Folded:
`M2M_fold = V·(S⁺·(Uᴴ·M2M))` is `889 × 7112` = 12.6 MFlop and no `P_up` for
nodes with children. 10× on the upward dense work; M2M storage 230 MB → 50 MB.
Childless nodes keep `s2m!` + `P_up` (as PVFMM's S2U: kernel evaluation to the
check surface then two GEMMs). `pinv_scale·m2m_scale` become one factor in
`rescale`.

**L2L is different for us.** Folding `P_down` into L2L with the local kept as
the large-surface density gives `(8·4056) × 4056` = 263 MFlop per group, worse
than today's 58 + 70 = 128, because `pc = 2p` makes the local 4.6× longer than
the check data. PVFMM does not have this asymmetry (equal counts). The
transferable version is #8 in the table: store the local as the small-surface
check data (889), fold `P_down` into L2L in the safe order (`(8·889) × 889`,
12.6 MFlop), and apply `P_down` (8.8 MFlop) only on real leaves that have
targets, before `l2t`. Per group ≈ 12.6 + 7·8.8 ≈ 74 MFlop (1.7×). Whether
`pc` can be smaller than `2p` for the stresslet/doublet (e.g. 12 → 728
points) is not answered by PVFMM: it reports accuracy only for single-layer
Laplace, and its double-layer Stokes kernel (`stokes_sym_dip`) has neither the
point-source augmentation nor the pressure pin we found necessary, so nothing
there bears on our `pc` finding.

### 2.3 One operator per class, applied through permutations

`InitList(1,1,2,U2U_Type)` lists the 8 children (rel. coords in {−1,1}³);
`class_hash` sends all 8 to class (1,1,1): **one M2M (and one L2L) matrix is
stored**; `PrecompPerm`/`equiv_surf_perm` build, for each of ReflecX/Y/Z,
SwapXY, SwapXZ (and Scaling), a signed permutation of the surface points
composed with the kernel's component permutation `perm_vec` (found
numerically in `Kernel::Initialize`: e.g. reflecting x flips the sign of the
x-component of a Stokes density). `Perm_R`/`Perm_C` compose these per child
(`PermutList`), and `EvalList` applies them as a gather-with-scaling on input
(`input_perm`) and a scatter-add-with-scaling on output (`output_perm`) around
**one GEMM for all children of all parents of the level** (all 8 classes share
`interac_mat`, so the GEMM loop groups them: `Mt(vec_cnt × M_dim1) =
Ms(vec_cnt × M_dim0) · M`). Flops are unchanged by the symmetry; it is a
storage device (÷8) and keeps the GEMM operand L2-resident. After #1 our M2M is
50 MB and L2L 230 MB; the L2L block is streamed once per level by BLAS, so the
gain for us is memory only — not recommended now.

### 2.4 Conditioning diagnostics

`Kernel::Initialize(verbose)` prints, per kernel, a "Multipole Error" and
"Local-exp Error" (m = 8, random sources in the box, targets outside; the
folded `(M_s2c·M_c2e0)(M_c2e1·M_e2t) − M_s2t` relative max) — the same test
our `conditioning(ops)` and the corner tests perform. Nothing beyond the SVD
cutoff is done about conditioning; Laplace's scale degeneracy and the
Helmholtz resonances we document are not addressed (their Helmholtz has a
fixed wavenumber `μ = 20π` and is flagged non-scale-invariant).

---

## 3. Kernel evaluation (near field, S2M check potentials, L2T)

### 3.1 Structure of `GenericKernel::Eval` (kernel.txx 421–530; SCTL
`generic-kernel.txx` is the newer equivalent)

Per call with `src_cnt` sources and `trg_cnt` targets:

1. AoS → SoA copy into a per-call scratch buffer, both counts padded to a
   multiple of `VecLen` (`DefaultVecLen<Real>`: 2 doubles on NEON, 4 on AVX2,
   8 on AVX-512), padding lanes zeroed.
2. Loop: source blocks of 500, then **targets in SIMD lanes** (`t += VecLen`),
   then sources one at a time broadcast with `Load1`; accumulators `tv[KDIM1]`
   stay in registers across all sources of the block; `uKerEval(tv, r, sv)`
   does `r² → rinv = approx_rsqrt<digits>(r², r² > 0) → FMA updates`.
3. Scale by the kernel constant once at the end (`FMA(tv, scale, old)`),
   SoA → AoS add into the output.

Self interaction: the mask `r² > 0` zeroes `rinv` for coincident points, so
there is no branch in the inner loop. Ours (`_direct_range`) has
`j == skip && continue` per source and accumulates `SVector`s through generic
`source_field` calls with `sqrt` and division; LLVM will not vectorise that
reduction across sources.

Micro-kernel flop counts (their `FLOPS` constants): Laplace 9, Laplace dipole
17, Laplace gradient 16, Stokes velocity 29 (SCTL's `Stokes3D_FxU::uKerApply`
23: identity + rank-one, `v_i += f_i·rinv + r_i·(f·r)·rinv³`), stresslet
("sym_dip") 35, velocity gradient 94.

`rsqrt` (SCTL `intrin-wrapper.txx`): `digits` is derived from the precision
(`SigBits·log10 2` = 15 for double, 7 for float). x86: `rsqrt14`/`rsqrtps`
estimate plus `log2(digits/4.2)` Newton steps (2 for double, 1 for float).
AArch64 (comment: "fastest on the M4"): float → `vrsqrteq_f32` + one
`vrsqrtsq_f32` step (≈ 4.9 digits) up to 4 digits, else `1/sqrt`; double →
plain `1/sqrt` for more than 6 digits (the bit-trick estimate
`0x5FE3FFFF20000000 − (bits >> 1)` with tuned Newton steps is used only up to
6 digits). So on Apple Silicon in double, PVFMM's inner loop is sqrt + divide,
the same instructions Julia emits; the gains to copy are the lane layout, the
mask, and the FMA accumulation, not a faster reciprocal square root.

### 3.2 U-list batching (`U_ListSetup` 5210–5605, `EvalListPts` 4176–4610)

Lists U0 (coarser leaf neighbours), U1 (same-level colleagues), U2 (finer leaf
neighbours) are built per target leaf. `EvalListPts` partitions **targets by
cumulative interaction cost** (`interac_cst`, `std::lower_bound` per thread)
and, per (target leaf, source leaf) pair, calls the SIMD kernel on the whole
source range of that leaf against the whole target range (`coord_shift`
applied for periodic images). Point data of all leaves live in one contiguous
level buffer (`CollectNodeData` packs `src_coord`, `src_value`, `trg_coord`
per node into `node_data_buff`), so a leaf's sources are one contiguous AoS
range — as ours (`tree.points[c]` ranges into `state.sources`).

Difference from ours: PVFMM evaluates per (target leaf, source leaf) pair with
targets vectorised, ours per target with a scalar loop over sources. The SIMD
form needs the targets of a leaf available as a block (we already bucket
targets by leaf in `evaluate!`, `bucket_by_node`).

### 3.3 S2M and L2T use the same kernel

`Source2UpSetup`: "targets" are the parent-level upward check surface points
(`tree->upwd_check_surf[depth]`, 296 at m = 8) and the output is the check
potential, then `M[2]=UC2UE0`, `M[3]=UC2UE1` are applied as two GEMMs batched
over as many leaves as fit the thread buffer (`vcnt`). `Down2TargetSetup`:
"sources" are the downward equivalent surface points with the local density as
values, targets the leaf's targets, kernel evaluation vectorised over targets.
Both map exactly onto our `s2m!` (sources → `large`, 1352 points at pc = 16)
and `l2t` (`large` → x). Note the cost: at p = 8, pc = 16 each target costs
1352 Stokeslet evaluations in `l2t` and each source 1352 in `s2m!`; with
ppl = 50 on uniform data the near field is ≈ 27 × 25 ≈ 700 evaluations per
target, so `l2t` + `s2m!` are the larger kernel-evaluation cost (≈ 135 M
evaluations each at 10⁵ points, versus ≈ 70 M in the near field). PVFMM's
equal counts make its `l2t`/`s2m` 4.6× cheaper at the same p. Vectorising
these two bodies pays more than vectorising the near field.

### 3.4 Kernel symmetries

`Kernel::Initialize` detects numerically, for each spatial transform
(ReflecX/Y/Z, SwapXY, SwapXZ) and for scaling by 2, the component
permutation-with-signs `P1, P2` such that `K(Rx) = P1·K(x)·P2` and the
homogeneity exponents per component (`src_scal`, `trg_scal`, solved by a small
least-squares system). These feed only `equiv_surf_perm` (M2M/L2L classes)
and the per-level scaling (`fft_scal`, `ifft_scal`, `scal[]` in the point
setups). Our `homogeneity(K)` trait and `rescale` cover the scaling part;
the reflection part is only needed for #9. Matrix assembly (`BuildMatrix`,
kernel.txx 346–418) is vectorised over targets or sources and OpenMP-parallel;
our `kernel_matrix` is scalar but runs only at operator build.

---

## 4. Real transforms, precision, spectra layout

- **R2C/C2R throughout the V-list** (`sctl::FFT<Real>` wraps FFTW `r2c`/`c2r`
  guru plans; `howmany = ker_dim·8`): `n3_ = (2m)²(m+1)` complex per box and
  component. Tables (`V_Type`) are transformed with the same R2C plan, so table
  and data spectra share the half-spectrum indexing. For us: `plan_rfft!` is
  not in FFTW.jl, but out-of-place `plan_rfft`/`plan_brfft` over the first
  dimension of the `(2p, 2p, 2p, ncp, nboxes)` array give `(p+1, 2p, 2p)`
  spectra; `embed` stays the same (it indexes the real grid), the blocked
  layout and `_hadamard_block!` only see `nfreq = (p+1)(2p)²` = 2304.
  Dual-number parts are real and go through the same path; complex expansions
  (Helmholtz) keep the complex transform (a type-level branch in
  `SpectralWorkspace`).
- **Single precision** is supported end to end (`float` instantiations,
  `Precomp_*_f.data`, `digits = 7` in the kernels, SSE/AVX float
  `matmult_8x8x2<float>`). The only precision-dependent algorithmic knob is
  the SVD cutoff `4·eps`. We already follow the kernel's `floattype`; nothing
  new to adopt beyond #3 and #4.
- Spectra memory per level: `fft_in` (sources of the block) + `fft_out`
  (targets) + per-thread buffers, bounded by the 1 GB block budget (§1.5).

---

## 5. Threading, scheduling, GPU boundary

- OpenMP, static contiguous partitions computed by hand (`a = n·tid/omp_p`),
  per-thread scratch from `ScratchBuf`. Pass order (`FMM_Tree::RunFMM`,
  `UpwardPass`, `DownwardPass`): S2U (all leaves), U2U level by level upward;
  then per level (one iteration for scale-invariant kernels): X, W, U, V
  lists; then D2D downward, D2T. Setup (lists, index tables, buffers) is done
  once in `SetupFMM` and reused for new densities (`ClearFMMData` +
  `PtFMM_Evaluate(tree, ..., &sl_den)`), like our `FMM`/`FMMState` split.
- V-list: FFTs partitioned by node index; Hadamard by frequency index (disjoint
  outputs); IFFT by node index. U-list: by cumulative cost. M2M/L2L
  (`EvalList`): input permutation by vector index; the GEMM is split by row
  ranges of the stacked-vector matrix per thread (each thread calls BLAS on its
  slice; no BLAS threading); output scatter boundaries are moved so that one
  node's contributions never straddle two threads (`out_ptr == output_perm[...]`
  loop). Equivalent to our `Threads.@threads :dynamic` over blocks and chunks.
- **GPU.** The only device code is `src/fmm_pts_gpu.cu` (`in_perm_k`,
  `out_perm_k`: permutation gathers/scatters with shared-memory staging) plus
  `CUBLASGEMM`, driven by `EvalListGPU`: the generic permute–GEMM–permute of a
  dense list. `DeviceMirror` pins host buffers (`cudaHostRegister`), copies a
  whole stage's stacked node vectors, precomputed matrices and index tables
  once per stage (`AllocDevice(..., copy=true)` in the `*Setup` functions and
  in `ClearFMMData`), and brings results back into a staging buffer that is
  added to the host array (`D2H_Wait:LocExp`, `D2H_Wait:Trg`) so CPU and GPU
  contributions sum. In this version `V_List` asserts `!device`, U/W/X go
  through `EvalListPts` whose device branch is `device && !have_gpu` (host
  pointers), and U2U/D2D setups are called with `device=false`; the CUDA path
  is effectively reachable only from the volume FMM (`fmm_cheb.txx`
  `EvalList(setup_data, device)`). There is no V-list GPU design to borrow.
  Our `m2l_level!` hook (level block of multipoles in, partial locals out,
  workspace in `state.scratch`) is already the right offload boundary; the
  data-movement pattern worth copying is "one H2D of the level's multipoles,
  compute, one D2H of the partial locals".

---

## 6. Parameters and accuracy, side by side

| | PVFMM | FlexibleFMM (3D cube) |
|---|---|---|
| order | `m` even, default 10 in `examples/src/fmm_pts.cpp` and `example1.cpp` (`-m 10`); docs: "typical 4–12, cost ∝ m³" | `p = 8` default (`StokesFMMPlan`), `pc = 2p` |
| surface points | `6(m−1)²+2` on both surfaces: 296 (m=8), 488 (m=10), 728 (m=12) | small 296 (p=8), large 1352 (pc=16) |
| radii | 1.05 / 2.95 (d = 0.05 / 0.025) | 1.1 / 2.8 (d = 0.1) |
| points per leaf | `max_pts` 350 (`fmm_pts.cpp -M`), 600 (`example1.cpp`); docs "~100–1000" | `ppl = 50` default; docs recommend ≈ 200 in 3D |
| pseudo-inverse | SVD cutoff `4·eps·σmax`, folded into M2M/L2L (safe order) | truncated SVD at `50·eps`, applied in factored form per level |
| M2L | FFT, R2C, 8×8 child blocks, 26 parent offsets | FFT, C2C, per child pair, 316 offsets, fused components |
| Laplace accuracy (relative max error, uniform random, from `docs/tutorials/particle-fmm-cpp.md`, gradient kernel) | 1e-3 (m=4), 1e-5 (6), 3e-7 (8), 6e-9 (10), 1e-10 (12) | worst-case corner test: 2e-5 (p=6), 1e-6 (8), 3e-8 (10); global ℓ∞ on uniform 10⁵ points ≈ 1e-7 at p=8 |
| Stokes accuracy | not reported in the repository | stokeslet 3e-6, stresslet 4e-5, doublet 1e-3 worst case at p=8 (global ≈ 1e-6) |
| throughput | no numbers in the repository or docs; see Malhotra & Biros, CiCP 18(3) 2015 (their table is per-node timings on Stampede, not reproduced here) | 12 Apple cores, 10⁵ uniform points, p=8: Laplace 0.5 s, Stokes 4.5 s matvec; Hadamard 70 / 50 GFlop/s |
| error metric | `CheckFMMOutput`: max |err| / max |value| over sampled targets, direct sum reference | same quantity in `test_3d.jl`, plus the corner-source worst case |

At equal p the two methods are the same discretisation (same lattice surfaces,
slightly different margins) and the accuracies are consistent (their m = 8
gradient 3e-7 vs our p = 8 potential 1e-7 global). Their m = 10 default buys
two more digits for 1.65× the surface points and 1.9× the frequencies.

---

## 7. Not applicable to the ghost-cell tree

- **W and X lists** (`W_Type`, `X_Type`, `W_ListSetup`, `X_ListSetup`): PVFMM
  evaluates them by kernel evaluation from the source's upward equivalent
  surface to the targets (W) and from the sources to the target's downward
  check surface (X). We have neither; the ghost construction turns them into
  same-level M2L pairs and colleague near-field sums (docs/tree.md).
- **U0/U2 lists** (coarser/finer leaf neighbours): our near field is
  colleague-only at the same level; the price is noted in docs/tree.md.
- **Periodic boundary operator** (`BC_Type`, the `PVFMM_BC_LEVELS = 45`
  recursion `M = M2L + M2M·(Pr·M·Pc)·L2L` with zero-average projections and
  the corner-polynomial correction): a periodic far-field operator applied at
  the root. Independent of our tree; only relevant if periodicity is added.
- **MPI / LET / hypercube multipole reduction** (`fmm_tree.txx`
  `MultipoleReduceBcast`, `mpi_tree.txx`): distributed memory only.
- **Volume FMM** (`fmm_cheb.*`, `cheb_utils.*`): Chebyshev sources.
- **CUDA offload**: see §5; nothing covers the V-list.
- **Symmetry-class permutations for V-list tables**: PVFMM does not do it
  either (§1.2). A cheaper variant that keeps our contiguous frequency runs is
  the offset parity of even kernels (`κ_{-o}(m) = κ_o(−m)`, hence
  `κ̂_{-o} = conj κ̂_o`): store 158 offsets, flip a sign in the generated kernel
  for the other 158 (#7). Single-axis reflections and axis swaps permute the
  frequency cube (reverse or re-stride the runs) and would break the SIMD
  inner loop, so the remaining 16-class reduction is not worth it.

---

## 8. Detailed mapping of the top recommendations onto our code

### 8.1 (#1) Fold `P_up` into M2M

`ki_common.jl` `_ki_dense_operators`: after `P_up = pinv_rtol(...)` and
`M2M = reduce(hcat, ...)`, form
`M2M_fold = P_up.V * (Diagonal(P_up.invS) * (P_up.Uh * M2M))` — three
products in that order, never `Matrix(P_up) * M2M`. `LevelOperators.M2M`
becomes `multipole_length × 8·multipole_length` (889 × 7112). `fmm.jl`
`upward_pass!`: the M2M product writes directly into `Mp[:, lr]` for group
parents; `partial_to_multipole!` runs only on the `s2m_nodes` columns (gather
those columns of `pM`, apply the factored `P_up`, scatter), so `pM` only needs
`partial_multipole_length` rows for childless nodes. `rescale`: multiply
`m2m_scale` by the pseudo-inverse factor. Check with the corner-source tests
and `test/test_homogeneous.jl`. Same for the circle method (`kifmm.jl`) if
wanted; there `nlarge = nsurface` so the fold is the plain 4× saving.

### 8.2 (#2) Sibling-group pair walk

`tree/lists.jl`: add to `InteractionList` (or alongside it) the parent-pair
list of level l: for each target group `g` (children of node `n`, `n` has
children and is not xlist), for each colleague slot `k` of `n` with
`c = colleagues[k, n]`, `c ≠ 0`, `has_children(c)`: source group
`child_first[c]`, parent offset index = `k` (the 3³ colleague slot; the self
slot 14 gives only near pairs and is skipped). A fixed table
`pattern[k][j1, j2] ∈ 0:316` (0 = near) comes from
`far_index[m2l_table_index(2·(cell[c]−cell[n]) + child_offsets[j2] −
child_offsets[j1])]`.

`spectral_m2l.jl` `_spectral_m2l_level!`, per frequency block: for each target
group in Morton order, for each of its ≤ 26 source groups, for each of the 27
values of `δ = j2 − j1 ∈ {−1,0,1}³` (one table `f = far(2O + δ)`): load the
table slice once (`tr_i[tt+q], ti_i[tt+q]`, 9 × 2B reals) and apply it to the
pairs `(j1, j1 + δ)` that are inside the group (up to 8) and have sources. This
needs `_hadamard_block!` to take a tuple of (source, target) column offsets
for one table, or to be called in a loop with the table slice already in
registers (a `@generated` body over the ≤ 8 pairs with the table values
hoisted). Reuse factors: table slice 64/27 ≈ 2.4× (weighted), source columns
up to 8× within the group pair, target columns (6 KB for the group) L1
resident for the whole group. Estimated L2 traffic per child pair: ≈ 960 B of
table + ≈ 100–200 B of source versus ≈ 3 KB now. Ghost targets are skipped at
group granularity (a group is all-ghost or all-real). The dense Stokes
corrections (`pin_row`, `src_col`) keep using the per-pair CSR list.

If the fuller PVFMM layout is wanted later (frequency-outermost with the 8
children as the SIMD lanes and 8×8-expanded tables), the costs to weigh are
the table memory (552 MB for Stokes p = 8 with R2C, versus ≈ 105 MB
unexpanded) and the 9.1 % of zero blocks; the benefit is that per frequency the
whole working set of a 100-parent block is ≈ 50 KB and L1-resident.

### 8.3 (#3) R2C

`SpectralWorkspace`: `Σ̂`, `Û` become `(p+1)·(2p)²` × ncp × nboxes complex;
add real staging arrays `(2p)³ × ncp × nboxes` for the embed/extract steps (or
embed directly into the real array and extract from the real array);
`fwd = plan_rfft(real_array, 1:3)`, `bwd = plan_brfft(complex_array, 2p, 1:3)`.
`_kernel_tables`: `rfft(t)` instead of `fft(t)`. `_block_size(nfreq, ncp)`
with `nfreq = 2304`: 2304 = 2⁸·9, so B = 16 still divides it (Laplace: 64
too). Branch on `S <: Complex` to keep the complex path for Helmholtz.

### 8.4 (#4) SIMD bodies

Write one kernel-specific "apply" per 3D kernel in the shape of
`Stokes3D_FxU::uKerApply`: inputs `r = (rx, ry, rz)` as `NTuple{L}`/`Vec{L}`
lanes of targets, `f` broadcast; `rinv = 1/sqrt(r²)` masked by `r² > 0`;
`v_i += f_i·rinv + r_i·(f·r)·rinv³`. Use it in: `s2m!` (lanes = check points
of `large`, sources broadcast; 1352 is a multiple of 4 and 8), `l2t` (lanes =
the leaf's targets, `large` points broadcast; targets are already bucketed per
leaf in `evaluate!`, so evaluate a leaf's targets as a block instead of per
target), `_direct_range` (lanes = targets of the leaf, sources of the colleague
range broadcast, the `skip` source handled by the mask on `r² > 0`, which also
removes the branch). The `Dual`/complex generic paths stay on the current
scalar code; only `Float64`/`Float32` `SVector` positions take the SIMD path
(dispatch on eltype).

### 8.5 (#5) FFT threading

`plan_fft!` on the whole level executes on FFTW's thread count, which is 1
unless `FFTW.set_num_threads` was called. Either set it to `Threads.nthreads()`
at workspace creation (plans must then be created after the call) or do as
PVFMM: one plan for a single box's `(2p,2p,2p,ncp)` array and
`Threads.@threads` over boxes executing it (FFTW `execute_dft` on different
arrays is thread-safe; planning is not). Measure first: for Laplace at 10⁵
points (≈ 4700 boxes × 2 × 0.25 MFlop) the transforms are ≈ 2.4 GFlop, which
is 0.25–0.5 s single-threaded — potentially most of the 0.5 s matvec.

### 8.6 (#6) Chunked spectra

In `_spectral_m2l_level!`, split `torder` (or the new target-group list) into
Morton-contiguous chunks with a byte budget; for each chunk gather the unique
sources into a compact index map, embed/transform only those, run the Hadamard
on the chunk, inverse-transform and extract the chunk's targets. `Σ̂`/`Û`
shrink from `nboxes` to `max(nsrc_chunk, ntgt_chunk)` columns. The extra
transforms are the chunk halo (≈ 25 % at 20 000 target groups per chunk).
