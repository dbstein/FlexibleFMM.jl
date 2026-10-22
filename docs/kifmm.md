# Kernel-independent operators

A box of width `w` carries two concentric circles with `Nequiv` equispaced points:

- the *small* circle, radius `r1 = w·√2/2`, the circumcircle of the box;
- the *large* circle, radius `r2 = w·(4 − √2)/2 ≈ 1.293·w`.

`r2` is exactly tangent to the small circle of the nearest interaction-list box: the
nearest such box has its center `2w` away and its small circle reaches back to
`2w − r1 = r2`. So the large circle is the largest check surface that still lies
inside the region where all far-field sources are, and the small circle the smallest
equivalent surface that encloses the box.

Upward, the multipole is a density on the small circle, determined from the check
potentials the sources (or the children's multipoles) produce on the large circle.
Downward, the local is a density on the large circle, determined from the check
potentials far multipoles (or the parent's local) produce on the small circle.
The translation operators map densities to check potentials and are built as
explicit kernel matrices in their own orientation (`kernel_matrix(K, sources,
targets)`, rows = targets), so nothing relies on kernel symmetry; the non-symmetric
test kernel `G − c∂₁G` exercises this.

## Accuracy

Measured worst case (a unit charge at a corner of the source box, targets on the
facing edge and corner of the nearest interaction-list box, error relative to that
field):

| points | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| worst case | 7e-5 | 3e-7 | 1e-8 | 2e-12 | 6e-15 |
| `0.547^N` | 6e-5 | 5e-7 | 4e-9 | 3e-13 | 4e-17 |
| global ℓ∞, uniform, 10⁵ points | 2e-5 | 2e-7 | 6e-9 | 6e-12 | 3e-14 |
| global ℓ², same | 1e-5 | 8e-8 | 1e-9 | 8e-13 | 1e-14 |

So the error decays like `(r1/r2)^N = 0.547^N` with `N` the number of points per
circle, about 2.6 digits per 10 points, until rounding stops it near 1e-14 at
`N ≈ 64`; global relative errors on realistic distributions (normalised by the
maximum of the whole potential) are one to two orders smaller than the single
worst interaction. This matches the ten digits at 32 points reported by Ying, Biros
and Zorin. Beyond ~100 points the check matrices are singular to working precision
and more points add nothing.

## The check-to-equivalent solve: factored, and folded in the safe order

The pseudo-inverse of the check map has norm `1/σ_min ≈ 0.547^(−Nequiv/2)`, `1e13` at
100 points. Forming it as one matrix means forming products whose terms are `1e13`
and whose results are `O(1)`: the rounding error is `eps/σ_min` in *every*
direction and the far field inherits it (`6e-4` at 100 points, measured; the first
design folded the dense pseudo-inverse into `M2M`/`M2L`/`L2L` this way). Applied in
factored form, `V·(Σ⁻¹·(Uᴴ·c))`, the rounding amplified by `1/σ_j` stays in mode
`j`, which the evaluation attenuates by `σ_j` again. The check data of the
childless nodes (upward) and of every node (downward) are therefore transformed per
level in factored form (`FactoredPinv`), with the singular values below
`pinv_rtol·σ₁` dropped.

The same order makes folding safe. `M2M` is stored as `V·(Σ⁻¹·(Uᴴ·M2Mᶜ))` and `L2L`
as `((L2Lᶜ·V)·Σ⁻¹)·Uᴴ`, with `M2Mᶜ`, `L2Lᶜ` the check matrices: the check matrix is
projected onto the singular vectors first, and since the columns of a physical check
matrix (fields of nearby densities) have components of size `σ_k` along the `k`-th
left vector, the scaled products stay `O(1)` and nothing cancels. `M2M` then maps
the children's multipoles straight to the parent's and `L2L` the parent's check
data to the children's, so the pseudo-inverses are applied only where the data
enter (the childless nodes' check potentials) and leave (the densities the
evaluation reads). For the cube method at `p = 8` this takes the Stokes `M2M` from
`4056 × 7112` to `889 × 7112` (230 to 50 MB, 58 to 12.6 MFlop per sibling group),
likewise `L2L`, and for a homogeneous kernel the folded operators are independent
of the box width. PVFMM's `Precomp` folds in the same order. An LU solve would be
equally stable but would not report or regularise the degeneracies below.

## Known degeneracies

- **Laplace scale degeneracy.** The check potential of a net charge `Q` on the large
  circle is `−Q·log(r2)/2π`, which vanishes at `r2 = 1`, i.e. `w = 2/(4 − √2) ≈
  0.7735`; a level at that width loses the net charge, and nearby widths lose digits.
  `conditioning(ops)` reports it; the cure is to rescale the problem (or the kernel
  by a constant), since a kernel-independent code cannot fix it.
- **Helmholtz resonances.** Mode `m` of the check map is proportional to
  `J_m(k·r1)·H_m(k·r2)`, singular when `k·r1` is a zero of `J_m`: `k·w = √2·j_{m,n}`,
  first at `k·w ≈ 3.40`. Expansions live on levels ≥ 3 (widths ≤ `W/4`), so `k·W ≤
  12` on the root box is safe; the test suite uses `k·W ≤ 10` and reports a scan of
  larger wavenumbers. Yukawa has no such zeros.

## Square and cube surfaces and spectral translation (`CubeKernelIndependent{D}`)

The second surface family, after Ying, Biros and Zorin, works in any dimension: the
small surface is the set of boundary points of the `p`-per-edge lattice on the
square or cube of half-width `(1+d)·r` (`4p − 4` points in 2D, `6p² − 12p + 8` in 3D),
the large one the boundary points of a `pc`-per-edge lattice on `(3 − 2d)·r`, with
`r` the box half-width and `pc = 2p` by default. The margin `d` keeps corner sources
off the equivalent surface and the large surface inside the nearest far box (whose
face is at `3r`). Everything above (factored solves, Stokes pin and source,
rescaling) is shared with the circle method.

Why the large surface is finer. With `pc = p` its lattice spacing is
`(3 − 2d)/(1 + d) ≈ 2.5` times that of the small one, so as the upward check it
undersamples the equivalent density: density modes whose field is small *at the
check samples* but not between them get singular values of `1e-11` (3D, `p = 8`),
far below their true size, and the fit either keeps them and amplifies rounding or
truncates them and loses what they carry. Laplace charges hardly notice (the data
are smooth), but anything with faster decay does: in 3D at `p = 8` the stresslet and
doublet translations were off by `0.2` and `0.8` in the worst case and no choice of
truncation tolerance brought them below `1e-3`. Doubling the points per edge on the
large surface (`pc = 2p`, 4.6× the points in 3D) removes the problem: `5e-5` and
`1e-3`, with the stokeslet at `3e-6` against `2e-3` before, and the truncation
tolerance no longer matters. A finer large surface is needed in both of its roles;
refining only the upward check (keeping the downward equivalent density coarse)
recovers a tenth of the gain. In 2D the same change improves the Stokes
translations ten to twenty-five times (stokeslet, stresslet, doublet worst cases
`5e-6, 6e-5, 5e-4` at `p = 8`, `2e-8, 1e-7, 2e-6` at `p = 12`, `3e-11, 2e-10, 5e-9`
at `p = 16`) and Laplace two to forty times (table below). Circles have no such
problem: the equispaced circle-to-circle map is diagonal in the Fourier modes, so
nothing is aliased. The
cost of `pc = 2p` is in `s2m!` and `l2t` (kernel evaluations at `nlarge` points per
source or target) and in the dense M2M/L2L sizes; the spectral translations, which
use only the small lattice, are unchanged.

What the lattice buys is the translation. The source's small surface and the
target's small surface are lattices with the same spacing `h`, so the kernel between
them depends only on the lattice index difference and the box offset, `κ_o(m) =
K(o·w, h·m)`, `m ∈ [−(p−1), p−1]^D`: the check potentials are a discrete convolution
of the density, embedded in the `p^D` lattice, with that table, which zero padding to
`(2p)^D` turns into a circular convolution computed by FFTs. Per level this is one
forward transform per source box, one Hadamard product with the offset's table per
interaction pair, and one inverse transform per target box, instead of a dense
`N × N` product per pair: `(2p)^D` flops per pair instead of `N²`, which in 3D is 13×
fewer at `p = 6` and 20× at `p = 8` and is the natural GPU formulation. The
driver's level-wise hook `m2l_level!` carries this; vector kernels use one table per
component pair; the Stokes pressure row and point-source column are small dense
corrections applied per pair. Dual-number and complex expansions are split into
their real or complex parts for the transforms (they are linear), so the path stays
differentiable.

The Hadamard stage is organised for the cache rather than for the flop count. The
spectra of a level are transposed into a frequency-blocked layout (`B` frequencies
of every box contiguous, real parts then imaginary parts, `B` = 64 divided by the
number of columns per box), threads own disjoint blocks, and within a block the
pairs are walked target by target with the targets in Morton order of their cells
(`InteractionList.torder`), so that a target's block stays in cache across its
pairs and neighbouring targets reuse their sources; the component products of a
vector kernel are fused into one pass per block, with the values of every stream
loaded before the stores so that a symmetric kernel (`symmetric_kernel`, the
Stokeslet) loads each of its 6 distinct tables once instead of 9. Walking the
pairs offset by offset instead, which is what the dense path wants, streams every
block once per pair and is bandwidth bound: it was 2.4× slower for Laplace in 3D.
Real expansions of a real kernel use real-input transforms: their spectra are
Hermitian, so only `(p+1)(2p)²` of the `(2p)³` frequencies are stored and
multiplied (0.56× the work and memory); complex kernels or densities take the
full complex transforms. Together with the table symmetry this halved the Stokes
translation. At `p = 8` and 200 points per leaf, 100 000 uniform points take
0.31 s for the Laplace matvec (0.21 s in the translations) and 1.4 s for Stokes
(1.1 s), the latter still the 3D hot spot; the Stokes Hadamard pass is load
bound (12 table and 6 source values per entry for 36 multiply-adds). In 2D the
spectral path does not pay: at 10⁶ uniform Laplace points the translations take
about twice as long as the dense circle operators (0.38 s against 0.17 s); the 2D
implementation exists to be tested, not to be fast.

## Against the Flatiron libraries

`bench/compare/compare.jl` runs the same points through FMM2D.jl and FMM3D.jl
(their own environment; the kernel normalisations are calibrated on a small
problem and reported). Uniform points, 12 Apple cores, errors at 2000 points (separate targets when these
rows were measured; the script now evaluates at the sources only)
against the direct sum; ours is the matvec with cached operators (tree build
excluded, 0.01–0.15 s), theirs the one-shot call including their tree build.

| case | ours | FMM2D / FMM3D |
|---|---|---|
| 2D Laplace, 10⁶ | circles 30: 0.17 s, 2e-8; circles 60: 0.31 s, 8e-14 | 0.69 s, 2e-9 (eps 1e-6); 0.64 s, 2e-13 (1e-9) |
| 2D Helmholtz k = 10, 10⁶ | circles 30: 0.75 s, 4e-8; circles 60: 1.17 s, 8e-14 | 1.48 s, 8e-9 (1e-6); 1.68 s, 3e-12 (1e-9) |
| 2D Stokes, 10⁶ | circles 30: 0.35 s, 1e-6; circles 60: 0.64 s, 8e-12 | 1.27 s, 8e-9 (1e-6); 1.34 s, 3e-12 (1e-9) |
| 3D Laplace, 10⁵ | cube 8: 0.14 s, 3e-8 (0.45 s before the fold and the real transforms) | 0.35 s, 1e-7 (1e-6); 0.26 s, 4e-11 (1e-9) |
| 3D Helmholtz k = 3, 10⁵ | cube 8: 0.65 s, 3e-8 | 0.75 s, 3e-7 (1e-6); 0.82 s, 3e-11 (1e-9) |
| 3D Yukawa k = 5, 10⁵ | cube 8: 0.36 s, 3e-8 | 0.59 s, 1e-7 (1e-6); 0.54 s, 4e-11 (1e-9) |
| 3D stokeslets, 10⁵ | cube 8: 1.08 s, 4e-7 (2.6 s before) | 1.15 s, 2e-6 (1e-6); 1.14 s, 1e-9 (1e-9) |
| 3D stresslets, 10⁵ | cube 8: 1.29 s, 6e-8 (2.5 s before) | 1.27 s, 4e-7 (1e-6); 1.48 s, 4e-10 (1e-9) |

The rows are warm matvecs (the best of three on an initialised state; the
comparison script reports the first application of a fresh state, which creates
the spectral workspaces, separately, after an external review noted that the
earlier rows had included it). In 2D the circle method is 2–4× faster than FMM2D
at equal or better accuracy. In 3D, after folding the pseudo-inverses and the
real-input transforms, the cube method is 2.5× faster than FMM3D for Laplace and
1.6× for Yukawa and level for Helmholtz and Stokes at comparable accuracy; the 3D
rows above are at 200 points per leaf on uniform points. On a fibre (a helix of
10⁵ points, 8 levels) the picture changes: the cube method takes 1.4 s for Laplace
and 13 s for Stokes against FMM3D's 1.8 and 4.6 s, and the harmonic methods 0.26
and 1.3 s (`docs/harmonics.md`): the deep tree with small boxes multiplies the
cube method's per-box surface evaluations and spectra.

With `device=metal` our rows run on the GPU (`DevicePlan` + `device_matvec!`, the
M3 Max in Float32, the libraries still on the CPU in Float64): Laplace cube 8 in
0.035 s and Stokes cube 8 in 0.121 s per matvec, 10× and 9× the library's time at
`eps = 1e-6`, with the errors against the direct sum at the Float32 floor
(`2e-5`, the same at `p = 6, 8, 10`). Helmholtz stays on the CPU on Metal, which
has no complex GEMM.

Three cheap measured gains from the bempp/kifmm study (`docs/notes/kifmm_report.md`):
the excluded source of the near field is masked rather than skipped, which keeps
the body branch-free (Stokes self evaluation 210 → 136 ms at 10⁵ points);
OpenBLAS's helper threads spin for about 0.1 s after every call (its thread
timeout) and compete with the Julia-threaded phases that follow the GEMMs, which
`OPENBLAS_THREAD_TIMEOUT=4` in the environment removes: Laplace matvec 0.34 →
0.21 s, Stokes 1.40 → 1.07 s, the near field alone 1.4× (forwarding BLAS to
Apple's Accelerate gives the same numbers on this machine; halving the BLAS
thread count gives half the gain; splitting the GEMMs over Julia threads with
single-threaded BLAS calls was tried and is slower, toggling the OpenBLAS pool
being expensive and Accelerate's own threading better, so that split is taken
only when BLAS is started with one thread, `column_blocks` in `docs/framework.md`);
and a reciprocal square
root by estimate and Newton steps gains nothing in Float64 on the CPU, while on
Metal the device's `rsqrt_fast` (the `FastMath` kernel policy, `fast=true` on the
device plan) gains 1.3× on the Laplace and 1.15× on the Stokes device evaluation at 10⁶ targets (0.076 → 0.058 s and 0.125 → 0.108 s on the M3 Max; the matvec 0.030 and 0.079 s) with deviations of 1e-7 from the precise kernel; the Stokes kernel in isolation gains 2×. The compressed translations of
kifmm were measured on
our operators: the far offsets see a smooth kernel, so at `p = 8` the per-offset
ranks are 25 (Laplace) and 69 (Stokes) at `1e-8` and 9 and 34 at `1e-6`, against
296 and 889, while the shared basis of all offsets barely compresses (250 and
776 at `1e-8`); on the CPU the compressed flops equal our half-spectrum Hadamard
flops, so the gain would be the GEMM-to-Hadamard rate ratio only, and the path
is used on the GPU instead (`DevicePlan(…; tol)`), where it replaces the dense
class products.

Two measured non-gains, recorded so they are not retried blindly: reshaping the
near field to run a chunk of targets in lanes against each source gains 1.3–1.4×
in isolation (the loops are bound by the square root and division on 2-wide
NEON), which is small against the invasiveness of the change; and the
sibling-group pair walk of PVFMM is expected to gain little here because the
tables of all 316 offsets already stay in L2 within a frequency block.

Points per leaf matter more in 3D than in 2D, because every box carries
`nlarge · nc` check values and its share of the translations while a surface
distribution fragments the octree (100 000 points on a sphere give 87 000 nodes
at `ppl = 50`). For Stokes on a sphere the matvec goes from 33 s at `ppl = 50` to
5.4 s at 100 and 1.6 s at 400, with the near field still a small share; start
from `ppl ≈ 200` for surface and fibre distributions in 3D.

Measured worst case (corner source, facing targets, Laplace, 2D) against circles
with the same number of small-surface points:

| small-surface points | 20 | 28 | 36 | 44 |
|---|---|---|---|---|
| square, d = 0.05 | 1e-5 | 3e-7 | 3e-9 | 2e-10 |
| square, d = 0.1 | 1e-5 | 6e-7 | 6e-9 | 3e-10 |
| square, d = 0.2 | 2e-5 | 7e-7 | 1e-8 | 3e-10 |
| circles | 3e-4 | 1e-5 | 2e-7 | 2e-8 |

In 3D (`d = 0.1`, the same worst case, with the large-surface count in parentheses):

| p (small / large points) | Laplace | stokeslet | stresslet | doublet |
|---|---|---|---|---|
| 6 (152 / 728) | 2e-5 | 1e-4 | 1e-3 | 2e-2 |
| 8 (296 / 1352) | 1e-6 | 3e-6 | 4e-5 | 1e-3 |
| 10 (488 / 2168) | 3e-8 | | | |

Global errors on uniform or surface distributions are one to three orders smaller
(Laplace `1e-7`, Stokes with all three singularities `1e-6` at `p = 8`); a tight
cluster of doublets can expose the doublet's worst case. The Stokes rows show that
the error grows with the singularity's order of decay (`1/r`, `1/r²`, `1/r³`), as it
does for Laplace dipoles (`1e-4` at `p = 8`); stresslets and doublets ask more of
the equivalent density than charges do.

Memory in 3D. The spectral path keeps the transformed multipoles and the
accumulated check data of a level in arrays of `(p+1)(2p)²` complex numbers per
box and component for real expansions of a real kernel (`(2p)³` otherwise; per
real/complex part for dual or complex expansions): `72 KB` per box for Laplace at
`p = 8`, three times that for Stokes, about `340 MB` for `10⁵` uniform points and
`3.4 GB` for `10⁶`. Single precision for the transforms and processing the target
boxes of a level in chunks are the remaining mitigations. The folded M2M and L2L
are `(nsurface·nc + 1) × 2³·(nsurface·nc + 1)` and its transpose shape, `50 MB`
each for Stokes at `p = 8`, built once for a homogeneous kernel; the tables of the
316 offsets are `290 MB` for Stokes (full and half spectra, real and imaginary
parts).
