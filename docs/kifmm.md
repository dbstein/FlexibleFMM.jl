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

## The check-to-equivalent solve must stay factored

The pseudo-inverse of the check map has norm `1/σ_min ≈ 0.547^(−Nequiv/2)`, `1e13` at
100 points. Forming it as one matrix, or folding it into `M2M`/`M2L`/`L2L`, means
forming products whose terms are `1e13` and whose results are `O(1)`: the rounding
error is `eps/σ_min` in *every* direction and the far field inherits it (`6e-4` at
100 points, measured; the folded design was tried first). Applied in factored form,
`V·(Σ⁻¹·(Uᴴ·c))`, the rounding amplified by `1/σ_j` stays in mode `j`, which the
evaluation attenuates by `σ_j` again. The transforms are therefore applied per level
on the whole block, in factored form (`FactoredPinv`), with the singular values
below `pinv_rtol·σ₁` dropped. An LU solve would be equally stable but would not
report or regularise the degeneracies below.

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
vector kernel are fused into one pass per block. Walking the pairs offset by
offset instead, which is what the dense path wants, streams every block once per
pair and is bandwidth bound: it was 2.4× slower for Laplace in 3D. The stage runs
at about 70 GFlop/s for Laplace and 50 for Stokes on 12 Apple cores (10 and 6 on
one), and the Stokes translation is the 3D hot spot: 100 000 uniform points take
0.5 s for the Laplace matvec and 4.5 s for Stokes at `p = 8`, almost all of it in
the `nc² = 9` products per pair. The planned reductions are the symmetry of the
Stokeslet tables (6 distinct products instead of 9), real-input transforms (a real
kernel's spectrum is Hermitian, halving the frequencies) and fewer table streams
per iteration. In 2D the spectral path does not pay: at 10⁶ uniform Laplace points
the translations take about 4× longer than the dense circle operators (0.67 s
against 0.17 s); the 2D implementation exists to be tested, not to be fast.

## Against the Flatiron libraries

`bench/compare/compare.jl` runs the same points through FMM2D.jl and FMM3D.jl
(their own environment; the kernel normalisations are calibrated on a small
problem and reported). Uniform points, 12 Apple cores, errors at 2000 targets
against the direct sum; ours is the matvec with cached operators (tree build
excluded, 0.01–0.15 s), theirs the one-shot call including their tree build.

| case | ours | FMM2D / FMM3D |
|---|---|---|
| 2D Laplace, 10⁶ | circles 30: 0.25 s, 2e-8; circles 60: 0.40 s, 8e-14 | 0.66 s, 2e-9 (eps 1e-6); 0.66 s, 2e-13 (1e-9) |
| 2D Helmholtz k = 10, 10⁶ | circles 60: 1.46 s, 1e-13 | 1.43 s, 5e-9 (1e-6); 1.69 s, 3e-12 (1e-9) |
| 2D Stokes, 10⁶ | circles 60: 0.76 s, 1e-11 | 1.28 s, 2e-8 (1e-6); 1.34 s, 4e-12 (1e-9) |
| 3D Laplace, 10⁵ | cube 6: 0.27 s, 2e-6; cube 8: 0.45 s, 3e-8; cube 10: 0.73 s, 6e-10 | 0.10 s, 7e-5 (1e-3); 0.36 s, 1e-7 (1e-6); 0.26 s, 4e-11 (1e-9) |
| 3D Helmholtz k = 3, 10⁵ | cube 8: 1.03 s, 3e-8 | 0.77 s, 2e-7 (1e-6); 0.82 s, 2e-11 (1e-9) |
| 3D stokeslets, 10⁵ | cube 6: 1.2 s, 2e-5; cube 8: 2.6 s, 4e-7 | 0.26 s, 5e-4 (1e-3); 1.15 s, 2e-6 (1e-6) |
| 3D stresslets, 10⁵ | cube 8: 2.5 s, 1e-7 | 1.3 s, 5e-7 (1e-6) |

In 2D the circle method is 1.5–2× faster than FMM2D at equal or better accuracy. In
3D FMM3D is 2–3× faster for Laplace and Stokes at comparable accuracy and about
even for Helmholtz; the gap is the Stokes translation cost discussed above and the
dense M2M/L2L of the finer large surface.

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
accumulated check data of a level in arrays of `(2p)³` complex numbers per box and
component (per real/complex part for dual or complex expansions): `128 KB` per box
for Laplace at `p = 8`, three times that for Stokes, about `600 MB` for `10⁵`
uniform points and `6 GB` for `10⁶`. Real-input transforms (halving the frequencies
of a real kernel), single precision for the transforms, and processing the target
boxes of a level in chunks are the planned mitigations. The dense operators of a
level are `nlarge·nc × 2³·(nsurface·nc + 1)` for M2M and its transpose shape for
L2L, `230 MB` each for Stokes at `p = 8`, built once for a homogeneous kernel.
