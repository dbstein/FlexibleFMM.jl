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

## Grid-boundary surfaces and spectral translation (`GridKernelIndependent{D}`)

The second surface family, after Ying, Biros and Zorin, works in any dimension: the
small surface is the set of boundary points of the `p`-per-edge lattice on the
square or cube of half-width `(1+d)·r` (`4p − 4` points in 2D, `6p² − 12p + 8` in 3D),
the large one the same lattice on `(3 − 2d)·r`, with `r` the box half-width. The
margin `d` keeps corner sources off the equivalent surface and the large surface
inside the nearest far box (whose face is at `3r`). Everything above (factored
solves, Stokes pin and source, rescaling) is shared with the circle method.

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

Measured worst case (corner source, facing targets, Laplace) at equal point counts:

| points | 20 | 28 | 36 | 44 |
|---|---|---|---|---|
| grid, d = 0.05 | 2e-5 | 2e-6 | 5e-8 | 5e-9 |
| grid, d = 0.1 | 6e-5 | 4e-6 | 1e-7 | 1e-8 |
| grid, d = 0.2 | 2e-4 | 1e-5 | 4e-7 | 2e-8 |
| circles | 3e-4 | 1e-5 | 2e-7 | 2e-8 |

Squares with a small margin are two to four times more accurate than circles at the
same number of points. For the Stokes kernel the random-box translation error at
`p = 8, 12, 16, 20` is `8e-5, 4e-8, 9e-11, 3e-12` (stresslet sources about ten
times worse), about a hundred times the Laplace error at the same `p`, the same as
with circles.
