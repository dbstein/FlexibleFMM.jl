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

Mode `m` of the small→large check map carries the factor `(r1/r2)^|m| = 0.547^|m|`
and `Nequiv` real points represent `Nequiv/2` harmonics, so the error of a
translation to the nearest box is about `0.547^(Nequiv/2)`: `1.4e-8` at 60 points,
`8e-14` at 100 (measured in `test/test_operators_kifmm.jl`). Beyond ~100 points the
check matrices are singular to working precision and more points add nothing.

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
