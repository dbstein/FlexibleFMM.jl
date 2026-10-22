# Analytic expansions: `SphericalHarmonics`

The classical multipole method for the 3D Laplace kernel in the framework's
five-operator form: expansions in solid harmonics of degree `p`, exact M2M and L2L,
dense M2L operators per far offset, charges and dipoles, potential and gradient
outputs. The basis, its normalisation and the derivation of the operators are in
the header of `src/operators/harmonics.jl`; this note records the design choices
and the measurements. The guides were fmm3d's Laplace routines
(`docs/notes/analytic_report.md`); nothing in the driver changed.

## Design

- **Operators from one identity.** At build time the complex Hobson harmonics
  `Υ_n^m = r^n P_n^m e^{imφ}/(n+m)!`, `Θ_n^m = (n−m)! P_n^m e^{imφ}/r^{n+1}` are
  used, in which the regular addition theorem is a plain convolution over `(n, m)`.
  M2M and L2L are its coefficients evaluated at the child centre `δ̂ ∈ {±1/4}³`,
  exact (triangular in the degree); the M2L, `L_n^m = Σ (−1)^k Θ_{n+k}^{m+l}(o) M_k^l`,
  follows from the addition theorem and the generating identity
  `1/|x − y| = Σ conj Υ(y) Θ(x)` by matching coefficients, and the tests check it
  as an identity, the three derivative identities (`∂_z`, `∂_x ∓ i∂_y` lower the
  degree and shift `m` by 0, ±1) and harmonicity by automatic differentiation.
  The three steps of fmm3d (rotate, shift along `z`, rotate back) are not needed:
  with dense operators built once per width the `O(p⁴)` closed form is cheaper.
- **Stored basis.** Schmidt semi-normalised and real: `Υ̃ = s Υ`, `Θ̃ = Θ/s`,
  `s = √((n−m)!(n+m)!)`, so that `|Υ̃_n^m| ≤ r^n`, `|Θ̃_n^m| ≤ r^{−n−1}` and every
  operator entry is `O(2^n/|o|^{n+1})`; the Hobson values carry `n!` and `1/n!`
  and overflow Float32 near `p = 30`. For real densities `M_n^{−m} = (−1)^m conj M_n^m`,
  so a column holds `(p+1)²` reals (`M_n^0`, then `Re, Im M_n^m`), the operators are
  real matrices (`real_operator` converts a complex operator that commutes with the
  conjugate symmetry), and complex densities promote the columns. fmm3d stores
  `(p+1)(2p+1)` complex numbers for the same information.
- **Per-point work.** The recurrences in the Schmidt normalisation need two rows of
  state and precomputed square-root coefficients (`HarmonicTables`), so `s2m!` and
  `l2t` are allocation-free folds for any `p`, which is therefore a runtime field,
  not a type parameter (nothing is unrolled). Dipoles and gradients use the
  derivative identities: a dipole scatters each `Υ̃_{n'}^{m'}` into three entries of
  degree `n'+1`, a gradient reads three coefficients per harmonic; no trigonometric
  functions anywhere.
- **Width independence.** In box coordinates `x̂ = (x − c)/w` the M2M and L2L are
  pure numbers and the M2L carries `c_K/w`: `homogeneity(Laplace3D) = −1` and
  `rescale` shares one reference set, as for the cube method.
- **Ordering of the far offsets, the children and the real coefficients** is the
  framework's; `multipole_field`, `check_m2m`, `check_l2l` and
  `check_m2l_all_offsets` ran unchanged.

## Accuracy

Translation error of one charge (relative, Float64, width 1) for the corner source
facing the corner target of the far box, the configuration that saturates the
bound, and for random boxes (40 points each, worst of the 316 offsets; charges and
dipoles):

| p | (2,0,0) corner | (2,1,1) corner | (3,3,3) corner | random, charges | random, dipoles |
|---|---|---|---|---|---|
| 8 | 2.1e-3 | 6.9e-4 | 1.0e-6 | 6.0e-4 | 3.7e-3 |
| 12 | 1.8e-4 | 1.4e-4 | 1.6e-9 | 3.0e-5 | 3.1e-4 |
| 16 | 1.6e-4 | 6.5e-6 | 2.6e-12 | 1.5e-6 | 2.7e-5 |
| 20 | 3.2e-5 | 1.5e-7 | 4e-15 | 2.4e-7 | 2.6e-6 |
| 22 | 1.4e-6 | 1.1e-7 | 8e-16 | 1.1e-8 | 4.9e-7 |
| 26 | 6.3e-6 | 3.2e-9 | 8e-16 | 2.9e-9 | 4.1e-7 |
| 30 | 8.1e-7 | 7.0e-10 | 8e-16 | | |

The far offsets reach roundoff, which confirms the operators; the nearest offset
converges only like `0.70^p`, irregularly. That is the classical M2L bound: with
`|M̃_k| ≤ ρ^k`, `|M2L̃_{nk}| ≤ C(n+k, n)/|o|^{n+k+1}` and `|Υ̃_n(x̂)| ≤ ρ^n`, the error
of truncating the `k`-series of every `L_n` at `p` sums to `≈ 3.7·(ρ/(|o| − ρ))^{p+1}
= 3.7·0.764^{p+1}` for `ρ = √3/2`, `|o| = 2`, and the corner pair nearly attains
it; the `0.577^p` rule of fmm3d (`l3dterms`) is the multipole truncation alone. The
cube method's check surfaces fit this near field far better: `1.1e-6` at `p = 8` on
the same configuration (`docs/kifmm.md`). Dipoles are a factor `≈ p` worse at equal
`p` (the derivative multiplies the degree-`n` coefficients by `n`), the same
mechanism as for the libraries. Global errors on data are two to three orders
smaller than the corner case: on `10⁵` uniform points the relative error against
direct sums is `3.1e-6`, `2.5e-7`, `1.9e-8` at `p = 12, 16, 20`.

## Cost

`10⁵` uniform points, `ppl = 200`, 12 threads, `OPENBLAS_THREAD_TIMEOUT=4`:

| method | up | down | self | matvec | error |
|---|---|---|---|---|---|
| harmonics p = 12 | 0.003 | 0.088 | 0.027 | 0.119 s | 3.1e-6 |
| harmonics p = 16 | 0.007 | 0.209 | 0.028 | 0.245 s | 2.5e-7 |
| harmonics p = 20 | 0.012 | 0.413 | 0.029 | 0.455 s | 1.9e-8 |
| cube p = 8 | 0.025 | 0.086 | 0.033 | 0.145 s | 3.3e-8 |

The upward pass is 4× cheaper than the cube method's (`O(p²)` per point against
1352 kernel evaluations), the near field is the same, and the dense M2L GEMMs,
`2(p+1)⁴` flops per pair, dominate: at matched global accuracy the dense analytic
path is 2–3× slower than the spectral cube path, as `docs/notes/analytic_report.md`
predicted. Operators: `316·(p+1)⁴·8` bytes, 69 MB at `p = 12`, 201 MB at 16,
469 MB at 20, built in 0.05–0.2 s. Default `p = 16`.

## Next

Low-rank factors of the M2L were measured and deferred: at `p = 16` the ranks of
the real operators (singular values above `tol·σ₁`) are, at `tol = 1e-6`, 133 for
`(2,0,0)`, 93 for `(2,1,1)`, 46 for `(2,2,2)`, 25 for `(3,3,3)`, mean 51 of 289
(at `1e-8`: 176/135/79/36, mean 82; `p = 20`: mean 54 and 87 of 441), so the
dominating stage would gain 2.8× at `1e-6` and 1.8× at `1e-8`, capped at that
accuracy, and the 316 SVDs take 24 s to build; worth doing once the 34 symmetry
classes of the real harmonic basis (every hyperoctahedral operation is a signed
permutation of it, mixing `Re` and `Im` of the same `(n, m)`) cut the build to 34
SVDs and also open the device plan to this method. Before that: the Stokes kernel
as four harmonic potentials (`u = ½[Φ − (x − c)·∇Φ + ∇Ψ]` with `Ψ` relative to
each box centre, so that the coupling enters M2M, M2L and L2L as `±¼`, `o`, `∓½`
times the scalar operators and no digits are lost at evaluation), then Yukawa.
