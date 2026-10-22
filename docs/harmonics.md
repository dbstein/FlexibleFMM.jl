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

## Stokes: `StokesHarmonics`

The 3D Stokeslet is carried on four harmonic expansions (`src/operators/harmonics_stokes.jl`,
`docs/stokes.md`): `u = ½[Φ − (x − c)·∇Φ + ∇Ψ_c]` with `Φ_k` the Laplace potentials of
the densities `f_k` and `Ψ_c` that of `(y − c)·f`, from `d·f = (x − c)·f − (y − c)·f`;
stresslets (the symmetric form of `∂_i∂_j∂_k r`) and doublets (`∂(−g·∇(1/r))`) are the
same form with dipole densities, the pressure is `−∇·Φ`, the velocity gradient needs
the Hessians (second derivatives as shifted coefficient sets, checked against
automatic differentiation). `Ψ` is kept relative to the centre of the box that holds
the expansion, so no digits are lost at evaluation; changing the reference point adds
a multiple of `Φ`, which enters the translations as `±¼` (M2M), `o` (M2L) and `∓½`
(L2L) times the scalar operators, applied blockwise (`HarmonicStokesM2L`: four GEMMs
with the scalar matrix plus the coupling; M2M and L2L dense `4N × 32N`). Rotlets, a
curl of three further potentials, are not supported. Every derivative at evaluation
costs a factor of about `p` in the translation error, which the measurements show:

| p | random, stokeslet | random, stresslet | random, doublet | corner (2,0,0), stokeslet | stresslet | doublet |
|---|---|---|---|---|---|---|
| 8 | 2.7e-3 | 2.4e-2 | 3.9e-2 | 1.4e-2 | 0.64 | 0.35 |
| 12 | 1.1e-4 | 6.8e-3 | 1.7e-3 | 4.9e-3 | 0.25 | 0.13 |
| 16 | 1.4e-5 | 1.5e-4 | 3.5e-4 | 1.6e-3 | 0.11 | 0.058 |
| 20 | 3.7e-6 | 7.4e-6 | 1.8e-4 | 5.0e-4 | 0.045 | 0.022 |
| 24 | 7.3e-8 | 3.2e-6 | 1.1e-6 | 1.5e-4 | 0.016 | 0.008 |

`10⁵` uniform points, `ppl = 200`, errors against direct sums on 1000 of them:

| density | harmonics p = 12 | p = 16 | p = 20 | cube p = 8 | cube p = 10 |
|---|---|---|---|---|---|
| stokeslets | 0.54 s, 3.0e-5 | 1.06 s, 1.1e-6 | 2.12 s, 1.3e-7 | 1.01 s, 3.0e-7 | 2.18 s, 1.0e-8 |
| stresslets | 0.75 s, 4.1e-6 | 1.23 s, 6.0e-7 | 2.46 s, 6.7e-8 | 1.25 s, 7.2e-8 | 2.46 s, 3.8e-9 |
| doublets | 0.59 s, 6.5e-8 | 1.0 s, 1.6e-8 | 2.1 s, 2.5e-9 | 1.04 s, 1.2e-8 | 2.08 s, 8.7e-10 |

So on the CPU the harmonic Stokes method is level with the cube method at matched
cost for stokeslets and doublets and a little behind for stresslets, with a far
worse worst case; its upward pass is 4–6× cheaper and its operators cost nothing
to build (no SVDs of `1352 × 889` matrices). The hoped-for gain for the singular
sources (`docs/notes/analytic_report.md`) does not materialise in this
representation, because the velocity needs gradients of the expansions.

## Low-rank translations

The far interaction is smooth, so each dense M2L has low ε-rank, and `tol > 0`
replaces it by its truncated SVD `U·Vt` (`LowRankM2L`, singular values below
`tol·σ₁` dropped), applied as two thin GEMMs, `4nr` flops per pair instead of
`2n²`. The intermediate product needs a workspace of `r` rows per pair, which the
default `m2l_level!` takes from the state's scratch (`workspace_rows(op)`), so the
operators stay immutable and shared between levels, states and threads; the
Stokes block operator applies the scalar factors to its four blocks. At `p = 16`
the ranks (`tol = 1e-6`) are 133 for `(2,0,0)`, 93 for `(2,1,1)`, 46 for `(2,2,2)`,
25 for `(3,3,3)`, mean 51 of 289 (at `1e-8`: 176/135/79/36, mean 82). The
translation error floors at a multiple of `tol` (the dropped directions act on the
whole multipole vector, the field is normalised by the attenuated far field):

| p = 20, tol | 0 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 |
|---|---|---|---|---|---|---|
| worst offset error (one sample) | 9.1e-8 | 3.1e-7 | 9.1e-8 | 9.1e-8 | 9.1e-8 | 9.1e-8 |
| mean rank of 441 | 441 | 54 | 70 | 87 | 106 | 127 |
| M2L memory | 469 MB | 116 | 149 | 185 | 225 | 270 |
| build (316 SVDs, `OPENBLAS_THREAD_TIMEOUT=4`) | 1.2 s | 10 s | 10 s | 10 s | 10 s | 10 s |

`10⁵` uniform points, `ppl = 200`, 12 threads (the error is against direct sums on
1000 points and is set by `p`, not by `tol`, in this range):

| | dense | tol = 1e-6 | 1e-7 | 1e-8 | cube p = 8 |
|---|---|---|---|---|---|
| Laplace p = 16 (down stage) | 0.251 s (0.217) | 0.181 (0.147) | 0.212 (0.178) | 0.229 (0.194) | 0.143 (0.089) |
| Laplace p = 20 | 0.483 (0.443) | | 0.301 (0.261) | 0.309 (0.270) | |
| Stokes p = 16 | 1.12 (0.98) | 0.88 (0.74) | 0.98 (0.83) | 1.05 (0.91) | 1.14 (0.88) |
| Stokes p = 20 | 2.22 (2.01) | | 1.49 (1.28) | 1.60 (1.39) | |

The downward stage gains 1.4–1.7×, well short of the 3–4× flop ratio: the two
thin GEMMs run below the square GEMM's efficiency, and the gather and scatter of
the pairs, which the rank does not shrink, are a large part of the stage. The
harmonic Laplace method at `p = 16` stays 1.3× behind the cube method at `p = 8`
(with 5× its error); the Stokes method at `p = 16`, `tol = 1e-6` is 1.3× ahead of
the cube method with 3× its error. What would reduce the stage further is the
gather/scatter traffic (one transform of the level's multipoles per symmetry
class, as the device plan does), not the rank.

The option applies to any method whose M2L is a dense GEMM (`BlackBox` and the
circle method could take it the same way; their 2D operators are small enough
not to need it); the cube method's CPU translation is spectral and its device
plan already uses low-rank class factors (`docs/kifmm.md`).

## Next

The 34 symmetry classes of the real harmonic basis (every hyperoctahedral
operation is a signed permutation of it, mixing `Re` and `Im` of the same
`(n, m)`): 34 SVDs instead of 316 at build time, and the class gather/GEMM/scatter
of the device plan for both harmonic methods. Then Yukawa 3D on the same basis.
