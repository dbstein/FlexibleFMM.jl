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
| build, per offset before the classes (316 SVDs) | 1.2 s | 10 s | 10 s | 10 s | 10 s | 10 s |

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

## Symmetry classes

Of the 48 signed permutations of the axes only the 16 that keep the `z` axis (the
group `D4h`) act on the real solid harmonics by signed permutations, and those act
within one `(n, m)`: writing `R = Rot_z(kπ/2) ∘ Refl_y^εy ∘ Refl_z^εz`,
`Υ_n^m(Rx) = i^{mk} (−1)^{(n+m)εz} [conj]^{εy} Υ_n^m(x)`, so a multipole transforms by
`M_n^m → conj(c)·[conj]^{εy} M_n^m` and a local by `L_n^m → c·[conj]^{εy} L_n^m`,
`c = i^{mk}(−1)^{(n+m)εz}` (the local of `Φ(R⁻¹x)`). The other 32 elements mix all
`m` of a degree (Wigner rotations) and are not used. Under `D4h` the 316 far
offsets fall into 34 classes (representative `(max(|o₁|,|o₂|), min(|o₁|,|o₂|), |o₃|)`,
orbits of 16, 8, 4, 2) and `M2L_o = P_t·M2L_rep·P_sᵀ` with *different* permutations
on the two sides (for a rotation `P_t = P_sᵀ`), which the cube method's classes
never needed; `SymmetryClasses` therefore carries both sides, the cube constructor
filling them identically, and `check_harmonic_classes` verifies the relation on
every offset against the dense matrices (to `1e-13` at `p = 16`). The level
operators hold the 34 representatives (dense or low rank) and one `PermutedM2L`
per offset referring to them; the harmonic methods' `m2l_level!` gathers the pairs
of a class with the source permutations (one column per pair and scalar expansion,
four for Stokes), applies the representative in one GEMM per chunk, and scatters
with the target permutations plus the Stokes coupling with each pair's own offset.
The gather is threaded over columns and the scatter over rows in chunks aligned
with the degree blocks, inside which every permutation acts (a chunk boundary
splitting an `(n, m)` pair let two threads write one row; the test against the
generic per-offset driver caught it). Build: 34 SVDs instead of 316; memory: 34
operators instead of 316.

Build time and memory of the 34 representatives (the per-offset build was 0.1–0.5 s
dense and 10 s with the SVDs, 201–469 MB dense):

| | p = 16 dense | p = 16, tol 1e-7 | p = 20 dense | p = 20, tol 1e-7 | p = 24 dense | p = 24, tol 1e-8 |
|---|---|---|---|---|---|---|
| build | 0.03 s | 0.42 s | 0.05 s | 1.1 s | 0.09 s | 2.3 s |
| M2L memory | 22 MB | 11 MB | 50 MB | 18 MB | 101 MB | 33 MB |

`10⁵` uniform points, `ppl = 200` (matvec, with the downward stage in parentheses;
errors as in the tables above, unchanged by the classes):

| | dense | tol = 1e-6 | tol = 1e-7 | tol = 1e-8 | cube p = 8 |
|---|---|---|---|---|---|
| Laplace p = 16 | 0.192 s (0.157) | 0.157 (0.123) | 0.168 (0.133) | | 0.152 (0.093) |
| Laplace p = 20 | 0.334 (0.294) | | 0.241 (0.202) | 0.260 (0.221) | |
| Stokes p = 16 | 0.735 (0.594) | 0.616 (0.474) | 0.654 (0.512) | | 1.035 (0.803) |
| Stokes p = 20 | | | 1.03 (0.83) | | |

The class-wise gather and scatter (threaded) and the bigger GEMMs take the
downward stage from 0.217 to 0.157 s dense and from 0.147 to 0.123 s at
`tol = 1e-6` for Laplace `p = 16`, and from 0.98 to 0.59 s and 0.74 to 0.47 s for
Stokes. The harmonic Laplace method at `p = 16`, `tol = 1e-6` now costs the same as
the cube method at `p = 8` (with 5× its error); the Stokes method is 1.7× faster
than the cube method (3× its error).

The fibre case is where the analytic method pays off. On a jittered helix of
`10⁵` points (`helix_points`, the slender-body geometry: 7–9 levels, few points per
box), errors against direct sums on 1000 points:

| helix, `ppl` | harmonics p = 16, tol 1e-6 | harmonics p = 20, tol 1e-7 | cube p = 8 |
|---|---|---|---|
| Laplace, 100 | 0.54 s, 7.7e-8 | 1.03 s, 8.8e-9 | 3.10 s, 2.1e-8 |
| Laplace, 200 | 0.29 s, 2.5e-8 | 0.47 s, 7.3e-9 | 1.42 s, 1.2e-8 |
| Laplace, 400 | 0.13 s, 2.8e-8 | 0.20 s, 1.7e-9 | 0.50 s, 6.5e-9 |
| Stokes, stokeslets + doublets, 200 | 1.91 s, 8.8e-10 | 4.26 s, 1.1e-10 | 10.5 s, 9.6e-10 (p = 10: 28.4 s, 3.0e-11) |

At equal accuracy the harmonic methods are 4–5× faster than the cube method on
the fibre, Laplace and Stokes alike: a deep tree with small boxes multiplies the
cube method's per-box costs (1352 kernel evaluations per point upward, the FFTs
and `(2p)³` spectra per box) while the harmonic per-box costs are `O(p²)` per
point and one small column per box. On uniform data (above) the two are level.

## On the device

`DevicePlan` takes the harmonic methods: the S2M kernel (one work-item per
childless node) and the evaluation kernel run the same folds as the CPU
(`_accumulate!`, `_harmonic_potential`, `_stokes_output`, …) on the coefficient
tables adapted to device arrays (`DeviceHarmonic`); the 34 class representatives,
dense or low rank, go through the class gather/GEMM/scatter of the
kernel-independent plan, generalised to apply the target-side permutations in the
scatter and, for the Stokes method, to lay the four expansions of a pair out as
four columns with the coupling `Ψ̂ += o·Φ` added in the scatter; there are no
pseudo-inverse stages. The CPU-backend tests compare the device multipoles with
the CPU pass to `1e-13` and the matvec and every output to `1e-11`.

Measured on an M3 Max in Float32 (`bench/bench.jl … device=metal`, `10⁵` points,
`ppl = 200`; the device matvec, and the evaluation at `10⁶` uniform targets):

| | harmonics p = 16, tol 1e-6 | harmonics p = 16 dense | cube p = 8 (dense classes) |
|---|---|---|---|
| Laplace, uniform: matvec / targets | 45 / 58 ms | 48 / 58 | 36 / 73 |
| Stokes, uniform | 204 / 124 | 218 / 125 | 122 / 101 |
| Laplace, helix | 72 / 7 | 77 / 7 | 75 / 21 |
| Stokes, helix | 369 / 145 | 390 / 144 | 296 / 56 |

Deviations from the CPU results are at the Float32 floor (`1e-6`–`1e-8`). On the
device the two methods are level for Laplace and the harmonic Stokes matvec is
1.3–1.7× slower, the opposite of the CPU picture on the helix (where the cube
method pays its per-box surface evaluations on 12 threads): on the GPU those
evaluations are the cheap part, and the harmonic S2M kernel runs one work-item
per childless node with a serial loop over its points, which leaves most of the
device idle during the upward stage. Target evaluation, one work-item per point
with the `O(p²)` fold, is 1.3–3× faster than the cube's surface sums.

## Next

The harmonic S2M kernel with one work-item per (node, order `m`) (each order's
chain is independent given its sectoral start; dipoles need the three
neighbouring chains), which is what the Stokes matvec on the device is waiting
for; the M2M and L2L of the Stokes method as block operators (7 of the 16 `N × N`
blocks are nonzero, so the dense `4N × 32N` form does 2.3× the flops, visible on
deep fibre trees); Yukawa 3D on the same basis.
