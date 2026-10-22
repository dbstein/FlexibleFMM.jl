# Analytic (Greengard–Rokhlin) expansions as a method of the FlexibleFMM framework

Design study, 2026-10-21. Sources: `fmm2d` and `fmm3d` (Flatiron, shallow clones under
`scratchpad/fmm2d_src`, `scratchpad/fmm3d_src`; file:line references below are into
those clones), and the FlexibleFMM method interface (`src/operators/interface.jl`),
driver (`src/fmm.jl`), cache (`src/operators/cache.jl`), the black-box model
(`src/operators/bbfmm.jl`), the symmetry classes (`src/operators/symmetry.jl`) and
the measured accuracies in `docs/kifmm.md`.

## 0. Verdict in one paragraph

Yes: an analytic method fits the five-operator interface without touching the tree,
the driver or the cache. The mapping is the one the black-box method already uses
(expansion = coefficient vector, partial transforms = identity, dense per-offset
M2L, dense M2M/L2L stacked by child, scaled coordinates `x/w`), with one difference
per family stated in the sections below. For the homogeneous 3D kernels the
analytic operators are width-independent in scaled coordinates exactly as the
black-box ones are (`rescale` with `m2l_scale = λ^{-1}`); for Laplace 2D a single
`log w` entry breaks that (built per width, trivially cheap); for Helmholtz/Yukawa
the operators depend on `k·w` and are built per level, which the cache already does
for non-homogeneous kernels. The gains are not raw CPU speed of the M2L (a dense
analytic M2L at the order needed for `1e-6` costs 1.5× the flops of our dense
`296²` Laplace operator and about 14× the flops of the spectral path; fmm3d itself
does not use dense or point-and-shoot M2L for Laplace, it uses plane waves). The
gains are: no pseudo-inverse and no check surfaces (exact operators, no `1/σ_min`
amplification, Float32 usable to `~1e-6` instead of the `1e-5`/`5e-3` floors of
the kernel-independent method), 5–10× less memory per box, 5–8× cheaper `s2m!`/`l2t`
per point, dipoles/stresslets/doublets converging at the charge rate instead of
the `1e-4`/`4e-5`/`1e-3` worst cases at `p = 8`, no Helmholtz resonance or Laplace
scale degeneracy, and a clean Stokes-via-Laplace that keeps our Stokes near field.
First target: Laplace 3D in a real spherical-harmonic basis with dense per-offset
operators (316 at `p ≤ 16`, or 34 signed-permutation classes beyond), ~500 lines
plus tests, everything else reused unchanged.

## 1. What a method must provide (recap of the interface, with the facts that matter)

From `src/operators/interface.jl`, `src/fmm.jl`, `src/operators/cache.jl`:

- `s2m!(col, m, ops, xs, τs, c)` fills the *partial* multipole of a childless node;
  `partial_to_multipole!`/`partial_to_local!` map a whole level block (for the
  black-box method: `copyto!`). `multipole_length`, `local_length`,
  `partial_*_length` all default to `expansion_length(m)`.
- `ops.M2M` is `Nm × 2^D·Nm` applied to the children's multipoles stacked in child
  slot order (`child_offsets`, coordinate 1 fastest); `ops.L2L` is `2^D·pll × pll`
  applied to the parent's *partial* local; both are applied as one GEMM per level
  (`threaded_mul!` in the current working tree).
- `ops.M2L[f]` for `f` in `far_offsets(Val(D))` (40 in 2D, 316 in 3D) is anything
  with `apply!(out, op, x, α)` (`M2LOperator{T}` is the dense default; the
  comment in the interface says a compressed form only needs `apply!`). The default
  `m2l_level!` gathers the source columns of one offset, one GEMM, scatter-adds.
  A method may override `m2l_level!` (the spectral cube method does).
- `l2t(m, ops, L, xrel, out)` with `out ∈ {Potential, Gradient, Pressure}`.
- `operators_type(m)` fixes the concrete `LevelOperators{T,R,X,L}`;
  `operator_eltype(m)`; `near_kernel(m)` (the near field never sees the method);
  `supports_density(m, T)`, `supports_output(m, out)`.
- `get_operators!(cache, e)` builds `level_operators(m, w)` per width, except when
  `homogeneity(near_kernel(m)) !== nothing`, in which case only the reference width
  is built and `rescale(ref, m, w)` must return operators sharing the matrices with
  per-width scalars `m2m_scale`, `l2l_scale`, `m2l_scale`.
- `expansion_eltype` promotes `operator_eltype(m)` with the kernel output and the
  density scalar, so a complex `operator_eltype` forces complex expansions for real
  data (this is the only place where "real formulation vs complex coefficients"
  shows up).
- Device passes (`src/device_passes.jl`) currently assert
  `m isa AbstractKernelIndependent` and build class operators through
  `class_operators(K, small, embed, w, p)` (cube lattices); they are *not* reused
  as-is by a new method. This is the one piece outside the method that a GPU path
  would need to generalise (identity pseudo-inverse stages, a method hook for class
  operators and permutations). It is not needed for the CPU first target.

Nothing above constrains what a multipole *is*. The geometry is also fixed and
identical to fmm2d/fmm3d's "list 2": same-width boxes at Chebyshev distance 2 or 3
in units of the width, nearest far center at `2w`. So the libraries' order
selection rules apply verbatim (they are written for exactly this configuration:
`l2dterms`/`l3dterms` use a unit box, worst source at the circum-radius, worst
target at distance 1.5).

## 2. Orders needed, common to all families

The libraries' rule (used globally per level in `cfmm2d.f:316`, `lfmm3d.f:394`):
first `n` with `ρ^n / r^{n+1} < eps`, `ρ` the box circum-radius (`√2/2`, `√3/2`),
`r = 1.5` (nearest face of the nearest far box).

| eps | Laplace 2D `l2dterms` (`l2dterms.f:38`) | Laplace 3D `l3dterms` (`l3dterms.f:21`) |
|---|---|---|
| 1e-3 | 9 | 12 |
| 1e-6 | 18 | 25 |
| 1e-9 | 28 | 37 |
| 1e-12 | 37 | 50 |

Our own worst case (the one measured in `docs/kifmm.md`: corner source, facing
corner target at offset 2) has ratio `ρ/r` with `r = |(1.5, 0.5[, 0.5])|`:
`0.447` in 2D, `0.522` in 3D, so `2·q^{p+1} < eps` gives `p ≈ 9/18/26/35` (2D) and
`11/22/32/43` (3D). The classical M2L bound with `c = R/ρ = 2.31`,
`(1/(c−1))^{p+1} = 0.764^{p+1}`, is far more pessimistic (`p ≈ 50` for `1e-6`); it
assumes sources and targets filling the whole spheres, which cubes do not. The
library rule and the corner estimate bracket what we will need; FMM3D measured
`1e-7` global at `eps = 1e-6`, `nterms = 25` in our comparison (`docs/kifmm.md`),
consistent with `p ≈ 20–22` giving `1e-6` worst case and `1e-7` global on uniform
data. **This must be measured** with `check_m2l_all_offsets` once the operators
exist; I quote both `p = 18` (CGR'99 practice) and `p = 25` (fmm3d) below.

Per-offset orders: `l3dterms_list2` (`l3dterms.f:115`) tabulates the order per
offset class using the distance to the nearest point of the far box. At
`eps = 1e-6`: offset `(2,0,0)` 25, `(2,1,1)` 21, `(2,2,0)` 15, `(2,2,2)` 12,
`(3,0,0)` 13, `(3,3,3)` 8. So a per-class truncation of the dense operators
(rectangular blocks mapping the first `(p_c+1)²` multipole entries to the first
`(p_c+1)²` local entries) cuts the dense M2L flops by roughly 3× summed over the
316 offsets (the nearest 6 offsets dominate). fmm3d does not use this table for
its M2L (it uses plane waves); it is a cheap optimisation for a dense per-class
scheme, where the class chunks are already grouped.

Expansion sizes at those orders, 3D `(p+1)²` real coefficients (see §4 for why
real):

| p | n = (p+1)² | dense M2L entries n² | f64 MB per operator | 2n² flops/pair | point-and-shoot ≈13p³ flops | rotation block entries Σ(2n+1)² |
|---|---|---|---|---|---|---|
| 9 | 100 | 10 000 | 0.08 | 20 000 | 9 500 | 1 330 |
| 12 | 169 | 28 561 | 0.23 | 57 000 | 22 500 | 2 925 |
| 18 | 361 | 130 321 | 1.0 | 261 000 | 76 000 | 9 139 |
| 20 | 441 | 194 481 | 1.6 | 389 000 | 104 000 | 12 341 |
| 25 | 676 | 456 976 | 3.7 | 914 000 | 203 000 | 23 426 |
| 37 | 1444 | 2 085 136 | 16.7 | 4 170 000 | 658 000 | 73 150 |

Our kernel-independent reference at `p = 8` (`docs/kifmm.md`): Laplace `296`
entries (`1e-6` worst, `1e-7` global), Stokes `889`; dense `2·296² = 175 000` and
`2·889² = 1 580 000` flops per pair; spectral half-spectrum `(p+1)(2p)² = 2304`
complex MACs ≈ `18 000` real flops per pair for Laplace, ×6 distinct tables for
Stokes ≈ `110–160 000`; memory `72 KB` per box (Laplace) for the transformed
spectra, `290 MB` of tables for Stokes.

## 3. Per kernel family

### 3.1 Laplace 2D (fmm2d `src/laplace`)

**Expansions.** With `z = (x − c)` as a complex number and `rscale = w`
(`cfmm2d.f:314`), the multipole is `u(z) = a_0 log|z| + Σ_{n=1}^{p} a_n (w/z)^n`
(`l2dmpevalp`, `laprouts2d.f:289`), `a_0 = Σ q_j`, `a_n = −Σ_j q_j (z_j/w)^n / n`
(`l2dformmpc`, `:84`; dipoles `l2dformmpd` `:156`, both `l2dformmpcd` `:219`); the
local is `Σ_{n=0}^{p} b_n (z/w)^n` (`l2dtaevalp` `:696`, gradient `l2dtaevalg`
`:759`, Hessian `:828`). The potential of real charges is the real part. Length:
`p+1` complex coefficients (`a_0` real), i.e. `2p+1` reals of information; fmm2d
keeps them complex because `cfmm2d` is the complex-charge engine (`lfmm2d`/`rfmm2d`
are wrappers).

**Operators.** `l2dmpmp` (`:905`): `b_i = (w_2/z_0)^i [ −a_0'/i + Σ_{j≤i} a_j' C(i−1,j−1) ]`
with `a_j' = a_j (z_0/w_1)^{−j}`; `l2dlocloc` (`:1020`): `b_i = Σ_{j≥i} a_j C(j,i) (z_0/w_1)^{j−i}·(…)`;
`l2dmploc` (`:1130`): `b_i = (w_2/z_0)^i Σ_{j≥1} a_j (−w_1/z_0)^j C(i+j−1, j−1) − a_0/(i)·(…)`,
`b_0 = a_0 log|z_0| + Σ_j a_j(…)`. All three are closed-form `(p+1)×(p+1)` complex
matrices with binomial entries (`carray`), applied per pair at `O(p²)` in
`cfmm2dmain` (`cfmm2d.f:1020–1025`, one `l2dmploc` per list-2 pair) — i.e. fmm2d's
2D M2L *is* a dense `(p+1)²` matrix-vector product per pair, same structure as ours.

**Mapping onto our interface.** Expansion vector of `p+1` complex numbers;
`operator_eltype = Complex{T}`; `M2M` `(p+1) × 4(p+1)`, `L2L` `4(p+1) × (p+1)`,
`M2L[f]` `(p+1)×(p+1)`, all built from the formulas above at the width `w` in
scaled coordinates (`z/w`); `s2m!` accumulates powers of `(x−c)/w` (charges) and
their derivatives (dipoles: `d·∇_s` of `−log|z−z_j|` gives `a_n += −(d_1 − i d_2)
(z_j/w)^{n−1}/w`); `l2t` is Horner on `(z/w)^n`, gradient from the derivative
series. Transforms identity.
Mismatch: every entry of `M2M`, `L2L`, `M2L` is width-independent *except*
`M2L[f][0,0] = log|z_0| = log|o| + log w` (the monopole-to-constant term). Options:
build per width (the cache does this when `homogeneity(Laplace2D) === nothing`,
which is the case; cost `40 × (p+1)²` complex entries per width, nothing), or
declare homogeneity and add `a_0·log w` as a per-level correction — not worth it.
Real formulation: none with fewer numbers (the `a_n` of real charges are genuinely
complex); the complex-linear maps can be written as real `2×2` blocks `[a −b; b a]`
on `(Re, Im)` pairs with the same flops, which is what a Metal path (no complex
GEMM) would want. For complex charges (`lfmm2d`) the same operators act on
complex `a_n`; our `expansion_eltype` promotion handles it.

**Sizes at our targets.** `p = 18` for `1e-6`: `19` complex = `38` reals per box
(circles: `24–30` reals for `3e-7`–`2e-8`); dense M2L `19² = 361` complex entries
≈ `2 900` real flops per pair against `2·30² = 1 800` for 30-point circles. 40
offsets × `361` × 16 B = `231 KB` per width. `s2m!`/`l2t`: `p` complex multiplies
per point against `N = 30` kernel evaluations (log, division). Comparable
throughout; the 2D circle method is already 1.5–2× faster than FMM2D
(`docs/kifmm.md`), so the Laurent method buys exactness and Float32 but not time.

**Effort.** ~200 lines (`level_operators`, `s2m!` for `Number`/`Dipole`/`Tuple`,
`l2t` for `Potential`/`Gradient`, `multipole_field`), tests through
`test/operator_checks.jl`. Low risk: the formulas are in `laprouts2d.f` and the
checks compare against direct sums.

### 3.2 Laplace 3D (fmm3d `src/Laplace`, `src/Common`)

**Expansions.** `Φ(x) = Σ_{n=0}^{p} Σ_{m=−n}^{n} M_n^m Y_n^m(θ,φ)/r^{n+1}` with
fmm3d's `Y_n^m = √(2n+1)√((n−m)!/(n+m)!) P_n^m(cosθ) e^{imφ}` (no Condon–Shortley,
`∫|Y|² = 4π`; header of `laprouts3d.f:1–60`), scaled as `M_n^m/w^n`
(`rscale = boxsize`, `lfmm3d.f:469`): `l3dmpevalp` (`laprouts3d.f:107`) evaluates
`Σ M_n^m (w^n/r^{n+1}) Y_n^m/√(2n+1)` using only `m ≥ 0` and `2·Re(M_n^m e^{imφ})`,
because for real charges `M_n^{−m} = conj(M_n^m)` (`l3dformmpc` `:437` writes both).
Local: `Σ L_n^m (r/w)^n Y_n^m/√(2n+1)` (`l3dtaevalp` `:999`, gradient `:1121`,
Hessian `:2203` with the precomputed `scarray` of `l3dtaevalhessdini` `:2132`).
Formation `l3dformmpc`/`l3dformmpd`/`l3dformmpcd` (`:437/:581/:787`): per source
`cart2polar`, `e^{imφ}` by recurrence, `P_n^m` by `ylgndrfw` (`Common/yrecursion.f`,
with the precomputed scaling table `wlege` to keep the recurrence in range), `O(p²)`
per source. fmm3d stores `(p+1)(2p+1)` complex numbers per expansion, twice the
information for real data; the minimal real storage is `(p+1)²` reals
(`Re M_n^0`; `Re, Im M_n^m`, `m = 1..n`).

**Operators (closed form, "point and shoot").** `l3dtrans.f`: `l3dmpmp` (`:13`),
`l3dmploc` (`:239`), `l3dlocloc` (`:463`) each rotate the expansion so that the
shift is along `z` (`rotviarecur`, `Common/rotviarecur.f:15`, recurrence, used for
`p < 30`; `rotviaproj`, `Common/rotproj.f:12`, FFT/projection, stable, for
`p ≥ 30`), apply the `z`-shift (`l3dmpmpzshift` `:149`, `l3dmploczshift` `:362`,
`l3dlocloczshift` `:589`: `L_j^k = Σ_{l≥|k|} M_l^k · (w/d)^{l+j}/d · dc(l+j, l−k)
dc(l+j, l+k) (−1)^{j+k}`, `dc` square-root binomials, coupling only equal `m`), and
rotate back. Cost per translation ≈ `2·Σ_n(2n+1)²` (rotations) + `Σ_m (p+1−|m|)²`
(shift) ≈ `(8/3 + 2/3) p³` complex MACs ≈ `13 p³` real flops (`76 000` at `p = 18`,
`203 000` at `p = 25`). **lfmm3d does not use `l3dmploc` for list 2**: `lfmm3dmain`
calls only `l3dmpmp` (`lfmm3d.f:1082, 1281`) and `l3dlocloc` (`:1886, 1927`); every
list-2 translation goes through the plane-wave representation (next paragraph).
`l3dmploc` exists for other callers.

**Operators (plane waves, what lfmm3d actually runs).** `lpwrouts.f`:
`mpoletoexp` (`:185`) converts a multipole into outgoing exponential moments
`M_λ(m) = i^m Σ_{n≥m} mpole(n,m) λ^n/√((n+m)!(n−m)!)` for the `+z`/`−z`
directions (`rlscini` `:5` precomputes `λ^n/√(...)`); `ftophys` (`:576`) goes from
Fourier modes to physical angles; the six directions come from the `z` pair plus the
`π/2` rotations `rotztoy`/`rotztox` (`Common/rotgen.f:5/178`) with precomputed real
rotation matrices `rdminus`/`rdplus` from `getpwrotmat` (`Common/fmmcommon.f:427`,
`fstrtn`); `processudexp`/`processnsexp`/`processewexp` (`:668/:1096/:1607`) do,
per target box and direction, the Hadamard products with the shift tables `xs, ys,
zs` of `mkexps` (`:36`) over the sources of that direction's list (with the
children-merging trick), and `exptolocal` (`:339`) converts the accumulated
incoming exponentials to a local expansion. Quadrature sizes (`lfmm3d.f:760–765`,
`isep = 1`, tables `numthetafour`/`numthetahalf` in `lwtsexp_sep1.f:2000/:2916`):

| eps | nlams | nexptotp (physical nodes) | nexptot (Fourier modes) |
|---|---|---|---|
| 1e-3 | 12 | 217 | 62 |
| 1e-6 | 20 | 705 | 186 |
| 1e-9 | 29 | 1627 | 405 |
| 1e-12 | 37 | 2770 | 670 |

Per pair the translation is one complex multiply per physical node, `≈ 8·705 ≈
5 600` real flops at `1e-6` (each pair belongs to exactly one of the six
directions), plus per box `6 × (mpoletoexp O(p²·nlams) + ftophys)` and one
`exptolocal`. That is 3× fewer flops per pair than our spectral path at `p = 8` and
~50× fewer than the dense `p = 18` matrix; it is the reason fmm3d's Laplace is as
fast as it is. It maps onto our `m2l_level!` hook exactly like the spectral
translation (per-box forward transform, per-pair Hadamard product, per-box inverse
transform; six directional streams instead of one `(2p)³` spectrum), but it is a
second step, not the first.

**Mapping onto our interface (dense first target).** Expansion = `(p+1)²` reals in
the real spherical-harmonic basis (`Re/Im` split with the conjugate symmetry;
equivalently real solid harmonics). All of M2M, L2L, M2L are real matrices for a
real kernel (they commute with conjugation), so `operator_eltype = T` and complex
densities simply promote the expansions (real matrix × complex columns). Scaled
coordinates `x̂ = (x − c)/w` (`rscale = w`) make `M2M` and `L2L` width-independent
(in `l3dmpmpzshift` the entries are `(−d/w_1)^l (w_1/w_2)^j` with `d ∝ w`; in
`l3dlocloczshift` `(d/w_1)^l (w_2/w_1)^j`) and `M2L ∝ 1/d = 1/(|o| w)` (the leading
`fr(0) = 1/zshift` in `l3dmploczshift`), i.e. exactly `m2l_scale = λ^{−1}` for
`homogeneity(Laplace3D) = −1`: the cache builds one width and `rescale` shares the
matrices, as for the black-box and cube methods. `s2m!`: per source, the regular
solid harmonics `ρ̂^n Y_n^m*` up to `p` (Legendre recurrence, `O(p²)`, ~`3–5` flops
per term) accumulated with the density: charges directly, dipoles through the
Cartesian derivative recurrences of the solid harmonics (what `l3dformmpd` does in
polar form; in Cartesian form `∂_z R_n^m` and `(∂_x ± i∂_y) R_n^m` are
`R_{n−1}^{m}` and `R_{n−1}^{m±1}` with constant coefficients, no trig functions at
all — preferable for the GPU per-point kernels). `l2t`: the same recurrence for
`(r/w)^n Y_n^m` at the target; `Gradient` from the derivative recurrences
(`l3dtaevalg`); a Hessian (for the Stokes velocity gradient, §3.6) from
`l3dtaevalh`. Transforms identity. `M2L[f]` dense `(p+1)² × (p+1)²` built once per
offset from the closed form (the Greengard–Rokhlin M2L coefficients
`A_n^m A_j^k / A_{n+j}^{m−k}`-type with `Y_{n+j}^{m−k}` of the offset direction
and `1/|o|^{n+j+1}`; or, less error-prone, by composing fmm3d's three steps —
rotation matrix, `z`-shift matrix, inverse rotation — as matrices at build time),
then converted to the real basis with the sparse unitary change of basis, and
verified against direct sums by the existing checks. `M2M`/`L2L` likewise (8
blocks). No mismatch with the interface.

Memory of the dense operators: `316 × (p+1)⁴ × 8 B` = `72 MB` at `p = 12`,
`211 MB` at `p = 16`, `330 MB` at `p = 18`, `1.16 GB` at `p = 25`. With symmetry
classes (§4.3): `34 × …` = `7.8 / 23 / 35 / 124 MB`, or `16 × …` with Wigner
transforms. So: 316 dense offsets are fine up to `p ≈ 16` (`1e-5` worst case),
classes beyond; the class machinery exists (`symmetry.jl`) and needs only a new
permutation constructor for the harmonic basis.

**Flops and time per pair, honest comparison with the cube method at `1e-6`.**
Dense analytic `p = 18`: `261 000` flops (GEMM rate); `p = 22`: `560 000`. Cube
`p = 8` dense: `175 000` (GEMM); spectral: `≈ 18 000` (Hadamard rate, load bound).
So the dense analytic M2L is 1.5–3× our dense operator and ~15–30× the spectral
flops; at a GEMM-to-Hadamard rate ratio of 5–10 the wall-clock gap to the spectral
path is 2–5× on the M2L stage, which is `0.21` of the `0.31 s` Laplace matvec at
`10⁵` points. Per-class low-rank factors (`class_factors` style, `4nk` flops): the
far interaction has the same ε-rank in any basis (≈ 25 at `1e-8`, 9 at `1e-6` for
Laplace per `docs/kifmm.md`), so `4·361·25 = 36 000` flops — at GEMM rate that
matches or beats the spectral stage. Per-class truncation (`l3dterms_list2`)
gives another ~3×. The dense analytic path will not beat the spectral path on
the CPU without one of these; the plane-wave `m2l_level!` would beat it by ~3× in
flops. Where the analytic method wins outright: `s2m!`/`l2t` cost `O(p²) ≈ 2–3
kflop` per point at `p = 20` against `nlarge = 1352` kernel evaluations (`≈ 15–20
kflop`, with `rsqrt`) per point for the cube method with `pc = 2p`; memory per
box `2·(p+1)²·8 B = 7 KB` at `p = 20` against `≈ 15 KB` of expansions plus `72 KB`
of spectra; and the operator build (no SVDs of `1352 × 296` matrices).

**Accuracy per coefficient.** This is where intuition fails: the cube method at
`p = 8` reaches `1e-6` worst case with `296` coefficients, while spherical
harmonics need `p ≈ 20–22` (`441–529` coefficients) for the same worst case,
because the sphere circumscribing a cube is a poor fit (`0.52` per order against
the cube surfaces' effective rate). In 2D the circle method is spectrally the
Laurent expansion and the counts agree. So "more compact" is not an argument in 3D;
"exact and well conditioned" is. For dipoles, stresslets and doublets the analytic
expansions converge at the charge rate (one power of `ρ/r` less) whereas the cube
method's check surfaces undersample faster-decaying fields (dipoles `1e-4`,
stresslet `4e-5`, doublet `1e-3` at `p = 8`): a 2–3 digit gain for the Stokes
singularities at equal order.

**Effort.** ~500 lines: a harmonics file (real solid harmonics and derivatives,
real-basis indexing, closed-form M2M/L2L/M2L builders) and the method file
(`level_operators`, `rescale`, `s2m!` per density type, `l2t` per output,
`multipole_field`), ~200 lines of tests. Risks: (i) normalisation and sign
conventions in the closed-form M2L (mitigated: every operator is checked against
direct sums over all 316 offsets by `check_m2l_all_offsets`, and the three-step
composition is an independent construction to cross-check); (ii) Float32 stability
of the Legendre recurrence at `p ≥ 20` (fmm3d's `wlege` scaling; in scaled
coordinates `ρ̂ ≤ 0.87` keeps the terms `O(1)`); (iii) `p` as a type parameter
(like `BlackBox{K,P}`) with per-point work buffers of size `(p+1)²` is too large for
`MVector` unrolling — the recurrence needs only two previous rows (`2(p+1)`
scalars), so `s2m!`/`l2t` stay allocation-free with small `MVector`s.

### 3.3 Helmholtz 2D and Yukawa 2D (fmm2d `src/helmholtz`, `src/modified-biharmonic`)

**Expansions.** Multipole `Σ_{n=−p}^{p} a_n H_n^{(1)}(k r) e^{inθ}`, local
`Σ b_n J_n(k r) e^{inθ}`, scaled as `H_n·rscale^n`, `J_n/rscale^n` with
`rscale = min(|k| w/2π, 1)` (`hfmm2d.f:322`; header of `helmrouts2d.f:20–35`
explains the convention: it keeps the small-argument growth of `H_n` and decay of
`J_n` out of the coefficients). `h2dformmpc` (`helmrouts2d.f:84`): `a_n += q
J_n(k r_j) e^{−inθ_j}/rscale^n`, Bessel values by `jbessel2d` per source;
dipoles `h2dformmpd` (`:144`). Evaluation `h2dmpevalp` (`:274`), `h2dtaevalp/g/h`
(`:711/:785/:869`). Length `2p+1` complex (indices `−p..p`); for real densities
with a complex kernel nothing is saved.

**Operators.** Graf's addition theorem: `h2dmpmp` (`:967`) and `h2dlocloc` (`:1096`)
with `J_{i−j}(k d) e^{i(i−j)θ_0}` (Toeplitz in the index difference), `h2dmploc`
(`:1221`) with `H_{i+j}` and `H_{i−j}` (`htemp(±(i±j))`) and the `rscale` powers:
dense `(2p+1)×(2p+1)` complex per pair, `O(p²)` per list-2 pair in `hfmm2dmain`.
Order `h2dterms(bsize, zk, eps)` (`h2dterms.f:35`): first `j` with
`(|J_j H_j| + |J_{j−1}H_{j−1}|)·|H_0| < eps·(|J_0H_0| + |J_1H_1|)` at arguments
`1.5 k w` and `(√2/2) k w`; tends to the Laplace values `9/18/28/37` (+1) as
`k w → 0` and grows like `(√2/2) k w + O(log 1/eps)` at high frequency.

**Mapping.** Identical to the Laurent method with Bessel/Hankel matrices; nothing
is width-independent (every entry depends on `k w`), so operators are built per
width, which `get_operators!` already does for `homogeneity === nothing`; per
width: `40 × (2p+1)²` complex entries, `0.5 MB` at `p = 20`. `rscale` is internal
to the method (coefficients are `a_n rscale^n`-scaled and the per-width factors sit
in the matrices). Two things the kernel-independent method cannot do that this
does: no resonance (`J_m(k r_1) = 0`, first at `k w ≈ 3.40`, `docs/kifmm.md`) —
the addition theorems are exact identities, the only conditioning issue is the
low-frequency growth handled by `rscale`; and no `k·W ≤ 12` restriction on the
root box. At high frequency the order grows with `k w` per level, as in fmm2d
(`nterms(i)` per level, `hfmm2d.f:323`), and our `expansion_length` would have to
vary per level — **this is a mismatch**: `expansion_length(m)` is one number,
`FMMState` allocates `Matrix{S}(Nexp, nnodes)` with it, and `M2M`/`L2L` are
`Nm × 2^D Nm`. The clean way within the interface is to use the maximum order over
the levels (zero-padded coefficients; wasteful at high frequency, exact otherwise)
— or, later, a per-level `expansion_length` which would touch `FMMState` (driver,
out of scope). For `k W ≲ 30` on the root box the waste is small.

**Yukawa 2D.** fmm2d has no standalone Yukawa FMM; `hfmm2d` accepts complex `zk`
(Yukawa is `k = iβ`: `H_n(iβr) ∝ K_n(βr)`, `J_n(iβr) ∝ I_n(βr)`), and the modified
biharmonic FMM carries a Yukawa part explicitly (`mbhrouts2d.f`: `ympole` with
`K_n`, `mbh2dmploc_vec` `:1456`, `mbh2dmpmp_vec` `:1260`, `mbh2dlocloc_vec` `:1633`,
evaluation `mbh2dmpevalp_vec` `:148` with `diffslogbk_fast` producing the scaled
`K_n`; `rscale = min(β w/2π, 1)`, `mbhfmm2d.f:336`, order from `l2dterms`). For
our `Yukawa2D` (`K_0(kr)/2π`): multipole `Σ a_n K_n(kr) e^{inθ}`, local
`Σ b_n I_n(kr) e^{inθ}`, `a_{−n} = conj(a_n)` for real data, so a real formulation
with `2p+1` reals exists and all operators are real (the `I/K` addition theorems
have real coefficients). Coarse levels where `e^{−k w}/w < eps` carry no far field
(fmm3d's `ilevcutoff`, `hfmm3d.f:813–818`); in our framework the method can return
zero `M2L` for such widths. Order: Laplace-like at small `k w`, decreasing with
`k w` (the far field is exponentially small).

**Effort.** Helmholtz 2D ~250 lines on top of the Laurent method (Bessel/Hankel
tables per width, `jbessel2d`/`h2dall`-style scaled recurrences — SpecialFunctions
gives `besselj`, `besselh` orders; Float32 Hankel computes in Float64 anyway, as
`kernels.jl` notes). Yukawa 2D ~150 lines (`besseli`, `besselk`). Value: removes the
resonance degeneracy, otherwise the circle method already matches FMM2D's accuracy
at equal time (`docs/kifmm.md`).

### 3.4 Helmholtz 3D and Yukawa 3D (fmm3d `src/Helmholtz`)

**Expansions.** `Σ M_n^m h_n^{(1)}(kr) Y_n^m`, local `Σ L_n^m j_n(kr) Y_n^m`, scaled
`h_n·scale^n`, `j_n/scale^n`, `scale = min(|k| w, 1)` (`hfmm3d.f:467–470`).
Formation `h3dformmpc/d/cd` (`helmrouts3d.f:417/559/772`, spherical Bessel
`besseljs3d`), evaluation `h3dmpevalp/g` (`:84/:223`), `h3dtaevalp/g` (`:989/:1120`).
Length `(p+1)(2p+1)` complex as stored, `(p+1)²` complex minimal (complex kernel: no
real formulation). Order `h3dterms(size, zk, eps)` (`h3dterms.f:16`), the
`j_n h_n` decay criterion at `1.5 k w` and `(√3/2) k w`: `12/25/37/50` at low
frequency, `≈ (√3/2) k w + O(log 1/eps)` at high frequency; per level
(`hfmm3d.f:477`) — the same per-level-order mismatch as in 2D, same resolution
(max order, zero padding).

**Operators.** `h3dmpmp` (`h3dtrans.f:10`), `h3dmploc` (`:289`), `h3dlocloc`
(`:545`): rotate (`rotviarecur`/`rotviaproj`) and shift along `z`, but the `z`-shift
is **not** a closed-form coefficient formula: `h3dmploczshift` (`:460`) evaluates
the multipole field and its normal derivative on a sphere of radius
`w√3/2` about the new center at `nquad ≈ 2.2–2.5·nterms` Legendre nodes
(`h3dmpevalsphere`, `projections.f:284`), projects onto `Y_n^m` (`h3dprojloc`
`:420`) and divides by `j_n(kR)` (`h3drescaleloc` `:62`). `O(p³)` with a larger
constant than Laplace. `hfmm3dmain` uses this point-and-shoot M2L only in the
"point and shoot regime" (`hfmm3d.f:1637–1685`, `Re(k w) > 16π` or `Im(k w) > 12π`,
`zkrupbound`/`zkiupbound` at `:807–808`); otherwise (`:1117–1125`) it builds
per-level plane-wave quadratures (`hwts3e`, `hnumfour`, `hnumphys`, tables per
`iprec` and per "domain" of `k w` in `hwts_3d_iprec*.txt`), rotates with
`getpwrotmat`'s matrices, and runs `hmpoletoexp`/`hprocessudexp`/…/`hexptolocal`
(`hpwrouts.f:235/794/413`). Levels with `e^{−Im(k) w}/w < eps` are cut off
(`ilevcutoff`, `:813–818`), which is how **Yukawa 3D** is handled: `hfmm3d` with
`zk = iβ` (`h_n(iβr) ∝ k_n(βr)`, `j_n ∝ i_n`); there is no separate Yukawa code.

**Mapping.** Expansion `(p+1)²` complex (Helmholtz) or `(p+1)²` real (Yukawa,
conjugate symmetry with real modified spherical Bessel functions);
`operator_eltype` complex / real. Operators per width (built by the cache for
`homogeneity === nothing`): `M2M` `(p+1)² × 8(p+1)²`, `M2L[f]` `(p+1)² × (p+1)²` —
`316 × 457k × 16 B = 2.3 GB` per level at `p = 25` complex: **dense per-offset is
not viable in 3D Helmholtz at `1e-6` without classes**; with 34 D4h classes `250
MB` per level (Float64), with 16 octahedral classes plus Wigner `117 MB`. The M2L
matrices themselves can be built from the closed-form Helmholtz M2L (the Gaunt
coefficient formula, `O(p⁵)` build per offset — `0.3 s` per offset at `p = 25` in
Julia, acceptable for 34 classes per level) or by composing fmm3d's three matrices.
Point-and-shoot per pair is the memory-free alternative (`≈ 13 p³` flops plus the
sphere evaluation; not a GEMM). Yukawa 3D is simpler than Helmholtz 3D (real,
exponentially small far field on coarse levels, order decreasing with `k w`) and
would come first.

**Effort.** Yukawa 3D ~300 lines on top of Laplace 3D (modified spherical Bessel
scaled recurrences, per-width build, level cutoff); Helmholtz 3D ~400 lines plus the
class machinery (§4.3) as a hard requirement. Both are the riskiest families
(memory, per-level orders, Float32 with complex expansions on Metal has no GEMM
anyway).

### 3.5 Stokes 2D via the biharmonic FMM (fmm2d `src/stokes`, `src/biharmonic`)

**What fmm2d does.** `stfmm2d` (`stokes/stfmm2d.f:27`) maps a stokeslet `σ` to two
complex charges `c_1 = (−iσ_1 + σ_2)/4`, `c_2 = conj(c_1)` (`:232`), a stresslet
`(μ, ν)` to three complex dipole parameters (`:240–244`), calls `bhfmm2d`
(`:265`; `biharmonic/bhfmm2d.f:1`) and reads the velocity as `u_1 = Im(w)`, `u_2 =
−Re(w)` of the complex field `w` (plus a global constant `zsum`), the pressure as
`−4 Im(∂w_1)` and the velocity gradient from the complex gradient components
(`:284`, `:294–297`). The biharmonic expansions (`bhrouts2d.f:51` header): **five** series
`w = Σ m_1(k)/z^k + Σ m_2(k)/z̄^k + z Σ m_3(k)/z̄^k + Re(m_4(0) log z + Σ m_4(k)/z^k)
+ i Re(m_5(0) log z + Σ m_5(k)/z^k)` (Goursat form); formation `bh2dformmpc/d/cd`
(`:560/:458/:670`), evaluation `bh2dmpevalp/g`, `bh2dtaevalp/g` (`:51/:137/:262/:340`),
translations `bh2dmpmp` (`:1192`), `bh2dmploc` (`:1322`), `bh2dlocloc` (`:1078`):
the same binomial matrices as Laplace applied to each series, with series 2 and 3
coupled (`h2a = (m_2 − m_3 z_0) conj(zpow)`), so the M2L is block-sparse: three
`(p+1)²` blocks plus one `2(p+1) × 2(p+1)` block ≈ `7(p+1)²` complex entries.
Order `bh2dterms` (`bh2dterms.f:15`) = the Laplace rule (`18` at `1e-6`);
`rscale = boxsize` (`bhfmm2d.f:352`). Length `5(p+1)` complex = `95` complex at
`p = 18` (series 4 and 5 are real-part series, so there is redundancy; the
information is roughly `3.5(p+1)` complex) against `121` reals for our circle method
at `nequiv = 60` (`1e-11`). Rotlet and doublet are "NOT YET IMPLEMENTED" in
`stfmm2d` (`:95–101`) but are Goursat-representable (rotlet = antisymmetric part
of the stresslet's dipole, doublet = a derivative of the stokeslet pair); modest
derivation.

**Mapping.** A complex-expansion method of length `5(p+1)`, `l2t` returning
`(Im, −Re)` and the pressure from the series-1 derivative; `s2m!` per singularity
through the complex parameters above. `operator_eltype` complex (inherent: `z` and
`z̄`). Width: the `log|z_0|` entries again, so per-width build. Near field unchanged
(`Stokeslet2D`). Effort ~400 lines; the `(−i, …)/4` bookkeeping is the risk, but
`check_m2l_all_offsets` with Stokes densities catches it. **Low priority**: our 2D
Stokes circle method is already faster than FMM2D at better accuracy
(`0.76 s` vs `1.28 s` at `10⁶`, `1e-11` vs `2e-8`).

### 3.6 Stokes 3D via Laplace (fmm3d `src/Stokes/stfmm3d.f`)

**What fmm3d does, exactly.** `stfmm3d` (`stfmm3d.f:38`) runs **one vectorised
Laplace FMM** (`lfmm3d` with `nd = ndper·nd`, `ifpgh = 3`: potential, gradient and
Hessian) with `ndper = 4` harmonic potentials for stokeslets and stresslets, `7`
when rotlets or doublets are present (`:223–232`). The sources (`:300–400`, the
stokeslet charge at `:321`), with
`y` the absolute source position:

- stokeslet `σ`: charges `σ_l/2` in potentials `l = 1,2,3` and `(σ·y)/2` in `l = 4`;
- stresslet `(μ, ν)`: dipole vectors `(μ_l ν + μ ν_l)/2` in `l = 1..3` and
  `(μ (ν·y) + ν (μ·y))/2` in `l = 4`;
- rotlet `(rlet, rvec)`: dipole vectors `−rvec_l rlet + rlet_l rvec` in `l = 5..7`;
- fmm3d's doublet: charge `(dlet·dvec)` in `l = 4` plus dipoles in `1..3` and `5..7`.

The unpacking (`:420–560`; `vel(l) += φ_l` at `:476`, `press −= 2∂_lφ_l` at
`:480`, Hessians into `velgrad` at `:516`), with `x` the absolute target position and
`φ_l, ∇φ_l, ∇∇φ_l` the Laplace outputs: `u = φ_{1..3} − Σ_l x_l ∇φ_l + ∇φ_4 +
φ_{5..7}`, `p = −2 Σ_l ∂_l φ_l`, `∇u` from the gradients and Hessians (`velgrad`).
This is the identity `(1/8π)(σ/r + r (r·σ)/r³) = φ_i − x_l ∂_i φ_l + ∂_i φ_4` with
`φ_l = σ_l/(8π r)`, `φ_4 = (σ·y)/(8π r)` (check: `−x_l σ_l ∂_i(1/r) + (σ·y)∂_i(1/r)
= −(σ·r)∂_i(1/r) = (σ·r) r_i/r³`). Note `x_l ∂_i φ_l − ∂_i φ_4` cancels to leading
order at large `|x|, |y|` (fmm3d rescales the tree to the unit box, `b0inv`); the
relative loss is `|x|/r`, harmless for far interactions and irrelevant for near
ones, which in our framework stay with the Stokes kernel.

Our singularities: stokeslet and stresslet exactly as above (JellySim's stresslet
is fmm3d's `T_{ijk}` up to sign); our rotlet `c ω × r/r³ = −c ε_{ijk} ω_j ∂_k(1/r)`
is three dipole *potentials* (blocks 5–7, no `x`-weighting, as in fmm3d); our
doublet `c(g/r³ − 3r(r·g)/r⁵) = ∇_x[c g·∇_y(1/r)]` is the **gradient of a single
dipole potential**, i.e. block 4 only (simpler than fmm3d's doublet, which is a
different singularity). So JellySim's trio (stokeslet, stresslet, doublet) needs
`ndper = 4`; rotlets need 7.

**In our framework: a method, not a driver composition.** A wrapper method
`StokesHarmonic{M}` over the Laplace 3D method with `expansion_length = 4·(p+1)²`
(`7·` with rotlets), expansions stored block-major (block `l` = the `l`-th harmonic
potential), `s2m!` writing the four blocks from the rules above with absolute
source positions (the block-4 charge `(σ·y)/2` is translation invariant; it is a
plain charge with a position-dependent strength, no center issue), `l2t` evaluating
four locals with gradients (and Hessians for `Gradient`) and assembling
`u, p, ∇u` with the absolute target position; `near_kernel = Stokeslet3D` so the
near field, `source_field`, `target_field` and the self-evaluation are untouched;
`supports_output(Pressure) = true`. Operators: `M2M`, `L2L`, `M2L[f]` are
`kron(I_4, A)`; storing them that way wastes 4× the GEMM flops on zeros, so the
method should (a) store the scalar `A` and provide an `M2LOperator` variant whose
`apply!` reshapes the gathered `4n × ncols` block to `n × 4·ncols` and does one GEMM
with `A` (the interface explicitly allows any `apply!`), and (b) for `M2M`/`L2L`,
which the driver applies as `ops.M2M * reshape(children block)`, either accept
`kron(I_4, ·)` (M2M/L2L are a small share) or lay the expansion out
component-minor so that the stacked children block reshapes to `n × (4·8·ngroups)`
— the latter interacts with the child-slot stacking and is a design choice to
make at implementation time, not a driver change. The alternative "composition at
the driver level" (four right-hand sides through the scalar method) would do four
M2L passes instead of one GEMM with four columns and would have to evaluate the
near field through the Laplace decomposition — strictly worse; rejected.

**Cost.** Multipole `4·361 = 1444` reals at `p = 18` (`2704` at `p = 25`) against
`889` for the cube method at `p = 8` (`3e-6`/`4e-5`/`1e-3` worst cases for
stokeslet/stresslet/doublet, which the analytic version turns into the charge
rate). Dense M2L flops per pair `4·2(p+1)⁴ = 1.04 M` at `p = 18` against `1.58 M`
dense and `≈ 0.16 M` spectral for the cube Stokes; with per-class rank-`k` factors
(`k ≈ 40` at `1e-6`) `4·4·361·40 = 231 000`. `l2t`: four local expansions with
gradients `≈ 4·2·(p+1)² ≈ 3 000` flops per target against `3 × 1352` Stokeslet
evaluations. Memory per box `8·(p+1)²·8 B = 23 KB` at `p = 18` against `≈ 45 KB`
of expansions plus `3 × 72 KB` of spectra. Pressure and velocity gradient are
natural outputs (fmm3d's `ifppreg = 2, 3`).

**Effort.** ~300 lines on top of Laplace 3D (the block layout, the four `s2m!`
rules, the `l2t` assembly with Hessians, the block `apply!`), tests against
`direct` with the Stokes singularities through the existing `check_*` functions
(`random_density` already produces `SVector{3}` for Stokes kernels; stresslets and
doublets need the density generators of `test/test_stokes.jl`).

## 4. Cross-cutting questions

### 4.1 Width independence and the per-level wavenumber

Scaled coordinates `x̂ = (x − c)/w` are fmm2d/fmm3d's `rscale = boxsize`
convention (`cfmm2d.f:314`, `bhfmm2d.f:352`, `lfmm3d.f:469`): coefficients
`a_n/w^n` (multipole) and `b_n w^n` (local). Then for a homogeneous kernel of degree
`a` the M2M and L2L matrices are pure numbers (binomials times `(child offset/w)^l
= (±1/4)^l` and `(w_child/w)^j = 2^{−j}`) and the M2L carries `w^a` as a whole:
`m2l_scale = λ^a`, exactly what `rescale` and the cache expect (Laplace 3D,
Stokeslet 3D). Laplace 2D: the same except the monopole's `log w` in one M2L entry
(§3.1). Helmholtz/Yukawa: the expansion functions are `J_n(k w ρ̂)`, `H_n(k w |o|)`
etc.; nothing is width independent and nothing can be made so (the libraries use
`rscale = min(k w, 1)` or `min(k w/2π, 1)` for *conditioning*, not sharing);
operators are built per width (our cache does this today for every 2D kernel and
for `Helmholtz3D`/`Yukawa3D`), and the plane-wave quadratures likewise depend on
`k w` (fmm3d rebuilds them per level, `hfmm3d.f:1117–1125`). The one genuine
mismatch is the per-level *order* of the Helmholtz expansions at high frequency
(§3.3): resolved inside the method by padding to the maximum order.

### 4.2 Element types: real densities with complex coefficients

| family | coefficients | real formulation | `operator_eltype` |
|---|---|---|---|
| Laplace 2D | `a_n ∈ ℂ`, no symmetry for real charges | `(Re, Im)` pairs with `2×2` real blocks; same flops | `Complex{T}` (or `T` with the block form) |
| Laplace 3D | `M_n^{−m} = conj(M_n^m)` for real data | yes, `(p+1)²` reals, all operators real | `T` |
| Helmholtz 2D/3D | complex kernel | none (as the KI) | `Complex{T}` |
| Yukawa 2D/3D | conjugate symmetry, real Bessel | yes | `T` |
| Stokes 2D (biharmonic) | `z`, `z̄` series | split only | `Complex{T}` |
| Stokes 3D via Laplace | real | yes | `T` |

With a real `operator_eltype`, complex densities promote through
`expansion_eltype` and the real matrices multiply complex columns; dual positions
go through the per-point bodies as for the other methods (the closed-form
recurrences are ordinary arithmetic, so `ForwardDiff.Dual` works unchanged; the
GEMMs are the generic fallback for `Dual`, as today).

### 4.3 Symmetry classes for spherical harmonics

The 48 signed axis permutations act on `Y_n^m` as follows (expansion coefficients
transform contragrediently; signs depend on the convention, the structure does
not): rotation by `π/2` about `z` multiplies `M_n^m` by `i^{±m}`, i.e. on `(Re, Im)`
a signed swap; `z → −z` multiplies by `(−1)^{n+m}`; `x → −x` and `y → −y` conjugate
with `(−1)^m`. So the 16-element subgroup fixing the `z` axis (`D4h`) acts by
**signed permutations** of the real basis — the same `(idx, sgn)` form as
`SymmetryClasses` — while the three cosets (cyclic axis permutations) need a
rotation by `±π/2` about `y`, a real block-diagonal matrix with blocks
`(2n+1)×(2n+1)` (fmm3d's `rdplus`/`rdminus` from `getpwrotmat`, used by
`rotztoy`/`rotztox`), `Σ(2n+1)² = 9 139` entries at `p = 18` against `130 321` for
the M2L. Two usable schemes:

- **34 classes, permutations only.** The 316 far offsets fall into 34 `D4h` orbits
  (counted: 40 orbits of the `7³` table minus 6 near orbits; against 16 under the
  full group). `SymmetryClasses{3}` with 34 representatives and the harmonic-basis
  `(idx, sgn)` drops into the existing device gather/GEMM/scatter without any
  change to its structure; memory `34 × (p+1)⁴ × 8 B` (`35 MB` at `p = 18`,
  `124 MB` at `p = 25`).
- **16 classes plus a small dense transform**: `M2L_o = (P_o R_o)·M2L_rep·(P_o R_o)ᵀ`
  with `R_o` one of `{I, R_y(π/2), R_y(−π/2)}` composed with an axis cycle — "class
  representative + a block-diagonal rotation" applied to the gathered columns
  (`≈ 9 k` flops per column at `p = 18`, 3% of the M2L). Needs a new apply path;
  halves the operator memory. Not needed first.

Point-and-shoot (fmm3d's `l3dmploc`) is the limit of the second scheme: rotate
every pair to the `z` axis (the rotation for a given direction is the same for all
pairs of that offset, so it could be precomputed per offset: `≈ 13` distinct
distances `|o|` and `≈ 26` distinct directions up to symmetry), shift along `z`
with the `m`-diagonal matrix (`Σ_m (p+1−|m|)² = 4 579` entries at `p = 18`,
30× sparser than dense), rotate back: `≈ 76 000` flops per pair at `p = 18`
against `261 000` dense, but as three block-diagonal products rather than one
GEMM. For our batched structure the dense class operator is the simpler and, at
GEMM rate, probably the faster choice up to `p ≈ 20`; per-class low-rank factors
(§3.2) are the compression that keeps the GEMM form. fmm3d's own answer for
Laplace is neither: plane waves (§3.2), `5 600` flops per pair at `1e-6`.

### 4.4 Where analytic expansions are better, where not (summary table)

| aspect | kernel-independent (cube, `p = 8`) | analytic Laplace 3D (`p = 18–22`) |
|---|---|---|
| coefficients for `1e-6` worst case | 296 (+1352 large-surface points) | 361–529 |
| conditioning | pseudo-inverse `1/σ_min`, truncation, Float32 floor `1e-5` (Laplace) / `5e-3` (Stokes) | exact recurrences, Float32 to `≈ 1e-6` expected (unmeasured) |
| dipole / stresslet / doublet worst case at the charge's `1e-6` | `1e-4` / `4e-5` / `1e-3` | charge rate (one power less) |
| `s2m!`, `l2t` per point | `1352` kernel evaluations | `O(p²) ≈ 400–500` terms |
| memory per box | `≈ 15 KB` + `72 KB` spectra (Laplace) | `3–4 KB` per expansion |
| M2L flops per pair | dense `175 k`; spectral `≈ 18 k` (Hadamard rate) | dense `261–560 k` (GEMM rate); rank-`k` `36–50 k`; plane waves `5.6 k` |
| M2L operator memory | 316 tables `≈ 100 MB` Laplace, `290 MB` Stokes; 16 class matrices | 316 dense `330 MB` at `p = 18`; 34 classes `35 MB` |
| near field | unchanged | unchanged |
| degeneracies | Laplace scale (`w ≈ 0.77`), Helmholtz resonances `k w ≈ 3.4` | none |
| operator build | SVDs of `1352×296`, `0.5 s`-class | closed form, milliseconds (Laplace), `O(p⁵)` per offset for Helmholtz |
| GPU | `DevicePlan` exists (KI only) | needs the ~50-line generalisation (identity transforms, class hook) |

## 5. Effort and risk, ranked recommendation

1. **Laplace 3D, real spherical harmonics, dense operators** (first target, below).
   ~500 lines + ~200 of tests; 1–2 weeks. Risk low-medium (conventions; verified
   by the checks). Immediate value: exact dipoles, Float32, memory, the base for 2–4.
2. **Stokes 3D via Laplace** (`ndper = 4`, rotlets as 7): ~300 lines; the
   accuracy gain for stresslets/doublets (`1e-3 → 1e-6`-class at equal effort) is
   the single largest practical gain of the whole exercise for the JellySim use.
3. **Device passes for analytic methods**: generalise `DevicePlan` (identity
   pseudo-inverse stages, a `class_operators(m, ops)` hook, the 34-class harmonic
   permutations); ~100 lines in `device_passes.jl` and `symmetry.jl`. This is where
   the Float32 conditioning pays (the Stokes Metal matvec is at the `5e-3` floor).
4. **Yukawa 3D**: ~300 lines; per-width operators, level cutoff.
5. **Laplace 2D (Laurent) and Yukawa 2D**: ~200 + 150 lines; cheap, exact,
   mostly for symmetry with 3D and Float32; **Helmholtz 2D** (~250) for the
   resonance problem at `k w > 3.4`.
6. **Plane-wave `m2l_level!` for Laplace 3D** (fmm3d's engine): ~800 lines
   (quadrature tables, six directional streams, the merge-and-shift lists); only
   if the M2L stage dominates after 1–3. This is the one piece that would make the
   analytic path faster than the spectral one on the CPU.
7. **Helmholtz 3D**: hardest (memory, per-level orders, complex Float32 on Metal
   impossible anyway); after 3 and 6.
8. **Stokes 2D biharmonic**: last; the circle method already wins.

## 6. The first target: Laplace 3D with dense per-offset operators

**New files** (nothing existing is modified except `FlexibleFMM.jl` includes/exports
and the test runner):

- `src/operators/harmonics.jl`: real solid-harmonic recurrences in Cartesian form
  (`R_n^m(x̂)` regular, `S_n^m(x̂)` irregular, their `∂_x, ∂_y, ∂_z`), the real-basis
  index map `(n, m, re/im) ↔ 1..(p+1)²`, the complex↔real change of basis (sparse,
  for building operators), closed-form `z`-shift matrices and the `y`-rotation
  matrices `d^n(±π/2)` (for the operator builders and, later, the classes),
  `harmonic_m2l(o, p)`, `harmonic_m2m(δ, p)`, `harmonic_l2l(δ, p)` returning real
  `(p+1)²`-square (or `×8`) matrices in scaled coordinates. Two independent
  constructions of M2L (closed form; rotation–shift–rotation composition) checked
  against each other in the tests.
- `src/operators/analytic_laplace3d.jl`: `struct SphericalHarmonics{K,P} <:
  AbstractMethod` (name to taste; `K = Laplace3D{T}`), `dimension = 3`,
  `expansion_length = (P+1)²`, `operator_eltype = floattype(K)`, `operators_type`,
  `level_operators(m, w)` (builds `M2M`, `L2L`, `M2L` as `Vector{M2LOperator{T}}`
  over `far_offsets(Val(3))` at width `w` with the `1/w` factor of the M2L in the
  matrices, extras holding `1/w`), `rescale(ref, m, w)` (shares all matrices,
  `m2l_scale = λ^{−1}`, scale in the extras — the `BlackBox` `rescale` is the
  template), `s2m!` with `accumulate_moments!`-style methods for `Number`, `Dipole`,
  `Tuple`, `l2t` for `Potential` and `Gradient`, `partial_to_multipole!` /
  `partial_to_local!` = `copyto!`, `multipole_field` (for the checks),
  `supports_density` for the above types only.
- `test/test_operators_analytic.jl`: `check_m2l_all_offsets`, `check_m2m`,
  `check_l2l`, `check_allocation_free` from `test/operator_checks.jl` unchanged;
  convergence in `p` (worst offset error vs `p`, to settle the order table of §2);
  dipoles at the charge rate; `rescale` consistency (operators at two widths agree
  with direct builds); Float32 at `p = 12, 16, 20`; a `Dual` position through
  `s2m!`/`l2t`; the two M2L constructions agree to `1e-13`.
- `test/test_fmm_analytic3d.jl` (or a block in `test_3d.jl`): end-to-end
  `FMM(SphericalHarmonics(Laplace3D(); p=16), points)` against `direct` on the
  distributions of `test_3d.jl`, cached operators across two trees
  (`test_cache.jl` pattern), `evaluate_at_sources`, Gradient output, and a
  side-by-side with `CubeKernelIndependent{3}(Laplace3D(); p=8)` on the same points.

**Reused unchanged**: `Tree`, lists, builder, `locate` (`src/tree/*`); `FMM`,
`FMMState`, `upward_pass!`, `downward_pass!`, the default `m2l_level!`
(gather/GEMM/scatter, `threaded_mul!`), `build_expansions!` (`src/fmm.jl`);
`OperatorCache`, `get_operators!`, `fit_box!` and the homogeneous sharing
(`src/operators/cache.jl`); `evaluate!`, `evaluate_at_sources!`, `evaluate_target`
(`src/evaluate.jl`); `source_field`/`target_field` for the near field and the
reference sums (`src/sources.jl`); `Laplace3D` with `homogeneity = −1`
(`src/kernels.jl`); `kernel_matrix`/`direct` (tests); `far_offsets`,
`child_offsets` (`src/util.jl`); `test/operator_checks.jl`; `bench/compare` for the
FMM3D comparison.

**Acceptance**: worst-offset M2L error vs `p` measured (expect `1e-6` at
`p ≈ 20–22`); dipole worst case within 3× of the charge's at the same `p`; `10⁵`
uniform points matvec time within 2× of the cube method at equal global error;
Float32 global error `≤ 1e-5` at `p = 16`.

**Then** (in order): the block `apply!` and `StokesHarmonic` (§3.6); the 34-class
harmonic `SymmetryClasses` and the `DevicePlan` generalisation; Yukawa 3D.

## 7. Uncertainties, stated

- The order needed for our worst case in 3D (`p ≈ 20–22` by the corner estimate,
  25 by fmm3d's rule, 18 by CGR'99 practice): to be measured; everything in the
  size tables scales as `(p+1)⁴`.
- CPU time of the dense analytic M2L against the spectral path: the flop ratio is
  15–30×, the rate ratio (GEMM vs load-bound Hadamard) 5–10×; I expect the dense
  path 2–5× slower on that stage and roughly even with the per-class low-rank
  factors; not measured.
- Float32 behaviour of the Legendre/solid-harmonic recurrences at `p ≥ 20` and of
  the dense operators (entries range over `|o|^{−(n+j+1)} ≈ 2^{−40}` at `p = 20`
  for the smallest terms; the scaled form keeps the columns balanced, but the
  Float32 error floor is an experiment).
- The exact count and the signs of the `D4h` signed permutations in a given real
  basis (structure is certain, conventions must be verified numerically as
  `check_symmetry_classes` does today).
- fmm3d's Helmholtz plane-wave tables are per `iprec` and per `k w` "domain"
  (`hwts_3d_iprec*.txt`); I did not quantify their sizes.
- Everything above about fmm2d/fmm3d is read from the sources listed; nothing was
  built or run.
