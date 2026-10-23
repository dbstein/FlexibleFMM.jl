# Heterogeneous sources and targets: several body kinds in one Stokes FMM

Design study, 2026-10-23, at commit 97fc54f. Sources read: JellySim
(`/Users/dstein/Documents/Projects/JellySim/JellySim.jl`, read only: the kernel
backends under `src/hydro/kernels/stokes/`, `represented_operators.jl`,
`solvers.jl`, `fiber_fiber_freespace_solver.jl`, `stokes_operators.jl`, the
points and boundaries, `bodies/rigid_body.jl`, the moving-sphere examples under
`examples/hydro/`, `docs/review/src/hydro/kernels/stokes/fmm3d_backend.md`,
`examples/drosophila/default_config.jl`) and this package (`src/sources.jl`,
`src/kernels.jl`, `src/evaluate.jl`, `src/fmm.jl`, `src/stokes_api.jl`,
`src/operators/{interface,ki_common,cubeki,kifmm,harmonics,harmonics_stokes,bbfmm}.jl`,
`src/device.jl`, `src/device_passes.jl`, `src/tree/*`, the docs). Measurements on
the M3 Max (12 Julia threads, Julia 1.12.5, `OPENBLAS_THREAD_TIMEOUT=4`; Metal
through `bench/gpu`) with the scripts in
`scratchpad/hetero_explore/` (`pair_cost.jl`, `pair_count.jl`, `mix_plans.jl`,
`metal_kinds.jl`, `grad_fix.jl`); nothing under `src/` was changed.

## 0. Verdict in one paragraph

Recommend **groups within one tree** (route c), with a source/target-object
front end in `src/stokes_api.jl` (`StokesBody`: points, the density type of the
group, the outputs wanted on it; a target-only body is a body whose density type
is `Nothing`). The tree needs no second sort key: the existing stable counting
sorts leave every childless node's points in input order, so if the bodies are
passed concatenated, every node (with or without children) owns one contiguous
range per group *in the group's own numbering* (the sorted order restricted to
the group), verified on a mixed tree (§3.5). The change is then a per-node
`G × nnodes` table of ranges, per-group `perm`, a state holding one typed vector
per group, and "for each group" around the four loops that touch a node's points
(S2M, the near field, the two device kernels), each iteration being today's typed
body unchanged. Its near field costs the ideal by construction and its far field
is paid once. Zero padding (route 1, the FMM3D way) needs no code and is the
right thing for a quick first run, but costs 2.0× on the velocity sweep after the
fixes below (5.7× with today's bodies) and 6–10× when every target also computes
the gradient and the pressure, on the near field that is 85% of the harmonic
matvec at 600 points per leaf. A tagged sum type (route d) costs 1.5–1.9× on the
CPU because the loop goes scalar. One pass per kind (route b) is route c's data
layout with the far field paid K times, and separate plans per kind (the
no-code baseline, what JellySim does today with DMK) lose on the near field,
not the far field: a small kind's own tree is coarse, so every target of the
big kind sums over whole coarse colleague ranges of it (4.1× the grouped near
field on the mix of §3.5; refining the small kinds' trees by hand to `ppl = 25`
brings that to 1.8× at three times the far field). Three cheap fixes matter
more than any route and
precede all of them: the 3D stresslet body's `ir^5` goes through Julia's
compensated `pow_body` (3.9× on every stresslet pair), the stresslet and doublet
velocity gradients are ForwardDiff Jacobians (4.7× and 3.5× against hand-written
ones, verified to 1.3e-15), and the harmonic device S2M does not compile for a
tuple density (`for τi in τ` over a heterogeneous tuple is a dynamic `getfield`
on the device; the recursive form compiles and runs).

## 1. What JellySim hands to a kernel backend today

Observed, not inferred:

- **The call.** `StokesKernel!(velocities, sources, CT; weights, targets,
  stokeslets, stresslets, stressvecs, doublets, backend, force_self_eval,
  fmm3d_eps, dmk_eps)` (`kernel_abstractions_wrappers.jl`). One source set
  (`Vector{SVector{3,T}}`), one density array per kind over the *whole* set, each
  `nothing` or of full length; `weights` separate (the quadrature weights are
  applied inside the kernel, never folded into the densities by the caller);
  `targets = nothing` or `sources === targets` or `force_self_eval` means self
  evaluation with `i = j` skipped; the output is a `Vector{SVector{3}}` of
  velocities in the targets' order, and nothing else: no backend returns a
  gradient or a pressure, and `grep` finds no traction anywhere in `src/`.
- **The backends.** The KernelAbstractions kernels (CPU/Metal/CUDA) take the
  per-kind arrays as `nothing` or an array, so `if stokeslets !== nothing` is a
  static branch and each kind present is summed for every pair, sharing
  `kernel_basics` (`1/r`, `d/r`). FMM3D: one `stfmm3d` call carrying `stoklet`
  *and* `strslet`/`strsvec` for every source when both kinds are present (the
  FMM3D model: every source carries every kind), plus a separate `lfmm3d` for the
  potential doublet. DMK: one kernel per tree, so stokeslets and stresslets are
  two trees, no doublet.
- **One body kind per call.** `ApplyNaiveQuadrature!` (`represented_operators.jl`)
  dispatches on the representation of the *represented point set*: the BIE
  flavour (periphery: `HydroBoundary` with nodes, weights, normals; stokeslets
  and/or stresslets per the representation's weights, `stressvecs = -normals`,
  the same density array passed as both `stokeslets` and `stresslets` for a
  single-plus-double layer) and the SBT flavour (fibres: `stokeslets = f`, and
  `doublets = (ε²/2)·f` when `include_doublet`, derived from the per-node cache
  `Geometry.Eps²Half`). Densities carry `vc = 1/μ` (`DrosophilaSolverMemory`'s
  convention block); the `1/(8π)` lives in the kernels, as in `Stokeslet3D`.
- **The per-matvec pattern** (`DrosophilaHydrodynamicF2F * f`, `solvers.jl`):
  (1) fibres → periphery (sources: fibre nodes with stokeslets + doublets;
  targets: periphery nodes), plus the fibre-to-periphery close correction;
  (2) the dense periphery solve `σₚ = 𝒟⁻¹(−uₚ)` on the linalg backend;
  (3) periphery → fibres (sources: periphery nodes with stresslets and/or
  stokeslets; targets: fibre nodes) and (4) fibres → fibres with
  `targets_are_fibers = true`, both in `HydroEvaluate!`, plus the close
  corrections and the self-interaction subtraction. So two or three N-body
  calls per matvec, each with a homogeneous source set and a different target
  set, and the periphery density is eliminated by a dense inverse rather than
  carried as an unknown.
- **The persistent-plan shape** (the DMK path, the closest to what the FMM will
  replace): `DMKHydroPlans` holds `DMKStokesPlan(fibres; targets=periphery)` and
  `DMKStokesPlan(periphery; targets=fibres)`, rebuilt by `HydroRefresh!` on a
  geometry capture (stamp-checked), with `dmk_eval_src_trg!` giving the fibre
  self term and the fibre → periphery term from one charge update, Float32
  buffers converted at the boundary. `StokesFMMPlan`/`stokes_fmm_eval_src_trg!`
  were written to this shape.
- **Mobile bodies today.** `MobileBody` (`src/bodies/rigid_body.jl`) is a rigid
  sphere with analytic Stokes drag (`6πμa`, `8πμa³`) and no layer potential; the
  moving-sphere examples (`examples/hydro/MovingBody3D_NBodies…`) represent each
  sphere by a single-plus-double layer with dense corrected quadrature matrices
  and take force and torque from the single-layer density (`𝐅 = μ Σ σ w`, the
  completed double-layer formulation), not from a traction evaluation. The
  traction route the question describes (velocity gradient and pressure at the
  surface, contracted with the normal) is new on both sides.
- **Sizes** (`examples/drosophila/default_config.jl`): 600 fibres before shaving,
  `h = 0.15` on the periphery; the representative mix of the question (10⁵
  periphery, 10⁶ fibre, 10⁵ object points) is an order of magnitude above the
  current runs, which is the point of an FMM.

What the integration will hand over, then: per geometry capture, for each body a
point array, its weights, its normals if it carries a double layer, which kinds
it carries and which outputs it wants; per matvec, one `SVector{3}` density
array per body (reused for the stokeslet and stresslet parts of a
single-plus-double body, plus `(ε²/2)f` for the fibres), and it expects back one
output array per target body in that body's point order, to which it adds its
close-evaluation corrections. Whether a matvec is one call (every body a source
and a target) or two (the block-eliminated periphery) is an open question (§7).

## 2. The package's model today

- **One density type per run.** `FMMState{S,P,Tτ}` holds `τ::Vector{Tτ}` in tree
  order; `Tτ` fixes the kernel bodies through dispatch:
  `source_field(K, s, t, τ)` and `target_field(K, s, t, τ, out)` in
  `src/sources.jl` (a `Number` is a charge, `Dipole`, for Stokes an `SVector{3}`
  stokeslet, `Stresslet(f, ν)`, `Doublet(g)`, `Rotlet(ω)`, a `Tuple` the sum of
  its parts through `_sum_fields`/`_sum_targets`). `expansion_eltype` promotes
  `density_scalar(Tτ)` into the expansions; `supports_density(m, Tτ)` is checked
  once per build.
- **S2M consumes `view(τ, tree.points[node])`** (`_s2m_level!` in `src/fmm.jl`):
  the kernel-independent `s2m!` (`ki_common.jl`) evaluates `source_field` of each
  source at the `nlarge` check points (1352 for the cube at `pc = 16`); the
  harmonic `s2m!` folds each source into `(p+1)²` or `4(p+1)²` coefficients
  (`_accumulate!`, `_stokes_accumulate!`, one method per density type, the tuple
  method a `for τi in τ` loop); the black box through `accumulate_moments!`.
  Each zeroes the column first and then accumulates.
- **The near field** (`_direct_range`, `src/evaluate.jl`; `_direct_target` in
  `kernels.jl` for the reference sums) is one `@simd` loop over a colleague's
  source range with the excluded source masked, `target_field` of the state's
  single density type, accumulation reassociated; it vectorises across sources
  only because the body is one type.
- **Outputs** are a type per `evaluate` call for the whole target set
  (`Potential()`/`Velocity()`, `Gradient()`, `Pressure()`), selecting the `l2t`
  method and the `target_field` method; `Gradient()` is hand-written for the
  stokeslet and a `ForwardDiff.jacobian` of `source_field` for the stresslet,
  doublet and rotlet; `Pressure()` is hand-written for all. There is no combined
  output and no per-target data (a normal) in an output.
- **Targets** are any point set (`evaluate(f, st, targets; output)`, located and
  bucketed by leaf) or the sources themselves (`evaluate_at_sources`, skipping
  the self term by the sorted index).
- **The device** (`src/device.jl`, `src/device_passes.jl`): `device_state` moves
  `state.τ` as one device vector; `_gather_input!` keeps one buffer per
  `(name, typeof(x))`; `_s2m_kernel!` and `_evaluate_kernel!` take `τ` and
  dispatch on its eltype inside the work-item; the harmonic S2M takes
  `Val(nb)`; the evaluation kernel's output type is the eltype of `out`.
- **The tree** sorts points by stable counting sorts into child slots
  (`_sort_range!`, serial and chunked, both stable; the device tree by one
  Morton `sortperm!`), so leaves mix every kind, and `tree.points[node]` is one
  range per node; the interaction lists prune on `has_source` per node.
- **The Stokes plan API** (`src/stokes_api.jl`): `StokesFMMPlan(sources;
  targets)` with one source set and at most one target set,
  `stokes_densities(n; stokeslets, stresslets, stressvecs, doublets, weights)`
  building one tuple per source with exactly the kinds present (no zeros: the
  tuple type is uniform over the sources, which is the model's constraint),
  `stokes_fmm_eval!`/`stokes_fmm_eval_src_trg!` with one `output` per call.

## 3. Measurements

### 3.1 One near-field pair (`pair_cost.jl`)

The package's own masked loop (`_direct_range`) over 600 sources at one target,
one core, best of five, nanoseconds per pair (millions of pairs per second in
parentheses). "stresslet′" is the stresslet body with `ir^5` written as
`ir²·ir²·ir`; "hand" gradients are the closed forms of `grad_fix.jl`, which agree
with the ForwardDiff ones to `1.3e-15` relative over 1000 random pairs.

| density \ output, Float64 | velocity | gradient | pressure | u+∇u+p (one pass) |
|---|---|---|---|---|
| stokeslet | 1.54 (650) | 4.10 (244) | 1.33 (751) | 5.91 (169) |
| stresslet, as written | 5.79 (173) | 22.7 ForwardDiff, 4.88 hand | 1.68 (595) | 30.5 |
| stresslet′ | 1.48 (676) | 4.94 hand (202) | 1.61 (621) | 6.85 (146) |
| doublet | 1.83 (546) | 14.2 ForwardDiff, 4.03 hand | 0.02 (compiled away) | 16.7 |
| (stokeslet, doublet) | 2.51 (399) | 21.5 (ForwardDiff doublet) | 1.41 (709) | 22.6 |
| (stokeslet, stresslet) as written | 7.33 (136) | 28.7 | 2.95 (339) | 37.0 |
| (stokeslet, stresslet′) | 3.68 (272) | 7.88 (127) | 2.94 (340) | 10.7 (93) |
| (stokeslet, stresslet, doublet) as written | 17.8 (56) | 40.6 | 3.03 | 50.6 |
| (stokeslet, stresslet′, doublet) | 5.21 (192) | 29.5 (ForwardDiff doublet) | 2.95 | 32.1 |
| flagged union, all stokeslet | 2.52 (397) | | | |
| flagged union, all stokeslet+doublet | 4.64 (215) | | | |
| flagged union, thirds sorted by kind | 3.74 (268) | | | |
| flagged union, kinds interleaved | 3.86 (259) | | | |

| density \ output, Float32 | velocity | gradient | pressure | u+∇u+p |
|---|---|---|---|---|
| stokeslet | 0.84 (1190) | 2.05 (488) | 0.67 | 4.12 |
| stresslet, as written | 3.51 (285) | 16.2 ForwardDiff, 2.77 hand | 1.13 | 28.1 |
| stresslet′ | 1.11 (902) | 2.76 hand | 1.15 | 4.99 |
| doublet | 0.85 (1179) | 11.3 ForwardDiff, 1.94 hand | 0.03 | 22.7 |
| (stokeslet, doublet) | 1.50 (668) | 25.7 | 0.92 | 26.3 |
| (stokeslet, stresslet′) | 1.83 (547) | 4.46 | 1.74 | 7.03 |
| (stokeslet, stresslet′, doublet) | 2.42 (414) | 23.0 | 1.78 | 28.3 |
| flagged union, all stokeslet / sorted thirds / interleaved | 2.25 / 3.68 / 3.65 | | | |

What the table says:

- **The stresslet body is 3.9× slower than it should be.** `x^5` on a `Float64`
  lowers to `Base.Math.pow_body` (the compensated integer power introduced for
  accuracy; `x^3` is special-cased to products, which is why the doublet and the
  pressures are fine): `@code_llvm` of `x^5` is a call, of `x²·x²·x` twelve
  lines. The fix is one line in each of the two stresslet bodies of
  `src/sources.jl` (3D velocity, and the 2D analogue uses `r2*r2`, fine) and
  brings the stresslet pair to the stokeslet's cost. The same `ir^5` is in the
  3D stresslet pressure (`ir3 * ir * ir`: fine) and in the cube `s2m!` through
  `source_field`: the cube stresslet S2M goes 7178 → 1347 ns per source (§3.2).
  On Metal the stresslet costs 2.0× the stokeslet (§3.4), the same mechanism.
- **The ForwardDiff gradients cost 4.7× (stresslet) and 3.5× (doublet)** against
  closed forms of a dozen lines; the closed forms are in `grad_fix.jl` with
  their check. The tractions on the mobile objects are gradients at every
  object target from fibre and object sources, so this is on the critical path.
- **Zero padding is a multiplier, not an additive cost**: the triple tuple costs
  5.21 ns against 2.51 for the fibres' own (stokeslet, doublet) pair (2.1×) and
  17.8 ns with today's stresslet body (7.1×); with the gradient it is 29.5
  against 21.5 (and 7.9 for (stokeslet, stresslet′): the ForwardDiff doublet
  dominates everything it enters).
- **A combined `(u, ∇u, p)` output saves 15% against three passes** (5.91 against
  1.54 + 4.10 + 1.33 for the stokeslet; 6.85 against 8.03 for the stresslet′),
  because the gradient dominates and the shared `d`, `1/r` are cheap; its real
  merit is one sweep over the colleagues instead of three (and one kernel launch
  on the device).
- **The tagged union (route d) runs scalar.** With every source carrying the same
  flag the loop costs 1.6× the typed stokeslet loop and 1.85× the typed
  (stokeslet, doublet) loop; sorted or interleaved kinds cost the same, so the
  branches predict fine and the loss is the vectorisation across sources that
  the one-type body gets (the 1.3–1.5× of `docs/kifmm.md`, here compounded with
  the three branches per pair).

### 3.2 S2M per source and `l2t` per target (`pair_cost.jl`)

Float64; `s2m!` of 600 sources in one node, nanoseconds per source; `l2t` of one
target, microseconds.

| method | s2m!: stokeslet / stresslet / stresslet′ / doublet / (stk, dbl) / (stk, strs) / (stk, strs′) / triple / triple′ | l2t: velocity / gradient / pressure |
|---|---|---|
| harmonics p = 8 | 185 / 512 / – / 115 / 340 / 735 / – / 902 / – | 0.75 / 2.29 / 0.77 |
| harmonics p = 16 | 631 / 1953 / – / 465 / 1183 / 2765 / – / 3281 / – | 2.82 / 9.31 / 2.85 |
| cube p = 8, pc = 16 | 1537 / 7178 / 1347 / 1603 / 2759 / 9519 / 2479 / 12063 / 3540 | 2.13 / 4.34 / 1.11 |

(The harmonic S2M has its own stresslet body, dipole folds, so the `ir^5` fix
does not reach it; its stresslet costs 2.8× a stokeslet because it folds four
dipoles.) Against the near field at 600 points per leaf, ≈ 5000 pairs per
target (§3.3): 5000 × 1.54 ns = 7.7 µs per target against 0.75 µs of `l2t` and
0.19 µs of S2M; the near field is 85% of the harmonic matvec there (§3.4), so the
per-pair costs are what decides.

### 3.3 The near-field pair structure of the representative mix (`pair_count.jl`)

10⁵ periphery points on a sphere of radius 0.48, 10⁶ fibre points as 2000
straight segments of 500 points (length 0.15, random positions and directions
inside radius 0.38), 10⁵ object points as 100 spheres of 1000 points (radius
0.02), in the unit box; one tree of all 1.2 × 10⁶ points.

| ppl | nodes / leaves / levels | leaves with > 1 kind (points in them) | near pairs (per target) | tree build |
|---|---|---|---|---|
| 600 | 23 345 / 9 759 / 7 | 913 (18.2%) | 5.79 × 10⁹ (4829) | 1.6 s (cold) |
| 200 | 103 105 / 31 186 / 8 | 1 357 (10.0%) | 2.28 × 10⁹ (1901) | 0.08 s |

Share of the near-field pairs by (target kind, source kind), `ppl = 600`:

| target \ source | periphery | fibres | objects | pairs per target |
|---|---|---|---|---|
| periphery | 2.42% | 1.88% | 0.04% | 2512 |
| fibres | 1.15% | 78.45% | 7.57% | 5052 |
| objects | 0.01% | 6.30% | 2.18% | 4920 |

By source kind: periphery (stresslets) 3.6%, fibres (stokeslet + doublet) 86.6%,
objects (stokeslet + stresslet) 9.8%. Object targets, the ones wanting the
gradient and the pressure, receive 8.5% of the pairs, three quarters of them
from fibre sources.

### 3.4 Matvec phases, CPU and Metal (`pair_cost.jl`, `metal_kinds.jl`)

2 × 10⁵ sphere points, `ppl = 600`, Float64, 12 threads, best of three; "self" is
`evaluate_at_sources` with the output named.

| method, density | up | down | self: velocity | gradient | pressure | u+∇u+p |
|---|---|---|---|---|---|---|
| harmonics p = 8, stokeslet | 0.010 | 0.017 | 0.159 | 0.417 | 0.145 | 0.636 |
| harmonics p = 8, stresslet (as written) | 0.016 | 0.019 | 0.521 | 1.993 | 0.195 | 2.916 |
| harmonics p = 8, (stokeslet, doublet) | 0.015 | 0.018 | 0.265 | 2.705 | 0.175 | 2.863 |
| harmonics p = 8, triple (as written) | 0.029 | 0.019 | 0.981 | 4.761 | 0.366 | 6.093 |
| cube p = 8, stokeslet | 0.155 | 0.248 | 0.300 | 0.702 | 0.196 | 0.778 |
| cube p = 8, stresslet (as written) | 0.317 | 0.249 | 0.738 | 2.856 | 0.223 | 3.229 |
| cube p = 8, (stokeslet, doublet) | 0.162 | 0.213 | 0.296 | 2.200 | 0.181 | 2.611 |
| cube p = 8, triple (as written) | 0.360 | 0.238 | 1.064 | 4.360 | 0.346 | 5.515 |

The self phase is 85% of the harmonic matvec and 43% of the cube's at this `ppl`
(the question's 50–80% bracket); the phases scale with the pair costs of §3.1
(the stokeslet self phase runs at 12.0× one core). The far field is small for the
harmonic method and barely depends on the density (the triple's S2M is 3× the
stokeslet's but S2M is a tenth of the upward pass); for the cube method the
upward pass doubles with the stresslet's S2M.

Metal, Float32, 10⁵ sphere points, `ppl = 600`, seconds; "passes" is
`device_build_expansions!` (S2M, the GEMMs, the class translations):

| method, density | passes | self: velocity | gradient | pressure |
|---|---|---|---|---|
| harmonics p = 8, stokeslet | 0.022 | 0.0079 | 0.0137 | 0.0076 |
| harmonics p = 8, stresslet | 0.026 | 0.0161 | 0.0376 | 0.0072 |
| harmonics p = 8, doublet | 0.020 | 0.0081 | 0.0179 | 0.0037 |
| harmonics p = 8, (stokeslet, doublet) | 0.024 | 0.0088 | 0.0237 | 0.0067 |
| harmonics p = 8, triple | 0.036 | 0.0196 | 0.0454 | 0.0087 |
| cube p = 8, stokeslet | 0.032 | 0.0087 | 0.0131 | 0.0081 |
| cube p = 8, (stokeslet, doublet) | 0.038 | 0.0100 | 0.0241 | 0.0086 |
| cube p = 8, triple | 0.038 | 0.0203 | 0.0510 | 0.0102 |

Zero padding costs 2.2× on the device velocity sweep against the fibres' kind
and 1.9× on the gradient; the ForwardDiff gradients compile and run on Metal.
The harmonic device S2M (`_s2m_harmonic_kernel!`) failed to compile for every
tuple density (`InvalidIRError: call to ijl_get_nth_field_checked` from
`_stokes_accumulate_order!`'s `for τi in τ`, `harmonics_stokes.jl:396`); the
rows above were obtained with the loop replaced by a recursive unroll over the
tuple in the script. The CPU `s2m!` tolerates the dynamic loop; the cube device
S2M goes through `source_field`'s recursive `_sum_fields` and is fine. The same
`for τi in τ` loops are in `_accumulate!`, `_accumulate_order!`,
`_stokes_accumulate!` and `accumulate_moments!`.

### 3.5 Three plans against one tree on the mix at 1/10 scale (`mix_plans.jl`)

10⁴ periphery, 10⁵ fibre, 10⁴ object points (the geometry of §3.3 scaled),
`ppl = 600`, 12 threads, seconds. "Three plans" is what the package does today
for several kinds: one `FMM` per kind sharing an `OperatorCache`,
`evaluate_at_sources` for the kind's own points and `evaluate` at the other
kinds' points (the sum of those is the near field a grouped tree would do
once). "Zero padded" is one tree with the triple tuple (zeros where a kind is
absent, today's bodies). "Stokeslets only" and "(stokeslet, doublet)
everywhere" are one tree with a homogeneous density, the latter the fibres'
kind that carries 87% of the pairs: a proxy for the grouped near field (its
far field is what the grouped tree pays once).

| harmonics p = 8 | tree | far | near (self + at the others) |
|---|---|---|---|
| three plans, every kind at ppl = 600 | 0.012 | 0.046 | 0.548 (0.162 + 0.386) |
| three plans, periphery and objects at ppl = 100 | 0.020 | 0.074 | 0.435 (0.145 + 0.290) |
| three plans, periphery and objects at ppl = 25 | 0.029 | 0.160 | 0.235 (0.140 + 0.095) |
| one tree, zero padded | 0.004 | 0.080 | 0.492 |
| one tree, stokeslets only | | 0.053 | 0.086 |
| one tree, (stokeslet, doublet) everywhere | | 0.057 | 0.132 |

| cube p = 8 | tree | far | near |
|---|---|---|---|
| three plans, every kind at ppl = 600 | 0.012 | 0.602 | 0.621 (0.176 + 0.444) |
| three plans, periphery and objects at ppl = 100 | 0.009 | 1.288 | 0.505 (0.158 + 0.347) |
| three plans, periphery and objects at ppl = 25 | 0.041 | 3.291 | 0.293 (0.163 + 0.130) |
| one tree, zero padded | 0.007 | 1.092 | 0.545 |
| one tree, stokeslets only | | 0.744 | 0.115 |
| one tree, (stokeslet, doublet) everywhere | | 0.771 | 0.167 |

Per plan (harmonics): periphery (10⁴ points, 3 levels, 73 nodes) self 0.014 s
but *0.217 s at the other kinds' points*; objects (3257 nodes) self 0.012 s and
0.125 s at the others; fibres self 0.136 s and 0.041 s at the others. The small
kinds' own trees are coarse (the periphery's leaves hold up to 600 points on a
sphere), so each of the 1.1 × 10⁵ fibre and object targets located in them sums
over the whole colleague ranges, the price `docs/tree.md` names for targets in
coarse leaves; in the combined tree the same periphery sources sit in refined
leaves next to the fibre targets and account for 1.15% of the pairs. Separate
plans therefore cost 4.1× the grouped near field (0.548 against the 0.132 s
proxy) unless each kind's tree is refined to the others' density by hand: at
`ppl = 100` the periphery's four levels are still coarse (0.228 s at the
others); at `ppl = 25` the near field is down to 1.8× the proxy, but the far
field has tripled for the harmonic method (0.160 against 0.057) and is 4.4× for
the cube (3.3 s, the objects' tree alone 2.5 s), so the best tuned three-plan
matvec is 0.40 s against the grouped tree's ≈ 0.19 s (0.057 + 0.132) and zero
padding's 0.57 s. The grouped tree gets the refinement for free: every kind's
sources are in the leaves the union of all kinds dictates.

## 4. Cost model for the production mix

From the pair structure of §3.3 (5.79 × 10⁹ pairs at `ppl = 600`) and the pair
costs of §3.1, Float64, one core, with the measured 12.0× for the self phase;
"fixed" means the `ir^5` and hand-written-gradient fixes applied. Per-pair
averages weight the source kinds (periphery 3.6%, fibres 86.6%, objects 9.8%),
and the ideal output cost gives the object targets' 8.5% of the pairs the
combined `(u, ∇u, p)` body (≈ 11 ns per pair with hand-written gradients: 10.7
measured for (stokeslet, stresslet′), the (stokeslet, doublet) one by composition)
and everyone else the velocity.

| route (near field only, harmonics p = 8) | ns per pair, today's bodies | ns per pair, fixed | seconds on 12 threads, fixed |
|---|---|---|---|
| ideal: each pair with its source's kind, velocity only | 3.10 | 2.59 | 1.25 |
| ideal with the objects' gradient and pressure | | 3.30 | 1.6 |
| zero padded, velocity everywhere | 17.8 | 5.21 | 2.5 |
| zero padded, every target `(u, ∇u, p)` | 50.6 | 21–32 (hand/ForwardDiff doublet gradient) | 10–15 |
| tagged union, velocity | | ≈ 3.8 (measured on the mix's kinds) | 1.85 |
| three plans at ppl = 600 (§3.5, scaled) | | 4.1× the (stokeslet, doublet) tree | ≈ 5 |
| three plans, small kinds at ppl = 25 (§3.5, scaled) | | 1.8× the (stokeslet, doublet) tree, far field 3× | ≈ 2.3 + 0.6 |

The far field of the harmonic method at `p = 8` is 0.14–0.24 µs per point (§3.4),
0.2–0.3 s at 1.2 × 10⁶ points, paid once by routes 1, c and d and once per kind
by route b (the periphery's and the objects' passes are over sparser trees, so
nearer 1.5× than 3×); the cube method's far field is 2 µs per point and would
make route b expensive. The tree builds in 1.6 s cold / 0.08 s warm for the
1.2 × 10⁶ points and is per geometry capture, not per matvec.

So, per matvec on this machine: the grouped tree ≈ 1.9 s; zero padding ≈ 2.8 s
on velocities but 10–15 s once the objects need their tractions; today's bodies
make everything 3–6× worse before any route is chosen. On the device the
multipliers are 2.2× (velocity) and 1.9× (gradient) for zero padding, §3.4.

## 5. The alternatives

### 5a. Zero padding (route 1, the FMM3D way)

Every source carries `(stokeslet, stresslet, doublet)` with zeros where absent
(JellySim's periphery stresslet needs a non-zero `ν` even where `f = 0`; the
doublet's pressure compiles away, nothing else does); every target gets every
output by three `evaluate` calls or a combined output.

- Changes: none in the core. `stokes_densities` already builds the tuple; a
  front end over `StokesFMMPlan` with all bodies concatenated runs today
  (after the device S2M tuple fix for the harmonic method on Metal).
- Cost: §4, 2.0× on the velocity sweep after the fixes, 6–10× with the outputs,
  memory 12 floats per source; on the device 2.2×/1.9×.
- API: today's. JellySim's three kinds map onto one concatenated source set with
  three padded arrays, and the objects' tractions onto `Gradient()` and
  `Pressure()` evaluations at all points (or at the objects' points through
  `evaluate`, which costs the ideal for the outputs; only the padding on the
  sources is unavoidable here).
- Verdict: the right thing for the first end-to-end run (it exists), not the
  design.

### 5b. One tree, one pass per kind

For kind k: an upward pass from kind k's sources (the others contribute zero),
a downward pass, a near-field sweep over kind k's sources, accumulated into the
outputs. The near field needs kind k's sources *and only those* per colleague,
so it needs the per-node per-kind ranges of 5c (or a mask per pair, which is
5d's branch without its benefit). Given those ranges, 5b is 5c with the far
field, `l2t` and the colleague walk repeated K times, with `has_source` per
kind to prune the lists (a per-kind `InteractionList`, or a per-kind
multipole-nonzero flag read by the gather). The only thing it buys is that the
S2M and evaluation bodies stay single-typed, which 5c also achieves by unrolling
over a tuple of typed vectors. Dominated by 5c; its separate-tree cousin
(three plans) is measured in §3.5 and loses on the near field, not the far
field.

### 5c. Groups within one tree (recommended)

**The layout.** The input is a tuple of point groups, concatenated for the
tree. The tree's stable sorts (`_sort_range!`: serial and chunk-parallel are
both stable, `split_nodes!` is called by refinement, restriction and ghosting
alike) leave the points of every *childless* node in input order, hence
group-contiguous; a node with children holds its children's ranges in slot
order, so its own range is child-major and not group-contiguous (552 of 7945
nodes on a 1.9 × 10⁵-point mixed tree). But in *group-local numbering*, the
sorted order restricted to group g, every node's group-g points are one
contiguous range, because the node's global range is contiguous and the
restriction of a contiguous range to a subsequence is contiguous. Checked on
that tree for every node and group. So no sort key changes: after `flatten`,
`gpoints[g, node] = (pre[g, lo-1]+1):pre[g, hi]` with `pre[g, j]` the count of
group-g points among sorted positions `1:j` (or recursively from the children's
ranges, which avoids the `G × N` table), `gperm[g]` the original indices of
group g's points in sorted order, `gpoint_node[g]` likewise; with `G = 1` these
are `points`, `perm`, `point_node`. On the device the same: the device tree
sorts by the full Morton key (so within a leaf the groups interleave), the
group id gathered by `perm`, one exclusive scan per group (`_exclusive_scan!`
exists), `gplo/gphi` as `G × nnodes` Int32 matrices from the scans at `plo-1`
and `phi`, `gperm[g]` by compaction. Nothing about the structure, the
colleagues, the lists or `locate` changes.

**The state.** `FMMState` holds `sources::NTuple{G,Vector{P}}` and
`τ::Tuple{Vector{T₁},…,Vector{T_G}}`, each in its group's numbering (gathered by
`gperm[g]`); the expansions are shared (one column per node) with the eltype
promoted over all groups' `density_scalar`. A target-only group has
`τ[g] = nothing`.

**The loops.** Every loop over a node's points becomes a loop over groups whose
body is today's typed body:

- `_s2m_level!` (`fmm.jl`): zero the column, then per group
  `s2m_accumulate!(col, m, ops, view(sources[g], r_g), view(τ[g], r_g), c)`,
  which is today's `s2m!` with its zeroing factored out (three methods:
  `ki_common.jl`, `harmonics.jl`/`harmonics_stokes.jl`, `bbfmm.jl`); `nothing`
  densities skip by dispatch.
- `evaluate_target` (`evaluate.jl`): per colleague, per group,
  `_direct_range(K, sources[g], τ[g], x, gpoints[g, c], skip_g, acc, out)` with
  `skip_g` the target's local index in its own group and 0 elsewhere; unrolled
  over the tuple (a recursive helper on `Base.tail`, so each group's loop is
  the vectorised one-type loop). `evaluate_at_sources!(out, f, st, g; output)`
  evaluates group g at its own points; `evaluate!` at arbitrary points is
  unchanged in form. `_output_eltype` promotes over the groups.
- The device kernels (`_evaluate_kernel!`, `_evaluate_harmonic_kernel!`,
  `_s2m_kernel!`, `_s2m_harmonic_kernel!`): `sources` and `τ` as tuples of
  device vectors (KernelAbstractions adapts tuples field by field), `gplo`,
  `gphi`; the same unrolled group loop, uniform across the work-items of a
  leaf, so no divergence. `evaluate_at_sources!` launches per target group with
  `Val(g)` and the group's local `skip`; `_sorted_inputs`/`_gather_input!` gather
  per group by `dev.gperm[g]` into one buffer per `(name, g)`.
- `build_expansions!` takes tuples of point groups and density groups;
  `FMM(method, groups::Tuple; …)` and `Tree(groups::Tuple; …)` record the
  boundaries. The one-vector forms stay as `G = 1`.

**Outputs per target group.** Separate `evaluate_at_sources!(…, g; output)`
calls per group (the sweeps are over disjoint target sets, so the total is the
ideal), with a combined `VelocityStress()`-type output for the bodies that
want tractions (§6).

**Costs.** The near field is the ideal of §4 by construction; the far field once;
the loop overhead is `G` range reads per colleague (`27G` per target, nothing
against 5000 pairs); the memory is `G × nnodes` ranges and the per-group
permutations. Mixed leaves (18% of the points at `ppl = 600`) cost nothing
extra: a group's loop over a colleague is the same loop whether or not other
groups share the box.

**Determinism and `G = 1`.** With one group the sorted order, the loops and the
accumulation order are today's, so results must stay bitwise identical (a test);
with several groups the near field accumulates group by group instead of in
sorted order, a rounding-level difference against a one-type run, and bitwise
reproducible across thread counts as today (per-target work, fixed order).

**JellySim's three kinds** map onto three groups, §6; the objects' tractions
onto the combined output at the objects' group, §6.

**Files**: `src/tree/tree.jl` (fields), `src/tree/builder.jl` (`flatten`:
the per-group ranges and permutations), `src/tree/device_builder.jl`
(`DeviceTree` fields, the scans in `_flatten`/`DeviceTree(tree, to)`),
`src/fmm.jl` (`FMMState`, `_s2m_level!`, `build_expansions!`, `FMM` on tuples),
`src/evaluate.jl` (`evaluate_target`, `evaluate_at_sources!` per group,
`_output_eltype`), the three `s2m!` files (the accumulate split),
`src/device.jl` and `src/device_passes.jl` (tuple arguments, per-group gathers,
per-group launches), `src/stokes_api.jl` (bodies), `docs/tree.md`,
`docs/framework.md` "Sources and targets", `docs/stokes.md`. `src/sources.jl`
needs nothing for the grouping itself (the fixes of §3.1 and the combined
output are separate).

### 5d. A sum type per element

One tagged struct per source (`tag`, `f`, `q`, `ν`, `g`: 12 floats and a byte)
with a branch per flag per pair. Measured (§3.1): 1.6–1.9× the typed loops on
the CPU because the loop no longer vectorises across sources, independent of
whether the kinds are sorted; on the device a `switch` per pair diverges only at
group boundaries once the leaves are sorted by kind, which 5c's layout gives
anyway, and the padded struct costs the bandwidth of zero padding. Device
arrays of `Union` eltype are not an option (a fixed-layout struct is required),
and the S2M bodies and `expansion_eltype` would dispatch on the tag at runtime.
One benefit: a single `τ` array type, so the kernels compile once per plan
rather than per group tuple. Dominated by 5c on cost and by zero padding on
simplicity.

### 5e. What else the measurements point at

1. **The three fixes** (§3.1, §3.4): stresslet `ir^5`, closed-form stresslet and
   doublet gradients (2D too, by the same algebra), the recursive tuple
   unroll in the four `for τi in τ` loops. Each is a few lines; together they
   are a 4–6× on the costs every route shares, and the device S2M fix is a
   correctness prerequisite for any tuple density on the harmonic device path.
2. **A combined output.** `VelocityStress()` (name open) with `target_field`
   returning a `StokesField{T}` (`u::SVector{3}`, `∇u::SMatrix{3,3}`, `p::T`;
   `+`, `zero`) hand-written for the three singularities, and `l2t` as one fold
   for the harmonic method (`_stokes_fold1` and `_stokes_fold2` share the
   recurrence) and one loop over the large surface for the cube. One sweep and
   one kernel launch instead of three, 15% on the pair cost.
3. **Group-local numbering** instead of a second sort key (5c): no change to the
   builders, the device tree's `sortperm!` stays one call, and the layout is a
   post-processing of `perm`.
4. **Active bodies per call.** `stokes_fmm_eval!` with `nothing` for a body's
   densities makes it target-only for that call (its S2M and near-field loops
   skip at runtime, its multipoles are zero and its M2L pairs wasted but
   correct). This serves JellySim's block-eliminated periphery (two calls per
   matvec with different source sets, one tree and one plan) without a second
   plan; pruning the lists per call can wait.

## 6. Recommendation

Route 5c, with the API below; the Stokes plan evolves rather than being
replaced: a `StokesFMMPlan` over one body is today's plan, `targets=` is a body
with `density = Nothing`, and `stokes_fmm_eval!`'s keyword form stays for one
body.

### 6.1 API

```julia
# a source/target object: points, the density type of the group (from the flags;
# `Nothing` for a target-only body), the output wanted on its points
struct StokesBody{D,T,Dens,Out}
    points::Vector{SVector{D,T}}
    density::Type{Dens}      # SVector{3,T}, Stresslet{3,T}, Tuple{SVector{3,T},Doublet{3,T}}, …, Nothing
    output::Out              # Velocity(), Gradient(), Pressure(), VelocityStress(), or nothing
end
StokesBody(points; stokeslets=false, stresslets=false, doublets=false, output=Velocity())

# the plan: one tree of all bodies' points (groups), one state; `bodies` fixes the
# group types for the run, so the state's type is fixed as today
StokesFMMPlan(bodies::Tuple{Vararg{StokesBody}}; method=:harmonics, p=8, ppl=600, bbox=nothing,
              precision=Float64, cache=nothing, tol=0)

# per matvec: one named tuple of density arrays per body (the keys the body
# declares: stokeslets, stresslets + stressvecs, doublets; `nothing` for a body that
# is target-only in this call), optional weights per body, one output array per
# body (`nothing` for a pure source); self terms skipped, everything else summed
stokes_fmm_eval!(outs::Tuple, plan, densities::Tuple; weights=nothing)

# the pieces underneath, generic (any kernel, any method)
Tree(groups::Tuple{Vararg{AbstractVector{<:SVector}}}; ppl, bbox)      # gpoints, gperm, gpoint_node
FMM(method, groups::Tuple; ppl, bbox, cache)
build_expansions!(state, f, groups::Tuple, τs::Tuple)                   # τs[g]::Vector or nothing
evaluate_at_sources!(out, f, state, g; output)                          # group g at its own points
evaluate!(out, f, state, targets; output)                               # unchanged
VelocityStress()                                                        # (u, ∇u, p) in one pass
```

JellySim's bodies:

| body | `StokesBody` | per matvec | output |
|---|---|---|---|
| periphery | `StokesBody(nodes; stresslets=true)` | `(stresslets=σₚ, stressvecs=-normals)`, `weights` | `Velocity()` |
| fibres | `StokesBody(nodes; stokeslets=true, doublets=include_doublet)` | `(stokeslets=f, doublets=(ε²/2)f)`, `weights` | `Velocity()` |
| mobile objects | `StokesBody(nodes; stokeslets=true, stresslets=true, output=VelocityStress())` | `(stokeslets=σ, stresslets=σ, stressvecs=-normals)`, `weights` | `(u, ∇u, p)` |

Tractions: with `G[a, j] = ∂u_a/∂x_j` and `p` from the FMM on densities that
carry `1/μ` (JellySim's `vc`), the FMM's pressure is `p/μ` and
`t = σ·n = μ(−p̃ n + (G + Gᵀ) n)`, formed by JellySim from `StokesField` and its
own normals (the normal is per-target data, which an output type cannot carry;
a `traction(field, n)` helper is a one-liner if wanted). What `target_field`
provides today for this: `Gradient()` as an `SMatrix` (hand-written stokeslet,
ForwardDiff otherwise) and `Pressure()` (hand-written), separately; the
combined output and the closed-form gradients are the additions. The
on-surface `∇u` and `p` of a body's own layer are singular; the FMM gives the
smooth-quadrature part, as it gives the velocity today, and the singular
correction is JellySim's (§7).

### 6.2 Stages

1. **Fixes, now** (`src/sources.jl`, `src/operators/harmonics*.jl`,
   `src/operators/bbfmm.jl`): `ir^5` as products; closed-form `Gradient()` for
   `Stresslet{3}` and `Doublet{3}` (and 2D), the ForwardDiff ones kept as the
   test oracle (the pattern of `test/test_stokes.jl`); the recursive tuple
   unroll in the four accumulate loops, with a device test on a tuple density
   for the harmonic method; `VelocityStress()` with its `l2t` for the
   kernel-independent and harmonic methods. Each measurable on its own.
2. **The grouped tree and state, CPU** (`src/tree`, `src/fmm.jl`,
   `src/evaluate.jl`, the `s2m!` split): `G = 1` bitwise identical to today;
   mixed groups against direct sums per group and output and against three
   plans; the tree invariants per group.
3. **The bodies API** (`src/stokes_api.jl`) over stage 2: `StokesBody`,
   `StokesFMMPlan(bodies)`, `stokes_fmm_eval!(outs, plan, densities)`, the
   one-body forms kept; `docs/stokes.md`. JellySim integrates here. (It could
   be written over three plans before stage 2, but §3.5 shows that costs the
   near field 4.1× unless the small kinds' trees are refined by hand, and
   then the far field 3×; not worth a throwaway.)
4. **The device** (`src/tree/device_builder.jl`, `src/device.jl`,
   `src/device_passes.jl`): the per-group scans in the device tree, tuple
   arguments in the four kernels, per-group gathers and launches; the CPU
   backend of KernelAbstractions reproduces the threaded path as today.
5. **Can wait**: per-call pruning of the lists for inactive bodies; rotlets in the
   harmonic method; `Nothing`-density bodies refining the tree (a flag, since a
   dense target set should refine it and a sparse one need not).

### 6.3 Tests that protect it

- `G = 1`: `Tree`, `FMMState`, `build_expansions!`, `evaluate_at_sources`,
  `evaluate`, `device_matvec!` bitwise equal to the current results on the
  existing suites (the suites run unchanged through the one-vector forms).
- Tree invariants per group (`test/test_tree.jl`, `test/test_3d.jl`): for every
  node and group, `gpoints[g, node]` partitions over the children in slot
  order, over childless nodes over the group, `gperm[g]` is a permutation of
  the group, `gpoint_node[g][k]` is the leaf `locate` returns; the chunked sort
  path forced on small inputs (`_PARALLEL_SORT_MIN`) gives the same ranges; the
  device tree's ranges equal the host's up to the within-leaf order.
- Mixed groups end to end (3D, harmonic and cube): three groups with the
  JellySim kinds, each output per group against `direct`/`direct_self` over all
  groups (the existing comparison, per group), and against three plans to
  rounding; Float32; dual positions in one group (the slender-body pattern of
  `test_harmonics_stokes.jl`); a `Nothing` group; allocation-free bodies per
  group (`@allocated` on `evaluate_target` with a tuple state).
- The fixes: closed-form gradients against ForwardDiff (`1e-13`), the stresslet
  body before and after on random pairs (bitwise differs; relative `1e-15`), the
  combined output against its three parts, the Stokes residual check of
  `test_stokes.jl` on the combined output, the harmonic device S2M on tuple
  densities against the CPU.
- Device: the grouped kernels against the CPU on the CPU backend and on Metal
  (`test_device*.jl` pattern), per-group launches with the self term skipped
  in the right group only.
- The bodies API: `StokesFMMPlan` over one body equals today's plan; the
  JellySim mapping table above as a test case with weights and `-normals`.

## 7. Open questions for the JellySim side

1. **One call or two per matvec.** With `σₚ` block-eliminated through the dense
   `𝒟⁻¹`, a matvec is fibres+objects → everyone, then periphery → fibres+objects:
   one plan, two `stokes_fmm_eval!` calls with different active bodies (5e.4).
   At 10⁵ periphery points the dense `𝒟⁻¹` (`3·10⁵` square) is out of reach, so
   the periphery density would become an unknown and the matvec one call with
   every body a source; the API serves both, the cost model assumed the latter.
2. **Tractions.** Which formulation needs `σ·n` on the mobile objects (today's
   examples take the force from the single-layer density and `MobileBody` uses
   analytic drag)? If a traction is needed, its on-surface singular part needs a
   corrected quadrature for the stress kernels, as `𝒟` is corrected for the
   velocity today; and the harmonic method loses a factor `≈ p` per derivative
   (`docs/harmonics.md`: stresslets at `p = 8` are `2e-2` on random boxes), so
   gradients at the objects may want `p = 12–16` or the cube method on their
   geometry; to be measured on it.
3. **Conventions.** Confirm the stresslet pressure
   `(1/2π)[(f·ν)/r³ − 3(d·f)(d·ν)/r⁵]` against JellySim's if it ever forms a
   pressure, the `μ` placement in `t = μ(−p̃ n + (G + Gᵀ) n)`, and that the
   doublet has no pressure.
4. **Precision on the device.** JellySim passes `Float32` to Metal; the
   harmonic Stokes method in Float32 is at the summation floor (`docs/framework.md`),
   fine for velocities; gradients in Float32 have not been measured.
5. **Geometry cadence.** The tree (0.08 s warm, 1.6 s cold at 1.2 × 10⁶ points)
   and the device plan are per `HydroRefresh!`; the fibre nodes move every
   Newton iteration when `update_hydro_solver` is set, so the plan build sits
   inside the step; acceptable against a 2 s matvec, to be watched on the GPU
   where the matvec is 20× faster.
6. **Weights and `-normals`**: kept as per-body inputs per call (JellySim passes
   them separately and the periphery's are constant per capture); folding them
   into the densities on the JellySim side would save one multiply per source,
   not worth a convention change.
