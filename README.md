# FlexibleFMM

A pure-Julia fast multipole method framework in two dimensions. One driver runs
several expansion schemes on an adaptive quadtree whose ghost-cell construction makes
every far-field interaction a same-width multipole-to-local translation and every
near-field interaction a same-level colleague sum. Two schemes are provided:

- `KernelIndependent` — the kernel-independent FMM of Ying, Biros and Zorin
  (equivalent densities on circles); needs only kernel evaluations, for kernels that
  are Green's functions of an elliptic PDE (Laplace, Helmholtz, Yukawa, …).
- `BlackBox` — the Chebyshev interpolation FMM of Fong and Darve; needs only kernel
  evaluations, for any smooth translation-invariant kernel.

The code is generic over element types: complex kernels and densities, and
`ForwardDiff.Dual` positions (directional/shape derivatives) go through unchanged.

## Usage

```julia
using FlexibleFMM, StaticArrays

sources = [SVector(rand(), rand()) for _ in 1:100_000]
τ       = randn(100_000)
targets = [SVector(rand(), rand()) for _ in 1:50_000]

m  = KernelIndependent(Laplace2D(); nequiv=60)      # or BlackBox(Laplace2D(); p=12)
f  = FMM(m, sources; ppl=50, bbox=(SVector(0.0, 0.0), 1.0))
st = build_expansions(f, sources, τ)                # multipoles and locals
u_sources = evaluate_at_sources(f, st)              # Σ_{j≠i} K(s_j, s_i) τ_j
u_targets = evaluate(f, st, targets)                # Σ_j K(s_j, t) τ_j
```

Kernels are callables `K(source, target)` on `SVector{2}`s; `Laplace2D()`,
`Helmholtz2D(k)` and `Yukawa2D(k)` are provided and any translation-invariant
function of the same shape works. `build_expansions!` reuses a state for new
densities; an `OperatorCache` passed to several `FMM`s shares their translation
operators (root boxes are snapped to cached widths). Targets must lie inside the
root box.

Accuracy: the kernel-independent method with `nequiv` points is accurate to about
`0.547^(nequiv/2)` (`1e-8` at 60, `1e-13` at 100); see `docs/kifmm.md` for why,
and for the two known degeneracies (a Laplace box width of `≈ 0.7735` and Helmholtz
resonances at `k·w ≈ 3.40, 5.42, …`). The black-box method converges geometrically
in `p`.

## Layout

- `src/tree/` — the ghost-cell quadtree (`docs/tree.md`)
- `src/operators/` — the method interface, the operator cache, and the two methods
  (`docs/framework.md`, `docs/kifmm.md`, `docs/bbfmm.md`)
- `src/fmm.jl`, `src/evaluate.jl` — the driver
- `test/` — run with `julia --project=. -t auto test/runtests.jl` (or `Pkg.test()`);
  includes a kernel-free combinatorial check of the tree's interaction structure,
  operator checks on all 40 far offsets, direct-sum and analytic (ring) comparisons,
  determinism, allocation and ForwardDiff tests
- `bench/bench.jl` — timings; `julia --project=. -t auto bench/bench.jl 1000000`

This is a port of the Python/numba `flexmm` package; the design decisions that
differ from it are recorded in the docs.
