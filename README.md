# FlexibleFMM

A pure-Julia fast multipole method framework in two and three dimensions. One
driver runs several expansion schemes on an adaptive quadtree or octree whose
ghost-cell construction makes every far-field interaction a same-width
multipole-to-local translation and every near-field interaction a same-level
colleague sum. Three schemes are provided:

- `KernelIndependent` — the kernel-independent FMM of Ying, Biros and Zorin
  (equivalent densities on circles); needs only kernel evaluations, for kernels that
  are Green's functions of an elliptic PDE (Laplace, Helmholtz, Yukawa, …).
- `CubeKernelIndependent{D}` (`SquareKernelIndependent` in 2D) — the same method on the
  surface of a square or cube (`p` points per edge, `2p` on the larger surface, margin
  `d`), with translation operators applied by FFT; the method for 3D.
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

K  = Laplace2D()
m  = KernelIndependent(K; nequiv=60)   # or SquareKernelIndependent(K; p=12), BlackBox(K; p=12)
f  = FMM(m, sources; ppl=50, bbox=(SVector(0.0, 0.0), 1.0))
st = build_expansions(f, sources, τ)                # multipoles and locals
u_sources = evaluate_at_sources(f, st)              # Σ_{j≠i} K(s_j, s_i) τ_j
u_targets = evaluate(f, st, targets)                # Σ_j K(s_j, t) τ_j
```

Densities say what a source is: a number is a charge, `Dipole(d)` a double-layer
source with moment `d` (`Dipole(μ, n)` for strength and normal), and a tuple of
densities sums its parts. `evaluate(f, st, targets; output=Gradient())` returns
`∇u` as `SVector`s. Dual-number positions, densities and targets go through
unchanged (directional and shape derivatives).

Stokes: `Stokeslet2D()` and `Stokeslet3D()` (JellySim's `1/8π` conventions) with
stokeslet (`SVector`), `Stresslet(f, ν)`, `Doublet(g)` and `Rotlet(ω)` densities,
velocity, pressure and velocity-gradient outputs, and a plan API in the shape of
JellySim's kernel backends (`StokesFMMPlan`, `stokes_fmm_eval!`, `StokesFMMKernel`);
see `docs/stokes.md`.

Kernels are callables `K(source, target)` on `SVector`s; `Laplace2D()`,
`Helmholtz2D(k)`, `Yukawa2D(k)` and their 3D versions `Laplace3D()`, `Helmholtz3D(k)`,
`Yukawa3D(k)` (`1/4πr` normalisation) are provided and any translation-invariant
function of the same shape works. In 3D use `CubeKernelIndependent{3}(K; p=8)` with
`SVector{3}` points. `build_expansions!` reuses a state for new
densities; an `OperatorCache` passed to several `FMM`s shares their translation
operators (root boxes are snapped to cached widths). Targets must lie inside the
root box.

Precision follows the kernel: `Laplace2D(Float32)` or `Helmholtz2D(10f0)` with Float32
points gives a single-precision tree, operators, expansions and output.

Accuracy: the kernel-independent method with `nequiv` points per circle has a
worst-case translation error of about `0.547^nequiv` (`1e-8` at 32, `2e-12` at 48,
rounding floor `~1e-14` near 64) and global relative errors one to two orders
smaller (10 digits at 32 points on uniform data); see `docs/kifmm.md`, also for the
two known degeneracies (a Laplace box width of `≈ 0.7735` and Helmholtz resonances
at `k·w ≈ 3.40, 5.42, …`). The black-box method converges geometrically in `p`. In
3D the cube method at `p = 8` has a worst-case translation error of `1e-6` for
Laplace (global errors about `1e-7`) and `3e-6`, `4e-5`, `1e-3` for stokeslets,
stresslets and doublets; `p = 10` gains a further 30× (`docs/kifmm.md`).

## Layout

- `src/tree/` — the ghost-cell quadtree (`docs/tree.md`)
- `src/operators/` — the method interface, the operator cache, and the methods
  (`docs/framework.md`, `docs/kifmm.md`, `docs/bbfmm.md`)
- `src/fmm.jl`, `src/evaluate.jl` — the driver
- `test/` — run with `julia --project=. -t auto test/runtests.jl` (or `Pkg.test()`);
  includes a kernel-free combinatorial check of the tree's interaction structure,
  operator checks on all 40 far offsets, direct-sum and analytic (ring) comparisons,
  determinism, allocation and ForwardDiff tests
- `bench/bench.jl` — timings; `julia --project=. -t auto bench/bench.jl N=1000000`
  (kernel, method, distribution and `ppl` options in its header)

This is a port of the Python/numba `flexmm` package; the design decisions that
differ from it are recorded in the docs.
