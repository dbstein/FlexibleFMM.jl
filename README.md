# FlexibleFMM

A pure-Julia fast multipole method framework in two and three dimensions. One
driver runs several expansion schemes on an adaptive quadtree or octree whose
ghost-cell construction makes every far-field interaction a same-width
multipole-to-local translation and every near-field interaction a same-level
colleague sum. Four schemes are provided:

- `KernelIndependent` — the kernel-independent FMM of Ying, Biros and Zorin
  (equivalent densities on circles); needs only kernel evaluations, for kernels that
  are Green's functions of an elliptic PDE (Laplace, Helmholtz, Yukawa, …).
- `CubeKernelIndependent{D}` (`SquareKernelIndependent` in 2D) — the same method on the
  surface of a square or cube (`p` points per edge, `2p` on the larger surface, margin
  `d`), with translation operators applied by FFT; the method for 3D.
- `BlackBox` — the Chebyshev interpolation FMM of Fong and Darve; needs only kernel
  evaluations, for any smooth translation-invariant kernel.
- `SphericalHarmonics` — the classical analytic multipole method for `Laplace3D`
  and `Yukawa3D`: real solid-harmonic expansions of degree `p` (with the modified
  spherical Bessel factors for Yukawa), exact M2M and L2L for Laplace and
  operators by projection per width for Yukawa, one M2L operator per symmetry
  class of the far offsets (34), dense or a truncated SVD (`tol`); charges and
  dipoles, potential and gradient; `StokesHarmonics` carries the 3D Stokeslet on
  four such expansions (`docs/harmonics.md`).

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

Kernels are callables `K(source, target)` on `SVector`s (a plain function works;
it takes the skipping self sum unless it declares `masks_self`, and the device
classes unless it declares `covariant`); `Laplace2D()`,
`Helmholtz2D(k)`, `Yukawa2D(k)` and their 3D versions `Laplace3D()`, `Helmholtz3D(k)`,
`Yukawa3D(k)` (`1/4πr` normalisation) are provided and any translation-invariant
function of the same shape works. In 3D use `CubeKernelIndependent{3}(K; p=8)` with
`SVector{3}` points. `build_expansions!` reuses a state for new
densities; an `OperatorCache` passed to several `FMM`s shares their translation
operators (root boxes are snapped to cached widths). Targets must lie inside the
root box.

Precision follows the kernel: `Laplace2D(Float32)` or `Helmholtz2D(10f0)` with Float32
points gives a single-precision tree, operators, expansions and output.
`fastmath(K)` switches the 3D Laplace and Stokes kernels to relaxed arithmetic
(the `FastMath` policy: 1.2× on the near field, deviations of 3e-7 in Float32;
`fast=true` in the bench scripts and on the device plan).

With OpenBLAS (Julia's default) start Julia with `OPENBLAS_THREAD_TIMEOUT=4` in the
environment: after each BLAS call its helper threads otherwise spin for ~0.1 s
against the package's Julia-threaded phases, which costs the 3D matvec 1.3–1.7×
(`docs/kifmm.md`); the variable makes them sleep at once and is read when OpenBLAS
starts, so it cannot be set from inside Julia. On Apple silicon, forwarding BLAS
to Accelerate instead (`BLAS.lbt_forward` on the Accelerate framework with the
ILP64 suffix) gives the same speed. On a node where a thread region costs
milliseconds (`bench/threads.jl` measures it; 30 ms on a 96-core Genoa node,
40 µs on a 64-core Icelake node) try `JULIA_THREAD_SLEEP_THRESHOLD=infinite
OPENBLAS_NUM_THREADS=1`: the Julia threads keep spinning between regions and the
large products are split over them (`docs/framework.md`, Threading). Measure it:
on the Icelake node, where regions are cheap, that configuration is 1.5× slower.

The whole matvec runs on GPUs through KernelAbstractions: `DevicePlan(f, MtlArray)`
(or `CuArray`) moves the operators once and builds the interaction lists on the
device, and `DevicePlan(m, xs; ppl, bbox)` with the points already on the device
builds the tree there too (`device_tree`, the same structure as the host tree);
`device_matvec!` takes sources and densities in their original order and runs the
expansion passes (dense translations by the symmetry classes of the far offsets)
and the self evaluation on the device, and the device methods of `evaluate!` and
`evaluate_at_sources!` take device arrays; backends are not dependencies. The
kernel-independent and the harmonic methods run there. On an M3 Max in Float32,
10⁵ Stokes points take 0.13 s per matvec with the cube method (7× the 12-thread
CPU time in the same precision); see `docs/framework.md`, `docs/harmonics.md` and
`bench/gpu/`.

Accuracy: the kernel-independent method with `nequiv` points per circle has a
worst-case translation error of about `0.547^nequiv` (`1e-8` at 32, `2e-12` at 48,
rounding floor `~1e-14` near 64) and global relative errors one to two orders
smaller (10 digits at 32 points on uniform data); see `docs/kifmm.md`, also for the
two known degeneracies (a Laplace box width of `≈ 0.7735` and Helmholtz resonances
at `k·w ≈ 3.40, 5.42, …`) and for timings against FMM2D/FMM3D (2D: 2–4× faster
than FMM2D; 3D: the cube method 2.5× faster than FMM3D for Laplace and level for
Helmholtz and Stokes on uniform points, the harmonic methods 2× faster for Stokes
and 3–7× on fibres). The
black-box method converges geometrically in `p`. In
3D the cube method at `p = 8` has a worst-case translation error of `1e-6` for
Laplace (global errors about `1e-7`) and `3e-6`, `1e-3`, `1e-3` for stokeslets,
stresslets and doublets on the corner configuration (sampled random boxes give
`4e-5` for stresslets); `p = 10` gains a further 30× (`docs/kifmm.md`). The
analytic method reaches `3e-6`, `3e-7`, `2e-8` global error at `p = 12, 16, 20` on
uniform data, with a worst case (corner source facing a corner target) converging
only as `0.7^p`; on fibre geometries the harmonic methods are 4–5× faster than the
cube method at equal accuracy, level on uniform data (`docs/harmonics.md`).

## Layout

- `src/tree/` — the ghost-cell quadtree (`docs/tree.md`)
- `src/operators/` — the method interface, the operator cache, and the methods
  (`docs/framework.md`, `docs/kifmm.md`, `docs/bbfmm.md`, `docs/harmonics.md`)
- `src/fmm.jl`, `src/evaluate.jl` — the driver
- `test/` — run with `julia --project=. -t auto test/runtests.jl` (or `Pkg.test()`);
  includes a kernel-free combinatorial check of the tree's interaction structure,
  operator checks on all 40 far offsets, direct-sum and analytic (ring) comparisons,
  determinism, allocation and ForwardDiff tests
- `bench/bench.jl` — timings; `julia --project=. -t auto bench/bench.jl N=1000000`
  (kernel, method, distribution and `ppl` options in its header; `cores=` for the
  per-core rates, half the hardware threads by default); with
  `device=metal` from the `bench/gpu` environment (`julia --project=bench/gpu -t auto
  bench/bench.jl dim=3 device=metal …`) it adds the device matvec per case;
  `device=ka_cpu` runs the same passes on KernelAbstractions' CPU backend (Julia
  threads under KernelAbstractions 0.9, the POCL OpenCL runtime under 0.10, where
  `POCL_DEBUG=all` explains a missing device);
  `device=cuda precision=float32` from the `bench/cuda` environment (instantiate
  it on the machine) on an NVIDIA card
- `bench/threads.jl` — threading diagnostics for a machine (the cost of a thread
  region, of compute and allocating tasks, of the class translations' small GEMMs,
  then the phases of one case), for reading cluster nodes whose matvec is far
  slower than their cores allow
- `bench/compare/` — timing and accuracy against the Flatiron libraries through
  FMM2D.jl and FMM3D.jl, in its own environment so they are not dependencies; the
  same `kernel`, `method`, `dist`, `ppl` and `device` options as the benchmark plus
  `eps` for the libraries (`julia --project=bench/compare -t auto
  bench/compare/compare.jl dim=3 N=100000 kernel=laplace,stokes method=cube:6,cube:8
  eps=1e-3,1e-6 device=metal`; the specifications are listed in `bench/common.jl`)
- `bench/gpu/` — the device evaluation and matvec on Apple Silicon through
  Metal.jl, in its own environment (`julia --project=bench/gpu -t auto
  bench/gpu/metal_matvec.jl`)

This is a port of the Python/numba `flexmm` package; the design decisions that
differ from it are recorded in the docs. MIT license (`LICENSE`).
