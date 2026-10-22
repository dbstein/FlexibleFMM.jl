# The operator framework

One driver (`src/fmm.jl`, `src/evaluate.jl`) runs any expansion scheme that provides
five operations per box width:

| operation | signature | meaning |
|---|---|---|
| `s2m!` | `(col, m, ops, xs, τs, c)` | sources of a childless node → its *partial* multipole |
| `M2M` | `Nm × 2^D·Nm` matrix | stacked children multipoles → parent multipole |
| `M2L[f]` | `Nexp × Nexp` matrix per far offset | source multipole at offset `f` → target partial local |
| `L2L` | `2^D·pll × pll` matrix | parent partial local → stacked children partial locals |
| `l2t` | `(m, ops, L, xrel)` | local column → potential at a point |

plus two per-level transforms, `partial_to_multipole!` and `partial_to_local!`,
that turn the accumulated partial expansions of a whole level into actual
expansions. The driver applies `M2M` and `L2L` through `m2m_apply!` and
`l2l_apply!`, matrix products by default; a method whose operators have block
structure overrides them (`StokesHarmonics` folds its coupling into the inputs
and runs four `N × N` GEMMs per child instead of a dense `4N × 32N` product). For the kernel-independent method the partials are check potentials and
the transforms are pseudo-inverses; for the black-box method they are the identity.
The downward transform runs on the real leaves of a level only, gathered from the
stored partial locals: the evaluation reads a local column for the leaf holding a
point and nothing else, and the level-to-level propagation uses the partials, so
the other columns of `locals` stay zero. On a surface or a fibre most nodes are
ghosts or interior (a 10⁵-point sphere at 200 per leaf: 19 689 nodes, 5 475 leaves;
the helix: 111 009 and 35 015), which made that transform a large share of the
downward pass: restricting it took the cube method's downward stage from 0.21 to
0.18 s (Laplace) and 1.36 to 1.07 s (Stokes) on the sphere and from 1.14 to 0.89 s
and 10.5 to 8.7 s on the helix, 15–20% of the matvec on those geometries and 5% on
uniform points; on Metal the Stokes cube's stage on the helix went 81 → 38 ms.
A method also states its `expansion_length`, its `operator_eltype`, and the kernel
to use in the near field. An M2L operator is dense (`M2LOperator`) or a truncated
SVD (`LowRankM2L`, the harmonic methods' `tol`); both are applied through `apply!`,
and the driver hands the low-rank form a workspace from the state (`workspace_rows`),
so operators stay immutable and shared.

Because the tree makes every translation a same-width one, `level_operators(m, w)`
builds everything for width `w` once and an `OperatorCache` keyed by the integer
exponent `e` (width `ldexp(W0, e)`) shares it between levels and between FMM
instances; root boxes are snapped to such widths (`fit_box!`). A method states its
`dimension(m)`; the tree, lists, driver and evaluation are generic in it.

For a kernel that is homogeneous, `K(λs, λt) = λᵃ K(s, t)` (declared by
`homogeneity(K) = a`; `Laplace3D` and `Stokeslet3D` have `a = −1`, no built-in 2D
kernel qualifies), the
operators of all widths are scalar multiples of one reference set: the cache builds
the reference width only and `rescale` makes the other widths from it, sharing the
matrices and carrying per-width factors (`m2m_scale`, `l2l_scale`, `m2l_scale`, and
the inverse factor on the pseudo-inverses) that the driver applies through the α
argument of `mul!`. For this to hold with the Stokes augmentations, the pinned
pressure row and the point-source column are nondimensionalised with the box width,
which also makes the pin's least-squares weight the same on every level.

## The passes

Expansions are stored one column per node, `Matrix{S}(Nexp, nnodes)`, so the
sibling-group reshape of a level block to `(4·Nexp, ngroups)` is free and BLAS
applies `M2M`/`L2L` to a whole level in one GEMM.

Upward, levels `L..3`: zero the partial block of the level; `s2m!` for every
childless node with sources (threaded over contiguous chunks of those nodes); the
method's transform of that block into the childless nodes' multipoles; then one
GEMM `M2M · reshape(children block)` assigned to the group parents' multipoles.
Downward, levels `3..L`: one GEMM `L2L · (parents' partial locals gathered by
group)` assigned to the level block; then, for each far offset with pairs, gather
the source columns, one GEMM with `M2L[f]`, scatter-add into the targets (a target
appears at most once per offset, so this is conflict-free); the block is kept as
the level's partial locals (what the next level's `L2L` consumes) and transformed
into the densities the evaluation reads.
Levels 1 and 2 carry nothing. The pair lists are stored in CSR form per level and
offset (`InteractionList`), so each GEMM touches only the multipoles actually used
(the Python applied each operator to every multipole of the level and gathered
afterwards).

The M2L step of the downward pass goes through a level-wise hook, `m2l_level!(pL,
m, ops, Mp, il, off, state)`, whose default is the gather/GEMM/scatter above; the
square/cube kernel-independent method overrides it with the FFT scheme described
in `docs/kifmm.md`, which is the shape a GPU wants (one transform per box, one
Hadamard product per pair). Method-specific workspaces live in `state.scratch`.

Evaluation: targets are located (`locate`), counting-sorted by leaf so that
consecutive targets reuse the same local expansion and source ranges, and processed
in contiguous chunks of that order; each target gets `l2t` of its leaf's local plus
the direct sum over its leaf's colleagues' source ranges. `evaluate_at_sources` uses
the tree's own `point_node` map and excludes the source itself by index.

Why batched GEMMs with gather/scatter are still the right structure in Julia: at
`Nexp ≈ 50` a level's M2L is thousands of `50 × 50` products, which as one GEMM per
offset runs at BLAS-3 speed; and the same layout (flat arrays, index lists,
per-item bodies for `s2m!`/`l2t`/near field) is what a GPU port needs.

## Element types and differentiability

Operators are built in `Float64` or `ComplexF64` from the kernel. Expansions use
`promote_type(operator eltype, kernel output on the position eltype, density eltype)`,
so complex densities with a real kernel, or `ForwardDiff.Dual` positions, propagate
through `mul!` (generic fallback for `Complex{Dual}`), the factored pseudo-inverse
and the per-point bodies. The tree holds only indices and box geometry; positions
and densities are passed to `build_expansions!` and `evaluate`, which for a fixed
tree makes the FMM a pure generic function of `(sources, τ, targets)`. One dual
partial gives a directional derivative, e.g. a shape derivative of a layer
potential in a given velocity field, at about twice the cost; see
`test/test_forwarddiff.jl`. This is valid for infinitesimal perturbations (a source
that leaves its box also leaves its equivalent surface).

## Precision

Everything follows the precision of the kernel: the built-in kernels store their
constants in a type parameter (`Laplace2D(Float32)`, `Helmholtz2D(10f0)`), `floattype(K)`
reports it (Float64 for a user kernel unless it defines the trait), and the operator
builders make surfaces, nodes and matrices in that precision. Points of the same
precision then give a Float32 tree, Float32 expansions and Float32 output, including
with `Dual{Float32}` positions, which is the combination a Metal GPU runs. A Float64
constant anywhere would silently promote a `Dual{Float32}` to `Dual{Float64}`, so
kernels must not write bare `π`-containing literals. Two of the special functions
(`hankelh1`, `besselk`) compute in Float64 regardless; Helmholtz is written through
`besselj0`/`bessely0`, Yukawa converts the result back.

In single precision the kernel-independent check maps can only resolve modes down to
about `50·eps ≈ 6e-6`, so the default `pinv_rtol` truncates there; the error floor
of about 1e-5 is reached by `nequiv ≈ 24–32` and more points add nothing
(`test/test_float32.jl`).

## Threading

`Threads.@threads :dynamic` over contiguous chunks of leaves (`s2m!`) and of
leaf-sorted targets (`evaluate`), with the threaded body in its own function:
a variable that is assigned both inside a `@threads` body and in the enclosing
function is one shared boxed variable, and tasks race on it. Every output element is
written by exactly one task in a fixed order, so results are bitwise independent of
the Julia thread count.

The number of thread regions per matvec matters as much as the work in them on
many-core nodes. A region spawns one task per thread and wakes the sleeping
threads one by one; on a 96-core Genoa node waking a thread takes 0.3 ms, so a
region costs 30 ms (21 µs on a 12-core M3 Max; `bench/threads.jl` measures it),
and a matvec made of 340 regions took 14 s there against 0.3 s of work. The
passes are therefore organised as few regions: one per level for S2M, one per
level for the class-wise M2L of the harmonic methods (tasks take blocks of 32
consecutive targets from a counter and do their own gather, product and scatter
with per-task workspaces; the block size is fixed so that the chunking of the
products, and hence the results, do not depend on the thread count), one for the
evaluation. The products inside that region run with one BLAS thread (the region
sets the count to one and restores it: many tasks calling a threaded BLAS serialise
on its queue). The large products between regions (M2M, L2L, the pseudo-inverse
transforms) go to BLAS with its own threads by default, which is faster than
splitting them over Julia threads on the M3 Max (`docs/kifmm.md`); with
`OPENBLAS_NUM_THREADS=1` they are split over Julia threads instead
(`column_blocks`), the configuration for a node whose Julia threads are kept
spinning (`JULIA_THREAD_SLEEP_THRESHOLD=infinite`, a region then costs 3 ms on the
Genoa node) and where OpenBLAS's own threads would be starved by them. It is a
configuration to measure, not a default: on a 64-core Icelake node, where a
region costs 40 µs, it is 1.5× slower (10⁶ sphere points, Stokes p = 6: matvec
1.0 → 1.6 s, the pseudo-inverse transforms of the cube method 10× slower split
over Julia threads than under OpenBLAS's 32 threads, and the spinning threads slow
the working ones in the evaluation, 0.72 → 1.2 s). The transforms of the cube
method's spectral translation are planned per box, single-threaded and
unaligned, and run box by box in one region per transform (planned over the
whole level with FFTW's threading, each execution spawned its tasks several
times: the cube method's down pass took 48 s on the Genoa node). The per-offset M2L of the circle and non-covariant methods still takes a region
per offset. With all of this, and the near-field loop vectorised (its accumulation
reassociated, `kernels.jl`), the Genoa node runs the Stokes harmonic matvec at 10⁶
sphere points, `p = 6`, 600 per leaf, in 0.20 s with spinning threads and one BLAS
thread (up 50, down 36, self 116 ms) and 0.50 s in the default configuration,
where the sleeping threads' wake-ups (one region per level) and OpenBLAS's
per-call wake-ups are the difference; it started at 14 s.

## Sources and targets

A source is a point with a density whose *type* says what singularity sits there:
a `Number` is a charge, `Dipole(d)` (moment `d`, i.e. strength times direction) a
double-layer source with field `d·∇ₛK`, and a tuple of densities the sum of its
parts, so a combined single-plus-double layer runs as one FMM. The output type
passed to `evaluate` (`Potential()` or `Gradient()`) says what is computed at a
target. Everything the FMM needs is two functions: `source_field(K, s, t, τ)`, the
field of one source at one point, used by `s2m!`, the near field and the reference
sums, and `target_field(K, s, t, τ, out)`, the output operator applied to it, used
by the near field; `l2t` has one method per output type. The translation operators
never see a source or target, so they are untouched by any of this.

Kernel derivatives (`gradient_t`, `gradient_s = -gradient_t`, `hessian_t`) are
hand-written for the built-in kernels (through a radial-kernel helper for Helmholtz
and Yukawa) and taken by ForwardDiff on the kernel for any other, so user kernels get
dipoles and gradients at once; nested differentiation through that fallback works
when the outer derivative uses ForwardDiff's own tags. The black-box method needs,
per density type, its action on the Chebyshev basis (`accumulate_moments!`; values
for charges, basis gradients for dipoles).

Two identities fall out of this and are tested against each other
(`test/test_sources.jl`): the derivative of the potential with respect to the source
positions along a velocity field V is the dipole FMM with moments τⱼVⱼ, and the
derivative with respect to the target positions is the gradient output dotted with
the target velocity. So a Jacobian-vector product of a layer potential with respect
to its geometry can be formed either with dual positions or as one extra FMM.

Adding a singularity type (a Stokes rotlet or stresslet, say) means a struct, a
`source_field` method, an `accumulate_moments!` method for the black-box method,
and `target_field` methods for the outputs it should support.

## Where extensions plug in

- **Another expansion scheme**: a new `AbstractMethod` with the five operations and
  identity transforms; nothing in the driver changes (this is how `BlackBox` was
  added after `KernelIndependent`, and `SphericalHarmonics`, the analytic Laplace
  method of `docs/harmonics.md`, after the 3D work).
- **Vector kernels** (Brinkman, say): as Stokes is done (`docs/stokes.md`), an
  `SMatrix`-valued kernel with `SVector` densities goes through the same
  `source_field`/`target_field` hooks; kernel matrices are assembled block-wise, the
  method owns the expansion length (the Stokes augmentations), and `Pressure()` is an
  output type evaluating the pressure kernel against the equivalent density.
- **3D**: done through `CubeKernelIndependent{3}` with `Laplace3D`, `Helmholtz3D`,
  `Yukawa3D` and `Stokeslet3D`, and `SphericalHarmonics` with `Laplace3D` and
  `Yukawa3D`; `KernelIndependent` (circles) and `BlackBox` are 2D.
  What 3D still needs is memory work on the spectral path (`docs/kifmm.md`).
- **GPU** (KernelAbstractions): the evaluation runs on a device (`src/device.jl`):
  `DeviceEvaluator(fmm, to)` moves the tree data once with a user-supplied array
  mover (`identity`, `MtlArray`, `CuArray`), `device_state(state, to)` the locals,
  sources and densities per evaluation, and the device methods of `evaluate!` and
  `evaluate_at_sources!` run one work-item per target (its leaf's local expansion
  plus the colleague sums, the arithmetic of `evaluate_target`); backends are not
  dependencies, the backend is that of the output array, and the CPU backend
  reproduces the threaded evaluation in the test suite. Measured on an M3 Max in
  Float32 (`bench/gpu/metal_eval.jl`, 10⁵ sources at 200 per leaf): 10⁶ targets in
  0.074 s for Laplace and 0.20 s for Stokes against 0.38 s and 1.65 s on 12 CPU
  threads, the self evaluation at the sources in 6 and 14 ms, uploads of 4–11 ms
  per evaluation. The expansion passes run on the device too
  (`src/device_passes.jl`): `DevicePlan(fmm, to)` holds the folded M2M and L2L,
  the pseudo-inverse factors, one dense translation operator per symmetry class
  of the far offsets (`src/operators/symmetry.jl`: 16 classes in 3D for the cube
  lattices, 34 under the subgroup fixing the `z` axis for the harmonic basis,
  applied by signed permutations of the expansion entries on the source and the
  target side, verified against every offset in the tests), the pairs of each class grouped by target (the classes rest on the kernel's
  covariance under the signed axis permutations, `covariant(K)`, true for the radial
  kernels and the Stokeslets; a merely translation-invariant kernel takes one dense
  operator per offset) in chunks of at most 4096 pairs, each padded to the power of two above its own
  size (padding every chunk to 4096 cost the 34-class harmonic methods 5× and the
  cube method 4× the real gather and GEMM volume), and exact-size work buffers
  per chunk size. The lists themselves are built on the device (`src/device_lists.jl`):
  from the tree's flat arrays (cells, parents, first children, the ghost and
  has-source flags, moved once) one work-item per target counts its far pairs, a
  prefix sum places them and a fill writes them; the pairs are counted per
  (target, class), placed class-major (the class lists of `symmetry.jl`, targets in
  node order) and filled; the runs (one per target and class) come from a
  compaction of the counts and the chunks, six integers each, from the per-class
  totals with their run ranges by binary search in the run starts. The host reads
  back three counts and the chunk descriptors per level, nothing else, and the
  gather and scatter kernels read the chunk's pairs and runs through its
  descriptor. (The first version cut the chunks on the host from the by-target
  lists and uploaded four arrays per chunk, 0.97 of the 1.15 s plan build of 10⁷
  points on the M3 Max; the version before it materialised every pair as a tuple
  and sorted it per class, 12.5 s of a 10⁷-point plan on an H100.) The inputs of
  `device_matvec!` are taken in their original order and gathered into the
  tree's by `perm` on the device (the output is written in the original order
  already); `sorted=true` takes them in the tree's order, as `device_state`
  provides them. The prefix sums of the list build are KernelAbstractions kernels
  of the package (block totals, a recursive scan of them, the blocks scanned with
  their offsets): Metal's `accumulate!` and `sum(…; dims)` go through MPSGraph,
  which compiles a graph per array size, 0.2–1.2 s of every plan build since every
  tree has new level sizes. `DevicePlan(…; verbose=true)` reports the build's
  stages. On the M3 Max (Metal, Float32, sphere points, Stokes cube `p = 4`, 800 per
  leaf) the plan builds in 0.07 s at 10⁶ points and 0.22 s at 10⁷ (lists 0.02 and
  0.05 s), against 1.3 and 1.8 s with the lists on the host, and the tree in 0.06 and
  0.32 s on 12 threads; the matvec takes 0.11 and 0.91 s. `device_build_expansions!` runs
  the S2M kernel and the GEMMs upward, and downward the L2L GEMMs, per chunk a
  gather, a GEMM and a target-owned scatter, and the pseudo-inverse to the
  densities of the level's real leaves (gathered, transformed, scattered);
  `device_matvec!` adds the self evaluation. Real expansions, or
  complex ones through `DevicePlan(f, to; eltype=ComplexF64)`, which promotes the
  operators (no complex GEMM on Metal); dual numbers take the CPU passes and the
  device evaluation. The harmonic
  methods take the same plan with their own far-field data (`DeviceHarmonic`:
  the scale per level and the recurrence tables, adapted to device arrays), an
  S2M kernel and an evaluation kernel that run the CPU folds, the class
  representatives as the translation operators, four columns per pair for the
  Stokes method with the coupling in the scatter, and no pseudo-inverse stages
  (`docs/harmonics.md`). Measured on an
  M3 Max in Float32 at 10⁵ points, p = 8, 200 per leaf (`bench/gpu/metal_matvec.jl`):
  the matvec in 0.036 s for Laplace and 0.129 s for Stokes against 0.15 s and
  0.90 s on 12 CPU threads in Float32, with the CPU's accuracy against direct
  sums. `DevicePlan(f, to; tol=1e-6)` replaces the class operators by low-rank
  factors (`class_factors`: the far offsets see a smooth kernel, so the Stokes
  operators of rank 889 compress to about 40 at `1e-6`), two GEMMs per chunk
  instead of one: Stokes 0.129 → 0.086 s on Metal at the same Float32 accuracy
  (at `1e-8` the factors keep Float32 noise and the path is slower; Laplace gains
  little, its matvec being elsewhere). The plan (`docs/notes/gpu_plan.md`,
  `gpu_plan_delta.md`) has the CUDA side, which is untested here, and the
  remaining items: dual numbers through the device passes and chunking the
  operators at 10⁶ points.
- **Reverse-mode AD**: the FMM is linear in `τ`, so the pullback with respect to `τ`
  is the FMM with sources and targets swapped and the kernel transposed; position
  pullbacks need the gradient kernel in `s2m!`/`l2t`/near field.
- **Local (near-field-only) evaluators**: a different refinement rule in the builder
  and a label filter in the near-field body.
