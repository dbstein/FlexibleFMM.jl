# Stokes

The Stokes layer follows the conventions of JellySim's kernel backends, so that a
three-dimensional FlexibleFMM can be dropped in there without changes on the
JellySim side: separate arrays for `stokeslets`, `stresslets` with `stressvecs`, and
`doublets`; optional per-source `weights`; `targets = nothing` for self-evaluation
with the `i = j` term skipped; a persistent plan for repeated solves on a fixed
geometry; a combined source-and-target evaluation; and the viscosity carried by the
densities. `Stokeslet3D` is JellySim's kernel suite exactly (its `StokesletKernel`,
`StressletKernel` and `DoubletKernel` are transcribed in `test/test_3d.jl` and
compared); the 2D kernels carry the 2D analogues of the same prefactors.

3D (`d = t − s`, `r = |d|`, `c = 1/8π`):

| singularity | density | velocity | pressure |
|---|---|---|---|
| stokeslet | `SVector{3}` f | c(f/r + d(d·f)/r³) | (d·f)/(4π r³) |
| stresslet | `Stresslet(f, ν)` | −(3/4π) d (d·f)(d·ν)/r⁵ | (1/2π)[(f·ν)/r³ − 3(d·f)(d·ν)/r⁵] |
| doublet | `Doublet(g)` | c(g/r³ − 3d(d·g)/r⁵) = ½∇²G·g | 0 |
| rotlet | `Rotlet(ω)`, ω an `SVector{3}` | c ω×d/r³ = ½ ε_jkl ω_j ∂_{s_k}G_il | 0 |

2D (`c = 1/4π`):

| singularity | density | velocity | pressure |
|---|---|---|---|
| stokeslet | `SVector{2}` f | c(−log r·f + d(d·f)/r²) | 2c(d·f)/r² |
| stresslet | `Stresslet(f, ν)` | −(1/π) d (d·f)(d·ν)/r⁴ | (1/π)[(f·ν)/r² − 2(d·f)(d·ν)/r⁴] |
| doublet | `Doublet(g)` | c(g/r² − 2d(d·g)/r⁴) = ½∇²G·g | 0 |
| rotlet | `Rotlet(ω)`, ω a scalar | 2cω(−d₂, d₁)/r² = −ω ε_jk ∂_{s_k}G_ij | 0 |

Each row satisfies the Stokes equations with its pressure (checked by automatic
differentiation in the tests). In 3D the single layer of a uniform traction on a
sphere reproduces the translating sphere, `(1 + a²/6 ∇²)G·F` outside and `F/(6πa)`
inside with the stokeslet's pressure outside and none inside, and the double layer
of a constant vector with the outward normal gives that vector inside and zero
outside; both hold to `1e-15` with the direct sums and to the FMM's accuracy through
it.

The doublet is the potential dipole that appears as the O(ε²) term of slender-body
theory; its normalisation matches the stokeslet's so that `(1/4π)[G + c·D]f` has one
prefactor, as in the 3D `(1/8π)[S + (ε²/2)D]`. Outputs: `Velocity()` (= `Potential()`),
`Gradient()` (the velocity gradient `∂u_a/∂t_j` as an `SMatrix`), `Pressure()`.

```julia
plan = StokesFMMPlan(sources; targets=targets, method=:ki, nequiv=60)   # 2D; or method=:square, p=12
plan = StokesFMMPlan(sources3d; targets=targets3d)                       # 3D: method=:cube, p=8, pc=16
stokes_fmm_eval!(u, plan; stokeslets=f, doublets=g, weights=w)        # at the targets
stokes_fmm_eval_src_trg!(u_src, u_trg, plan; stokeslets=f)            # self and targets, one build
u = StokesFMMKernel(sources; stokeslets=f, stresslets=q, stressvecs=ν) # one shot, self
```

Dual-number positions, densities and weights go through both forms (the plan is
built from the values of the positions); a Newton–Krylov Jacobian-vector product
with respect to the fibre geometry is one evaluation with one-partial duals.

## How the methods handle the singularities

The kernel-independent method needs nothing per singularity beyond its field: the
singularity type enters only through `source_field` in the check potentials and the
near field. Two things about the single-layer representation itself have to be
handled, and both come from the same fact, that a stokeslet density on a closed
curve has zero net flux through every enclosing curve.

- **Upward, a point-source strength per box.** The stresslet (double-layer) kernel
  is not flux-free: its flux through any circle or sphere around the source is
  −(f·ν) in both dimensions, i.e. it carries a point mass source, which no
  single-layer density can produce (a box of stresslets represented by a density
  alone had a 0.4 relative far-field error). The multipole therefore carries one
  extra entry, the strength of a point source at the box center (`multipole_length
  = nc·nsurface + 1`, the source's velocity `d/(2π r²)` in 2D and `d/(4π r³)` in
  3D), fitted together with the density from the check velocities: the source's
  velocity on the check surface is purely radial, exactly the direction the single
  layer cannot produce, so the fit is well posed. Stokeslets, doublets and rotlets
  are flux-free and get strength 0. The source is a potential flow and contributes
  nothing to the pressure.
- **Downward, the pressure at the box center.** The null space of the single layer
  (the normal density) has zero velocity inside the curve but a constant pressure,
  so a local expansion fitted to velocities alone has an undetermined pressure
  constant per box. The downward check data carry the pressure at the center as one
  extra row (`partial_local_length = 2·nequiv + 1`), which is what the Python
  "pressure slot" did. No source term is needed downward: the field inside a box
  due to sources outside it has zero flux. The upward check map is rank deficient by
  one and the truncated SVD drops that mode.

Both extras are nondimensionalised with the box width (the row is `w·p`, the
source strength is per unit width), so that they scale with the width exactly like
the velocity kernel: the augmented operators of a homogeneous kernel stay scalar
multiples of a reference set (`docs/framework.md`), and the pressure pin carries
the same least-squares weight on every level.

In 3D the cube method needs its large surface finer than its small one (`pc = 2p`,
the default) for the Stokes singularities: with equal lattices the stresslet and
doublet translations were wrong by order one at `p = 8`; `docs/kifmm.md` has the
mechanism and the measured accuracy per singularity.

The black-box method interpolates the stokeslet, so a singularity must be a
differential operator in the source applied to the stokeslet: stokeslets (values of
the basis), rotlets (basis gradient) and doublets (half the basis Laplacian) are
handled through `accumulate_moments!`. The stresslet's pressure term is not of that
form with vector moments; supporting it needs an extra dyad-valued moment block per
node, which is not implemented, so stresslets and the pressure output error with
the black-box method.

In single precision (`precision=Float32`) the Stokes kernel-independent method
reaches about 5e-3 relative accuracy at `nequiv = 40`, coarser than the scalar
kernels' 1e-4, because the pinned downward solve and the two-component check maps
leave fewer usable modes below the Float32 truncation.
