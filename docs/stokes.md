# Stokes

The Stokes layer follows the conventions of JellySim's kernel backends, so that a
three-dimensional FlexibleFMM can be dropped in there without changes on the
JellySim side: separate arrays for `stokeslets`, `stresslets` with `stressvecs`, and
`doublets`; optional per-source `weights`; `targets = nothing` for self-evaluation
with the `i = j` term skipped; a persistent plan for repeated solves on a fixed
geometry; a combined source-and-target evaluation; and the viscosity carried by the
densities. The 2D kernels carry the 2D analogues of JellySim's 3D prefactors.

| singularity | density | velocity (d = t − s, r = |d|, c = 1/4π) | pressure |
|---|---|---|---|
| stokeslet | `SVector{2}` f | c(−log r·f + d(d·f)/r²) | 2c(d·f)/r² |
| stresslet | `Stresslet(f, ν)` | −(1/π) d (d·f)(d·ν)/r⁴ | (1/π)[(f·ν)/r² − 2(d·f)(d·ν)/r⁴] |
| doublet | `Doublet(g)` | c(g/r² − 2d(d·g)/r⁴) = ½∇²G·g | 0 |
| rotlet | `Rotlet(ω)` | 2cω(−d₂, d₁)/r² = −ω ε_jk ∂_{s_k}G_ij | 0 |

The doublet is the potential dipole that appears as the O(ε²) term of slender-body
theory; its normalisation matches the stokeslet's so that `(1/4π)[G + c·D]f` has one
prefactor, as in the 3D `(1/8π)[S + (ε²/2)D]`. Outputs: `Velocity()` (= `Potential()`),
`Gradient()` (the velocity gradient `∂u_a/∂t_j` as an `SMatrix`), `Pressure()`.

```julia
plan = StokesFMMPlan(sources; targets=targets, method=:ki, nequiv=60)
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
  is not flux-free: its flux through any circle around the source is −(f·ν), i.e.
  it carries a point mass source, which no single-layer density can produce (a box
  of stresslets represented by a density alone had a 0.4 relative far-field error).
  The multipole therefore carries one extra entry, the strength of a point source at
  the box center (`multipole_length = 2·nequiv + 1`), fitted together with the density
  from the check velocities: the source's velocity on the check circle is purely
  radial, exactly the direction the single layer cannot produce, so the fit is well
  posed. Stokeslets, doublets and rotlets are flux-free and get strength 0. The
  source is a potential flow and contributes nothing to the pressure.
- **Downward, the pressure at the box center.** The null space of the single layer
  (the normal density) has zero velocity inside the curve but a constant pressure,
  so a local expansion fitted to velocities alone has an undetermined pressure
  constant per box. The downward check data carry the pressure at the center as one
  extra row (`partial_local_length = 2·nequiv + 1`), which is what the Python
  "pressure slot" did. No source term is needed downward: the field inside a box
  due to sources outside it has zero flux. The upward check map is rank deficient by
  one and the truncated SVD drops that mode.

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
