# Black-box (Chebyshev) operators

A box of width `w` maps to `[−1, 1]²` by `x̂ = 2(x − c)/w` and carries the `p × p`
tensor grid of first-kind Chebyshev points, coordinate 1 fastest; tensor index `n`
pairs `T_j(x̂₁)T_k(x̂₂)` (`n = j + 1 + k·p`) with node `(x_j, x_k)`.

- **Multipole** = moments `m_n = Σ_i T_n(x̂_i) τ_i` (`s2m!`, allocation-free from the
  three-term recurrence). The equivalent nodal weights are `w = VIᵀ m`, with `VI` the
  inverse tensor Vandermonde (values at nodes → coefficients), which on first-kind
  points is the closed-form `kron(D·Vᵀ, D·Vᵀ)`, `D = diag(1/p, 2/p, …)`.
- **Local** = Chebyshev coefficients of the far field in `x̂`; `l2t` is a nested
  Clenshaw evaluation.
- `M2L[f] = VI · K(shifted nodes → nodes) · VIᵀ`.
- `M2M` block `k` = `(VI·V_k)ᵀ` and `L2L` block `k` = `VI·V_k`, with `V_k` the tensor
  Vandermonde at child `k`'s nodes in parent coordinates `x̂ᴾ = x̂ᵏ/2 + (δ_k − ½)`.
- The partial→actual transforms are the identity.

Accuracy is governed by Chebyshev interpolation of the kernel between well-separated
boxes and converges geometrically in `p`; the nearest interaction-list box is the
hardest (see `test/test_operators_bbfmm.jl`). Any smooth kernel works, which is the
method's point; it does not need the kernel to satisfy a PDE.
