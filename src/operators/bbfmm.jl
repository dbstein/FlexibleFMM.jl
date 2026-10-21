# Black-box FMM operators (Fong & Darve), 2D, on first-kind Chebyshev tensor grids.
#
# Coordinates: a box of width w maps to [-1,1]² by x̂ = 2(x - c)/w. Tensor index
# n ↔ (j, k) with j fastest, pairing T_j(x̂₁)T_k(x̂₂) with node (x_j, x_k).
# Multipole = moments m_n = Σ_i T_n(x̂_i) τ_i. The equivalent nodal weights are
# w = VIᵀ m, where VI is the inverse tensor Vandermonde (values at nodes →
# coefficients). Local = Chebyshev coefficients of the field in x̂ coordinates.
#   M2L[f] = VI · K(shifted nodes → nodes) · VIᵀ
#   M2M block k = (VI·V_k)ᵀ,  L2L block k = VI·V_k,
# with V_k the tensor Vandermonde at child k's nodes expressed in parent
# coordinates, x̂ᴾ = x̂ᵏ/2 + (δ_k - 1/2). The partial→actual transforms are the identity.

struct BlackBox{K,P} <: AbstractMethod
    kernel::K
end

"""
    BlackBox(kernel; p=8)

The black-box (Chebyshev interpolation) method with `p` points per dimension.
"""
function BlackBox(kernel; p::Integer=8)
    p >= 2 || throw(ArgumentError("p must be at least 2"))
    return BlackBox{typeof(kernel),Int(p)}(kernel)
end

order(::BlackBox{K,P}) where {K,P} = P
expansion_length(m::BlackBox{K,P}) where {K,P} = ncomp(m.kernel) * P * P
near_kernel(m::BlackBox) = m.kernel
function operator_eltype(m::BlackBox)
    R = floattype(m.kernel)
    return eltype(kernel_eltype(m.kernel, SVector{2,R}, SVector{2,R}))
end

struct BBExtras{R}
    nodes::Vector{SVector{2,R}}         # tensor nodes in physical units, relative to the center
    scale::R                            # 2/w
    VI::Matrix{R}                       # inverse tensor Vandermonde
end

"""
    tensor_vandermonde(points, p)

`V[i, j+1 + k p] = T_j(x_i) T_k(y_i)` for points in `[-1,1]²`.
"""
function tensor_vandermonde(points::AbstractVector{SVector{2,R}}, p::Int) where {R}
    V = Matrix{R}(undef, length(points), p * p)
    for (i, x) in enumerate(points)
        tx = chebyshev_vandermonde([x[1]], p)
        ty = chebyshev_vandermonde([x[2]], p)
        for k in 1:p, j in 1:p
            V[i, j + (k - 1) * p] = tx[1, j] * ty[1, k]
        end
    end
    return V
end

# For a kernel with nc components the interpolation acts per component: with the
# component-major layout of `kernel_matrix`, that is kron(I_nc, VI).
function level_operators(m::BlackBox{K,P}, w::Real) where {K,P}
    Kf = m.kernel
    R = floattype(Kf)
    w = R(w)
    nc = ncomp(Kf)
    nodes = chebyshev_tensor_nodes(Val(2), P, R)
    VI1 = chebyshev_vandermonde_inverse(P, R)
    VI = kron(VI1, VI1)
    Inc = Matrix{R}(I, nc, nc)
    VIc = kron(Inc, VI)
    phys = [n * (w / 2) for n in nodes]
    blocksM = Matrix{R}[]
    blocksL = Matrix{R}[]
    for δ in child_offsets(Val(2))
        shift = SVector{2,R}(δ) .- R(0.5)
        A = kron(Inc, VI * tensor_vandermonde([n / 2 + shift for n in nodes], P))
        push!(blocksL, A)
        push!(blocksM, Matrix(transpose(A)))
    end
    M2L = [M2LOperator(VIc * kernel_matrix(Kf, phys .+ Ref(SVector{2,R}(off) * w), phys) * transpose(VIc))
           for off in far_offsets(Val(2))]
    T = eltype(M2L[1].A)
    return LevelOperators{T,BBExtras{R}}(w, Matrix{T}(reduce(hcat, blocksM)),
                                         Matrix{T}(reduce(vcat, blocksL)), M2L,
                                         BBExtras{R}(phys, 2 / w, VI))
end

"""
    s2m!(col, m, ops, xs, τs, c)

Chebyshev moments of the sources of a childless node: each density type adds its
action on the basis through `accumulate_moments!`.
"""
function s2m!(col::AbstractVector, m::BlackBox{K,P}, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{2}) where {K,P}
    s = ops.extra.scale
    @inbounds for n in eachindex(col)
        col[n] = zero(eltype(col))
    end
    @inbounds for i in eachindex(xs)
        accumulate_moments!(col, Val(P), (xs[i] - c) * s, s, τs[i])
    end
    return col
end

"""
    accumulate_moments!(col, ::Val{P}, x̂, scale, τ)

Add the moments of one source at `x̂ ∈ [-1,1]²` with density `τ`: `τ·T_n(x̂)` for a
charge, `d·∇ₛT_n` (with `scale = 2/w` for the chain rule) for a dipole, the sum for
a tuple. A new singularity type adds a method here.
"""
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, τ::Number) where {P}
    tx = chebyshev_values(Val(P), x̂[1])
    ty = chebyshev_values(Val(P), x̂[2])
    @inbounds for k in 1:P
        tyk = ty[k] * τ
        for j in 1:P
            col[j + (k - 1) * P] += tx[j] * tyk
        end
    end
    return col
end
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, d::Dipole) where {P}
    tx, dx = chebyshev_values_derivatives(Val(P), x̂[1])
    ty, dy = chebyshev_values_derivatives(Val(P), x̂[2])
    d1 = d.d[1] * scale
    d2 = d.d[2] * scale
    @inbounds for k in 1:P
        a = ty[k] * d1
        b = dy[k] * d2
        for j in 1:P
            col[j + (k - 1) * P] += dx[j] * a + tx[j] * b
        end
    end
    return col
end
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, τ::Tuple) where {P}
    for τi in τ
        accumulate_moments!(col, Val(P), x̂, scale, τi)
    end
    return col
end
# Stokes: vector moments, component major; a singularity that is a differential
# operator applied to the Stokeslet contributes that operator applied to the basis.
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, f::SVector{2}) where {P}
    tx = chebyshev_values(Val(P), x̂[1])
    ty = chebyshev_values(Val(P), x̂[2])
    @inbounds for k in 1:P, j in 1:P
        n = j + (k - 1) * P
        t = tx[j] * ty[k]
        col[n] += t * f[1]
        col[P * P + n] += t * f[2]
    end
    return col
end
# rotlet: u = -ω ε_{jk} ∂_{s_k} G_{ij}  ⇒  moment ω·scale·(-∂₂T_n, ∂₁T_n)
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, τ::Rotlet) where {P}
    tx, dx = chebyshev_values_derivatives(Val(P), x̂[1])
    ty, dy = chebyshev_values_derivatives(Val(P), x̂[2])
    ws = τ.ω * scale
    @inbounds for k in 1:P, j in 1:P
        n = j + (k - 1) * P
        col[n] += -ws * tx[j] * dy[k]
        col[P * P + n] += ws * dx[j] * ty[k]
    end
    return col
end
# doublet: u = ½∇²_s G·g  ⇒  moment ½·scale²·(∇²T_n)·g
@inline function accumulate_moments!(col, ::Val{P}, x̂::SVector{2}, scale, τ::Doublet{2}) where {P}
    tx, _, ex = chebyshev_values_derivatives2(Val(P), x̂[1])
    ty, _, ey = chebyshev_values_derivatives2(Val(P), x̂[2])
    g = τ.g * (scale * scale / 2)
    @inbounds for k in 1:P, j in 1:P
        n = j + (k - 1) * P
        lap = ex[j] * ty[k] + tx[j] * ey[k]
        col[n] += lap * g[1]
        col[P * P + n] += lap * g[2]
    end
    return col
end
supports_density(::BlackBox, ::Type{<:Stresslet}) = false
supports_output(::BlackBox, ::Pressure) = false

partial_to_multipole!(M::AbstractVecOrMat, ::BlackBox, ::LevelOperators, pM::AbstractVecOrMat, work) = copyto!(M, pM)
partial_to_local!(L::AbstractVecOrMat, ::BlackBox, ::LevelOperators, pL::AbstractVecOrMat, work) = copyto!(L, pL)

"""
    l2t(m, ops, L, xrel, out=Potential())

The local Chebyshev series (or its gradient) at `xrel` relative to the box center.
"""
@inline function l2t(m::BlackBox{K,P}, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Potential=Potential()) where {K,P}
    x̂ = xrel * ops.extra.scale
    return _bb_value(Val(ncomp(m.kernel)), L, P, x̂)
end
@inline _bb_value(::Val{1}, L, P, x̂) = clenshaw2d(L, P, x̂[1], x̂[2])
@inline function _bb_value(::Val{nc}, L, P, x̂) where {nc}
    return SVector(ntuple(a -> clenshaw2d(view(L, (a - 1) * P * P + 1:a * P * P), P, x̂[1], x̂[2]), Val(nc)))
end
@inline function l2t(m::BlackBox{K,P}, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}, ::Gradient) where {K,P}
    s = ops.extra.scale
    x̂ = xrel * s
    return _bb_gradient(Val(ncomp(m.kernel)), L, P, x̂) * s
end
@inline function _bb_gradient(::Val{1}, L, P, x̂)
    gx, gy = clenshaw2d_gradient(L, P, x̂[1], x̂[2])
    return SVector(gx, gy)
end
# velocity gradient ∂u_a/∂x_j as an SMatrix (rows a, columns j)
@inline function _bb_gradient(::Val{2}, L, P, x̂)
    g1 = clenshaw2d_gradient(view(L, 1:P * P), P, x̂[1], x̂[2])
    g2 = clenshaw2d_gradient(view(L, P * P + 1:2 * P * P), P, x̂[1], x̂[2])
    return SMatrix{2,2}(g1[1], g2[1], g1[2], g2[2])
end


"""
    multipole_field(m, ops, M, xrel)

Far field of the moments `M` at `xrel`: the equivalent nodal weights `VIᵀ M` acting
through the kernel. Used by tests.
"""
function multipole_field(m::BlackBox, ops::LevelOperators, M::AbstractVector, xrel::SVector{2})
    K = m.kernel
    nodes = ops.extra.nodes
    nn = length(nodes)
    nc = Val(ncomp(K))
    wts = kron(Matrix{eltype(ops.extra.VI)}(I, ncomp(K), ncomp(K)), transpose(ops.extra.VI)) * M
    S = promote_type(eltype(wts), eltype(kernel_eltype(K, eltype(nodes), typeof(xrel))))
    acc = _field_zero(S, nc)
    for n in 1:nn
        acc += K(nodes[n], xrel) * _load(wts, n, nn, nc)
    end
    return acc
end
