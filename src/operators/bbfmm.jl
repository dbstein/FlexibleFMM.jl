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
expansion_length(::BlackBox{K,P}) where {K,P} = P * P
near_kernel(m::BlackBox) = m.kernel
function operator_eltype(m::BlackBox)
    R = floattype(m.kernel)
    return kernel_eltype(m.kernel, SVector{2,R}, SVector{2,R})
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

function level_operators(m::BlackBox{K,P}, w::Real) where {K,P}
    Kf = m.kernel
    R = floattype(Kf)
    w = R(w)
    nodes = chebyshev_tensor_nodes(Val(2), P, R)
    VI1 = chebyshev_vandermonde_inverse(P, R)
    VI = kron(VI1, VI1)
    phys = [n * (w / 2) for n in nodes]
    blocksM = Matrix{R}[]
    blocksL = Matrix{R}[]
    for δ in child_offsets(Val(2))
        shift = SVector{2,R}(δ) .- R(0.5)
        A = VI * tensor_vandermonde([n / 2 + shift for n in nodes], P)
        push!(blocksL, A)
        push!(blocksM, Matrix(transpose(A)))
    end
    M2L = [M2LOperator(VI * kernel_matrix(Kf, phys .+ Ref(SVector{2,R}(off) * w), phys) * transpose(VI))
           for off in far_offsets(Val(2))]
    T = eltype(M2L[1].A)
    return LevelOperators{T,BBExtras{R}}(w, Matrix{T}(reduce(hcat, blocksM)),
                                         Matrix{T}(reduce(vcat, blocksL)), M2L,
                                         BBExtras{R}(phys, 2 / w, VI))
end

"""
    s2m!(col, m, ops, xs, τs, c)

Chebyshev moments of the sources of a childless node.
"""
function s2m!(col::AbstractVector, m::BlackBox{K,P}, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{2}) where {K,P}
    s = ops.extra.scale
    @inbounds for n in eachindex(col)
        col[n] = zero(eltype(col))
    end
    @inbounds for i in eachindex(xs)
        x̂ = (xs[i] - c) * s
        tx = chebyshev_values(Val(P), x̂[1])
        ty = chebyshev_values(Val(P), x̂[2])
        τ = τs[i]
        for k in 1:P
            tyk = ty[k] * τ
            for j in 1:P
                col[j + (k - 1) * P] += tx[j] * tyk
            end
        end
    end
    return col
end

partial_to_multipole!(M::AbstractVecOrMat, ::BlackBox, ::LevelOperators, pM::AbstractVecOrMat, work) = copyto!(M, pM)
partial_to_local!(L::AbstractVecOrMat, ::BlackBox, ::LevelOperators, pL::AbstractVecOrMat, work) = copyto!(L, pL)

"""
    l2t(m, ops, L, xrel)

The local Chebyshev series evaluated at `xrel` relative to the box center.
"""
@inline function l2t(m::BlackBox{K,P}, ops::LevelOperators, L::AbstractVector, xrel::SVector{2}) where {K,P}
    x̂ = xrel * ops.extra.scale
    return clenshaw2d(L, P, x̂[1], x̂[2])
end

"""
    multipole_field(m, ops, M, xrel)

Far field of the moments `M` at `xrel`: the equivalent nodal weights `VIᵀ M` acting
through the kernel. Used by tests.
"""
function multipole_field(m::BlackBox, ops::LevelOperators, M::AbstractVector, xrel::SVector{2})
    K = m.kernel
    nodes = ops.extra.nodes
    wts = transpose(ops.extra.VI) * M
    acc = zero(promote_type(eltype(wts), kernel_eltype(K, eltype(nodes), typeof(xrel))))
    for n in eachindex(nodes)
        acc += K(nodes[n], xrel) * wts[n]
    end
    return acc
end
