# Chebyshev polynomials of the first kind on first-kind points, as used by the
# black-box FMM. Conventions: coefficient vectors are 1-based, `c[j+1]` multiplies
# `T_j`; tensor-product quantities index coordinate 1 fastest.

"""
    chebyshev_points(p, T=Float64) -> Vector{T}

The `p` Chebyshev points of the first kind, `cos((2k-1)π / 2p)`, in ascending order.
"""
function chebyshev_points(p::Integer, ::Type{T}=Float64) where {T}
    p >= 1 || throw(ArgumentError("p must be positive"))
    return [cos((2k - 1) * T(π) / (2p)) for k in p:-1:1]
end

"""
    chebyshev_values(::Val{P}, x) -> SVector{P}

`T_0(x), …, T_{P-1}(x)` by the three-term recurrence. Allocation free.
"""
@inline function chebyshev_values(::Val{P}, x) where {P}
    t = MVector{P,typeof(x)}(undef)
    @inbounds begin
        t[1] = one(x)
        P >= 2 && (t[2] = x)
        for n in 3:P
            t[n] = 2x * t[n-1] - t[n-2]
        end
    end
    return SVector(t)
end

"""
    chebyshev_values_derivatives(::Val{P}, x) -> (SVector{P}, SVector{P})

`T_0(x), …, T_{P-1}(x)` and their derivatives, from `T_n' = 2T_{n-1} + 2x T_{n-1}' - T_{n-2}'`.
"""
@inline function chebyshev_values_derivatives(::Val{P}, x) where {P}
    t = MVector{P,typeof(x)}(undef)
    d = MVector{P,typeof(x)}(undef)
    @inbounds begin
        t[1] = one(x)
        d[1] = zero(x)
        if P >= 2
            t[2] = x
            d[2] = one(x)
        end
        for n in 3:P
            t[n] = 2x * t[n-1] - t[n-2]
            d[n] = 2t[n-1] + 2x * d[n-1] - d[n-2]
        end
    end
    return SVector(t), SVector(d)
end

"""
    chebyshev_values_derivatives2(::Val{P}, x) -> (T, T′, T″)

Values, first and second derivatives of `T_0, …, T_{P-1}` at `x`
(`T_n″ = 4T′_{n-1} + 2x T″_{n-1} - T″_{n-2}`).
"""
@inline function chebyshev_values_derivatives2(::Val{P}, x) where {P}
    t = MVector{P,typeof(x)}(undef)
    d = MVector{P,typeof(x)}(undef)
    e = MVector{P,typeof(x)}(undef)
    @inbounds begin
        t[1] = one(x); d[1] = zero(x); e[1] = zero(x)
        if P >= 2
            t[2] = x; d[2] = one(x); e[2] = zero(x)
        end
        for n in 3:P
            t[n] = 2x * t[n-1] - t[n-2]
            d[n] = 2t[n-1] + 2x * d[n-1] - d[n-2]
            e[n] = 4d[n-1] + 2x * e[n-1] - e[n-2]
        end
    end
    return SVector(t), SVector(d), SVector(e)
end

"""
    chebyshev_vandermonde(x, p) -> Matrix

`V[i, j+1] = T_j(x[i])` for `j = 0:p-1`.
"""
function chebyshev_vandermonde(x::AbstractVector{T}, p::Integer) where {T}
    V = Matrix{T}(undef, length(x), p)
    for i in eachindex(x)
        V[i, 1] = one(T)
        p >= 2 && (V[i, 2] = x[i])
        for j in 3:p
            V[i, j] = 2x[i] * V[i, j-1] - V[i, j-2]
        end
    end
    return V
end

"""
    chebyshev_vandermonde_inverse(p, T=Float64) -> Matrix{T}

Inverse of `chebyshev_vandermonde(chebyshev_points(p, T), p)`. On first-kind points
the discrete orthogonality of the `T_j` gives it in closed form,
`V⁻¹ = Diagonal(1/p, 2/p, …, 2/p) Vᵀ`, with condition number √2.
"""
function chebyshev_vandermonde_inverse(p::Integer, ::Type{T}=Float64) where {T}
    V = chebyshev_vandermonde(chebyshev_points(p, T), p)
    c = fill(T(2) / p, p)
    c[1] = one(T) / p
    return Diagonal(c) * transpose(V)
end

"""
    chebyshev_tensor_nodes(::Val{D}, p, T=Float64) -> Vector{SVector{D,T}}

The `p^D` tensor grid of first-kind points in `[-1, 1]^D`, coordinate 1 fastest, so
node `m` pairs with the coefficient `m` of the tensor basis `T_{j₁}(x₁)⋯T_{j_D}(x_D)`
enumerated the same way. The tensor Vandermonde is `kron(V, …, V)` (innermost last).
"""
function chebyshev_tensor_nodes(::Val{D}, p::Integer, ::Type{T}=Float64) where {D,T}
    x = chebyshev_points(p, T)
    return vec([SVector{D,T}(ntuple(d -> x[I[d]], Val(D)))
                for I in CartesianIndices(ntuple(_ -> p, Val(D)))])
end

"""
    clenshaw(c, x)

`Σ_{j=0}^{n-1} c[j+1] T_j(x)` by Clenshaw's recurrence, for any `n ≥ 1`.
"""
@inline clenshaw(c::AbstractVector, x) = _clenshaw_range(c, firstindex(c), lastindex(c), x)

# Clenshaw on the coefficient slice `c[lo:hi]` without forming a view.
@inline function _clenshaw_range(c::AbstractVector, lo::Int, hi::Int, x)
    @inbounds begin
        hi == lo && return c[lo] * one(x)
        b1 = c[hi] * one(x)      # b_{k+1}
        b2 = zero(b1)            # b_{k+2}
        for k in hi-1:-1:lo+1
            b1, b2 = c[k] + 2x * b1 - b2, b1
        end
        return c[lo] + x * b1 - b2
    end
end

"""
    clenshaw_derivative(c, x)

Derivative of `Σ c[j+1] T_j` at `x`, by differentiating Clenshaw's recurrence.
"""
@inline clenshaw_derivative(c::AbstractVector, x) =
    _clenshaw_derivative_range(c, firstindex(c), lastindex(c), x)

# Streamed Clenshaw for coefficients c[lo:hi]: feeding the coefficients from the top,
# b_k = c_k + 2x b_{k+1} - b_{k+2} and b'_k = 2b_{k+1} + 2x b'_{k+1} - b'_{k+2};
# the value is c_0 + x b_1 - b_2 and the derivative b_1 + x b'_1 - b'_2.
@inline function _clenshaw_derivative_range(c::AbstractVector, lo::Int, hi::Int, x)
    @inbounds begin
        b1 = zero(c[hi] * one(x)); b2 = b1
        d1 = b1; d2 = b1
        for k in hi:-1:lo+1
            d1, d2 = 2b1 + 2x * d1 - d2, d1
            b1, b2 = c[k] + 2x * b1 - b2, b1
        end
        return b1 + x * d1 - d2
    end
end

"""
    clenshaw2d(c, p, x, y)

Evaluate the 2D tensor series with coefficients `c` (length `p²`, `c[j+1 + k p]`
multiplying `T_j(x) T_k(y)`) at `(x, y)`: Clenshaw in `y` whose coefficients
`w_k = Σ_j c[j+1 + k p] T_j(x)` are themselves computed by Clenshaw in `x`.
Allocation free.
"""
@inline function clenshaw2d(c::AbstractVector, p::Int, x, y)
    @boundscheck length(c) == p * p || throw(DimensionMismatch("expected p² coefficients"))
    @inbounds begin
        base = firstindex(c) - 1
        p == 1 && return c[base+1] * one(x) * one(y)
        b1 = _clenshaw_range(c, base + (p-1)*p + 1, base + p*p, x) * one(y)
        b2 = zero(b1)
        for k in p-2:-1:1
            wk = _clenshaw_range(c, base + k*p + 1, base + (k+1)*p, x)
            b1, b2 = wk + 2y * b1 - b2, b1
        end
        w0 = _clenshaw_range(c, base + 1, base + p, x)
        return w0 + y * b1 - b2
    end
end

"""
    clenshaw2d_gradient(c, p, x, y) -> (∂x, ∂y)

Gradient of the 2D tensor series of `clenshaw2d` at `(x, y)`, in the `[-1,1]²`
coordinates. Allocation free: the inner sums in `x` (values `w_k` and derivatives
`v_k`) feed one Clenshaw value recurrence and one derivative recurrence in `y`.
"""
@inline function clenshaw2d_gradient(c::AbstractVector, p::Int, x, y)
    @boundscheck length(c) == p * p || throw(DimensionMismatch("expected p² coefficients"))
    @inbounds begin
        base = firstindex(c) - 1
        z = zero(c[base+1] * one(x) * one(y))
        b1 = z; b2 = z            # value recurrence in y on the v_k
        g1 = z; g2 = z            # Clenshaw in y on the w_k …
        d1 = z; d2 = z            # … and its derivative
        for k in p-1:-1:1
            lo = base + k * p + 1
            wk = _clenshaw_range(c, lo, lo + p - 1, x)
            vk = _clenshaw_derivative_range(c, lo, lo + p - 1, x)
            b1, b2 = vk + 2y * b1 - b2, b1
            d1, d2 = 2 * g1 + 2y * d1 - d2, d1
            g1, g2 = wk + 2y * g1 - g2, g1
        end
        v0 = _clenshaw_derivative_range(c, base + 1, base + p, x)
        return v0 + y * b1 - b2, g1 + y * d1 - d2
    end
end
