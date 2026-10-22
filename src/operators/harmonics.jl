# Analytic Laplace expansions in solid harmonics (3D): the classical multipole method
# in the five-operator form of the framework, with dense per-offset translations.
#
# Complex basis, Hobson normalisation (build time only; P_n^m with the Condon–Shortley phase):
#   Υ_n^m(x) = r^n P_n^m(cos θ) e^{imφ}/(n+m)!,   Θ_n^m(x) = (n−m)! P_n^m(cos θ) e^{imφ}/r^{n+1},
#   Υ_n^{−m} = (−1)^m conj Υ_n^m, likewise Θ, with the identities
#   1/|x − y| = Σ_{n,m} conj(Υ_n^m(y)) Θ_n^m(x)                              (|y| < |x|)
#   Υ_n^m(a + b) = Σ_{k≤n} Σ_l Υ_k^l(a) Υ_{n−k}^{m−l}(b)                       (regular addition theorem)
#   Θ_n^m(a + b) = Σ_k Σ_l (−1)^k conj(Υ_k^l(b)) Θ_{n+k}^{m+l}(a)              (|b| < |a|; from the two above
#                                                                              by matching coefficients)
#   ∂_z Υ_n^m = Υ_{n−1}^m,  (∂_x − i∂_y) Υ_n^m = −Υ_{n−1}^{m−1},  (∂_x + i∂_y) Υ_n^m = Υ_{n−1}^{m+1}.
# In box coordinates x̂ = (x − c)/w, for the kernel c_K/|x − y| (c_K = 1/4π):
#   multipole   M_n^m = Σ_j τ_j conj Υ_n^m(ŷ_j),     field  (c_K/w) Σ M_n^m Θ_n^m(x̂)
#   local       Φ(x) = Σ L_n^m conj Υ_n^m(x̂)           (the c_K/w absorbed into the M2L)
#   M2M   M^p_n^m = Σ_{k,l} 2^{−(n−k)} conj Υ_k^l(δ̂) M^c_{n−k}^{m−l}      δ̂ = child centre/w ∈ {±1/4}³
#   L2L   L^c_n^m = 2^{−n} Σ_{k,l} conj Υ_k^l(δ̂) L^p_{n+k}^{m+l}
#   M2L   L_n^m = (c_K/w) Σ_{k,l} (−1)^k Θ_{n+k}^{m+l}(o) M_k^l               o = far offset, k, n ≤ p
# M2M and L2L are exact; the M2L truncation at degree p is the only approximation.
#
# Stored basis: Schmidt semi-normalised, Υ̃ = s Υ and Θ̃ = Θ/s with s_n^m = √((n−|m|)!(n+|m|)!),
# so that |Υ̃_n^m| ≤ r^n, |Θ̃_n^m| ≤ r^{−n−1} and the operator entries are O(2^n/|o|^{n+1}),
# safe in Float32 (the Hobson values carry n! and 1/n!). Real form: for real densities
# M_n^{−m} = (−1)^m conj M_n^m, so a column holds M_n^0 at index n²+1 and Re, Im M_n^m at
# n²+2m, n²+2m+1 for m = 1..n, (p+1)² reals in all; the operators are real matrices and
# complex densities simply promote the columns. The per-point recurrences run in this
# normalisation with precomputed square-root coefficients (`HarmonicTables`) and two rows
# of state, so `s2m!` and `l2t` are allocation-free for any p.

@inline cidx(n::Int, m::Int) = n * n + n + m + 1          # complex index, −n ≤ m ≤ n
@inline tidx(n::Int, m::Int) = (n * (n + 1)) >> 1 + m + 1   # triangular index, 0 ≤ m ≤ n

"""
    regular_hobson(x, p), irregular_hobson(x, p)

`Υ_n^m(x)` and `Θ_n^m(x)` for `0 ≤ n ≤ p`, `|m| ≤ n`, at the complex indices `cidx(n, m)`.
"""
function regular_hobson(x::SVector{3}, p::Int)
    T = float(eltype(x))
    Y = zeros(Complex{T}, (p + 1)^2)
    r2 = dot(x, x); z = x[3]; xy = complex(x[1], x[2])
    s = one(Complex{T})                                  # Υ_m^m
    for m in 0:p
        Y[cidx(m, m)] = s
        prev = zero(s); cur = s                          # Υ_{n−2}^m, Υ_{n−1}^m
        for n in m+1:p
            nxt = ((2n - 1) * z * cur - r2 * prev) / ((n - m) * (n + m))
            Y[cidx(n, m)] = nxt
            prev = cur; cur = nxt
        end
        s = -xy / (2(m + 1)) * s
    end
    for n in 1:p, m in 1:n
        Y[cidx(n, -m)] = (-1)^m * conj(Y[cidx(n, m)])
    end
    return Y
end
function irregular_hobson(x::SVector{3}, p::Int)
    T = float(eltype(x))
    Y = zeros(Complex{T}, (p + 1)^2)
    r2 = dot(x, x); ir2 = 1 / r2; z = x[3]; xy = complex(x[1], x[2])
    s = complex(1 / sqrt(r2))                            # Θ_m^m
    for m in 0:p
        Y[cidx(m, m)] = s
        prev = zero(s); cur = s
        for n in m+1:p
            nxt = ((2n - 1) * z * cur - ((n - 1)^2 - m^2) * prev) * ir2
            Y[cidx(n, m)] = nxt
            prev = cur; cur = nxt
        end
        s = -(2m + 1) * xy * ir2 * s
    end
    for n in 1:p, m in 1:n
        Y[cidx(n, -m)] = (-1)^m * conj(Y[cidx(n, m)])
    end
    return Y
end

"""
    scaling_factors(p)

`s_n^m = √((n−|m|)!(n+|m|)!)` at the complex indices, the Hobson → Schmidt rescaling.
"""
function scaling_factors(p::Int)
    s = zeros(Float64, (p + 1)^2)
    for n in 0:p, m in -n:n
        a = 1.0
        for k in 2:n-abs(m); a *= k; end
        for k in 2:n+abs(m); a *= k; end
        s[cidx(n, m)] = sqrt(a)
    end
    return s
end

"""
    complex_coefficients(v, p), real_coefficients(c, p)

The real storage of a conjugate-symmetric coefficient vector and back.
"""
function complex_coefficients(v::AbstractVector, p::Int)
    c = zeros(complex(float(eltype(v))), (p + 1)^2)
    for n in 0:p
        c[cidx(n, 0)] = v[n*n+1]
        for m in 1:n
            z = complex(v[n*n+2m], v[n*n+2m+1])
            c[cidx(n, m)] = z
            c[cidx(n, -m)] = (-1)^m * conj(z)
        end
    end
    return c
end
function real_coefficients(c::AbstractVector, p::Int)
    v = zeros(real(eltype(c)), (p + 1)^2)
    for n in 0:p
        v[n*n+1] = real(c[cidx(n, 0)])
        for m in 1:n
            v[n*n+2m] = real(c[cidx(n, m)]); v[n*n+2m+1] = imag(c[cidx(n, m)])
        end
    end
    return v
end

# The real matrix of a complex operator that commutes with the conjugate symmetry:
# column j is the real storage of A applied to the complex image of the j-th real basis vector.
function real_operator(A::AbstractMatrix{<:Complex}, p::Int)
    N = (p + 1)^2
    R = zeros(Float64, N, N)
    v = zeros(ComplexF64, N)
    for n′ in 0:p, m′ in 0:n′, part in 0:(m′ == 0 ? 0 : 1)
        j = m′ == 0 ? n′ * n′ + 1 : n′ * n′ + 2m′ + part
        if m′ == 0
            v .= view(A, :, cidx(n′, 0))
        else
            sg = (-1)^m′
            if part == 0
                v .= view(A, :, cidx(n′, m′)) .+ sg .* view(A, :, cidx(n′, -m′))
            else
                v .= im .* (view(A, :, cidx(n′, m′)) .- sg .* view(A, :, cidx(n′, -m′)))
            end
        end
        for n in 0:p
            R[n*n+1, j] = real(v[cidx(n, 0)])
            for m in 1:n
                R[n*n+2m, j] = real(v[cidx(n, m)]); R[n*n+2m+1, j] = imag(v[cidx(n, m)])
            end
        end
    end
    return R
end

# ---- the per-point recurrences in the Schmidt normalisation ----

"""
    HarmonicTables{T}

Coefficients of the recurrences, at the triangular indices `tidx(n, m)`:
`Υ̃_n^m = a z Υ̃_{n−1}^m − b r² Υ̃_{n−2}^m` (`a = (2n−1)/√((n−m)(n+m))`,
`b = √((n−m−1)(n+m−1))/√((n−m)(n+m))`), `Υ̃_{m}^{m} = −e_m (x + iy) Υ̃_{m−1}^{m−1}` with
`e_m = √((2m−1)/2m)`, and the derivative factors `∂_z Υ̃_n^m = cz Υ̃_{n−1}^m`,
`(∂_x − i∂_y) Υ̃_n^m = −cm Υ̃_{n−1}^{m−1}`, `(∂_x + i∂_y) Υ̃_n^m = cp Υ̃_{n−1}^{m+1}`.
"""
struct HarmonicTables{T}
    p::Int
    a::Vector{T}
    b::Vector{T}
    e::Vector{T}
    cz::Vector{T}
    cm::Vector{T}
    cp::Vector{T}
end
function harmonic_tables(p::Int, ::Type{T}) where {T}
    nt = tidx(p, p)
    a = zeros(T, nt); b = zeros(T, nt); cz = zeros(T, nt); cm = zeros(T, nt); cp = zeros(T, nt)
    for n in 0:p, m in 0:n
        t = tidx(n, m)
        c = sqrt(Float64((n - m) * (n + m)))
        cz[t] = c
        if n > m
            a[t] = (2n - 1) / c
            b[t] = sqrt(Float64((n - m - 1) * (n + m - 1))) / c
        end
        cm[t] = n + m >= 2 ? sqrt(Float64((n + m) * (n + m - 1))) : zero(T)
        cp[t] = n - m >= 2 ? sqrt(Float64((n - m) * (n - m - 1))) : zero(T)
    end
    e = [sqrt((2m - 1) / (2m)) for m in 1:p]
    return HarmonicTables{T}(p, a, b, T.(e), cz, cm, cp)
end

"""
    _fold_regular(f, acc, x̂, nmax, tb)

`acc = f(acc, n, m, Re Υ̃_n^m(x̂), Im Υ̃_n^m(x̂))` over `0 ≤ m ≤ n ≤ nmax`, m-major, with two
rows of state (the values of negative `m` follow by conjugation).
"""
@inline function _fold_regular(f::F, acc, x̂::SVector{3}, nmax::Int, tb::HarmonicTables) where {F}
    x, y, z = x̂[1], x̂[2], x̂[3]
    r2 = x * x + y * y + z * z
    sr = one(x); si = zero(x)                                # Υ̃_m^m
    @inbounds for m in 0:nmax
        acc = f(acc, m, m, sr, si)
        pr = zero(sr); pi = zero(si); cr = sr; ci = si       # Υ̃_{n−2}^m, Υ̃_{n−1}^m
        for n in m+1:nmax
            t = tidx(n, m)
            az = tb.a[t] * z; br = tb.b[t] * r2
            nr = az * cr - br * pr; ni = az * ci - br * pi
            acc = f(acc, n, m, nr, ni)
            pr = cr; pi = ci; cr = nr; ci = ni
        end
        if m < nmax
            e = tb.e[m+1]
            sr, si = -e * (x * sr - y * si), -e * (x * si + y * sr)
        end
    end
    return acc
end

# add the complex value (re, im) to the (n, m) entry of a real column
@inline function _acc!(col, n::Int, m::Int, re, im)
    @inbounds if m == 0
        col[n*n+1] += re
    else
        col[n*n+2m] += re
        col[n*n+2m+1] += im
    end
    return nothing
end
# the (n, m) entry of a real column as (re, im), zero outside the triangle
@inline function _get(L, n::Int, m::Int)
    @inbounds if m > n
        return zero(eltype(L)), zero(eltype(L))
    elseif m == 0
        return L[n*n+1], zero(eltype(L))
    else
        return L[n*n+2m], L[n*n+2m+1]
    end
end

# M_n^m += τ conj Υ̃_n^m(ŷ)
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Number)
    _fold_regular(nothing, x̂, p, tb) do _, n, m, vr, vi
        _acc!(col, n, m, τ * vr, -τ * vi)
        nothing
    end
    return col
end
# dipole: M_n^m += conj(d̂·∇Υ̃_n^m(ŷ)), d̂ = d/w, by the derivative identities; the value
# Υ̃_{n'}^{m'} feeds (n'+1, m'+1), (n'+1, m') and (n'+1, m'−1), the last folded into
# the real entry (n'+1, 0) when m' = 1
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, d::Dipole{3})
    dx, dy, dz = d.d[1], d.d[2], d.d[3]
    _fold_regular(nothing, x̂, p - 1, tb) do _, n′, m′, ur, ui
        n = n′ + 1
        @inbounds begin
            c = tb.cm[tidx(n, m′ + 1)] / 2                               # −½ cm d₋ conj u
            _acc!(col, n, m′ + 1, -c * (dx * ur - dy * ui), c * (dx * ui + dy * ur))
            c = tb.cz[tidx(n, m′)] * dz                                  # cz d_z conj u
            _acc!(col, n, m′, c * ur, -c * ui)
            if m′ >= 2
                c = tb.cp[tidx(n, m′ - 1)] / 2                           # ½ cp d₊ conj u
                _acc!(col, n, m′ - 1, c * (dx * ur + dy * ui), c * (dy * ur - dx * ui))
            elseif m′ == 1
                c = tb.cp[tidx(n, 0)]                                     # cp Re(d₋ u)
                _acc!(col, n, 0, c * (dx * ur + dy * ui), zero(ur))
            end
        end
        nothing
    end
    return col
end
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple)
    for τi in τ
        _accumulate!(col, x̂, p, tb, τi)
    end
    return col
end

# ---- the method ----

struct SphericalHarmonics{K<:Laplace3D,T} <: AbstractMethod
    kernel::K
    p::Int
end

"""
    SphericalHarmonics(kernel::Laplace3D; p=16)

The analytic multipole method for the 3D Laplace kernel: expansions in real solid
harmonics up to degree `p` (`(p+1)²` coefficients), exact M2M and L2L, dense M2L
operators per far offset. Charges and dipoles; potential and gradient outputs.
"""
function SphericalHarmonics(kernel::Laplace3D{T,PreciseMath}; p::Integer=16) where {T}
    p >= 1 || throw(ArgumentError("p must be at least 1"))
    return SphericalHarmonics{typeof(kernel),T}(kernel, Int(p))
end

order(m::SphericalHarmonics) = m.p
dimension(::SphericalHarmonics) = 3
expansion_length(m::SphericalHarmonics) = (m.p + 1)^2
near_kernel(m::SphericalHarmonics) = m.kernel
operator_eltype(::SphericalHarmonics{K,T}) where {K,T} = T
operators_type(::SphericalHarmonics{K,T}) where {K,T} = LevelOperators{T,T,HarmonicExtras{T},M2LOperator{T}}
supports_density(::SphericalHarmonics, ::Type) = false
supports_density(::SphericalHarmonics, ::Type{<:Number}) = true
supports_density(::SphericalHarmonics, ::Type{<:Dipole{3}}) = true
supports_density(m::SphericalHarmonics, ::Type{T}) where {T<:Tuple} = all(S -> supports_density(m, S), fieldtypes(T))
supports_output(::SphericalHarmonics, out) = out isa Potential || out isa Gradient

struct HarmonicExtras{T}
    scale::T                 # 1/w
    tb::HarmonicTables{T}
end

function level_operators(m::SphericalHarmonics{K,T}, w::Real) where {K,T}
    p = m.p
    w = T(w)
    N = (p + 1)^2
    s = scaling_factors(2p)
    blocksM = Matrix{T}[]; blocksL = Matrix{T}[]
    A = zeros(ComplexF64, N, N); B = zeros(ComplexF64, N, N)
    for δ in child_offsets(Val(3))
        δ̂ = (SVector{3,Float64}(δ) .- 0.5) / 2
        Y = regular_hobson(δ̂, p)
        fill!(A, 0); fill!(B, 0)
        for n′ in 0:p, m′ in -n′:n′, n in n′:p
            d = n - n′
            for mm in max(-n, m′ - d):min(n, m′ + d)
                v = conj(Y[cidx(d, mm - m′)]) * (s[cidx(n, mm)] / s[cidx(n′, m′)])
                A[cidx(n, mm), cidx(n′, m′)] = v * 2.0^(-n′)             # M2M
                B[cidx(n′, m′), cidx(n, mm)] = v * 2.0^(-n′)             # L2L
            end
        end
        push!(blocksM, Matrix{T}(real_operator(A, p)))
        push!(blocksL, Matrix{T}(real_operator(B, p)))
    end
    c = Float64(m.kernel.c) / Float64(w)
    M2L = Vector{M2LOperator{T}}(undef, 7^3 - 3^3)
    for (f, off) in enumerate(far_offsets(Val(3)))
        Θ = irregular_hobson(SVector{3,Float64}(off), 2p)
        for k in 0:p, l in -k:k
            sk = (-1)^k * c / s[cidx(k, l)]
            for n in 0:p, mm in -n:n
                A[cidx(n, mm), cidx(k, l)] = sk * Θ[cidx(n + k, mm + l)] / s[cidx(n, mm)]
            end
        end
        M2L[f] = M2LOperator(Matrix{T}(real_operator(A, p)))
    end
    return LevelOperators(w, reduce(hcat, blocksM), reduce(vcat, blocksL), M2L,
                          HarmonicExtras{T}(1 / w, harmonic_tables(p, T)))::operators_type(m)
end

# M2M and L2L are width independent; the M2L carries 1/w.
function rescale(ref::LevelOperators{T,R,HarmonicExtras{R}}, m::SphericalHarmonics, w::Real) where {T,R}
    λ = R(w) / ref.width
    return typeof(ref)(R(w), ref.M2M, ref.L2L, ref.M2L, ref.m2m_scale, ref.l2l_scale,
                       ref.m2l_scale / λ, HarmonicExtras{R}(ref.extra.scale / λ, ref.extra.tb))
end

function s2m!(col::AbstractVector, m::SphericalHarmonics, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{3})
    s = ops.extra.scale
    tb = ops.extra.tb
    p = m.p
    @inbounds for n in eachindex(col)
        col[n] = zero(eltype(col))
    end
    @inbounds for i in eachindex(xs)
        _accumulate!(col, (xs[i] - c) * s, p, tb, _scaled_density(τs[i], s))
    end
    return col
end
@inline _scaled_density(τ::Number, s) = τ
@inline _scaled_density(d::Dipole, s) = Dipole(d.d * s)
@inline _scaled_density(τ::Tuple, s) = map(t -> _scaled_density(t, s), τ)

partial_to_multipole!(M::AbstractVecOrMat, ::SphericalHarmonics, ::LevelOperators, pM::AbstractVecOrMat, work) = copyto!(M, pM)
partial_to_local!(L::AbstractVecOrMat, ::SphericalHarmonics, ::LevelOperators, pL::AbstractVecOrMat, work) = copyto!(L, pL)

"""
    l2t(m, ops, L, xrel, out=Potential())

`Σ L_n^m conj Υ̃_n^m(x̂)` or its gradient at `xrel` relative to the box center.
"""
@inline function l2t(m::SphericalHarmonics, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Potential=Potential())
    x̂ = xrel * ops.extra.scale
    S = promote_type(eltype(L), typeof(x̂[1]))
    return _fold_regular(zero(S), x̂, m.p, ops.extra.tb) do acc, n, mm, vr, vi
        lr, li = _get(L, n, mm)
        mm == 0 ? acc + lr * vr : acc + 2 * (lr * vr + li * vi)
    end
end
# ∇Φ = (1/w) Σ_{n',m'} A conj Υ̃_{n'}^{m'}(x̂), the three coefficient sets A^x, A^y, A^z of degree
# p−1 following from the derivative identities (A_{n'}^{m'} combines L_{n'+1}^{m'±1}, L_{n'+1}^{m'})
@inline function l2t(m::SphericalHarmonics, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Gradient)
    s = ops.extra.scale
    x̂ = xrel * s
    tb = ops.extra.tb
    S = promote_type(eltype(L), typeof(x̂[1]))
    g = _fold_regular(zero(SVector{3,S}), x̂, m.p - 1, tb) do acc, n′, m′, vr, vi
        n = n′ + 1
        @inbounds if m′ == 0
            lr, li = _get(L, n, 1)
            c = tb.cm[tidx(n, 1)]
            ax = -c * lr; ay = -c * li; az = tb.cz[tidx(n, 0)] * L[n*n+1]
            return acc + SVector(ax * vr, ay * vr, az * vr)
        else
            ur, ui = _get(L, n, m′ + 1); dr, di = _get(L, n, m′ - 1)
            cu = tb.cm[tidx(n, m′ + 1)] / 2; cd = tb.cp[tidx(n, m′ - 1)] / 2
            axr = -cu * ur + cd * dr; axi = -cu * ui + cd * di            # A^x = ½(−cm L⁺ + cp L⁻)
            ayr = -(cu * ui + cd * di); ayi = cu * ur + cd * dr           # A^y = (i/2)(cm L⁺ + cp L⁻)
            cz = tb.cz[tidx(n, m′)]
            zr, zi = _get(L, n, m′)
            return acc + 2 * SVector(axr * vr + axi * vi, ayr * vr + ayi * vi, cz * (zr * vr + zi * vi))
        end
    end
    return g * s
end

"""
    multipole_field(m, ops, M, xrel)

`(c_K/w) Σ M_n^m Θ̃_n^m(x̂)` at `xrel` relative to the box center. Used by tests.
"""
function multipole_field(m::SphericalHarmonics, ops::LevelOperators, M::AbstractVector, xrel::SVector{3})
    p = m.p
    x̂ = xrel * ops.extra.scale
    r = norm(x̂)
    Y = regular_hobson(x̂, p)
    s = scaling_factors(p)
    c = complex_coefficients(M, p)
    acc = zero(promote_type(eltype(c), eltype(Y)))
    for n in 0:p, mm in -n:n
        acc += c[cidx(n, mm)] * Y[cidx(n, mm)] * s[cidx(n, mm)] / r^(2n + 1)
    end
    v = acc * (m.kernel.c * ops.extra.scale)
    return eltype(M) <: Real ? real(v) : v
end
