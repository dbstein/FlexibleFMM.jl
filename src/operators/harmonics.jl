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
struct HarmonicTables{T,V<:AbstractVector{T}}       # V: a host or a device vector type
    p::Int
    a::V
    b::V
    e::V
    cz::V
    cm::V
    cp::V
end
HarmonicTables(p::Int, a::V, b::V, e::V, cz::V, cm::V, cp::V) where {V} = HarmonicTables{eltype(V),V}(p, a, b, e, cz, cm, cp)
Adapt.@adapt_structure HarmonicTables                 # so that a kernel receives the tables as device arrays
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
    return HarmonicTables(p, a, b, T.(e), cz, cm, cp)
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

# M_n^m += w(n) τ conj Υ̃_n^m(ŷ); the degree weight w is 1 for Laplace and the radial
# factor u_n(κ̂ r) for Yukawa
@inline _unit(n) = 1
@inline _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ) = _accumulate!(col, x̂, p, tb, τ, _unit)
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Number, w::F) where {F}
    _fold_regular(nothing, x̂, p, tb) do _, n, m, vr, vi
        t = τ * w(n)
        _acc!(col, n, m, t * vr, -t * vi)
        nothing
    end
    return col
end
# dipole: M_n^m += w(n) conj(d̂·∇Υ̃_n^m(ŷ)), d̂ = d/w, by the derivative identities; the
# value Υ̃_{n'}^{m'} feeds (n'+1, m'+1), (n'+1, m') and (n'+1, m'−1), the last folded
# into the real entry (n'+1, 0) when m' = 1
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, d::Dipole{3}, w::F) where {F}
    dx, dy, dz = d.d[1], d.d[2], d.d[3]
    _fold_regular(nothing, x̂, p - 1, tb) do _, n′, m′, ur, ui
        n = n′ + 1
        wn = w(n)
        @inbounds begin
            c = wn * tb.cm[tidx(n, m′ + 1)] / 2                          # −½ cm d₋ conj u
            _acc!(col, n, m′ + 1, -c * (dx * ur - dy * ui), c * (dx * ui + dy * ur))
            c = wn * tb.cz[tidx(n, m′)] * dz                             # cz d_z conj u
            _acc!(col, n, m′, c * ur, -c * ui)
            if m′ >= 2
                c = wn * tb.cp[tidx(n, m′ - 1)] / 2                      # ½ cp d₊ conj u
                _acc!(col, n, m′ - 1, c * (dx * ur + dy * ui), c * (dy * ur - dx * ui))
            elseif m′ == 1
                c = wn * tb.cp[tidx(n, 0)]                                # cp Re(d₋ u)
                _acc!(col, n, 0, c * (dx * ur + dy * ui), zero(ur))
            end
        end
        nothing
    end
    return col
end
# tuples by recursion (a loop over a tuple is a dynamic getfield, which the device
# compilers reject in the S2M kernels)
@inline function _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple, w::F) where {F}
    _accumulate!(col, x̂, p, tb, first(τ), w)
    return _accumulate!(col, x̂, p, tb, Base.tail(τ), w)
end
@inline _accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, ::Tuple{}, w) = col

# ---- the method ----

abstract type HarmonicMethod <: AbstractMethod end            # the methods on solid harmonics (Laplace, Stokes)

struct SphericalHarmonics{K<:Union{Laplace3D,Yukawa3D},T,L} <: HarmonicMethod      # L: the M2L operator type
    kernel::K
    p::Int
    tol::Float64
end

"""
    SphericalHarmonics(kernel::Laplace3D; p=16, tol=0)

The analytic multipole method for the 3D Laplace kernel: expansions in real solid
harmonics up to degree `p` (`(p+1)²` coefficients), exact M2M and L2L, dense M2L
operators per far offset, or, for `tol > 0`, their truncated SVDs with the singular
values below `tol·σ₁` dropped (`LowRankM2L`; the translation error then floors near
`tol`). Charges and dipoles; potential and gradient outputs.
"""
function SphericalHarmonics(kernel::Laplace3D{T}; p::Integer=16, tol::Real=0) where {T}
    p >= 1 || throw(ArgumentError("p must be at least 1"))
    0 <= tol < 1 || throw(ArgumentError("tol must lie in [0, 1)"))
    L = tol > 0 ? LowRankM2L{T} : M2LOperator{T}
    return SphericalHarmonics{typeof(kernel),T,L}(kernel, Int(p), Float64(tol))
end

order(m::SphericalHarmonics) = m.p
_nblocks(::SphericalHarmonics) = 1                         # scalar expansions per box
dimension(::SphericalHarmonics) = 3
expansion_length(m::SphericalHarmonics) = (m.p + 1)^2
near_kernel(m::SphericalHarmonics) = m.kernel
operator_eltype(::SphericalHarmonics{K,T}) where {K,T} = T
operators_type(::SphericalHarmonics{K,T,L}) where {K,T,L} = LevelOperators{T,T,HarmonicExtras{T,L},PermutedM2L{L}}
supports_density(::SphericalHarmonics, ::Type) = false
supports_density(::SphericalHarmonics, ::Type{<:Number}) = true
supports_density(::SphericalHarmonics, ::Type{<:Dipole{3}}) = true
supports_density(m::SphericalHarmonics, ::Type{T}) where {T<:Tuple} = all(S -> supports_density(m, S), fieldtypes(T))
supports_output(::SphericalHarmonics, out) = out isa Potential || out isa Gradient

"""
    PermutedM2L{L}

The translation operator of one far offset as its class representative `op` (of
type `L`, dense or low rank, shared by the offsets of the class) and the signed
permutations of `SymmetryClasses`, `M2L_o·m = P_t·(op·(P_sᵀ·m))`. The class-wise
`m2l_level!` of the harmonic methods applies the permutations in its gather and
scatter; `apply!` here permutes per call (allocating), for the checks.
"""
struct PermutedM2L{L}
    op::L
    idx_s::Vector{Int}
    sgn_s::Vector{Int8}
    idx_t::Vector{Int}
    sgn_t::Vector{Int8}
end
Base.size(op::PermutedM2L) = size(op.op)
workspace_rows(op::PermutedM2L) = workspace_rows(op.op)
function apply!(out::AbstractVecOrMat, op::PermutedM2L, x::AbstractVecOrMat, α=true)
    xp = similar(x); y = similar(out)
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        xp[i, j] = op.sgn_s[i] * x[op.idx_s[i], j]
    end
    apply!(y, op.op, xp, α)
    @inbounds for j in axes(y, 2), i in axes(y, 1)
        out[op.idx_t[i], j] = op.sgn_t[i] * y[i, j]
    end
    return out
end
apply!(out::AbstractVecOrMat, op::PermutedM2L, x::AbstractVecOrMat, α, work) = apply!(out, op, x, α)

struct HarmonicExtras{T,L}
    scale::T                 # 1/w
    tb::HarmonicTables{T,Vector{T}}
    sc::SymmetryClasses{3}   # the 34 classes of the far offsets and their permutations
    reps::Vector{L}          # the class representatives' operators (what `M2L` refers to)
    offs::Vector{SVector{3,Int}}
    κ̂::T                     # κ·w for Yukawa, 0 for Laplace
end
_rescaled(e::HarmonicExtras{T,L}, λ) where {T,L} = HarmonicExtras{T,L}(e.scale / λ, e.tb, e.sc, e.reps, e.offs, e.κ̂)

# the dense real M2L matrix of the far offset `o` for the kernel constant `c` (c_K/w)
function _harmonic_m2l(o::SVector{3,Int}, p::Int, c::Float64, s::Vector{Float64}, A::Matrix{ComplexF64})
    Θ = irregular_hobson(SVector{3,Float64}(o), 2p)
    for k in 0:p, l in -k:k
        sk = (-1)^k * c / s[cidx(k, l)]
        for n in 0:p, mm in -n:n
            A[cidx(n, mm), cidx(k, l)] = sk * Θ[cidx(n + k, mm + l)] / s[cidx(n, mm)]
        end
    end
    return real_operator(A, p)
end

function level_operators(m::SphericalHarmonics{K,T,L}, w::Real) where {K<:Laplace3D,T,L}
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
    sc = harmonic_symmetry_classes(p)
    reps = L[_m2l_operator(L, _harmonic_m2l(r, p, c, s, A), m.tol) for r in sc.reps]
    M2L = [PermutedM2L{L}(reps[sc.class[f]], sc.idx[f], sc.sgn[f], sc.idx_t[f], sc.sgn_t[f]) for f in eachindex(sc.class)]
    return LevelOperators(w, reduce(hcat, blocksM), reduce(vcat, blocksL), M2L,
                          HarmonicExtras{T,L}(1 / w, harmonic_tables(p, T), sc, reps, far_offsets(Val(3)), zero(T)))::operators_type(m)
end

_m2l_operator(::Type{M2LOperator{T}}, R::Matrix{Float64}, tol) where {T} = M2LOperator(Matrix{T}(R))
function _m2l_operator(::Type{LowRankM2L{T}}, R::Matrix{Float64}, tol) where {T}
    F = svd(R)
    r = max(1, count(σ -> σ >= tol * F.S[1], F.S))
    return LowRankM2L{T}(Matrix{T}(F.U[:, 1:r] .* transpose(F.S[1:r])), Matrix{T}(F.Vt[1:r, :]))
end

# M2M and L2L are width independent; the M2L carries 1/w.
function rescale(ref::LevelOperators{T,R,<:HarmonicExtras{R}}, m::SphericalHarmonics, w::Real) where {T,R}
    λ = R(w) / ref.width
    return typeof(ref)(R(w), ref.M2M, ref.L2L, ref.M2L, ref.m2m_scale, ref.l2l_scale,
                       ref.m2l_scale / λ, _rescaled(ref.extra, λ))
end

function s2m!(col::AbstractVector, m::SphericalHarmonics, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{3})
    fill!(col, zero(eltype(col)))
    return s2m_accumulate!(col, m, ops, xs, τs, c)
end
function s2m_accumulate!(col::AbstractVector, m::SphericalHarmonics, ops::LevelOperators, xs::AbstractVector,
                         τs::AbstractVector, c::SVector{3})
    s = ops.extra.scale
    tb = ops.extra.tb
    p = m.p
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
@inline l2t(m::SphericalHarmonics, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Potential=Potential()) =
    _harmonic_potential(L, xrel * ops.extra.scale, m.p, ops.extra.tb)
@inline function l2t(m::SphericalHarmonics, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Gradient)
    s = ops.extra.scale
    return _harmonic_gradient(L, xrel * s, m.p, ops.extra.tb) * s
end
# the bodies, shared with the device kernels: the value and the scaled gradient at x̂,
# with a degree weight (1 for Laplace, the radial factors for Yukawa)
@inline _harmonic_potential(L, x̂::SVector{3}, p::Int, tb::HarmonicTables) = _harmonic_potential(L, x̂, p, tb, _unit)
@inline function _harmonic_potential(L, x̂::SVector{3}, p::Int, tb::HarmonicTables, w::F) where {F}
    S = promote_type(eltype(L), typeof(x̂[1]))
    return _fold_regular(zero(S), x̂, p, tb) do acc, n, mm, vr, vi
        lr, li = _get(L, n, mm)
        mm == 0 ? acc + w(n) * (lr * vr) : acc + w(n) * (2 * (lr * vr + li * vi))
    end
end
# ∇̂Φ = Σ_{n',m'} A conj Υ̃_{n'}^{m'}(x̂), the three coefficient sets A^x, A^y, A^z of degree
# p−1 following from the derivative identities (A_{n'}^{m'} combines L_{n'+1}^{m'±1}, L_{n'+1}^{m'})
@inline _harmonic_gradient(L, x̂::SVector{3}, p::Int, tb::HarmonicTables) = _harmonic_gradient(L, x̂, p, tb, _unit)
@inline function _harmonic_gradient(L, x̂::SVector{3}, p::Int, tb::HarmonicTables, w::F) where {F}
    S = promote_type(eltype(L), typeof(x̂[1]))
    return _fold_regular(zero(SVector{3,S}), x̂, p - 1, tb) do acc, n′, m′, vr, vi
        n = n′ + 1
        wn = w(n)
        @inbounds if m′ == 0
            lr, li = _get(L, n, 1)
            c = tb.cm[tidx(n, 1)]
            ax = -c * lr; ay = -c * li; az = tb.cz[tidx(n, 0)] * L[n*n+1]
            return acc + wn * SVector(ax * vr, ay * vr, az * vr)
        else
            ur, ui = _get(L, n, m′ + 1); dr, di = _get(L, n, m′ - 1)
            cu = tb.cm[tidx(n, m′ + 1)] / 2; cd = tb.cp[tidx(n, m′ - 1)] / 2
            axr = -cu * ur + cd * dr; axi = -cu * ui + cd * di            # A^x = ½(−cm L⁺ + cp L⁻)
            ayr = -(cu * ui + cd * di); ayi = cu * ur + cd * dr           # A^y = (i/2)(cm L⁺ + cp L⁻)
            cz = tb.cz[tidx(n, m′)]
            zr, zi = _get(L, n, m′)
            return acc + (2 * wn) * SVector(axr * vr + axi * vi, ayr * vr + ayi * vi, cz * (zr * vr + zi * vi))
        end
    end
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

# ---- the class-wise translation of a level ----
#
# For each of the 34 classes, the pairs of its offsets are gathered with the source
# permutations into one matrix (one column per pair and scalar expansion; the Stokes
# method has four, laid out as consecutive columns), translated by the class
# representative in one GEMM (two for the low-rank form) and scattered with the
# target permutations, plus the Stokes coupling Ψ̂ += o·Φ with each pair's own
# offset. The pairs of a class are listed in the interaction list's target order
# (`ClassList`, built once per state and level; the device plan cuts its chunks from
# the same lists). The level runs as ONE thread region: the targets are cut into
# blocks of `_TARGET_BLOCK` consecutive targets, and each task takes blocks from a
# counter and, per class, gathers the block's pairs (a contiguous range of the
# class list) into its own workspace, applies the representative, and scatters into
# the block's targets, which no other task touches. The products run with a single
# BLAS thread (the region sets the count to one and restores it): many tasks calling
# a threaded BLAS serialise on its queue. A region per class and level (the first
# version: 340 per matvec) cost 30 ms each on a 96-core node where waking a sleeping
# thread takes 0.3 ms (`bench/threads.jl`), 10 s of a 14 s matvec. The block size is
# fixed, not a function of the thread count, so that the chunking of the products and
# hence the results do not depend on it.
const _TARGET_BLOCK = 32
const _TASK_COLUMNS = 512         # pairs per gather/product/scatter chunk of a task

function m2l_level!(pL::AbstractMatrix, m::HarmonicMethod, ops::LevelOperators,
                    Mp::AbstractMatrix, il::InteractionList, off::Int, state::FMMState)
    nb = _nblocks(m)
    N = (m.p + 1)^2
    e = ops.extra
    nt = length(il.torder)
    nt == 0 && return pL
    lists = _class_lists(state, il, e.sc.class, length(e.reps), off + 1)
    rmax = maximum(workspace_rows, e.reps)
    ws = _task_workspaces(state, N, nb * _TASK_COLUMNS, rmax)
    nblocks = cld(nt, _TARGET_BLOCK)
    next = Threads.Atomic{Int}(1)
    nblas = BLAS.get_num_threads()
    nblas > 1 && BLAS.set_num_threads(1)
    try
        Threads.@threads :dynamic for t in 1:length(ws)         # one task per workspace
            G, Rw, work = ws[t]
            while true
                blk = Threads.atomic_add!(next, 1)
                blk > nblocks && break
                g1 = (blk - 1) * _TARGET_BLOCK + 1
                g2 = min(blk * _TARGET_BLOCK, nt)
                for c in eachindex(e.reps)
                    _translate_targets!(pL, Mp, e.reps[c], e, lists[c], g1, g2, G, Rw, work, ops.m2l_scale, N, Val(nb))
                end
            end
        end
    finally
        nblas > 1 && BLAS.set_num_threads(nblas)
    end
    return pL
end
# the class lists of the level starting at node `lo`, built once per state (they
# depend on the tree and the classes only)
function _class_lists(state::FMMState, il::InteractionList, class::Vector{Int}, nclass::Int, lo::Int)
    key = (:class_lists, lo)
    v = get(state.scratch, key, nothing)
    if v === nothing
        v = class_lists(il, class, nclass, lo)
        state.scratch[key] = v
    end
    return v::Vector{ClassList}
end
# one workspace per task: the gathered columns, the product, and the low-rank
# intermediate (`rows × cols` each, `r × cols` for the last; `r = 0`: none)
function _task_workspaces(state::FMMState{S}, rows::Int, cols::Int, r::Int) where {S}
    key = (:task_workspaces, rows, cols, r)
    v = get(state.scratch, key, nothing)
    if v === nothing
        W = r == 0 ? Nothing : Matrix{S}
        v = Tuple{Matrix{S},Matrix{S},W}[(Matrix{S}(undef, rows, cols), Matrix{S}(undef, rows, cols),
                                          r == 0 ? nothing : Matrix{S}(undef, r, cols)) for _ in 1:Threads.nthreads()]
        state.scratch[key] = v
    end
    return v::Vector{<:Tuple{Matrix{S},Matrix{S},Union{Nothing,Matrix{S}}}}
end
# the pairs of class list `cl` whose targets are g1:g2 of the level's target order,
# in chunks that fit the task's workspace: gather, product, scatter
function _translate_targets!(pL, Mp, rep, e::HarmonicExtras, cl::ClassList, g1::Int, g2::Int, G, Rw, work, α, N, ::Val{nb}) where {nb}
    r1, r2 = _target_runs(cl, g1, g2)
    r1 > r2 && return pL
    a = Int(cl.ptr[r1])
    b = Int(cl.ptr[r2+1]) - 1
    cap = size(G, 2) ÷ nb
    sc = e.sc
    for ca in a:cap:b
        cb = min(ca + cap - 1, b)
        ncol = cb - ca + 1
        @inbounds for q in 1:ncol
            f = Int(cl.f[ca+q-1])
            _gather_column!(G, Mp, Int(cl.src[ca+q-1]), sc.idx[f], sc.sgn[f], q, N, Val(nb))
        end
        apply!(view(Rw, :, 1:nb*ncol), rep, view(G, :, 1:nb*ncol), α, work)
        s1, s2 = _run_range(cl, ca, cb)
        for r in s1:s2
            _scatter_target!(pL, Rw, sc, e.offs, cl, r, ca, cb, N, Val(nb))
        end
    end
    return pL
end
@inline function _gather_column!(G, Mp, src::Int, idx, sgn, q::Int, N::Int, ::Val{nb}) where {nb}
    @inbounds for a in 1:nb
        col = (q - 1) * nb + a
        o = (a - 1) * N
        for i in 1:N
            G[i, col] = sgn[i] * Mp[o+idx[i], src]
        end
    end
    return G
end
# the pairs of run r that lie in the chunk a:b (a target may span two chunks), into its column of pL
function _scatter_target!(pL, Rw, sc::SymmetryClasses, offs, cl::ClassList, r::Int, a::Int, b::Int, N::Int, ::Val{nb}) where {nb}
    @inbounds begin
        t = Int(cl.tgt[r])
        for q in max(Int(cl.ptr[r]), a):min(Int(cl.ptr[r+1]) - 1, b)
            f = Int(cl.f[q])
            idx = sc.idx_t[f]; sgn = sc.sgn_t[f]
            col = (q - a) * nb
            for blk in 1:nb
                o = (blk - 1) * N
                if nb == 4 && blk == 4                     # Ψ̂ += o·Φ, with the pair's own offset
                    ov = offs[f]
                    for i in 1:N
                        pL[o+idx[i], t] += sgn[i] * (Rw[i, col+4] + ov[1] * Rw[i, col+1] + ov[2] * Rw[i, col+2] + ov[3] * Rw[i, col+3])
                    end
                else
                    for i in 1:N
                        pL[o+idx[i], t] += sgn[i] * Rw[i, col+blk]
                    end
                end
            end
        end
    end
    return pL
end

"""
    check_harmonic_classes(p; rtol=1e-12) -> maximum relative error

Verifies `M2L_o = P_t·M2L_rep·P_sᵀ` on every far offset against the dense matrices
of the harmonic basis of degree `p`.
"""
function check_harmonic_classes(p::Int; rtol=1e-12)
    sc = harmonic_symmetry_classes(p)
    s = scaling_factors(2p)
    A = zeros(ComplexF64, (p + 1)^2, (p + 1)^2)
    reps = [_harmonic_m2l(r, p, 1.0, s, A) for r in sc.reps]
    worst = 0.0
    for (f, o) in enumerate(far_offsets(Val(3)))
        M = _harmonic_m2l(o, p, 1.0, s, A)
        R = reps[sc.class[f]]
        B = similar(M)
        it = sc.idx_t[f]; st = sc.sgn_t[f]; is = sc.idx[f]; ss = sc.sgn[f]
        for j in axes(M, 2), i in axes(M, 1)
            B[it[i], is[j]] = st[i] * ss[j] * R[i, j]
        end
        worst = max(worst, maximum(abs.(M - B)) / maximum(abs.(M)))
    end
    worst <= rtol || throw(ArgumentError("the harmonic symmetry classes do not reproduce the translation operators (error $worst)"))
    return worst
end

# ---- one order at a time (the device S2M kernel: one work-item per node and order) ----

"""
    _fold_chain(f, acc, x̂, m, nmax, tb)

`acc = f(acc, n, m, Re Υ̃_n^m(x̂), Im Υ̃_n^m(x̂))` over `n = m..nmax` for one order `m`,
the sectoral start `Υ̃_m^m` built directly (a product of `m` factors), so that the
chains of different orders are independent.
"""
@inline function _fold_chain(f::F, acc, x̂::SVector{3}, m::Int, nmax::Int, tb::HarmonicTables) where {F}
    x, y, z = x̂[1], x̂[2], x̂[3]
    r2 = x * x + y * y + z * z
    sr = one(x); si = zero(x)
    @inbounds for k in 1:m
        e = tb.e[k]
        sr, si = -e * (x * sr - y * si), -e * (x * si + y * sr)
    end
    acc = f(acc, m, m, sr, si)
    pr = zero(sr); pi = zero(si); cr = sr; ci = si
    @inbounds for n in m+1:nmax
        t = tidx(n, m)
        az = tb.a[t] * z; br = tb.b[t] * r2
        nr = az * cr - br * pr; ni = az * ci - br * pi
        acc = f(acc, n, m, nr, ni)
        pr = cr; pi = ci; cr = nr; ci = ni
    end
    return acc
end

# the contributions of one source to the entries of order `m` only; a dipole's come
# from the chains m−1, m, m+1 (the three terms of the derivative identities)
@inline function _accumulate_order!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Number, m::Int)
    _fold_chain(nothing, x̂, m, p, tb) do _, n, mm, vr, vi
        _acc!(col, n, mm, τ * vr, -τ * vi)
        nothing
    end
    return col
end
@inline function _accumulate_order!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, d::Dipole{3}, m::Int)
    dx, dy, dz = d.d[1], d.d[2], d.d[3]
    if m >= 1                                                    # from the chain m−1: the (n′+1, m′+1) term
        _fold_chain(nothing, x̂, m - 1, p - 1, tb) do _, n′, m′, ur, ui
            n = n′ + 1
            c = (@inbounds tb.cm[tidx(n, m)]) / 2
            _acc!(col, n, m, -c * (dx * ur - dy * ui), c * (dx * ui + dy * ur))
            nothing
        end
    end
    if m <= p - 1                                                # from the chain m: the (n′+1, m′) term
        _fold_chain(nothing, x̂, m, p - 1, tb) do _, n′, m′, ur, ui
            n = n′ + 1
            c = (@inbounds tb.cz[tidx(n, m)]) * dz
            _acc!(col, n, m, c * ur, -c * ui)
            nothing
        end
    end
    if m + 1 <= p - 1                                            # from the chain m+1: the (n′+1, m′−1) term
        _fold_chain(nothing, x̂, m + 1, p - 1, tb) do _, n′, m′, ur, ui
            n = n′ + 1
            if m >= 1
                c = (@inbounds tb.cp[tidx(n, m)]) / 2
                _acc!(col, n, m, c * (dx * ur + dy * ui), c * (dy * ur - dx * ui))
            else
                c = @inbounds tb.cp[tidx(n, 0)]
                _acc!(col, n, 0, c * (dx * ur + dy * ui), zero(ur))
            end
            nothing
        end
    end
    return col
end
@inline function _accumulate_order!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple, m::Int)
    _accumulate_order!(col, x̂, p, tb, first(τ), m)
    return _accumulate_order!(col, x̂, p, tb, Base.tail(τ), m)
end
@inline _accumulate_order!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, ::Tuple{}, m::Int) = col
# zero the entries of order m of a column
@inline function _zero_order!(col, p::Int, m::Int)
    @inbounds for n in m:p
        if m == 0
            col[n*n+1] = zero(eltype(col))
        else
            col[n*n+2m] = zero(eltype(col)); col[n*n+2m+1] = zero(eltype(col))
        end
    end
    return col
end
