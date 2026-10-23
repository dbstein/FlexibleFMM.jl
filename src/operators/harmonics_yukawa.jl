# The Yukawa kernel e^{−κr}/(4πr) on the harmonic basis. With the modified spherical
# Bessel functions normalised to the Laplace limit,
#   u_n(z) = (2n+1)!! i_n(z)/z^n,   v_n(z) = (2/π) z^{n+1} k_n(z)/(2n−1)!!     (→ 1 as z → 0),
# the generating identity keeps the Laplace form with per-degree radial factors,
#   e^{−κ|x−y|}/|x−y| = Σ_n u_n(κ r_<) v_n(κ r_>) Σ_m S̃_n^m(x̂) conj S̃_n^m(ŷ),
# so the regular basis is u_n(κ̂ r) Υ̃_n^m and the irregular one v_n(κ̂ r) Θ̃_n^m in box
# coordinates (κ̂ = κ w), the multipoles and locals are the Laplace folds with the
# weight u_n, and the symmetry classes still hold (a radial kernel). The translations
# couple all degrees (the Yukawa addition theorems carry Gaunt coefficients), so M2M,
# L2L and the 34 M2L representatives are built per width by projection on a sphere,
# where the real surface harmonics are orthogonal: a Gauss × trapezoid rule of
# (2p+10) × (4p+20) points integrates the products exactly up to degree 4p+19, and
# the angular content of the fields decays geometrically beyond. For κ̂ r ≳ p the
# Yukawa series converge slowly relative to their own far field (the polynomial
# factors of k_n), which only matters for that exponentially small part. The radial factors
# follow stable recurrences: u by Miller's backward recurrence normalised with
# u_0 = sinh z / z, v forward from v_0 = e^{−z}, v_1 = e^{−z}(1 + z).

"""
    SphericalHarmonics(kernel::Yukawa3D; p=16, tol=0)

The harmonic method for the Yukawa kernel `e^{−κr}/(4πr)`: the Laplace expansions
with the radial factors of the modified spherical Bessel functions, translation
operators built per width by projection. Charges and dipoles; potential and
gradient outputs; `p ≤ 60`.
"""
function SphericalHarmonics(kernel::Yukawa3D{T}; p::Integer=16, tol::Real=0) where {T}
    1 <= p <= 60 || throw(ArgumentError("p must lie in 1:60"))
    0 <= tol < 1 || throw(ArgumentError("tol must lie in [0, 1)"))
    L = tol > 0 ? LowRankM2L{T} : M2LOperator{T}
    return SphericalHarmonics{typeof(kernel),T,L}(kernel, Int(p), Float64(tol))
end

# ---- the radial factors ----

@inline _yk_value(x::Real) = x
@inline _yk_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# u[n+1] = u_n(z), n = 0..nmax: backward recurrence u_n = u_{n+1} + z² u_{n+2}/((2n+3)(2n+5))
# from a start above max(nmax, z), normalised with u_0 = sinh z / z
# u[n+1] = u_n(z) from z² (u_n is even in z, and the squared radius keeps the
# derivative smooth at the box centre, where |x̂| is not differentiable)
@inline function _yukawa_u!(u, z2, nmax::Int)
    if _yk_value(z2) < 1e-8                                    # the series' first two terms
        @inbounds for n in 0:nmax
            u[n+1] = 1 + z2 / (2 * (2n + 3))
        end
        return u
    end
    z = sqrt(z2)
    N = max(nmax, ceil(Int, _yk_value(z))) + 25
    a = zero(z); b = one(z)                                    # u_{N+1}, u_N (unnormalised)
    @inbounds for n in N-1:-1:0
        c = b + z2 * a / ((2n + 3) * (2n + 5))
        n <= nmax && (u[n+1] = c)
        a = b; b = c
    end
    s = (sinh(z) / z) / b
    @inbounds for n in 0:nmax
        u[n+1] *= s
    end
    return u
end
# v[n+1] = v_n(z): v_{n+1} = v_n + z² v_{n−1}/((2n−1)(2n+1))
@inline function _yukawa_v!(v, z, nmax::Int)
    e = exp(-z)
    z2 = z * z
    @inbounds begin
        v[1] = e
        nmax >= 1 && (v[2] = e * (1 + z))
        for n in 1:nmax-1
            v[n+2] = v[n+1] + z2 * v[n] / ((2n - 1) * (2n + 1))
        end
    end
    return v
end

# ---- the operators by projection (build time, Float64) ----

function _gauss_legendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    F = eigen(SymTridiagonal(zeros(n), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end
# the product rule on the unit sphere: unit vectors and weights
function _sphere_quadrature(p::Int)
    nθ = 2p + 10; nφ = 4p + 20                                 # exact to degree 4p+19: the aliasing of the
                                                               # geometrically decaying angular content is far below the truncation
    t, wt = _gauss_legendre(nθ)
    pts = Vector{SVector{3,Float64}}(undef, nθ * nφ); wts = Vector{Float64}(undef, nθ * nφ)
    q = 0
    for (c, w) in zip(t, wt), k in 0:nφ-1
        φ = 2π * k / nφ
        s = sqrt(max(0.0, 1 - c^2))
        q += 1
        pts[q] = SVector(s * cos(φ), s * sin(φ), c)
        wts[q] = w * 2π / nφ
    end
    return pts, wts
end
# the regular real basis at y: u_n(κ̂|y|) [Υ̃_n^0 | 2 Re Υ̃ | 2 Im Υ̃](y)
function _yukawa_regular!(b, y::SVector{3}, κ̂, p::Int, tb::HarmonicTables, u)
    _yukawa_u!(u, (κ̂ * norm(y))^2, p)
    _fold_regular(nothing, y, p, tb) do _, n, m, vr, vi
        w = u[n+1]
        if m == 0
            b[n*n+1] = w * vr
        else
            b[n*n+2m] = 2w * vr; b[n*n+2m+1] = 2w * vi
        end
        nothing
    end
    return b
end
# the irregular real basis at y: v_n(κ̂|y|) [Θ̃_n^0 | 2 Re Θ̃ | −2 Im Θ̃](y), Θ̃ = Υ̃/|y|^{2n+1}
function _yukawa_irregular!(b, y::SVector{3}, κ̂, p::Int, tb::HarmonicTables, v)
    r = norm(y)
    _yukawa_v!(v, κ̂ * r, p)
    ir = 1 / r
    _fold_regular(nothing, y, p, tb) do _, n, m, vr, vi
        w = v[n+1] * ir^(2n + 1)
        if m == 0
            b[n*n+1] = w * vr
        else
            b[n*n+2m] = 2w * vr; b[n*n+2m+1] = -2w * vi
        end
        nothing
    end
    return b
end
# the projection onto the regular (or irregular) basis on the sphere of radius ρ:
# coefficient_i = Σ_q Φ(ρ x̂_q) β_i(x̂_q) w_q / (γ_i f_{n_i}), f = u_n(κ̂ρ) ρ^n or v_n(κ̂ρ) ρ^{−n−1},
# γ_i = 4π/(2n+1) for m = 0 and 8π/(2n+1) otherwise (the imaginary rows of the
# irregular basis carry the sign of its definition)
function _projection(p::Int, κ̂::Float64, ρ::Float64, irregular::Bool, pts, wts, tb)
    N = (p + 1)^2
    B = zeros(N, length(pts))
    buf = zeros(N)
    for (q, x) in enumerate(pts)
        _fold_regular(nothing, x, p, tb) do _, n, m, vr, vi
            if m == 0
                buf[n*n+1] = vr
            else
                buf[n*n+2m] = 2vr; buf[n*n+2m+1] = 2vi
            end
            nothing
        end
        B[:, q] = buf .* wts[q]
    end
    rad = zeros(p + 1)
    irregular ? _yukawa_v!(rad, κ̂ * ρ, p) : _yukawa_u!(rad, (κ̂ * ρ)^2, p)
    for n in 0:p
        f = irregular ? rad[n+1] * ρ^(-n - 1) : rad[n+1] * ρ^n
        γ0 = 4π / (2n + 1)
        for i in n*n+1:(n+1)^2
            k = i - n * n
            γ = k == 1 ? γ0 : 2γ0
            sg = (irregular && k > 1 && isodd(k)) ? -1.0 : 1.0
            B[i, :] .*= sg / (γ * f)
        end
    end
    return B
end
# the operators of one width in box coordinates (κ̂ = κ w); the M2L carries c_K/w
function _yukawa_operators(p::Int, κ̂::Float64, cw::Float64, sc::SymmetryClasses, tb::HarmonicTables)
    N = (p + 1)^2
    pts, wts = _sphere_quadrature(p)
    nq = length(pts)
    # the projection spheres: the target's local at ρ = ½ and the parent's multipole at
    # ρ = 2 keep the angular content of the projected fields decaying like ¼ and 0.22
    # per degree (the source is at |o| ≥ 2, the child centre at |δ̂| = 0.43), far from
    # the rule's exactness limit; the child's local at ρ = 1
    ρt = 0.5; ρp = 2.0; ρc = 1.0
    Preg = _projection(p, κ̂, ρt, false, pts, wts, tb)         # onto the regular basis at the target
    Pirr = _projection(p, κ̂, ρp, true, pts, wts, tb)          # onto the parent's irregular basis
    Pregc = _projection(p, κ̂ / 2, ρc, false, pts, wts, tb)    # onto a child's regular basis
    F = zeros(N, nq)
    u = zeros(p + 1); v = zeros(p + 1)
    col = zeros(N)
    # M2L: the source's irregular basis at the target sphere
    reps = Matrix{Float64}[]
    for o in sc.reps
        oc = SVector{3,Float64}(o)
        for q in 1:nq
            _yukawa_irregular!(col, ρt * pts[q] - oc, κ̂, p, tb, v)
            F[:, q] = col
        end
        push!(reps, (cw * Preg) * transpose(F))
    end
    # M2M: the child's irregular basis (κ̂/2, child coordinates 2(x̂ − δ̂)) at the parent sphere,
    # with the child's 2/w against the parent's 1/w; L2L: the parent's regular basis at the child sphere
    blocksM = Matrix{Float64}[]; blocksL = Matrix{Float64}[]
    for δ in child_offsets(Val(3))
        δ̂ = (SVector{3,Float64}(δ) .- 0.5) / 2
        for q in 1:nq
            _yukawa_irregular!(col, 2 * (ρp * pts[q] - δ̂), κ̂ / 2, p, tb, v)
            F[:, q] = 2 .* col
        end
        push!(blocksM, Pirr * transpose(F))
        for q in 1:nq
            _yukawa_regular!(col, δ̂ + ρc * pts[q] / 2, κ̂, p, tb, u)
            F[:, q] = col
        end
        push!(blocksL, Pregc * transpose(F))
    end
    return reps, reduce(hcat, blocksM), reduce(vcat, blocksL)
end

function level_operators(m::SphericalHarmonics{K,T,L}, w::Real) where {K<:Yukawa3D,T,L}
    p = m.p
    w = T(w)
    κ̂ = Float64(m.kernel.k) * Float64(w)
    tb = harmonic_tables(p, Float64)
    sc = harmonic_symmetry_classes(p)
    reps64, M2M, L2L = _yukawa_operators(p, κ̂, Float64(m.kernel.c) / Float64(w), sc, tb)
    reps = L[_m2l_operator(L, R, m.tol) for R in reps64]
    M2L = [PermutedM2L{L}(reps[sc.class[f]], sc.idx[f], sc.sgn[f], sc.idx_t[f], sc.sgn_t[f]) for f in eachindex(sc.class)]
    return LevelOperators(w, Matrix{T}(M2M), Matrix{T}(L2L), M2L,
                          HarmonicExtras{T,L}(1 / w, harmonic_tables(p, T), sc, reps, far_offsets(Val(3)), T(κ̂)))::operators_type(m)
end

# ---- the per-point bodies: the Laplace folds with the radial weights ----
# A dipole adds the radial derivative: d·∇(u_n(κ̂r) Υ̃) = u_n d·∇Υ̃ + Υ̃ κ̂² u_{n+1} (d·x̂)/(2n+3),
# from u_n'(z) = z u_{n+1}(z)/(2n+3); the gradient of a local has the same two terms.

# u_0..u_nmax as an immutable vector, so that the weight functors are isbits and the
# bodies stay allocation-free
@inline function _radial_weights(z2, nmax::Int)
    u = MVector{64,typeof(z2)}(undef)
    _yukawa_u!(u, z2, nmax)
    return SVector(u)
end

struct _DegreeWeight{V}
    u::V
end
@inline (w::_DegreeWeight)(n) = @inbounds w.u[n+1]
struct _RadialWeight{V,T}
    u::V
    κ2::T
end
@inline (w::_RadialWeight)(n) = @inbounds w.κ2 * w.u[n+2] / (2n + 3)

@inline function _yukawa_accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Number, u, κ̂)
    return _accumulate!(col, x̂, p, tb, τ, _DegreeWeight(u))
end
@inline function _yukawa_accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, d::Dipole{3}, u, κ̂)
    _accumulate!(col, x̂, p, tb, d, _DegreeWeight(u))
    return _accumulate!(col, x̂, p, tb, sum(d.d .* x̂), _RadialWeight(u, κ̂ * κ̂))
end
@inline function _yukawa_accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple, u, κ̂)
    _yukawa_accumulate!(col, x̂, p, tb, first(τ), u, κ̂)
    return _yukawa_accumulate!(col, x̂, p, tb, Base.tail(τ), u, κ̂)
end
@inline _yukawa_accumulate!(col, x̂::SVector{3}, p::Int, tb::HarmonicTables, ::Tuple{}, u, κ̂) = col

function s2m!(col::AbstractVector, m::SphericalHarmonics{<:Yukawa3D}, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{3})
    fill!(col, zero(eltype(col)))
    return s2m_accumulate!(col, m, ops, xs, τs, c)
end
function s2m_accumulate!(col::AbstractVector, m::SphericalHarmonics{<:Yukawa3D}, ops::LevelOperators, xs::AbstractVector,
                         τs::AbstractVector, c::SVector{3})
    s = ops.extra.scale
    tb = ops.extra.tb
    κ̂ = ops.extra.κ̂
    p = m.p
    @inbounds for i in eachindex(xs)
        x̂ = (xs[i] - c) * s
        u = _radial_weights(κ̂ * κ̂ * sum(abs2, x̂), p + 1)
        _yukawa_accumulate!(col, x̂, p, tb, _scaled_density(τs[i], s), u, κ̂)
    end
    return col
end

@inline function l2t(m::SphericalHarmonics{<:Yukawa3D}, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Potential=Potential())
    κ̂ = ops.extra.κ̂
    x̂ = xrel * ops.extra.scale
    u = _radial_weights(κ̂ * κ̂ * sum(abs2, x̂), m.p)
    return _harmonic_potential(L, x̂, m.p, ops.extra.tb, _DegreeWeight(u))
end
@inline function l2t(m::SphericalHarmonics{<:Yukawa3D}, ops::LevelOperators, L::AbstractVector, xrel::SVector{3}, ::Gradient)
    κ̂ = ops.extra.κ̂
    s = ops.extra.scale
    x̂ = xrel * s
    u = _radial_weights(κ̂ * κ̂ * sum(abs2, x̂), m.p + 1)
    g = _harmonic_gradient(L, x̂, m.p, ops.extra.tb, _DegreeWeight(u))
    return (g + x̂ * _harmonic_potential(L, x̂, m.p, ops.extra.tb, _RadialWeight(u, κ̂ * κ̂))) * s
end

function multipole_field(m::SphericalHarmonics{<:Yukawa3D}, ops::LevelOperators, M::AbstractVector, xrel::SVector{3})
    p = m.p
    x̂ = xrel * ops.extra.scale
    r = norm(x̂)
    Y = regular_hobson(x̂, p)
    s = scaling_factors(p)
    c = complex_coefficients(M, p)
    v = zeros(typeof(r), p + 1)
    _yukawa_v!(v, ops.extra.κ̂ * r, p)
    acc = zero(promote_type(eltype(c), eltype(Y)))
    for n in 0:p, mm in -n:n
        acc += c[cidx(n, mm)] * Y[cidx(n, mm)] * s[cidx(n, mm)] * v[n+1] / r^(2n + 1)
    end
    val = acc * (m.kernel.c * ops.extra.scale)
    return eltype(M) <: Real ? real(val) : val
end
