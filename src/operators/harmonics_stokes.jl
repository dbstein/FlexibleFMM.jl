# The 3D Stokeslet through four harmonic potentials (the fmm3d stfmm3d construction
# in the translation-covariant form). With d = x − y, ∂_i(1/r) = −d_i/r³ and the
# Laplace potentials Φ_k = Σ f_k/(4πr) (charges) and Ψ_c = Σ (y − c)·f/(4πr), for any
# point c,
#   u_i = (1/8π) Σ [f_i/r + d_i (d·f)/r³] = ½[Φ_i − (x − c)_k ∂_iΦ_k + ∂_iΨ_c],
# because d·f = (x − c)·f − (y − c)·f. The stresslet −(3/4π) d (d·f)(d·ν)/r⁵ and the
# doublet (1/8π)(g/r³ − 3d(d·g)/r⁵) = (1/8π)∂(−g·∇(1/r)) have the same form with
# dipole densities (Dipole(d) is the Laplace potential −d·∇ₓ(1/4πr)): the stresslet
# with Φ_k from Dipole(−(f_k ν + ν_k f)) and Ψ_c from Dipole(−((ν·(y−c)) f + (f·(y−c)) ν)),
# the symmetric form of ∂_i∂_j∂_k r expanded in harmonic potentials; the doublet with
# Ψ_c from Dipole(g). The pressure is −∂_kΦ_k for all three; the rotlet ω×d/r³ = ∇×(ω/r)
# would need three more potentials and is not supported here.
#
# The expansion of a box with centre c and width w holds the four scalar expansions
# [Φ₁; Φ₂; Φ₃; Ψ̂], Ψ̂ = Ψ_c/w, in the box's own coordinates, so that at evaluation
# u = ½[Φ − x̂·∇̂Φ + ∇̂Ψ̂] involves no large cancellation (|x̂| ≤ √3/2). Changing the
# reference point, Ψ_{c'} = Ψ_c + (c − c')·Φ, couples Ψ̂ to Φ in the translations:
#   M2M  Ψ̂_P = ½ Ψ̂_C + δ̂·Φ          δ̂ = (c_C − c_P)/w_P ∈ {±1/4}³
#   M2L  Ψ̂_B = Ψ̂_A + o·Φ             o = far offset (c_A − c_B = o w)
#   L2L  Ψ̂_C = 2 Ψ̂_P − 2 δ̂·Φ
# applied with the scalar operators of `SphericalHarmonics` on each block. Gradients
# and Hessians of the local expansions come from the derivative identities of
# harmonics.jl as shifted coefficient sets.

struct StokesHarmonics{K<:Stokeslet3D,T,S} <: HarmonicMethod      # S: the scalar M2L operator type
    kernel::K
    p::Int
    tol::Float64
end

"""
    StokesHarmonics(kernel::Stokeslet3D; p=16, tol=0)

The analytic multipole method for the 3D Stokeslet: four solid-harmonic expansions
of degree `p` (`4(p+1)²` coefficients), with the scalar translations dense or, for
`tol > 0`, in truncated-SVD form as for `SphericalHarmonics`. Stokeslets, stresslets
and doublets with real densities; velocity, velocity-gradient and pressure outputs.
"""
function StokesHarmonics(kernel::Stokeslet3D{T}; p::Integer=16, tol::Real=0) where {T}
    p >= 1 || throw(ArgumentError("p must be at least 1"))
    0 <= tol < 1 || throw(ArgumentError("tol must lie in [0, 1)"))
    S = tol > 0 ? LowRankM2L{T} : M2LOperator{T}
    return StokesHarmonics{typeof(kernel),T,S}(kernel, Int(p), Float64(tol))
end

order(m::StokesHarmonics) = m.p
_nblocks(::StokesHarmonics) = 4
dimension(::StokesHarmonics) = 3
expansion_length(m::StokesHarmonics) = 4 * (m.p + 1)^2
near_kernel(m::StokesHarmonics) = m.kernel
operator_eltype(::StokesHarmonics{K,T}) where {K,T} = T
operators_type(::StokesHarmonics{K,T,S}) where {K,T,S} = LevelOperators{T,T,HarmonicExtras{T,S},HarmonicStokesM2L{T,PermutedM2L{S}}}
supports_density(::StokesHarmonics, ::Type) = false
# real densities only: the evaluation bodies combine the real pairs of the storage
# in complex arithmetic, which a complex density would conflate with its own parts
supports_density(::StokesHarmonics, ::Type{SVector{3,T}}) where {T<:Real} = true
supports_density(::StokesHarmonics, ::Type{Stresslet{3,T}}) where {T<:Real} = true
supports_density(::StokesHarmonics, ::Type{Doublet{3,T}}) where {T<:Real} = true
supports_density(m::StokesHarmonics, ::Type{T}) where {T<:Tuple} = all(S -> supports_density(m, S), fieldtypes(T))
supports_output(::StokesHarmonics, out) = out isa Potential || out isa Gradient || out isa Pressure
_scalar_method(m::StokesHarmonics{K,T}, p) where {K,T} = SphericalHarmonics(Laplace3D(T); p=p, tol=m.tol)

"""
    HarmonicStokesM2L{T,S}

The M2L of one far offset `o`: the scalar operator (of type `S`, a `PermutedM2L`
of the class representative) on each of the four blocks and the coupling
`Ψ̂ += o·Φ`; applied by the class-wise `m2l_level!` of harmonics.jl in the driver,
block by block here for the checks.
"""
struct HarmonicStokesM2L{T,S}
    op::S
    o::SVector{3,T}
end
Base.size(op::HarmonicStokesM2L) = 4 .* size(op.op)
workspace_rows(op::HarmonicStokesM2L) = workspace_rows(op.op)
@inline _block(x::AbstractVector, N::Int, a::Int) = view(x, (a-1)*N+1:a*N)
@inline _block(x::AbstractMatrix, N::Int, a::Int) = view(x, (a-1)*N+1:a*N, :)
function apply!(out::AbstractVecOrMat, op::HarmonicStokesM2L, x::AbstractVecOrMat, α, work)
    N = size(op.op)[1]
    for a in 1:4
        apply!(_block(out, N, a), op.op, _block(x, N, a), α, work)
    end
    o4 = _block(out, N, 4)
    for k in 1:3
        ok = op.o[k]
        iszero(ok) && continue
        o4 .+= ok .* _block(out, N, k)
    end
    return out
end
apply!(out::AbstractVecOrMat, op::HarmonicStokesM2L, x::AbstractVecOrMat, α=true) = apply!(out, op, x, α, nothing)

function level_operators(m::StokesHarmonics{K,T,S}, w::Real) where {K,T,S}
    p = m.p
    N = (p + 1)^2
    sops = level_operators(_scalar_method(m, p), w)
    # M2M and L2L hold the scalar matrices (N × 8N, 8N × N); m2m_apply!/l2l_apply! below
    # apply them blockwise with the coupling folded into the inputs
    M2L = [HarmonicStokesM2L{T,PermutedM2L{S}}(sops.M2L[f], SVector{3,T}(off)) for (f, off) in enumerate(far_offsets(Val(3)))]
    return LevelOperators(sops.width, sops.M2M, sops.L2L, M2L, sops.extra)::operators_type(m)
end

# The block structure of the Stokes M2M: for child j, blocks 1–3 of the parent get
# A_j M_a and block 4 gets A_j (½ M_4 + δ̂·M), so the coupling is folded into the
# input and each child costs four N × N GEMMs on the stacked group columns. L2L is
# the mirror image: child blocks 1–3 are B_j L_a, block 4 is B_j (2 L_4 − 2 δ̂·L).
# The dense 4N × 32N forms do four times the flops; the device plan keeps them
# (`_stokes_dense_operators`), because there one large GEMM per level beats eight
# small ones with their kernels (measured: M2M + L2L 74 → 109 ms on the helix).
function _stokes_dense_operators(ops::LevelOperators, N::Int)
    T = eltype(ops.M2M)
    M2M = zeros(T, 4N, 32N); L2L = zeros(T, 32N, 4N)
    for (j, δ) in enumerate(child_offsets(Val(3)))
        δ̂ = (SVector{3,T}(δ) .- T(0.5)) / 2
        Aj = view(ops.M2M, :, (j-1)*N+1:j*N)
        Bj = view(ops.L2L, (j-1)*N+1:j*N, :)
        cj = (j - 1) * 4N
        for a in 1:3
            M2M[(a-1)*N+1:a*N, cj+(a-1)*N+1:cj+a*N] = Aj
            L2L[cj+(a-1)*N+1:cj+a*N, (a-1)*N+1:a*N] = Bj
            M2M[3N+1:4N, cj+(a-1)*N+1:cj+a*N] = δ̂[a] * Aj
            L2L[cj+3N+1:cj+4N, (a-1)*N+1:a*N] = (-2 * δ̂[a]) * Bj
        end
        M2M[3N+1:4N, cj+3N+1:cj+4N] = Aj / 2
        L2L[cj+3N+1:cj+4N, 3N+1:4N] = 2 * Bj
    end
    return M2M, L2L
end
# The block products as four GEMMs per level each: M2M applies the scalar M2M
# (N × 8N) to the stacked children's component a (the fourth with the coupling
# folded in), L2L applies the scalar L2L (8N × N) to the parents' component a and
# the children's fourth component is the combination 2·(L2L·G₄) − 2·δ̂·(L2L·G₁..₃).
# (One product per child and component, 32 per level, cost 50 ms per level on a
# 96-core node where every threaded BLAS call wakes its threads.)
function m2m_apply!(out, m::StokesHarmonics, ops::LevelOperators, B, α, work)
    N = (m.p + 1)^2
    ng = size(B, 2)
    X = reshape(view(work, 1:8N*ng), 8N, ng)          # the children's component a, stacked
    T = eltype(ops.M2M)
    offs = child_offsets(Val(3))
    column_blocks(ng) do cols                                  # the tasks own disjoint columns of X and out
        for a in 1:4
            @inbounds for (j, δ) in enumerate(offs)
                δ̂ = (SVector{3,Float64}(δ) .- 0.5) / 2
                r0 = (j - 1) * 4N
                x0 = (j - 1) * N
                if a < 4
                    for g in cols, i in 1:N
                        X[x0+i, g] = B[r0+(a-1)*N+i, g]
                    end
                else
                    for g in cols, i in 1:N
                        X[x0+i, g] = B[r0+3N+i, g] / 2 + δ̂[1] * B[r0+i, g] + δ̂[2] * B[r0+N+i, g] + δ̂[3] * B[r0+2N+i, g]
                    end
                end
            end
            mul!(view(out, (a-1)*N+1:a*N, cols), ops.M2M, view(X, :, cols), T(α), false)
        end
    end
    return out
end
function l2l_apply!(out, m::StokesHarmonics, ops::LevelOperators, G, α, work)
    N = (m.p + 1)^2
    ng = size(G, 2)
    R = reshape(view(work, 1:8N*4ng), 8N, 4ng)        # R[:, (a-1)·ng + g] = L2L · G_a[:, g]
    T = eltype(ops.L2L)
    offs = child_offsets(Val(3))
    column_blocks(ng) do cols
        for a in 1:4
            mul!(view(R, :, (a-1)*ng .+ cols), ops.L2L, view(G, (a-1)*N+1:a*N, cols), T(α), false)
        end
        @inbounds for (j, δ) in enumerate(offs)
            δ̂ = (SVector{3,Float64}(δ) .- 0.5) / 2
            r0 = (j - 1) * 4N
            x0 = (j - 1) * N
            for g in cols, i in 1:N
                r1 = R[x0+i, g]; r2 = R[x0+i, ng+g]; r3 = R[x0+i, 2ng+g]
                out[r0+i, g] = r1
                out[r0+N+i, g] = r2
                out[r0+2N+i, g] = r3
                out[r0+3N+i, g] = 2 * R[x0+i, 3ng+g] - 2 * (δ̂[1] * r1 + δ̂[2] * r2 + δ̂[3] * r3)
            end
        end
    end
    return out
end
function rescale(ref::LevelOperators{T,R,<:HarmonicExtras{R},<:HarmonicStokesM2L{T}}, m::StokesHarmonics, w::Real) where {T,R}
    λ = R(w) / ref.width
    return typeof(ref)(R(w), ref.M2M, ref.L2L, ref.M2L, ref.m2m_scale, ref.l2l_scale,
                       ref.m2l_scale / λ, _rescaled(ref.extra, λ))
end

partial_to_multipole!(M::AbstractVecOrMat, ::StokesHarmonics, ::LevelOperators, pM::AbstractVecOrMat, work) = copyto!(M, pM)
partial_to_local!(L::AbstractVecOrMat, ::StokesHarmonics, ::LevelOperators, pL::AbstractVecOrMat, work) = copyto!(L, pL)

# the four blocks of one source; the dipole moments are passed scaled by 1/w (`s`)
@inline function _stokes_accumulate!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, f::SVector{3}, s)
    for k in 1:3
        _accumulate!(_block(col, N, k), ŷ, p, tb, f[k])
    end
    _accumulate!(_block(col, N, 4), ŷ, p, tb, sum(ŷ .* f))
    return col
end
@inline function _stokes_accumulate!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Stresslet{3}, s)
    f, ν = τ.f, τ.ν
    for k in 1:3
        _accumulate!(_block(col, N, k), ŷ, p, tb, Dipole(-(f[k] * ν + ν[k] * f) * s))
    end
    _accumulate!(_block(col, N, 4), ŷ, p, tb, Dipole(-(sum(ν .* ŷ) * f + sum(f .* ŷ) * ν) * s))
    return col
end
@inline function _stokes_accumulate!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Doublet{3}, s)
    _accumulate!(_block(col, N, 4), ŷ, p, tb, Dipole(τ.g * (s * s)))      # Ψ̂ = Ψ/w
    return col
end
@inline function _stokes_accumulate!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple, s)
    for τi in τ
        _stokes_accumulate!(col, N, ŷ, p, tb, τi, s)
    end
    return col
end

function s2m!(col::AbstractVector, m::StokesHarmonics, ops::LevelOperators, xs::AbstractVector,
              τs::AbstractVector, c::SVector{3})
    s = ops.extra.scale
    tb = ops.extra.tb
    p = m.p
    N = (p + 1)^2
    @inbounds for n in eachindex(col)
        col[n] = zero(eltype(col))
    end
    @inbounds for i in eachindex(xs)
        _stokes_accumulate!(col, N, (xs[i] - c) * s, p, tb, τs[i], s)
    end
    return col
end

# ---- derivatives of a local expansion as shifted coefficient sets ----
# The scaled gradient of Φ = Σ L conj Υ̃ is Σ A conj Υ̃ with, from the derivative
# identities, (D₋L)_n^m = −cm(n+1,m+1) L_{n+1}^{m+1}, (D₊L)_n^m = cp(n+1,m−1) L_{n+1}^{m−1},
# (D_zL)_n^m = cz(n+1,m) L_{n+1}^m and ∂̂_x ↔ ½(D₋ + D₊), ∂̂_y ↔ (i/2)(D₊ − D₋), ∂̂_z ↔ D_z;
# second derivatives compose them. The coefficients at negative m follow from the
# conjugate symmetry, and cm(n,−m) = cp(n,m), cz(n,−m) = cz(n,m).
@inline function _cget(L, o::Int, p::Int, n::Int, m::Int)
    z = zero(eltype(L))
    (n > p || abs(m) > n) && return complex(z, z)
    @inbounds if m == 0
        return complex(L[o+n*n+1], z)
    elseif m > 0
        return complex(L[o+n*n+2m], L[o+n*n+2m+1])
    else
        return (iseven(m) ? 1 : -1) * complex(L[o+n*n-2m], -L[o+n*n-2m+1])
    end
end
@inline _cz(tb::HarmonicTables, n::Int, m::Int) = abs(m) > n ? zero(eltype(tb.cz)) : (@inbounds tb.cz[tidx(n, abs(m))])
@inline _cm(tb::HarmonicTables, n::Int, m::Int) = abs(m) > n ? zero(eltype(tb.cm)) :
    m >= 0 ? (@inbounds tb.cm[tidx(n, m)]) : (@inbounds tb.cp[tidx(n, -m)])
@inline _cp(tb::HarmonicTables, n::Int, m::Int) = abs(m) > n ? zero(eltype(tb.cp)) :
    m >= 0 ? (@inbounds tb.cp[tidx(n, m)]) : (@inbounds tb.cm[tidx(n, -m)])
# coefficients at (n′, m′) of the scaled gradient of block `o`
@inline function _dcoef(L, o::Int, p::Int, tb::HarmonicTables, n′::Int, m′::Int)
    n = n′ + 1
    dm = -_cm(tb, n, m′ + 1) * _cget(L, o, p, n, m′ + 1)
    dp = _cp(tb, n, m′ - 1) * _cget(L, o, p, n, m′ - 1)
    dz = _cz(tb, n, m′) * _cget(L, o, p, n, m′)
    return (dm + dp) / 2, im * (dp - dm) / 2, dz
end
# coefficients at (n″, m″) of the scaled Hessian of block `o`: (xx, yy, zz, xy, xz, yz)
@inline function _d2coef(L, o::Int, p::Int, tb::HarmonicTables, n″::Int, m″::Int)
    n1 = n″ + 1; n2 = n″ + 2
    L0 = _cget(L, o, p, n2, m″)
    DmDm = _cm(tb, n1, m″ + 1) * _cm(tb, n2, m″ + 2) * _cget(L, o, p, n2, m″ + 2)
    DpDp = _cp(tb, n1, m″ - 1) * _cp(tb, n2, m″ - 2) * _cget(L, o, p, n2, m″ - 2)
    DmDp = -_cm(tb, n1, m″ + 1) * _cp(tb, n2, m″) * L0
    DzDz = _cz(tb, n1, m″) * _cz(tb, n2, m″) * L0
    DzDm = -_cz(tb, n1, m″) * _cm(tb, n2, m″ + 1) * _cget(L, o, p, n2, m″ + 1)
    DzDp = _cz(tb, n1, m″) * _cp(tb, n2, m″ - 1) * _cget(L, o, p, n2, m″ - 1)
    return ((DmDm + 2DmDp + DpDp) / 4, (2DmDp - DpDp - DmDm) / 4, DzDz,
            im * (DpDp - DmDm) / 4, (DzDm + DzDp) / 2, im * (DzDp - DzDm) / 2)
end
@inline _re(c, v) = real(c * conj(v))

# values Φ (3), scaled gradients G[i,k] = ∂̂_iΦ_k and ∇̂Ψ̂ at x̂
@inline function _stokes_fold1(L, N::Int, x̂::SVector{3}, p::Int, tb::HarmonicTables)
    S = promote_type(eltype(L), typeof(x̂[1]))
    init = (zero(SVector{3,S}), zero(SMatrix{3,3,S}), zero(SVector{3,S}))
    return _fold_regular(init, x̂, p, tb) do acc, n, mm, vr, vi
        Φ, G, gΨ = acc
        v = complex(vr, vi)
        wt = mm == 0 ? 1 : 2
        Φ += wt * SVector(_re(_cget(L, 0, p, n, mm), v), _re(_cget(L, N, p, n, mm), v), _re(_cget(L, 2N, p, n, mm), v))
        if n < p
            a1 = _dcoef(L, 0, p, tb, n, mm); a2 = _dcoef(L, N, p, tb, n, mm); a3 = _dcoef(L, 2N, p, tb, n, mm)
            G += wt * SMatrix{3,3,S}(_re(a1[1], v), _re(a1[2], v), _re(a1[3], v),
                                     _re(a2[1], v), _re(a2[2], v), _re(a2[3], v),
                                     _re(a3[1], v), _re(a3[2], v), _re(a3[3], v))
            a4 = _dcoef(L, 3N, p, tb, n, mm)
            gΨ += wt * SVector(_re(a4[1], v), _re(a4[2], v), _re(a4[3], v))
        end
        (Φ, G, gΨ)
    end
end
# scaled gradients G and Hessians H[k] (k = 1..4, symmetric) at x̂
@inline function _stokes_fold2(L, N::Int, x̂::SVector{3}, p::Int, tb::HarmonicTables)
    S = promote_type(eltype(L), typeof(x̂[1]))
    Z = zero(SMatrix{3,3,S})
    init = (Z, Z, Z, Z, Z)
    return _fold_regular(init, x̂, p - 1, tb) do acc, n, mm, vr, vi
        G, H1, H2, H3, H4 = acc
        v = complex(vr, vi)
        wt = mm == 0 ? 1 : 2
        a1 = _dcoef(L, 0, p, tb, n, mm); a2 = _dcoef(L, N, p, tb, n, mm); a3 = _dcoef(L, 2N, p, tb, n, mm)
        G += wt * SMatrix{3,3,S}(_re(a1[1], v), _re(a1[2], v), _re(a1[3], v),
                                 _re(a2[1], v), _re(a2[2], v), _re(a2[3], v),
                                 _re(a3[1], v), _re(a3[2], v), _re(a3[3], v))
        if n < p - 1
            H1 += wt * _hess(_d2coef(L, 0, p, tb, n, mm), v)
            H2 += wt * _hess(_d2coef(L, N, p, tb, n, mm), v)
            H3 += wt * _hess(_d2coef(L, 2N, p, tb, n, mm), v)
            H4 += wt * _hess(_d2coef(L, 3N, p, tb, n, mm), v)
        end
        (G, H1, H2, H3, H4)
    end
end
@inline function _hess(h, v)
    xx, yy, zz, xy, xz, yz = _re(h[1], v), _re(h[2], v), _re(h[3], v), _re(h[4], v), _re(h[5], v), _re(h[6], v)
    return SMatrix{3,3}(xx, xy, xz, xy, yy, yz, xz, yz, zz)
end

"""
    l2t(m::StokesHarmonics, ops, L, xrel, out)

Velocity `½[Φ − x̂·∇̂Φ + ∇̂Ψ̂]`, pressure `−∇·Φ` or velocity gradient
`½[∇Φᵀ − ∇Φ − x̂·∇̂∇̂Φ + ∇̂∇̂Ψ̂]/w` at `xrel` relative to the box center.
"""
@inline function l2t(m::StokesHarmonics, ops::LevelOperators, L::AbstractVector, xrel::SVector{3},
                     out::Union{Potential,Pressure,Gradient}=Potential())
    s = ops.extra.scale
    return _stokes_output(L, (m.p + 1)^2, xrel * s, s, m.p, ops.extra.tb, out)
end
# the bodies, shared with the device kernels
@inline function _stokes_output(L, N::Int, x̂::SVector{3}, s, p::Int, tb::HarmonicTables, ::Potential)
    Φ, G, gΨ = _stokes_fold1(L, N, x̂, p, tb)
    return (Φ - G * x̂ + gΨ) / 2
end
@inline function _stokes_output(L, N::Int, x̂::SVector{3}, s, p::Int, tb::HarmonicTables, ::Pressure)
    _, G, _ = _stokes_fold1(L, N, x̂, p, tb)
    return -s * tr(G)
end
@inline function _stokes_output(L, N::Int, x̂::SVector{3}, s, p::Int, tb::HarmonicTables, ::Gradient)
    G, H1, H2, H3, H4 = _stokes_fold2(L, N, x̂, p, tb)
    return (s / 2) * (transpose(G) - G - x̂[1] * H1 - x̂[2] * H2 - x̂[3] * H3 + H4)
end

"""
    multipole_field(m::StokesHarmonics, ops, M, xrel)

The velocity of the four multipole expansions at `xrel`, with the gradients taken by
ForwardDiff on the scalar fields. Used by tests.
"""
function multipole_field(m::StokesHarmonics, ops::LevelOperators, M::AbstractVector, xrel::SVector{3})
    N = (m.p + 1)^2
    sh = _scalar_method(m, m.p)
    s = ops.extra.scale
    Φ(x) = SVector(ntuple(k -> multipole_field(sh, ops, view(M, (k-1)*N+1:k*N), x), Val(3)))
    Ψ(x) = multipole_field(sh, ops, view(M, 3N+1:4N), x)
    J = ForwardDiff.jacobian(Φ, xrel)             # J[k, i] = ∂Φ_k/∂x_i
    g = ForwardDiff.gradient(Ψ, xrel)
    return (Φ(xrel) - transpose(J) * xrel + g / s) / 2
end

# the four blocks of one source, order m only (the device S2M kernel)
@inline function _stokes_accumulate_order!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, f::SVector{3}, s, m::Int)
    for k in 1:3
        _accumulate_order!(_block(col, N, k), ŷ, p, tb, f[k], m)
    end
    _accumulate_order!(_block(col, N, 4), ŷ, p, tb, sum(ŷ .* f), m)
    return col
end
@inline function _stokes_accumulate_order!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Stresslet{3}, s, m::Int)
    f, ν = τ.f, τ.ν
    for k in 1:3
        _accumulate_order!(_block(col, N, k), ŷ, p, tb, Dipole(-(f[k] * ν + ν[k] * f) * s), m)
    end
    _accumulate_order!(_block(col, N, 4), ŷ, p, tb, Dipole(-(sum(ν .* ŷ) * f + sum(f .* ŷ) * ν) * s), m)
    return col
end
@inline function _stokes_accumulate_order!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Doublet{3}, s, m::Int)
    _accumulate_order!(_block(col, N, 4), ŷ, p, tb, Dipole(τ.g * (s * s)), m)
    return col
end
@inline function _stokes_accumulate_order!(col, N::Int, ŷ::SVector{3}, p::Int, tb::HarmonicTables, τ::Tuple, s, m::Int)
    for τi in τ
        _stokes_accumulate_order!(col, N, ŷ, p, tb, τi, s, m)
    end
    return col
end
