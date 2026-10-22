# Operator-level checks shared by the kernel-independent and black-box tests. They
# use only the method interface, so a method that passes them is usable by the
# driver without further changes. Generic in the dimension of the method.
using FlexibleFMM: level_operators, s2m!, l2t, multipole_field, expansion_length,
    multipole_length, local_length, partial_multipole_length, partial_local_length,
    far_offsets, child_offsets, apply!, partial_to_multipole!, partial_to_local!,
    operator_eltype, density_scalar, ncomp, dimension, m2m_apply!, l2l_apply!

# M2M and L2L through the method's hooks (the matrix products by default)
function m2m_product(m, ops, stacked::AbstractVector)
    Nm = multipole_length(m)
    out = zeros(eltype(stacked), Nm, 1)
    m2m_apply!(out, m, ops, reshape(stacked, :, 1), 1, zeros(eltype(stacked), Nm, 1 << dimension(m)))   # work: the children's block
    return vec(out)
end
function l2l_product(m, ops, pL::AbstractVector)
    pll = partial_local_length(m)
    out = zeros(eltype(pL), (1 << dimension(m)) * pll, 1)
    l2l_apply!(out, m, ops, reshape(pL, :, 1), 1, zeros(eltype(pL), pll, 1 << dimension(m)))
    return vec(out)
end

box_points(rng, c::SVector{D}, w, n) where {D} = [c + w * (rand(rng, SVector{D,Float64}) .- 0.5) for _ in 1:n]
# random densities of the kind the kernel takes: scalars, or SVectors for Stokes
kernel_dim(K) = 2
kernel_dim(::Union{Laplace3D,Helmholtz3D,Yukawa3D,Stokeslet3D}) = 3
function random_density(rng, K, n)
    D = kernel_dim(K)
    return ncomp(K, Val(D)) == 1 ? randn(rng, n) : [randn(rng, SVector{D,Float64}) for _ in 1:n]
end

function multipole_of(m, ops, xs, τ, c)
    S = promote_type(density_scalar(eltype(τ)), operator_eltype(m))
    pM = zeros(S, partial_multipole_length(m))
    s2m!(pM, m, ops, xs, τ, c)
    M = zeros(S, multipole_length(m))
    partial_to_multipole!(M, m, ops, pM, zeros(S, max(length(pM), length(M))))
    return M
end
function local_of(m, ops, pL)
    L = zeros(eltype(pL), local_length(m))
    return partial_to_local!(L, m, ops, pL, zeros(eltype(pL), max(length(pL), length(L))))
end

# Max relative error of the translated field over the target box, per far offset.
function m2l_errors(rng, m, w; ns=40, nt=40, density=random_density)
    D = dimension(m)
    ops = level_operators(m, w)
    errs = Float64[]
    for (f, off) in enumerate(far_offsets(Val(D)))
        cs = SVector{D,Float64}(off) * w          # source box center; target box at the origin
        xs = box_points(rng, cs, w, ns)
        τ = density(rng, m.kernel, ns)
        ts = box_points(rng, zero(cs), w, nt)
        M = multipole_of(m, ops, xs, τ, cs)
        pL = zeros(eltype(M), partial_local_length(m))
        apply!(pL, ops.M2L[f], M)
        L = local_of(m, ops, pL)
        u = [l2t(m, ops, L, t) for t in ts]
        push!(errs, mrd(u, direct(m.kernel, xs, τ, ts)))
    end
    return errs
end

function check_m2l_all_offsets(rng, m, w, tol; density=random_density)
    D = dimension(m)
    errs = m2l_errors(rng, m, w; density=density)
    @test length(errs) == 7^D - 3^D
    @test maximum(errs) < tol
    nearest = [maximum(abs, o) == 2 for o in far_offsets(Val(D))]
    @test maximum(errs[nearest]) >= maximum(errs[.!nearest]) / 10
    return maximum(errs)
end

# children multipoles → parent multipole, against the parent's own s2m and the exact field
function check_m2m(rng, m, w, tol; density=random_density)
    D = dimension(m)
    nch = 1 << D
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    Nm = multipole_length(m)
    S = operator_eltype(m)
    centers = [(SVector{D,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(D))]
    stacked = zeros(S, nch * Nm)
    allx = SVector{D,Float64}[]; allτ = eltype(density(rng, m.kernel, 1))[]
    for k in 1:nch
        xs = box_points(rng, centers[k], w / 2, 25)
        τ = density(rng, m.kernel, 25)
        stacked[(k-1)*Nm+1:k*Nm] = multipole_of(m, cops, xs, τ, centers[k])
        append!(allx, xs); append!(allτ, τ)
    end
    Mp = m2m_product(m, ops, stacked)                        # M2M yields the parent's multipole
    Md = multipole_of(m, ops, allx, allτ, zero(SVector{D,Float64}))
    ts = [2.5w * normalize(randn(rng, SVector{D,Float64})) for _ in 1:50]
    uex = direct(m.kernel, allx, allτ, ts)
    @test mrd([multipole_field(m, ops, Mp, t) for t in ts], uex) < tol
    @test mrd([multipole_field(m, ops, Md, t) for t in ts], uex) < tol
    return nothing
end

# parent local → children locals, against the exact field of a far source box
function check_l2l(rng, m, w, tol; density=random_density)
    D = dimension(m)
    nch = 1 << D
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    pll = partial_local_length(m)
    off = SVector{D}(ntuple(k -> (3, -2, 1)[k], D))
    f = findfirst(==(off), far_offsets(Val(D)))
    cs = SVector{D,Float64}(off) * w
    xs = box_points(rng, cs, w, 40)
    τ = density(rng, m.kernel, 40)
    pLp = zeros(operator_eltype(m), pll)
    apply!(pLp, ops.M2L[f], multipole_of(m, ops, xs, τ, cs))
    Lp = local_of(m, ops, pLp)
    pLc = l2l_product(m, ops, pLp)                           # L2L works on the parent's partial local
    centers = [(SVector{D,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(D))]
    for k in 1:nch
        ts = box_points(rng, centers[k], w / 2, 20)
        uex = direct(m.kernel, xs, τ, ts)
        Lk = local_of(m, cops, pLc[(k-1)*pll+1:k*pll])
        @test mrd([l2t(m, cops, Lk, t - centers[k]) for t in ts], uex) < tol
        @test mrd([l2t(m, ops, Lp, t) for t in ts], uex) < tol
    end
    return nothing
end

function check_allocation_free(rng, m, w; m2l=true)
    D = dimension(m)
    ops = level_operators(m, w)
    S = operator_eltype(m)
    o = zero(SVector{D,Float64})
    xs = box_points(rng, o, w, 30); τ = random_density(rng, m.kernel, 30)
    col = zeros(S, partial_multipole_length(m))
    s2m!(col, m, ops, xs, τ, o)
    @test @allocated(s2m!(col, m, ops, xs, τ, o)) == 0
    x = SVector{D}(ntuple(k -> (0.1w, 0.05w, 0.02w)[k], D))
    L = zeros(S, local_length(m))
    l2t(m, ops, L, x)
    @test @allocated(l2t(m, ops, L, x)) == 0
    if m2l
        Mv = zeros(S, multipole_length(m))
        out = zeros(S, partial_local_length(m))
        apply!(out, ops.M2L[1], Mv)
        @test @allocated(apply!(out, ops.M2L[1], Mv)) == 0
    end
    return nothing
end
