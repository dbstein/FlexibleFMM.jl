# Operator-level checks shared by the kernel-independent and black-box tests. They
# use only the method interface, so a method that passes them is usable by the
# driver without further changes.
using FlexibleFMM: level_operators, s2m!, l2t, multipole_field, expansion_length,
    multipole_length, local_length, partial_multipole_length, partial_local_length,
    far_offsets, child_offsets, apply!, partial_to_multipole!, partial_to_local!,
    operator_eltype, density_scalar, ncomp

box_points(rng, c, w, n) = [c + w * (P2(rand(rng), rand(rng)) .- 0.5) for _ in 1:n]
# random densities of the kind the kernel takes: scalars, or SVectors for Stokes
random_density(rng, K, n) = ncomp(K, Val(2)) == 1 ? randn(rng, n) : [P2(randn(rng), randn(rng)) for _ in 1:n]

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
    ops = level_operators(m, w)
    errs = Float64[]
    for (f, off) in enumerate(far_offsets(Val(2)))
        cs = SVector{2,Float64}(off) * w          # source box center; target box at the origin
        xs = box_points(rng, cs, w, ns)
        τ = density(rng, m.kernel, ns)
        ts = box_points(rng, P2(0, 0), w, nt)
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
    errs = m2l_errors(rng, m, w; density=density)
    @test length(errs) == 40
    @test maximum(errs) < tol
    nearest = [maximum(abs, o) == 2 for o in far_offsets(Val(2))]
    @test maximum(errs[nearest]) >= maximum(errs[.!nearest]) / 10
    return maximum(errs)
end

# children multipoles → parent multipole, against the parent's own s2m and the exact field
function check_m2m(rng, m, w, tol; density=random_density)
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    Nm = multipole_length(m)
    S = operator_eltype(m)
    centers = [(SVector{2,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(2))]
    stacked = zeros(S, 4Nm)
    allx = P2[]; allτ = eltype(density(rng, m.kernel, 1))[]
    for k in 1:4
        xs = box_points(rng, centers[k], w / 2, 25)
        τ = density(rng, m.kernel, 25)
        stacked[(k-1)*Nm+1:k*Nm] = multipole_of(m, cops, xs, τ, centers[k])
        append!(allx, xs); append!(allτ, τ)
    end
    pM = ops.M2M * stacked
    Mp = partial_to_multipole!(zeros(S, Nm), m, ops, pM, zeros(S, max(length(pM), Nm)))
    Md = multipole_of(m, ops, allx, allτ, P2(0, 0))
    ts = [2.5w * P2(cos(θ), sin(θ)) for θ in 2π .* rand(rng, 50)]
    uex = direct(m.kernel, allx, allτ, ts)
    @test mrd([multipole_field(m, ops, Mp, t) for t in ts], uex) < tol
    @test mrd([multipole_field(m, ops, Md, t) for t in ts], uex) < tol
    return nothing
end

# parent local → children locals, against the exact field of a far source box
function check_l2l(rng, m, w, tol; density=random_density)
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    pll = partial_local_length(m)
    off = SVector(3, -2)
    f = findfirst(==(off), far_offsets(Val(2)))
    xs = box_points(rng, SVector{2,Float64}(off) * w, w, 40)
    τ = density(rng, m.kernel, 40)
    pLp = zeros(operator_eltype(m), pll)
    apply!(pLp, ops.M2L[f], multipole_of(m, ops, xs, τ, SVector{2,Float64}(off) * w))
    Lp = local_of(m, ops, pLp)
    pLc = ops.L2L * Lp
    centers = [(SVector{2,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(2))]
    for k in 1:4
        ts = box_points(rng, centers[k], w / 2, 20)
        uex = direct(m.kernel, xs, τ, ts)
        Lk = local_of(m, cops, pLc[(k-1)*pll+1:k*pll])
        @test mrd([l2t(m, cops, Lk, t - centers[k]) for t in ts], uex) < tol
        @test mrd([l2t(m, ops, Lp, t) for t in ts], uex) < tol
    end
    return nothing
end

function check_allocation_free(rng, m, w; m2l=true)
    ops = level_operators(m, w)
    S = operator_eltype(m)
    xs = box_points(rng, P2(0, 0), w, 30); τ = random_density(rng, m.kernel, 30)
    col = zeros(S, partial_multipole_length(m))
    s2m!(col, m, ops, xs, τ, P2(0, 0))
    @test @allocated(s2m!(col, m, ops, xs, τ, P2(0, 0))) == 0
    x = P2(0.1w, 0.05w)
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
