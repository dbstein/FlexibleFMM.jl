# Operator-level checks shared by the kernel-independent and black-box tests. They
# use only the method interface, so a method that passes them is usable by the
# driver without further changes.
using FlexibleFMM: level_operators, s2m!, l2t, multipole_field, expansion_length,
    far_offsets, child_offsets, apply!, partial_to_multipole!, partial_to_local!,
    operator_eltype

box_points(rng, c, w, n) = [c + w * (P2(rand(rng), rand(rng)) .- 0.5) for _ in 1:n]

function multipole_of(m, ops, xs, τ, c)
    S = promote_type(eltype(τ), operator_eltype(m))
    N = expansion_length(m)
    pM = zeros(S, N)
    s2m!(pM, m, ops, xs, τ, c)
    M = similar(pM)
    partial_to_multipole!(M, m, ops, pM, zeros(S, N))
    return M
end
local_of(m, ops, pL) = partial_to_local!(similar(pL), m, ops, pL, similar(pL))

# Max relative error of the translated field over the target box, per far offset.
function m2l_errors(rng, m, w; ns=40, nt=40)
    ops = level_operators(m, w)
    errs = Float64[]
    for (f, off) in enumerate(far_offsets(Val(2)))
        cs = SVector{2,Float64}(off) * w          # source box center; target box at the origin
        xs = box_points(rng, cs, w, ns)
        τ = randn(rng, ns)
        ts = box_points(rng, P2(0, 0), w, nt)
        M = multipole_of(m, ops, xs, τ, cs)
        pL = similar(M)
        apply!(pL, ops.M2L[f], M)
        L = local_of(m, ops, pL)
        u = [l2t(m, ops, L, t) for t in ts]
        push!(errs, max_relative_diff(u, direct(m.kernel, xs, τ, ts)))
    end
    return errs
end

function check_m2l_all_offsets(rng, m, w, tol)
    errs = m2l_errors(rng, m, w)
    @test length(errs) == 40
    @test maximum(errs) < tol
    nearest = [maximum(abs, o) == 2 for o in far_offsets(Val(2))]
    @test maximum(errs[nearest]) >= maximum(errs[.!nearest]) / 10
    return maximum(errs)
end

# children multipoles → parent multipole, against the parent's own s2m and the exact field
function check_m2m(rng, m, w, tol)
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    N = expansion_length(m)
    S = operator_eltype(m)
    centers = [(SVector{2,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(2))]
    stacked = zeros(S, 4N)
    allx = P2[]; allτ = Float64[]
    for k in 1:4
        xs = box_points(rng, centers[k], w / 2, 25)
        τ = randn(rng, 25)
        stacked[(k-1)*N+1:k*N] = multipole_of(m, cops, xs, τ, centers[k])
        append!(allx, xs); append!(allτ, τ)
    end
    Mp = partial_to_multipole!(zeros(S, N), m, ops, ops.M2M * stacked, zeros(S, N))
    Md = multipole_of(m, ops, allx, allτ, P2(0, 0))
    ts = [2.5w * P2(cos(θ), sin(θ)) for θ in 2π .* rand(rng, 50)]
    uex = direct(m.kernel, allx, allτ, ts)
    @test max_relative_diff([multipole_field(m, ops, Mp, t) for t in ts], uex) < tol
    @test max_relative_diff([multipole_field(m, ops, Md, t) for t in ts], uex) < tol
    return nothing
end

# parent local → children locals, against the exact field of a far source box
function check_l2l(rng, m, w, tol)
    ops = level_operators(m, w)
    cops = level_operators(m, w / 2)
    N = expansion_length(m)
    off = SVector(3, -2)
    f = findfirst(==(off), far_offsets(Val(2)))
    xs = box_points(rng, SVector{2,Float64}(off) * w, w, 40)
    τ = randn(rng, 40)
    Lp = local_of(m, ops, ops.M2L[f].A * multipole_of(m, ops, xs, τ, SVector{2,Float64}(off) * w))
    pLc = ops.L2L * Lp
    centers = [(SVector{2,Float64}(δ) .- 0.5) * (w / 2) for δ in child_offsets(Val(2))]
    for k in 1:4
        ts = box_points(rng, centers[k], w / 2, 20)
        uex = direct(m.kernel, xs, τ, ts)
        Lk = local_of(m, cops, pLc[(k-1)*N+1:k*N])
        @test max_relative_diff([l2t(m, cops, Lk, t - centers[k]) for t in ts], uex) < tol
        @test max_relative_diff([l2t(m, ops, Lp, t) for t in ts], uex) < tol
    end
    return nothing
end

function check_allocation_free(rng, m, w)
    ops = level_operators(m, w)
    S = operator_eltype(m)
    N = expansion_length(m)
    xs = box_points(rng, P2(0, 0), w, 30); τ = randn(rng, 30)
    col = zeros(S, N)
    s2m!(col, m, ops, xs, τ, P2(0, 0))
    @test @allocated(s2m!(col, m, ops, xs, τ, P2(0, 0))) == 0
    x = P2(0.1w, 0.05w)
    l2t(m, ops, col, x)
    @test @allocated(l2t(m, ops, col, x)) == 0
    out = similar(col)
    apply!(out, ops.M2L[1], col)
    @test @allocated(apply!(out, ops.M2L[1], col)) == 0
    return nothing
end
