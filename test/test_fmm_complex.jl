# Complex-valued kernels end to end (Helmholtz) and complex densities with a real
# kernel; Yukawa as the real-valued decaying kernel. Helmholtz is tested in the
# low-frequency regime k·W ≤ 10: with expansions living on levels ≥ 3 (widths ≤ W/4)
# this stays below the first check-map resonance at k·w ≈ 3.40.
using FlexibleFMM: nlevels

rng = MersenneTwister(9)

# Σ_{j≠i} K(s_j, s_i) τ_j for a subset of source indices
function direct_self_subset(K, pts, τ, idx)
    out = Vector{promote_type(FlexibleFMM.kernel_eltype(K, eltype(pts), eltype(pts)), eltype(τ))}(undef, length(idx))
    Threads.@threads :dynamic for q in eachindex(idx)
        i = idx[q]
        acc = zero(eltype(out))
        for j in eachindex(pts)
            j == i && continue
            acc += K(pts[j], pts[i]) * τ[j]
        end
        out[q] = acc
    end
    return out
end

@testset "$(kname), $(mname)" for (kname, K) in (("Helmholtz k=10", Helmholtz2D(10.0)),
                                                ("Yukawa k=5", Yukawa2D(5.0))),
                                   (mname, m) in (("KI nequiv=60", KernelIndependent(K; nequiv=60)),
                                                  ("BB p=14", BlackBox(K; p=14)))
    N = 5000
    pts = mixed_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 1500)
    idx = rand(rng, 1:N, 1500)
    f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
    st = build_expansions(f, pts, τ)
    @test eltype(st.multipoles) == FlexibleFMM.operator_eltype(m)
    ut = evaluate(f, st, targets)
    @test max_relative_diff(ut, direct(K, pts, τ, targets)) < 1e-7
    us = evaluate_at_sources(f, st)
    @test max_relative_diff(us[idx], direct_self_subset(K, pts, τ, idx)) < 1e-7
end

@testset "complex densities with a real kernel" begin
    K = Laplace2D()
    N = 4000
    pts = uniform_points(rng, N)
    τ = randn(rng, ComplexF64, N)
    targets = uniform_points(rng, 1000)
    for m in (KernelIndependent(K; nequiv=60), BlackBox(K; p=14))
        f = FMM(m, pts; ppl=40, bbox=(P2(0, 0), 1.0))
        st = build_expansions(f, pts, τ)
        @test eltype(st.multipoles) == ComplexF64
        @test max_relative_diff(evaluate(f, st, targets), direct(K, pts, τ, targets)) < 1e-7
        @test max_relative_diff(evaluate_at_sources(f, st), direct_self(K, pts, τ)) < 1e-7
    end
end

@testset "Helmholtz wavenumber scan (reported, asserted only at k·W ≤ 10)" begin
    N = 4000
    pts = uniform_points(rng, N)
    τ = randn(rng, N)
    targets = uniform_points(rng, 500)
    rows = String[]
    for k in (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
        K = Helmholtz2D(k)
        uex = direct(K, pts, τ, targets)
        errs = Float64[]
        for n in (60, 100)
            f = FMM(KernelIndependent(K; nequiv=n), pts; ppl=40, bbox=(P2(0, 0), 1.0))
            st = build_expansions(f, pts, τ)
            push!(errs, max_relative_diff(evaluate(f, st, targets), uex))
        end
        push!(rows, "k·W = $(k): nequiv 60 → $(errs[1]), nequiv 100 → $(errs[2])")
        if k <= 10
            @test errs[1] < 1e-6
            @test errs[2] < 1e-9
        end
    end
    @info "Helmholtz KIFMM error vs wavenumber on the unit box\n" * join(rows, "\n")
end
