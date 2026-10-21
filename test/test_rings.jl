# Analytic tests: concentric rings of N_i equispaced equal charges. Outside the
# largest ring the field is that of point charges at the center up to aliasing of
# order (R/r)^N_i; inside the smallest ring it is the corresponding regular solution.
using SpecialFunctions: besselj0, bessely0, besseli, besselk, hankelh1

rng = MersenneTwister(7)

function rings(radii, counts, charges)
    pts = P2[]; τ = Float64[]
    for (R, n, q) in zip(radii, counts, charges)
        for k in 0:n-1
            θ = 2π * k / n + 0.1
            push!(pts, R * P2(cos(θ), sin(θ)))
            push!(τ, q)
        end
    end
    return pts, τ
end

radii = (0.3, 0.55, 0.8, 1.0)
counts = (3000, 4000, 2500, 3500)
charges = (0.7, -1.3, 0.4, 2.0)
pts, τ = rings(radii, counts, charges)
Q = [n * q for (n, q) in zip(counts, charges)]
# exterior targets (r > 1.6) and interior targets (r < 0.15)
θe = 2π .* rand(rng, 400)
re = 1.6 .+ 0.8 .* rand(rng, 400)
ext = [r * P2(cos(t), sin(t)) for (r, t) in zip(re, θe)]
θi = 2π .* rand(rng, 300)
ri = 0.15 .* rand(rng, 300)
int = [r * P2(cos(t), sin(t)) for (r, t) in zip(ri, θi)]
box = (P2(-2.5, -2.5), 5.0)

@testset "Laplace rings" begin
    m = KernelIndependent(Laplace2D(); nequiv=80)
    f = FMM(m, pts; ppl=40, bbox=box)
    st = build_expansions(f, pts, τ)
    ue = evaluate(f, st, ext)
    exact_e = [-sum(Q) * log(norm(x)) / (2π) for x in ext]
    @test max_relative_diff(ue, exact_e) < 1e-9
    ui = evaluate(f, st, int)
    exact_i = fill(-sum(Qk * log(R) for (Qk, R) in zip(Q, radii)) / (2π), length(int))
    @test max_relative_diff(ui, exact_i) < 1e-9
end

@testset "Yukawa rings" begin
    k = 2.0
    m = KernelIndependent(Yukawa2D(k); nequiv=80)
    f = FMM(m, pts; ppl=40, bbox=box)
    st = build_expansions(f, pts, τ)
    ue = evaluate(f, st, ext)
    exact_e = [sum(Qk * besseli(0, k * R) for (Qk, R) in zip(Q, radii)) * besselk(0, k * norm(x)) / (2π) for x in ext]
    @test max_relative_diff(ue, exact_e) < 1e-9
    ui = evaluate(f, st, int)
    exact_i = [sum(Qk * besselk(0, k * R) for (Qk, R) in zip(Q, radii)) * besseli(0, k * norm(x)) / (2π) for x in int]
    @test max_relative_diff(ui, exact_i) < 1e-9
end

@testset "Helmholtz rings" begin
    k = 1.5     # k·W = 7.5 on the root box, well below the first resonance of any level with expansions
    m = KernelIndependent(Helmholtz2D(k); nequiv=80)
    f = FMM(m, pts; ppl=40, bbox=box)
    st = build_expansions(f, pts, τ)
    ue = evaluate(f, st, ext)
    exact_e = [sum(Qk * besselj0(k * R) for (Qk, R) in zip(Q, radii)) * (im / 4) * hankelh1(0, k * norm(x)) for x in ext]
    @test max_relative_diff(ue, exact_e) < 1e-9
    ui = evaluate(f, st, int)
    exact_i = [sum(Qk * hankelh1(0, k * R) for (Qk, R) in zip(Q, radii)) * (im / 4) * besselj0(k * norm(x)) for x in int]
    @test max_relative_diff(ui, exact_i) < 1e-9
end
