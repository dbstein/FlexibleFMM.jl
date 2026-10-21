# Shared helpers for the test suite: point distributions and comparison utilities.

using Random

const P2 = SVector{2,Float64}

max_relative_diff(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
max_abs_diff(a, b) = maximum(abs.(a .- b))
mrd(a, b) = maximum(norm.(a .- b)) / maximum(norm.(b))     # scalars, SVectors or SMatrices

# Point distributions in (or near) the unit square; all return Vector{P2}.
uniform_points(rng, N) = [P2(rand(rng), rand(rng)) for _ in 1:N]

function clustered_points(rng, N; nclusters=10, spread=0.01)
    centers = [P2(0.1 + 0.8rand(rng), 0.1 + 0.8rand(rng)) for _ in 1:nclusters]
    return [centers[1 + (i - 1) % nclusters] + spread * P2(randn(rng), randn(rng)) for i in 1:N]
end

function circle_points(rng, N; center=P2(0.5, 0.5), radius=0.4)
    θ = 2π .* rand(rng, N)
    return [center + radius * P2(cos(t), sin(t)) for t in θ]
end

line_points(rng, N) = [P2(rand(rng), 0.5 + 1e-3 * randn(rng)) for _ in 1:N]

one_quadrant_points(rng, N) = [P2(0.3rand(rng), 0.3rand(rng)) for _ in 1:N]

# Circle + a tight cluster + a sparse background: forces level restriction and ghosts.
function mixed_points(rng, N)
    nc = N ÷ 2
    nk = N ÷ 4
    nb = N - nc - nk
    c = circle_points(rng, nc; center=P2(0.5, 0.5), radius=0.3)
    k = [P2(0.9, 0.9) + 0.002 * P2(randn(rng), randn(rng)) for _ in 1:nk]
    b = uniform_points(rng, nb)
    return vcat(c, k, b)
end

duplicate_points(N) = fill(P2(0.3, 0.7), N)

const DISTRIBUTIONS = (
    uniform = uniform_points,
    clustered = clustered_points,
    circle = circle_points,
    line = line_points,
    one_quadrant = one_quadrant_points,
    mixed = mixed_points,
)

# 3D distributions in (or near) the unit cube; all return Vector{P3}.
const P3 = SVector{3,Float64}
uniform_points3(rng, N) = [rand(rng, P3) for _ in 1:N]
function clustered_points3(rng, N; nclusters=10, spread=0.01)
    centers = [0.1 .+ 0.8 .* rand(rng, P3) for _ in 1:nclusters]
    return [centers[1 + (i - 1) % nclusters] + spread * randn(rng, P3) for i in 1:N]
end
sphere_points(rng, N; center=P3(0.5, 0.5, 0.5), radius=0.4) = [center + radius * normalize(randn(rng, P3)) for _ in 1:N]
# a curve (the fibre case): a jittered helix through the cube
helix_points(rng, N) = [P3(0.5 + 0.3cos(θ), 0.5 + 0.3sin(θ), 0.05 + 0.9θ / (6π)) + 1e-3 * randn(rng, P3) for θ in 6π .* rand(rng, N)]
one_octant_points(rng, N) = [0.3 * rand(rng, P3) for _ in 1:N]
# sphere + a tight cluster + a sparse background: forces level restriction and ghosts
function mixed_points3(rng, N)
    ns = N ÷ 2
    nk = N ÷ 4
    s = sphere_points(rng, ns; radius=0.3)
    k = [P3(0.9, 0.9, 0.9) + 0.002 * randn(rng, P3) for _ in 1:nk]
    return vcat(s, k, uniform_points3(rng, N - ns - nk))
end
const DISTRIBUTIONS3 = (
    uniform = uniform_points3,
    clustered = clustered_points3,
    sphere = sphere_points,
    helix = helix_points,
    one_octant = one_octant_points,
    mixed = mixed_points3,
)
