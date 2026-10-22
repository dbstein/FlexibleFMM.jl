# The tree built on the device (here KernelAbstractions' CPU backend, `Array`s) against
# the host builder: every structural field, the interaction lists, the leaves' point
# sets, and the invariants.
using FlexibleFMM: device_tree, DeviceTree, nlevels

rng = MersenneTwister(41)
same_structure(a::Tree, b::Tree) = all(getfield(a, f) == getfield(b, f) for f in fieldnames(Tree) if f ∉ (:perm, :ilists)) &&
    all(all(getfield(x, g) == getfield(y, g) for g in fieldnames(FlexibleFMM.InteractionList)) for (x, y) in zip(a.ilists, b.ilists))

@testset "device tree equals the host tree: $(name)" for (name, pts, ppl) in (
        ("mixed 2D", mixed_points(rng, 8000), 30), ("one quadrant 2D", [0.5 .* p for p in uniform_points(rng, 3000)], 20),
        ("mixed 3D", mixed_points3(rng, 8000), 40), ("sphere 3D", sphere_points(rng, 8000), 60),
        ("helix 3D", helix_points(rng, 8000), 50), ("clustered 3D", clustered_points3(rng, 6000), 30))
    box = (zero(eltype(pts)), 1.0)
    ref = Tree(pts; ppl=ppl, bbox=box)
    dt = device_tree(pts; ppl=ppl, bbox=box)
    t = Tree(dt)
    @test same_structure(t, ref)
    @test all(sort(t.perm[r]) == sort(ref.perm[r]) for r in ref.points[ref.leaf])   # the points of each leaf
    check_tree_invariants(t, pts)
    @test nlevels(dt) == nlevels(ref) && FlexibleFMM.nnodes(dt) == length(ref.level)
    @test Array(dt.sources) == pts[t.perm]
    @test device_tree(pts; ppl=ppl, bbox=box).perm == dt.perm                       # deterministic
end

@testset "device tree: small cases and guards" begin
    pts = uniform_points3(rng, 10)
    dt = device_tree(pts; ppl=50, bbox=(P3(0, 0, 0), 1.0))                          # the root alone
    @test nlevels(dt) == 1 && same_structure(Tree(dt), Tree(pts; ppl=50, bbox=(P3(0, 0, 0), 1.0)))
    dup = fill(P3(0.3, 0.3, 0.3), 20)
    @test_throws ArgumentError device_tree(dup; ppl=5, bbox=(P3(0, 0, 0), 1.0))    # coincident points
    @test_throws ArgumentError device_tree(P3[]; ppl=5, bbox=(P3(0, 0, 0), 1.0))
    f32 = SVector{3,Float32}.(sphere_points(rng, 5000))
    dt32 = device_tree(f32; ppl=40, bbox=(zero(SVector{3,Float32}), 1f0))
    @test same_structure(Tree(dt32), Tree(f32; ppl=40, bbox=(zero(SVector{3,Float32}), 1f0)))
end

@testset "the plan from points on the device equals the plan from the host tree" begin
    pts = mixed_points3(rng, 20000)
    box = (P3(0, 0, 0), 1.0)
    for (m, τ) in ((CubeKernelIndependent{3}(Laplace3D(); p=4), randn(rng, 20000)),
                   (CubeKernelIndependent{3}(Stokeslet3D(); p=4), [randn(rng, P3) for _ in 1:20000]),
                   (StokesHarmonics(Stokeslet3D(); p=6), [randn(rng, P3) for _ in 1:20000]))
        cache = OperatorCache(m)
        f = FMM(m, pts; ppl=60, bbox=box, cache=cache)
        plan_h = DevicePlan(f)
        plan_d = DevicePlan(m, pts; ppl=60, bbox=box, cache=cache)           # Arrays: the CPU backend, no host tree
        @test plan_d.nlevels == plan_h.nlevels && plan_d.lo == plan_h.lo && plan_d.nl == plan_h.nl
        @test all(length(plan_d.s2m_nodes[l]) == length(plan_h.s2m_nodes[l]) for l in 1:plan_h.nlevels)
        @test all(plan_d.leaf_nodes[l] == plan_h.leaf_nodes[l] for l in 1:plan_h.nlevels)
        u_h = device_matvec!(similar(τ), plan_h, (sources=pts, τ=τ))
        u_d = device_matvec!(similar(τ), plan_d, (sources=pts, τ=τ))
        @test mrd(u_d, u_h) < 1e-10                                            # the leaves' points in another order only
        @test mrd(u_d, evaluate_at_sources(f, build_expansions(f, pts, τ))) < 1e-10
    end
    @test_throws ArgumentError DevicePlan(CubeKernelIndependent{3}(Laplace3D(); p=4), pts; ppl=60, bbox=(P3(0, 0, 0), 0.5))
end
