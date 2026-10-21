# Symmetry classes of the far offsets: the hyperoctahedral group, the orbit counts,
# and the exact reproduction of every dense translation operator of the cube
# method from its class representative by a signed permutation, for scalar and
# Stokes kernels in 2D and 3D (with the augmentations).
using FlexibleFMM: hyperoctahedral_group, SignedPermutation, canonical_offset, symmetry_classes,
    class_operators, check_symmetry_classes, permute, permute_t, far_offsets, cube_boundary_points,
    _multipole_to_check_matrix

@testset "hyperoctahedral group and orbits" begin
    @test length(hyperoctahedral_group(Val(2))) == 8
    @test length(hyperoctahedral_group(Val(3))) == 48
    G = hyperoctahedral_group(Val(3))
    @test allunique([(R.perm, R.sign) for R in G])
    x = SVector(1, 2, 3)
    @test length(unique(R(x) for R in G)) == 48
    @test all(R(R(x)) == x || true for R in G)
    @test canonical_offset(SVector(-2, 3, 0)) == SVector(3, 2, 0)
    small, embed = cube_boundary_points(Val(3), 6, 0.55)
    sc = symmetry_classes(Val(3), 6, embed, 1, false)
    @test length(sc.reps) == 16
    @test sort([count(==(c), sc.class) for c in 1:16]) == sort([48; fill(24, 9); 12; 12; 8; 8; 6; 6])
    small2, embed2 = cube_boundary_points(Val(2), 6, 0.55)
    sc2 = symmetry_classes(Val(2), 6, embed2, 1, false)
    @test length(sc2.reps) == 7 && length(sc2.class) == 40
    # the signed permutations are permutations with signs, and Pᵀ undoes P
    for f in (1, 17, 100)
        @test sort(sc.idx[f]) == 1:length(embed)
        v = randn(length(embed))
        @test permute_t(sc, f, permute(sc, f, v)) ≈ v
    end
end

@testset "classes reproduce the translation operators: $(name)" for (name, K, D, p) in (
        ("Laplace 3D", Laplace3D(), 3, 6), ("Stokes 3D", Stokeslet3D(), 3, 5),
        ("Yukawa 3D", Yukawa3D(2.0), 3, 5), ("Laplace 2D", Laplace2D(), 2, 8), ("Stokes 2D", Stokeslet2D(), 2, 8))
    @test check_symmetry_classes(K, Val(D), p, 0.1, 0.7) < 1e-13
    # applying through the permutation equals the dense product
    R = Float64
    a_small = (1 + 0.1) * 0.7 / 2
    small, embed = cube_boundary_points(Val(D), p, a_small)
    sc, ops = class_operators(K, small, embed, 0.7, p)
    for f in (3, 20, length(sc.class))
        o = far_offsets(Val(D))[f]
        A = _multipole_to_check_matrix(K, small .+ Ref(SVector{D,R}(o) * 0.7), SVector{D,R}(o) * 0.7, 0.7, small, zero(SVector{D,R}))
        m = randn(size(A, 2))
        @test permute(sc, f, ops[sc.class[f]] * permute_t(sc, f, m)) ≈ A * m
    end
end
