# Black-box operators in isolation, through the same checks as the kernel-independent
# ones. Chebyshev interpolation of the kernel converges geometrically in p; the
# nearest interaction-list box is the hardest.
using FlexibleFMM: BlackBox, level_operators, tensor_vandermonde, chebyshev_tensor_nodes,
    chebyshev_vandermonde_inverse, order

rng = MersenneTwister(8)

@testset "tensor Vandermonde" begin
    p = 5
    nodes = chebyshev_tensor_nodes(Val(2), p)
    V = tensor_vandermonde(nodes, p)
    VI = kron(chebyshev_vandermonde_inverse(p), chebyshev_vandermonde_inverse(p))
    @test norm(VI * V - I) < 1e-12
    @test order(BlackBox(Laplace2D(); p=7)) == 7
    @test_throws ArgumentError BlackBox(Laplace2D(); p=1)
end

@testset "M2L on all 40 offsets: $(name)" for (name, K) in (
        ("Laplace", Laplace2D()),
        ("Laplace dipole test (non-symmetric)", LaplaceDipoleTest(0.7)),
        ("Yukawa", Yukawa2D(3.0)),
        ("Helmholtz", Helmholtz2D(4.0)))
    check_m2l_all_offsets(rng, BlackBox(K; p=14), 0.37, 1e-7)
end

@testset "convergence in p" begin
    e = [maximum(m2l_errors(rng, BlackBox(Laplace2D(); p=p), 0.5)) for p in (6, 10, 14, 18)]
    @test e[1] > 10e[2] > 10e[3] > 10e[4]
    @test e[4] < 1e-10
end

@testset "M2M and L2L: $(name)" for (name, K) in (("Laplace", Laplace2D()),
        ("non-symmetric", LaplaceDipoleTest(0.4)), ("Helmholtz", Helmholtz2D(3.0)))
    m = BlackBox(K; p=14)
    check_m2m(rng, m, 0.61, 1e-7)
    check_l2l(rng, m, 0.61, 1e-7)
end

@testset "allocation-free bodies" begin
    for K in (Laplace2D(), Helmholtz2D(2.0))
        check_allocation_free(rng, BlackBox(K; p=10), 0.3)
    end
end
