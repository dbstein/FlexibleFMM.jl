using FlexibleFMM: chebyshev_points, chebyshev_values, chebyshev_vandermonde,
    chebyshev_vandermonde_inverse, chebyshev_tensor_nodes, clenshaw, clenshaw_derivative,
    clenshaw2d

rng = MersenneTwister(2)
T_exact(j, x) = cos(j * acos(x))

@testset "points" begin
    for p in (1, 2, 7, 12)
        x = chebyshev_points(p)
        @test length(x) == p
        @test issorted(x)
        @test all(-1 .< x .< 1)
        @test x ≈ [cos((2k - 1) * π / (2p)) for k in p:-1:1]
    end
    @test_throws ArgumentError chebyshev_points(0)
end

@testset "values and Vandermonde" begin
    for p in (1, 2, 3, 9)
        x = 2rand(rng, 5) .- 1
        for xi in x
            v = chebyshev_values(Val(p), xi)
            @test v ≈ [T_exact(j, xi) for j in 0:p-1]
        end
        V = chebyshev_vandermonde(x, p)
        @test V ≈ [T_exact(j, xi) for xi in x, j in 0:p-1]
        VI = chebyshev_vandermonde_inverse(p)
        Vp = chebyshev_vandermonde(chebyshev_points(p), p)
        @test norm(VI * Vp - I) < 1e-13
        @test norm(Vp * VI - I) < 1e-13
    end
    chebyshev_values(Val(8), 0.3)
    @test @allocated(chebyshev_values(Val(8), 0.3)) == 0
    # Dual input
    d = ForwardDiff.derivative(x -> chebyshev_values(Val(5), x)[5], 0.3)
    @test d ≈ ForwardDiff.derivative(x -> T_exact(4, x), 0.3)
end

@testset "clenshaw" begin
    for n in (1, 2, 3, 4, 11)
        c = randn(rng, n)
        for x in (2rand(rng, 4) .- 1)
            exact = sum(c[j+1] * T_exact(j, x) for j in 0:n-1)
            @test clenshaw(c, x) ≈ exact
            dexact = ForwardDiff.derivative(y -> sum(c[j+1] * T_exact(j, y) for j in 0:n-1), x)
            @test clenshaw_derivative(c, x) ≈ dexact atol = 1e-12
            @test clenshaw_derivative(c, x) ≈ ForwardDiff.derivative(y -> clenshaw(c, y), x) atol = 1e-12
        end
    end
    # complex coefficients, Dual argument
    cc = randn(rng, ComplexF64, 6)
    @test clenshaw(cc, 0.2) ≈ sum(cc[j+1] * T_exact(j, 0.2) for j in 0:5)
    @test ForwardDiff.derivative(x -> real(clenshaw(cc, x)), 0.2) ≈ real(clenshaw_derivative(cc, 0.2))
    c = randn(rng, 8)
    clenshaw(c, 0.1)
    @test @allocated(clenshaw(c, 0.1)) == 0
end

@testset "interpolation exactness" begin
    # a degree-(p-1) polynomial is recovered exactly from its values at the points
    p = 10
    x = chebyshev_points(p)
    f(t) = 1 + 2t - 3t^2 + 0.5t^5 - t^9
    coeffs = chebyshev_vandermonde_inverse(p) * f.(x)
    for t in (2rand(rng, 5) .- 1)
        @test clenshaw(coeffs, t) ≈ f(t)
    end
end

@testset "tensor nodes and 2D Clenshaw" begin
    p = 6
    nodes = chebyshev_tensor_nodes(Val(2), p)
    x = chebyshev_points(p)
    @test nodes isa Vector && length(nodes) == p^2
    @test nodes[2] == SVector(x[2], x[1])          # coordinate 1 fastest
    @test nodes[p+1] == SVector(x[1], x[2])
    V = chebyshev_vandermonde(x, p)
    V2 = kron(V, V)
    # V2[m, n] = T_{j}(x_a) T_{k}(y_b) with m ↔ (a,b), n ↔ (j,k), first index fastest
    for (m, nd) in enumerate(nodes), n in 1:p^2
        j, k = (n - 1) % p, (n - 1) ÷ p
        @test V2[m, n] ≈ T_exact(j, nd[1]) * T_exact(k, nd[2]) atol = 1e-14
    end
    c = randn(rng, p^2)
    for q in (2rand(rng, 3) .- 1, 2rand(rng, 3) .- 1)
        xq, yq = q[1], q[2]
        exact = sum(c[j+1 + k*p] * T_exact(j, xq) * T_exact(k, yq) for j in 0:p-1, k in 0:p-1)
        @test clenshaw2d(c, p, xq, yq) ≈ exact
    end
    clenshaw2d(c, p, 0.1, 0.2)
    @test @allocated(clenshaw2d(c, p, 0.1, 0.2)) == 0
    # interpolation through the tensor Vandermonde inverse: exact for a polynomial of
    # degree < p in each variable
    VI2 = kron(chebyshev_vandermonde_inverse(p), chebyshev_vandermonde_inverse(p))
    g(v) = 1 + v[1]^3 * v[2]^2 - 2v[2]^5 + v[1]^4 - 0.3v[1] * v[2]
    coeffs = VI2 * g.(nodes)
    @test clenshaw2d(coeffs, p, 0.3, -0.4) ≈ g(SVector(0.3, -0.4))
    # and spectrally accurate for a smooth function (coefficients of cos(2y) decay like J_n(2))
    p16 = 16
    nodes16 = chebyshev_tensor_nodes(Val(2), p16)
    VI16 = kron(chebyshev_vandermonde_inverse(p16), chebyshev_vandermonde_inverse(p16))
    h(v) = exp(v[1]) * cos(2v[2]) + v[1] * v[2]^2
    coeffs16 = VI16 * h.(nodes16)
    @test clenshaw2d(coeffs16, p16, 0.3, -0.4) ≈ h(SVector(0.3, -0.4)) atol = 1e-10
end
