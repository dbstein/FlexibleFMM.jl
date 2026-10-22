using FlexibleFMM
using Test
using LinearAlgebra
using StaticArrays
using SpecialFunctions
using ForwardDiff

include("testutils.jl")
include("operator_checks.jl")

@testset "FlexibleFMM" begin
    @testset "util" begin
        include("test_util.jl")
    end
    @testset "kernels" begin
        include("test_kernels.jl")
    end
    @testset "chebyshev" begin
        include("test_chebyshev.jl")
    end
    @testset "tree" begin
        include("test_tree.jl")
    end
    @testset "coverage" begin
        include("test_coverage.jl")
    end
    @testset "KIFMM operators" begin
        include("test_operators_kifmm.jl")
    end
    @testset "BBFMM operators" begin
        include("test_operators_bbfmm.jl")
    end
    @testset "FMM, real kernels" begin
        include("test_fmm_laplace.jl")
    end
    @testset "FMM, complex kernels" begin
        include("test_fmm_complex.jl")
    end
    @testset "rings" begin
        include("test_rings.jl")
    end
    @testset "threads and allocations" begin
        include("test_threads.jl")
    end
    @testset "ForwardDiff" begin
        include("test_forwarddiff.jl")
    end
    @testset "cache" begin
        include("test_cache.jl")
    end
    @testset "Float32" begin
        include("test_float32.jl")
    end
    @testset "sources and targets" begin
        include("test_sources.jl")
    end
    @testset "Stokes" begin
        include("test_stokes.jl")
    end
    @testset "homogeneous kernels" begin
        include("test_homogeneous.jl")
    end
    @testset "square surfaces and spectral translation" begin
        include("test_cubeki.jl")
    end
    @testset "three dimensions" begin
        include("test_3d.jl")
    end
    @testset "spherical harmonics" begin
        include("test_harmonics.jl")
    end
    @testset "Stokes harmonics" begin
        include("test_harmonics_stokes.jl")
    end
    @testset "Yukawa harmonics" begin
        include("test_harmonics_yukawa.jl")
    end
    @testset "device evaluation" begin
        include("test_device.jl")
    end
    @testset "symmetry classes" begin
        include("test_symmetry.jl")
    end
    @testset "device passes" begin
        include("test_device_passes.jl")
    end
    @testset "device tree" begin
        include("test_device_tree.jl")
    end
    @testset "device passes, harmonic methods" begin
        include("test_device_harmonics.jl")
    end
end
