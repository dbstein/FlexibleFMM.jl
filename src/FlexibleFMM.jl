module FlexibleFMM

using LinearAlgebra
using StaticArrays
using SpecialFunctions: besselj0, bessely0, besselj1, bessely1, besselk
using FFTW: fft, fft!, ifft!, plan_fft!, plan_bfft!, plan_rfft, plan_brfft
import ForwardDiff

include("util.jl")
include("kernels.jl")
include("sources.jl")
include("chebyshev.jl")
include("tree/tree.jl")
include("tree/lists.jl")
include("tree/builder.jl")
include("tree/locate.jl")
include("operators/interface.jl")
include("operators/cache.jl")
include("operators/ki_common.jl")
include("operators/kifmm.jl")
include("operators/spectral_m2l.jl")
include("operators/bbfmm.jl")
include("fmm.jl")
include("operators/cubeki.jl")
include("evaluate.jl")
include("stokes_api.jl")
include("device.jl")

export Laplace2D, Helmholtz2D, Yukawa2D, LaplaceDipoleTest, Stokeslet2D, pressure_kernel, homogeneity
export Laplace3D, Helmholtz3D, Yukawa3D, Stokeslet3D
export Dipole, Stresslet, Doublet, Rotlet, Potential, Velocity, Gradient, Pressure
export source_field, target_field, gradient_s, gradient_t, hessian_t
export StokesFMMPlan, stokes_fmm_eval!, stokes_fmm_eval_src_trg!, StokesFMMKernel, StokesFMMKernel!, stokes_densities
export kernel_matrix, direct, direct!, direct_self, direct_self!
export Tree, locate, locate!, square_bbox
export KernelIndependent, CubeKernelIndependent, SquareKernelIndependent, BlackBox, OperatorCache
export FMM, FMMState, build_expansions, build_expansions!
export evaluate, evaluate!, evaluate_at_sources, evaluate_at_sources!
export DeviceEvaluator, device_state

end
