module FlexibleFMM

using LinearAlgebra
using StaticArrays
using SpecialFunctions: besselj0, bessely0, besselj1, bessely1, besselk
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
include("operators/kifmm.jl")
include("operators/bbfmm.jl")
include("fmm.jl")
include("evaluate.jl")

export Laplace2D, Helmholtz2D, Yukawa2D, LaplaceDipoleTest
export Dipole, Potential, Gradient, source_field, target_field, gradient_s, gradient_t, hessian_t
export kernel_matrix, direct, direct!, direct_self, direct_self!
export Tree, locate, locate!, square_bbox
export KernelIndependent, BlackBox, OperatorCache
export FMM, FMMState, build_expansions, build_expansions!
export evaluate, evaluate!, evaluate_at_sources, evaluate_at_sources!

end
