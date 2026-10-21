# The method interface. A method (kernel-independent, black-box, …) turns a kernel
# into the five operators the driver needs, per box width:
#
#   s2m!(col, m, ops, xs, τs, c)           sources of a childless node → its partial multipole
#   ops.M2M   (Nexp × 2^D·Nexp)            stacked children multipoles → parent partial multipole
#   ops.M2L[f] (Nexp × Nexp)               source multipole at far offset f → target partial local
#   ops.L2L   (2^D·Nexp × Nexp)            parent local → stacked children partial locals
#   l2t(m, ops, L, xrel)                   local column → potential at a point relative to the center
#   partial_to_multipole!(M, m, ops, pM, work), partial_to_local!(L, m, ops, pL, work)
#                                          partial → actual expansions of a whole level
#
# plus `expansion_length(m)`, `operator_eltype(m)` and `near_kernel(m)`. "Partial"
# expansions are the accumulated contributions before the method's per-level
# transform (check potentials for the kernel-independent method, where the transform
# is a pseudo-inverse; the identity for the black-box method). Multipoles and locals
# are whatever the method chooses them to be; the driver only moves columns around
# and multiplies by these matrices.

abstract type AbstractMethod end

"""
    M2LOperator{T}

A same-width translation operator. Dense for now; a compressed form only needs to
provide `apply!`.
"""
struct M2LOperator{T}
    A::Matrix{T}
end
@inline apply!(out::AbstractVecOrMat, op::M2LOperator, x::AbstractVecOrMat) = mul!(out, op.A, x)
Base.size(op::M2LOperator) = size(op.A)

"""
    LevelOperators{T,X}

The translation operators of one box width, with method-specific extras `X`.
"""
struct LevelOperators{T,X}
    width::Float64
    M2M::Matrix{T}
    L2L::Matrix{T}
    M2L::Vector{M2LOperator{T}}
    extra::X
end

function expansion_length end
function operator_eltype end
function level_operators end
function s2m! end
function l2t end
function near_kernel end
function partial_to_multipole! end
function partial_to_local! end

"""
    expansion_eltype(m, Tpos, Tτ)

Element type of expansions built from positions of eltype `Tpos` and densities of
eltype `Tτ`: the operator eltype promoted with the kernel's output on such positions.
"""
function expansion_eltype(m::AbstractMethod, ::Type{Tpos}, ::Type{Tτ}) where {Tpos,Tτ}
    K = near_kernel(m)
    return promote_type(operator_eltype(m),
                        kernel_eltype(K, SVector{2,Tpos}, SVector{2,floattype(K)}), Tτ)
end
