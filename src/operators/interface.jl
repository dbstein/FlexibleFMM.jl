# The method interface. A method (kernel-independent, black-box, …) turns a kernel
# into the five operators the driver needs, per box width:
#
#   s2m!(col, m, ops, xs, τs, c)           sources of a childless node → its partial multipole
#                                          (any density type with a `source_field`)
#   ops.M2M   (Nm × 2^D·Nm)                stacked children multipoles → parent multipole
#   ops.M2L[f] (Nexp × Nexp)               source multipole at far offset f → target partial local
#   ops.L2L   (2^D·pll × pll)              parent partial local → stacked children partial locals
#   l2t(m, ops, L, xrel, out)              local column → output `out` at a point relative to the center
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

A same-width translation operator, dense. Another form needs `apply!(out, op, x, α[,
work])`, computing `out = α·op·x`, and `workspace_rows(op)`.
"""
struct M2LOperator{T}
    A::Matrix{T}
end
@inline apply!(out::AbstractVecOrMat, op::M2LOperator{T}, x::AbstractVecOrMat, α=true) where {T} =
    mul!(out, op.A, x, T(α), false)
@inline apply!(out::AbstractVecOrMat, op::M2LOperator, x::AbstractVecOrMat, α, work) = apply!(out, op, x, α)
Base.size(op::M2LOperator) = size(op.A)

"""
    LowRankM2L{T}

A translation operator in truncated-SVD form `U·Vt`, `U` of size `n × r` with the
singular values folded in: `4nr` flops per column instead of `2n²`. The intermediate
`Vt·x` goes to `work` (at least `r` rows and as many columns as `x`; the driver
takes it from the state's scratch, so the operator can be shared between levels,
states and threads); the form without `work` allocates it.
"""
struct LowRankM2L{T}
    U::Matrix{T}
    Vt::Matrix{T}
end
Base.size(op::LowRankM2L) = (size(op.U, 1), size(op.Vt, 2))
function apply!(out::AbstractVector, op::LowRankM2L{T}, x::AbstractVector, α, work::AbstractVector) where {T}
    w = view(work, 1:size(op.U, 2))
    mul!(w, op.Vt, x)
    return mul!(out, op.U, w, T(α), false)
end
function apply!(out::AbstractMatrix, op::LowRankM2L{T}, x::AbstractMatrix, α, work::AbstractMatrix) where {T}
    w = view(work, 1:size(op.U, 2), 1:size(x, 2))
    mul!(w, op.Vt, x)
    return mul!(out, op.U, w, T(α), false)
end
apply!(out::AbstractVecOrMat, op::LowRankM2L, x::AbstractVecOrMat, α=true) =
    apply!(out, op, x, α, similar(x, size(op.U, 2), size(x)[2:end]...))
apply!(out::AbstractVecOrMat, op::LowRankM2L, x::AbstractVecOrMat, α, ::Nothing) = apply!(out, op, x, α)

"""
    workspace_rows(op)

Rows of the workspace `apply!` needs for `op`, 0 for a dense operator.
"""
workspace_rows(::M2LOperator) = 0
workspace_rows(op::LowRankM2L) = size(op.U, 2)

"""
    LevelOperators{T,R,X,L}

The translation operators of one box width, with method-specific extras `X` and
M2L operators of type `L` (dense `M2LOperator` or spectral). The matrices may be
shared with other widths (homogeneous kernels); the per-width factors `m2m_scale`,
`l2l_scale`, `m2l_scale` are applied when they are used.
"""
struct LevelOperators{T,R<:Real,X,L}
    width::R
    M2M::Matrix{T}
    L2L::Matrix{T}
    M2L::Vector{L}
    m2m_scale::R
    l2l_scale::R
    m2l_scale::R
    extra::X
end
LevelOperators(width::R, M2M::Matrix{T}, L2L::Matrix{T}, M2L::Vector{L}, extra::X) where {T,R,X,L} =
    LevelOperators{T,R,X,L}(width, M2M, L2L, M2L, one(R), one(R), one(R), extra)

"""
    m2l_level!(pL, m, ops, Mp, il, off, state)

Add the far-field contributions of one level to its partial locals `pL` (columns
local to the level, global node `i` at column `i - off`), from the multipoles `Mp`
and the level's interaction list. The default applies each far offset's operator to
the gathered sources (gather → GEMM → scatter-add); a method may override it with a
level-wise scheme (the spectral translation of the square/cube method).
"""
function m2l_level! end

"""
    rescale(ops, m, w) -> LevelOperators

Operators for width `w` sharing the matrices of `ops` (built for a homogeneous
kernel at another width): scaled surfaces or nodes and per-width scalar factors.
"""
function rescale end

"""
    dimension(m)

Spatial dimension the method's operators are built for.
"""
function dimension end

"""
    operators_type(m)

The concrete `LevelOperators` type `level_operators(m, w)` returns; used to type the
operator cache without relying on inference.
"""
function operators_type end

"""
    supports_density(m, T), supports_output(m, out)

Whether a method can form expansions from densities of type `T` and evaluate the
output `out`; checked before the threaded passes so that unsupported combinations
raise a plain `ArgumentError`.
"""
supports_density(m::AbstractMethod, ::Type) = true
supports_density(m::AbstractMethod, ::Type{T}) where {T<:Tuple} = all(S -> supports_density(m, S), fieldtypes(T))
supports_output(m::AbstractMethod, out) = true

function expansion_length end
"""
    multipole_length(m), local_length(m)
    partial_multipole_length(m), partial_local_length(m)

Lengths of the multipole and local expansions and of the partial expansions (the
accumulated check data before the method's transform). All equal
`expansion_length(m)` unless a method augments one of them, as the kernel-independent
Stokes method does: its multipoles carry a point-source strength (the single-layer
density cannot represent the net flux of stresslet sources) and its downward check
data carry the pressure at the box center.
"""
multipole_length(m::AbstractMethod) = expansion_length(m)
local_length(m::AbstractMethod) = expansion_length(m)
partial_multipole_length(m::AbstractMethod) = expansion_length(m)
partial_local_length(m::AbstractMethod) = expansion_length(m)
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
    D = dimension(m)
    return promote_type(operator_eltype(m),
                        eltype(kernel_eltype(K, SVector{D,Tpos}, SVector{D,floattype(K)})),
                        density_scalar(Tτ))
end
