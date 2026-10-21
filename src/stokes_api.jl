# A Stokes front end in the shape of JellySim's kernel backends: separate arrays for
# stokeslets, stresslets (+ stressvecs) and doublets, optional per-source weights,
# `targets = nothing` for self-evaluation (the i = j term skipped), a persistent
# plan for repeated solves on a fixed geometry, and a combined source+target
# evaluation. μ is carried by the densities, as in JellySim.

"""
    StokesFMMPlan(sources; targets=nothing, method=:ki, nequiv=60, p=12, ppl=50,
                  bbox=nothing, precision=Float64, cache=nothing)

Tree and operators for a fixed geometry (`sources` and, if given, `targets`, both
vectors of `SVector` or `D × N` matrices). `method` is `:ki` (kernel independent on
circles, `nequiv` points; 2D), `:grid` (kernel independent on grid-boundary
surfaces, `p` points per edge and margin `d`, spectral translations; any
dimension) or `:bb` (black box, order `p`; stokeslets, doublets and rotlets,
velocity only; 2D). The first two support every singularity and the pressure.
"""
mutable struct StokesFMMPlan{D,F<:FMM,T}
    fmm::F
    sources::Vector{SVector{D,T}}
    targets::Union{Nothing,Vector{SVector{D,T}}}
    state::Any
end

"""
    stokeslet_kernel(::Val{D}, precision)

The Stokes kernel of the Stokes front end in `D` dimensions.
"""
stokeslet_kernel(::Val{2}, ::Type{T}) where {T} = Stokeslet2D(T)
stokeslet_kernel(::Val{D}, ::Type{T}) where {D,T} = throw(ArgumentError("no Stokes kernel in $D dimensions yet"))

_value(x::Real) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_values(pts::AbstractVector{<:SVector{D}}) where {D} = [SVector{D}(map(_value, p)) for p in pts]

function StokesFMMPlan(sources; targets=nothing, method::Symbol=:ki, nequiv::Integer=60,
                       p::Integer=12, d::Real=0.1, ppl::Integer=50, bbox=nothing,
                       precision::Type{<:AbstractFloat}=Float64, cache=nothing)
    D = point_dimension(sources)
    K = stokeslet_kernel(Val(D), precision)
    m = method === :ki ? KernelIndependent(K; nequiv=nequiv) :
        method === :grid ? GridKernelIndependent{D}(K; p=p, d=d) :
        method === :bb ? BlackBox(K; p=p) :
        throw(ArgumentError("method must be :ki, :grid or :bb"))
    pts = _values(as_svectors(Val(D), sources))
    tgts = targets === nothing ? nothing : _values(as_svectors(Val(D), targets))
    if bbox === nothing
        bbox = square_bbox(tgts === nothing ? pts : vcat(pts, tgts))
    end
    f = cache === nothing ? FMM(m, pts; ppl=ppl, bbox=bbox) : FMM(m, pts; ppl=ppl, bbox=bbox, cache=cache)
    return StokesFMMPlan{D,typeof(f),eltype(eltype(pts))}(f, pts, tgts, nothing)
end

# One density entry per source: a tuple of the singularities present (weighted).
@inline _wt(w::Nothing, j) = true
@inline _wt(w::AbstractVector, j) = @inbounds w[j]
@inline _parts(j, w, stk::AbstractVector, strs, nu, dub) = ((@inbounds stk[j] * _wt(w, j)), _parts(j, w, nothing, strs, nu, dub)...)
@inline _parts(j, w, ::Nothing, strs::AbstractVector, nu::AbstractVector, dub) = (Stresslet((@inbounds strs[j] * _wt(w, j)), (@inbounds nu[j])), _parts(j, w, nothing, nothing, nothing, dub)...)
@inline _parts(j, w, ::Nothing, ::Nothing, ::Nothing, dub::AbstractVector) = (Doublet((@inbounds dub[j] * _wt(w, j))),)
@inline _parts(j, w, ::Nothing, ::Nothing, ::Nothing, ::Nothing) = ()

function stokes_densities(n::Int; weights=nothing, stokeslets=nothing, stresslets=nothing,
                          stressvecs=nothing, doublets=nothing)
    (stresslets === nothing) == (stressvecs === nothing) ||
        throw(ArgumentError("stresslets and stressvecs must be given together"))
    for (name, a) in (("weights", weights), ("stokeslets", stokeslets), ("stresslets", stresslets),
                      ("stressvecs", stressvecs), ("doublets", doublets))
        a === nothing || length(a) == n || throw(DimensionMismatch("$name has length $(length(a)), expected $n"))
    end
    (stokeslets !== nothing || stresslets !== nothing || doublets !== nothing) ||
        throw(ArgumentError("provide at least one of stokeslets, stresslets, doublets"))
    return [_parts(j, weights, stokeslets, stresslets, stressvecs, doublets) for j in 1:n]
end

function _state!(plan::StokesFMMPlan{D}, srcs, τ) where {D}
    Tp = eltype(eltype(srcs))
    Tτ = eltype(τ)
    S = expansion_eltype(plan.fmm.method, Tp, Tτ)
    st = plan.state
    if !(st isa FMMState{S,SVector{D,Tp},Tτ})
        st = FMMState(plan.fmm, S, Tp, Tτ)
        plan.state = st
    end
    return st::FMMState{S,SVector{D,Tp},Tτ}
end

"""
    stokes_fmm_eval!(velocities, plan; stokeslets=nothing, stresslets=nothing,
                     stressvecs=nothing, doublets=nothing, weights=nothing,
                     sources=nothing, targets=nothing, output=Velocity())

Velocity (or `Pressure()` / `Gradient()`) at the plan's targets, or at the sources
themselves for a self plan, from the given densities. `sources`/`targets` may
override the positions (same points, e.g. dual numbers whose values match the
plan's geometry) for directional derivatives.
"""
function stokes_fmm_eval!(velocities::AbstractVector, plan::StokesFMMPlan{D};
                          stokeslets=nothing, stresslets=nothing, stressvecs=nothing, doublets=nothing,
                          weights=nothing, sources=nothing, targets=nothing, output=Velocity()) where {D}
    f = plan.fmm
    n = npoints(f.tree)
    srcs = sources === nothing ? _plan_sources(plan) : as_svectors(Val(D), sources)
    τ = stokes_densities(n; weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    st = _state!(plan, srcs, τ)
    build_expansions!(st, f, srcs, τ)
    if plan.targets === nothing && targets === nothing
        return evaluate_at_sources!(velocities, f, st; output=output)
    else
        tg = targets === nothing ? plan.targets : as_svectors(Val(D), targets)
        return evaluate!(velocities, f, st, tg; output=output)
    end
end

"""
    stokes_fmm_eval_src_trg!(vel_src, vel_trg, plan; ...)

One expansion build, two evaluations: at the sources (self, i = j skipped) into
`vel_src` and at the plan's targets into `vel_trg`.
"""
function stokes_fmm_eval_src_trg!(vel_src::AbstractVector, vel_trg::AbstractVector, plan::StokesFMMPlan{D};
                                  stokeslets=nothing, stresslets=nothing, stressvecs=nothing, doublets=nothing,
                                  weights=nothing, sources=nothing, targets=nothing, output=Velocity()) where {D}
    plan.targets === nothing && targets === nothing && throw(ArgumentError("the plan has no targets"))
    f = plan.fmm
    n = npoints(f.tree)
    srcs = sources === nothing ? _plan_sources(plan) : as_svectors(Val(D), sources)
    τ = stokes_densities(n; weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    st = _state!(plan, srcs, τ)
    build_expansions!(st, f, srcs, τ)
    evaluate_at_sources!(vel_src, f, st; output=output)
    tg = targets === nothing ? plan.targets : as_svectors(Val(D), targets)
    evaluate!(vel_trg, f, st, tg; output=output)
    return vel_src, vel_trg
end

_plan_sources(plan::StokesFMMPlan) = plan.sources

"""
    StokesFMMKernel!(velocities, sources; weights=nothing, targets=nothing,
                     stokeslets=nothing, stresslets=nothing, stressvecs=nothing,
                     doublets=nothing, force_self_eval=false, output=Velocity(), plan kwargs...)
    StokesFMMKernel(sources; ...)

One-shot evaluation in the shape of JellySim's `StokesKernel!`: builds a plan from
the (values of the) positions and evaluates. Dual-number positions, densities and
weights are allowed.
"""
function StokesFMMKernel!(velocities::AbstractVector, sources; weights=nothing, targets=nothing,
                          stokeslets=nothing, stresslets=nothing, stressvecs=nothing, doublets=nothing,
                          force_self_eval::Bool=false, output=Velocity(), kwargs...)
    D = point_dimension(sources)
    srcs = as_svectors(Val(D), sources)
    self = targets === nothing || sources === targets || force_self_eval
    tgts = self ? nothing : as_svectors(Val(D), targets)
    plan = StokesFMMPlan(srcs; targets=tgts, kwargs...)
    return stokes_fmm_eval!(velocities, plan; stokeslets=stokeslets, stresslets=stresslets,
                            stressvecs=stressvecs, doublets=doublets, weights=weights,
                            sources=srcs, targets=tgts, output=output)
end
function StokesFMMKernel(sources; weights=nothing, targets=nothing, stokeslets=nothing, stresslets=nothing,
                         stressvecs=nothing, doublets=nothing, force_self_eval::Bool=false, output=Velocity(), kwargs...)
    D = point_dimension(sources)
    srcs = as_svectors(Val(D), sources)
    self = targets === nothing || sources === targets || force_self_eval
    tgts = self ? srcs : as_svectors(Val(D), targets)
    τ = stokes_densities(length(srcs); weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    K = stokeslet_kernel(Val(D), get(kwargs, :precision, Float64))
    OT = Base.promote_op(target_field, typeof(K), eltype(srcs), eltype(tgts), eltype(τ), typeof(output))
    velocities = Vector{OT}(undef, length(tgts))
    return StokesFMMKernel!(velocities, srcs; weights=weights, targets=self ? nothing : tgts,
                            stokeslets=stokeslets, stresslets=stresslets, stressvecs=stressvecs,
                            doublets=doublets, output=output, kwargs...)
end
