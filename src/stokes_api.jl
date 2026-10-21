# A Stokes front end in the shape of JellySim's kernel backends: separate arrays for
# stokeslets, stresslets (+ stressvecs) and doublets, optional per-source weights,
# `targets = nothing` for self-evaluation (the i = j term skipped), a persistent
# plan for repeated solves on a fixed geometry, and a combined source+target
# evaluation. μ is carried by the densities, as in JellySim.

"""
    StokesFMMPlan(sources; targets=nothing, method=:ki, nequiv=60, p=12, ppl=50,
                  bbox=nothing, precision=Float64, cache=nothing)

Tree and operators for a fixed geometry (`sources` and, if given, `targets`, both
vectors of `SVector{2}` or `2 × N` matrices). `method` is `:ki` (kernel independent,
`nequiv` points per circle; supports every singularity and the pressure) or `:bb`
(black box, order `p`; stokeslets, doublets and rotlets, velocity only).
"""
mutable struct StokesFMMPlan{F<:FMM,T}
    fmm::F
    sources::Vector{SVector{2,T}}
    targets::Union{Nothing,Vector{SVector{2,T}}}
    state::Any
end

_value(x::Real) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_values(pts::AbstractVector{<:SVector{D}}) where {D} = [SVector{D}(map(_value, p)) for p in pts]

function StokesFMMPlan(sources; targets=nothing, method::Symbol=:ki, nequiv::Integer=60,
                       p::Integer=12, ppl::Integer=50, bbox=nothing, precision::Type{<:AbstractFloat}=Float64,
                       cache=nothing)
    K = Stokeslet2D(precision)
    m = method === :ki ? KernelIndependent(K; nequiv=nequiv) :
        method === :bb ? BlackBox(K; p=p) :
        throw(ArgumentError("method must be :ki or :bb"))
    pts = _values(as_svectors(Val(2), sources))
    tgts = targets === nothing ? nothing : _values(as_svectors(Val(2), targets))
    if bbox === nothing
        bbox = square_bbox(tgts === nothing ? pts : vcat(pts, tgts))
    end
    f = cache === nothing ? FMM(m, pts; ppl=ppl, bbox=bbox) : FMM(m, pts; ppl=ppl, bbox=bbox, cache=cache)
    return StokesFMMPlan{typeof(f),eltype(eltype(pts))}(f, pts, tgts, nothing)
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

function _state!(plan::StokesFMMPlan, srcs, τ)
    Tp = eltype(eltype(srcs))
    Tτ = eltype(τ)
    S = expansion_eltype(plan.fmm.method, Tp, Tτ)
    st = plan.state
    if !(st isa FMMState{S,SVector{2,Tp},Tτ})
        st = FMMState(plan.fmm, S, Tp, Tτ)
        plan.state = st
    end
    return st::FMMState{S,SVector{2,Tp},Tτ}
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
function stokes_fmm_eval!(velocities::AbstractVector, plan::StokesFMMPlan;
                          stokeslets=nothing, stresslets=nothing, stressvecs=nothing, doublets=nothing,
                          weights=nothing, sources=nothing, targets=nothing, output=Velocity())
    f = plan.fmm
    n = npoints(f.tree)
    srcs = sources === nothing ? _plan_sources(plan) : as_svectors(Val(2), sources)
    τ = stokes_densities(n; weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    st = _state!(plan, srcs, τ)
    build_expansions!(st, f, srcs, τ)
    if plan.targets === nothing && targets === nothing
        return evaluate_at_sources!(velocities, f, st; output=output)
    else
        tg = targets === nothing ? plan.targets : as_svectors(Val(2), targets)
        return evaluate!(velocities, f, st, tg; output=output)
    end
end

"""
    stokes_fmm_eval_src_trg!(vel_src, vel_trg, plan; ...)

One expansion build, two evaluations: at the sources (self, i = j skipped) into
`vel_src` and at the plan's targets into `vel_trg`.
"""
function stokes_fmm_eval_src_trg!(vel_src::AbstractVector, vel_trg::AbstractVector, plan::StokesFMMPlan;
                                  stokeslets=nothing, stresslets=nothing, stressvecs=nothing, doublets=nothing,
                                  weights=nothing, sources=nothing, targets=nothing, output=Velocity())
    plan.targets === nothing && targets === nothing && throw(ArgumentError("the plan has no targets"))
    f = plan.fmm
    n = npoints(f.tree)
    srcs = sources === nothing ? _plan_sources(plan) : as_svectors(Val(2), sources)
    τ = stokes_densities(n; weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    st = _state!(plan, srcs, τ)
    build_expansions!(st, f, srcs, τ)
    evaluate_at_sources!(vel_src, f, st; output=output)
    tg = targets === nothing ? plan.targets : as_svectors(Val(2), targets)
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
    srcs = as_svectors(Val(2), sources)
    self = targets === nothing || sources === targets || force_self_eval
    tgts = self ? nothing : as_svectors(Val(2), targets)
    plan = StokesFMMPlan(srcs; targets=tgts, kwargs...)
    return stokes_fmm_eval!(velocities, plan; stokeslets=stokeslets, stresslets=stresslets,
                            stressvecs=stressvecs, doublets=doublets, weights=weights,
                            sources=srcs, targets=tgts, output=output)
end
function StokesFMMKernel(sources; weights=nothing, targets=nothing, stokeslets=nothing, stresslets=nothing,
                         stressvecs=nothing, doublets=nothing, force_self_eval::Bool=false, output=Velocity(), kwargs...)
    srcs = as_svectors(Val(2), sources)
    self = targets === nothing || sources === targets || force_self_eval
    tgts = self ? srcs : as_svectors(Val(2), targets)
    τ = stokes_densities(length(srcs); weights=weights, stokeslets=stokeslets, stresslets=stresslets,
                         stressvecs=stressvecs, doublets=doublets)
    K = Stokeslet2D(get(kwargs, :precision, Float64))
    OT = Base.promote_op(target_field, typeof(K), eltype(srcs), eltype(tgts), eltype(τ), typeof(output))
    velocities = Vector{OT}(undef, length(tgts))
    return StokesFMMKernel!(velocities, srcs; weights=weights, targets=self ? nothing : tgts,
                            stokeslets=stokeslets, stresslets=stresslets, stressvecs=stressvecs,
                            doublets=doublets, output=output, kwargs...)
end
