# Operators depend only on the box width. A cache stores them by integer exponent
# relative to a base width, `w = ldexp(W0, e)`, which is exact, and root boxes are
# snapped to such widths so that every FMM built with the same cache shares them.

mutable struct OperatorCache{M<:AbstractMethod,O}
    method::M
    W0::Float64            # NaN until the first box is fitted
    ops::Dict{Int,O}
end

"""
    OperatorCache(method)

An empty operator cache for `method`. Pass the same cache to several `FMM`s to reuse
their translation operators.
"""
function OperatorCache(method::AbstractMethod)
    O = operators_type(method)
    return OperatorCache{typeof(method),O}(method, NaN, Dict{Int,O}())
end
has_base_width(c::OperatorCache) = !isnan(c.W0)
Base.length(c::OperatorCache) = length(c.ops)

"""
    get_operators!(cache, e)

Operators for width `ldexp(cache.W0, e)`, built on first use. For a homogeneous
kernel only the reference width (exponent 0) is built; every other width shares its
matrices through `rescale`.
"""
function get_operators!(c::OperatorCache, e::Int)
    has_base_width(c) || error("fit a box before requesting operators")
    haskey(c.ops, e) && return c.ops[e]
    m = c.method
    w = ldexp(c.W0, e)
    if homogeneity(near_kernel(m)) === nothing || e == 0
        ops = level_operators(m, w)
    else
        ref = get_operators!(c, 0)
        ops = rescale(ref, m, w)
    end
    c.ops[e] = ops
    return ops
end

"""
    fit_box!(cache, points, bbox) -> (origin, W, e)

A square box containing `bbox = (origin, W)` (or, if `nothing`, the points' own
square box) whose width is `ldexp(cache.W0, e)`. The first call fixes `cache.W0`.
The box keeps the requested center except for the lower-edge adjustment that exact
containment of the points may need; if that leaves the upper edge short, the width
is doubled.
"""
function fit_box!(c::OperatorCache, points::AbstractVector{SVector{D,T}}, bbox) where {D,T}
    lo = reduce((a, b) -> min.(a, b), points)
    hi = reduce((a, b) -> max.(a, b), points)
    return fit_box!(c, lo, hi, bbox === nothing ? square_bbox(points) : bbox)
end
# from the points' bounds (`lo`, `hi`) and the requested box
function fit_box!(c::OperatorCache, lo::SVector{D,T}, hi::SVector{D,T}, bbox) where {D,T}
    origin, W = SVector{D,T}(bbox[1]), T(bbox[2])
    W > 0 || throw(ArgumentError("box width must be positive"))
    all(origin .<= lo) && all(origin .+ W .>= hi) || throw(ArgumentError("points lie outside the requested box"))
    if !has_base_width(c)
        c.W0 = Float64(W)
        return origin, W, 0
    end
    f, e = frexp(Float64(W) / c.W0)      # W/W0 = f·2^e, f ∈ [1/2, 1)
    e = f == 0.5 ? e - 1 : e              # smallest e with ldexp(W0, e) ≥ W
    cen = origin .+ W / 2
    for _ in 1:2
        Wn = T(ldexp(c.W0, e))
        on = min.(cen .- Wn / 2, lo)      # exact containment at the lower edges
        all(on .+ Wn .>= hi) && return on, Wn, e
        e += 1                            # an ulp short: take the next width
    end
    error("could not fit a box")
end
