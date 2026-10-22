# The FastMath policy on Metal: the device's fast reciprocal square root for
# Float32 and for its dual numbers (the derivative rule as in JellySim). Loaded
# automatically with Metal; the CPU has no FastMath method on purpose.
module FlexibleFMMMetalExt

using FlexibleFMM, Metal, ForwardDiff
import FlexibleFMM: _rinv, FastMath

@inline _rinv(::FastMath, r2::Float32) = Metal.rsqrt_fast(r2)
@inline function _rinv(::FastMath, x::ForwardDiff.Dual{Tg,Float32,N}) where {Tg,N}
    v = Metal.rsqrt_fast(ForwardDiff.value(x))
    return ForwardDiff.Dual{Tg}(v, ForwardDiff.partials(x) * (-v * v * v / 2))
end

end
