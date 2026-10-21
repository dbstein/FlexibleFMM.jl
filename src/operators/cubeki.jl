# Kernel-independent FMM on square/cube surfaces (any dimension), after Ying,
# Biros & Zorin: the surfaces are the boundary points of p-per-edge lattices on the
# squares/cubes of half-width (1+d)·r (small: upward equivalent and downward check)
# and (3-2d)·r (large: upward check and downward equivalent), r the box half-width.
# The margin d keeps the box's corner sources off the equivalent surface and the
# large surface inside the nearest far box (whose face is at 3r). Because the small
# surfaces of all boxes are lattices of the same spacing, the M2L operators are
# spectral (src/operators/spectral_m2l.jl); M2M, L2L and the check solves are dense.
#
# The large surface carries pc > p points per edge. With pc = p its lattice is
# (3-2d)/(1+d) ≈ 2.5 times coarser than the small one, so as an upward check it
# undersamples the equivalent density: modes the check cannot see get tiny singular
# values and either amplify rounding (kept) or are lost (truncated). In 3D at p = 8
# this cost the stresslet and doublet translations three orders of magnitude; pc = 2p
# recovers them (docs/kifmm.md). Circles have no such problem: the equispaced
# circle-to-circle map is diagonal in the Fourier modes.

struct CubeKernelIndependent{D,K} <: AbstractKernelIndependent
    kernel::K
    p::Int
    pc::Int
    d::Float64
    pinv_rtol::Float64
end

"""
    CubeKernelIndependent{D}(kernel; p=8, pc=2p, d=0.1, pinv_rtol=50·eps(floattype(kernel)))
    SquareKernelIndependent(kernel; p=8, pc=2p, d=0.1, pinv_rtol=50·eps(floattype(kernel)))

The kernel-independent method on the surface of a square (`D = 2`) or cube (`D = 3`)
with `p` points per edge on the small surface (`p^D - (p-2)^D` points: `4p - 4` in 2D,
`6p² - 12p + 8` in 3D), `pc` per edge on the large one, and margin `d`. Translation
operators are applied by FFT; the order of the method is set by `p`, and `pc = 2p`
keeps the large surface from undersampling the equivalent density (see the file
header). `SquareKernelIndependent` is `CubeKernelIndependent{2}`.
"""
function CubeKernelIndependent{D}(kernel; p::Integer=8, pc::Integer=2p, d::Real=0.1,
                                  pinv_rtol::Real=50 * eps(floattype(kernel))) where {D}
    p >= 2 || throw(ArgumentError("p must be at least 2"))
    pc >= p || throw(ArgumentError("pc must be at least p"))
    0 < d < 1 || throw(ArgumentError("d must lie in (0, 1)"))
    return CubeKernelIndependent{D,typeof(kernel)}(kernel, Int(p), Int(pc), Float64(d), Float64(pinv_rtol))
end
const SquareKernelIndependent = CubeKernelIndependent{2}

dimension(::CubeKernelIndependent{D}) where {D} = D
nsurface(m::CubeKernelIndependent{D}) where {D} = m.p^D - (m.p - 2)^D
nlarge(m::CubeKernelIndependent{D}) where {D} = m.pc^D - (m.pc - 2)^D
function operators_type(m::CubeKernelIndependent{D}) where {D}
    T = operator_eltype(m)
    R = floattype(m.kernel)
    return LevelOperators{T,R,KIExtras{T,R,D},SpectralM2L{R,D}}
end

function level_operators(m::CubeKernelIndependent{D}, w::Real) where {D}
    K = m.kernel
    R = floattype(K)
    w = R(w)
    p = m.p
    a_small = R(1 + m.d) * w / 2
    a_large = R(3 - 2m.d) * w / 2
    small, embed = cube_boundary_points(Val(D), p, a_small)
    large, _ = cube_boundary_points(Val(D), m.pc, a_large)
    child_small, _ = cube_boundary_points(Val(D), p, a_small / 2)
    P_up, P_down, M2M, L2L = _ki_dense_operators(m, w, small, large, child_small)
    T = operator_eltype(m)
    h = 2a_small / (p - 1)
    nc = ncomp(K, Val(D))
    origin = zero(SVector{D,R})
    M2L = SpectralM2L{R,D}[]
    for off in far_offsets(Val(D))
        o = SVector{D,R}(off)
        tables = _kernel_tables(K, Val(D), p, h, o, w)
        pin = pins_pressure(K) ? vec(_pressure_row(K, small .+ Ref(o * w), origin, w)) : R[]
        has_flux_source(K) && !isempty(pin) && push!(pin, zero(R))
        src = has_flux_source(K) ? vec(_source_column(K, small, o * w, w)) : R[]
        push!(M2L, SpectralM2L{R,D}(tables, nc, p, embed, pin, src))
    end
    extra = KIExtras{T,R,D}(small, large, P_up, P_down, one(R), p, h, embed)
    return LevelOperators(w, Matrix{T}(M2M), Matrix{T}(L2L), M2L, extra)::operators_type(m)
end

function m2l_level!(pL::AbstractMatrix, m::CubeKernelIndependent{D}, ops::LevelOperators, Mp::AbstractMatrix{S},
                    il::InteractionList, off::Int, state::FMMState) where {D,S}
    nc = ncomp(m.kernel, Val(D))
    N = length(ops.extra.small)
    np = _nparts(S)
    R = _basereal(S)
    nl = size(pL, 2)
    key = (:spectral, R, D, ops.extra.p, nc * np, nl)
    ws = get!(state.scratch, key) do
        SpectralWorkspace(R, Val(D), ops.extra.p, nc * np, nl)
    end::SpectralWorkspace{R,D}
    return spectral_m2l_level!(pL, ops, Mp, il, off, ws, ops.m2l_scale, N, nc)
end
