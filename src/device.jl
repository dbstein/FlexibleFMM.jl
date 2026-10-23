# Device evaluation with KernelAbstractions: one work-item per target computes the
# local expansion of its leaf (the density on the large surface, as `l2t`) plus the
# direct sum over the leaf's colleagues (as `evaluate_target`), on the flat arrays
# of the tree. The caller chooses the device through a function that moves arrays
# (`identity` for the CPU backend, `MtlArray`, `CuArray`, …), so backends and their
# array types are not dependencies of the package. The same body runs on every
# backend; on the CPU backend it reproduces the threaded evaluation to rounding.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

"""
    DeviceEvaluator(fmm, to=identity; fast=false)

The tree data the device evaluation needs, moved once with `to`: the colleagues,
source ranges, levels and centers of the nodes, the method's far-field data (the
large surface of every level for the kernel-independent methods, `DeviceSurface`;
the scale per level and the recurrence tables for the harmonic methods,
`DeviceHarmonic`), the leaf and the output slot of every source, and the kernel,
with the `FastMath` policy if `fast` (relaxed arithmetic, `fastmath(K)`; the
kernel's own policy otherwise). `to` maps a CPU array to the device's array type.
"""
struct DeviceEvaluator{D,K,AI,VI,VP,F}
    kernel::K
    colleagues::AI        # 3^D × nnodes, Int32
    level::VI
    centers::VP           # SVector{D,T} per node
    far::F                # DeviceSurface or DeviceHarmonic
    gplo::AI              # per group and node: its points' range in the group's numbering, Int32
    gphi::AI
    gpoint_node::Vector{VI}   # per group: the leaf of each group-local sorted point
    gperm::Vector{VI}         # per group: the original index within the group of each sorted point
end
ngroups(dev::DeviceEvaluator) = length(dev.gperm)

# the far-field data per method, passed to the kernels as one argument (adapted
# to device arrays field by field)
struct DeviceSurface{AR}
    ysoa::AR              # large surface per level, nlarge × D × nlevels
end
struct DeviceHarmonic{V,TB}
    p::Int
    nb::Int               # scalar expansions per box (1, or 4 for Stokes)
    scales::V             # 1/w per level
    tb::TB                # HarmonicTables on the device
end
Adapt.@adapt_structure DeviceSurface
Adapt.@adapt_structure DeviceHarmonic

function _device_far(ops::Vector, ::AbstractKernelIndependent, L::Int, to, ::Val{D}) where {D}
    ys = ops[1].extra.ysoa
    ysoa = Array{eltype(ys),3}(undef, size(ys, 1), D, L)
    for l in 1:L
        ysoa[:, :, l] = ops[l].extra.ysoa
    end
    return DeviceSurface(to(ysoa))
end
function _device_far(ops::Vector, m::HarmonicMethod, L::Int, to, ::Val{D}) where {D}
    tb = ops[1].extra.tb
    return DeviceHarmonic(m.p, _nblocks(m), to([ops[l].extra.scale for l in 1:L]),
                          HarmonicTables(tb.p, to(tb.a), to(tb.b), to(tb.e), to(tb.cz), to(tb.cm), to(tb.cp)))
end
_device_far(ops::Vector, m, L::Int, to, ::Val) = throw(ArgumentError("the device evaluation is written for the kernel-independent and harmonic methods"))
_device_far(ops::Vector, m::SphericalHarmonics{<:Yukawa3D}, L::Int, to, ::Val) = throw(ArgumentError("the Yukawa harmonic method is not on the device yet"))

# the centers of the nodes from their cells and levels, on the device
@kernel function _centers!(centers, @Const(cell), @Const(level), origin, W0)
    i = @index(Global, Linear)
    @inbounds centers[i] = _center(origin, SVector{length(origin),Int}(cell[i]), W0 * typeof(W0)(0.5)^(Int(level[i]) - 1))
end

"""
    device_state(state, to=identity)

The per-evaluation data: the locals, sorted sources and densities of a built state,
moved with `to`.
"""
device_state(state::FMMState, to=identity) =
    (locals=to(state.locals), sources=Tuple(map(to, state.sources)), τ=map(x -> x === nothing ? nothing : to(x), state.τ))
# the inputs of a device evaluation as tuples of per-group arrays (one group: plain arrays accepted)
_group_tuple(x::Tuple) = x
_group_tuple(x) = (x,)

@inline _load_col(L, k::Int, i::Int, n::Int, ::Val{1}) = @inbounds L[k, i]
@inline _load_col(L, k::Int, i::Int, n::Int, ::Val{nc}) where {nc} = SVector(ntuple(a -> (@inbounds L[(a - 1) * n + k, i]), Val(nc)))

# the near field of one colleague over the groups (tuples of per-group arrays,
# unrolled by recursion; `nothing` densities: no sources); the self term is excluded
# in the target's own group `sg` only
@inline function _device_near(K, sources::Tuple, τ::Tuple, gplo, gphi, node, x, sg, skip, acc, output, g::Int)
    τg = first(τ)
    if τg !== nothing
        xs = first(sources)
        @inbounds for j in Int(gplo[g, node]):Int(gphi[g, node])
            (g == sg && j == skip) && continue
            acc += target_field(K, xs[j], x, τg[j], output)
        end
    end
    return _device_near(K, Base.tail(sources), Base.tail(τ), gplo, gphi, node, x, sg, skip, acc, output, g + 1)
end
@inline _device_near(K, ::Tuple{}, ::Tuple{}, gplo, gphi, node, x, sg, skip, acc, output, g::Int) = acc

@kernel function _evaluate_kernel!(out, @Const(oidx), @Const(targets), @Const(tnode), @Const(tskip), @Const(locals),
                                   @Const(ysoa), @Const(level), @Const(centers), @Const(colleagues), @Const(gplo),
                                   @Const(gphi), sources, τ, sg::Int32, K, output, ::Val{D}, ::Val{nc}) where {D,nc}
    q = @index(Global, Linear)
    @inbounds begin
        x = targets[q]
        i = Int(tnode[q])
        l = Int(level[i])
        xrel = x - centers[i]
        acc = zero(eltype(out))
        N = size(ysoa, 1)
        for k in 1:N
            y = SVector(ntuple(d -> ysoa[k, d, l], Val(D)))
            acc += target_field(K, y, xrel, _load_col(locals, k, i, N, Val(nc)), output)
        end
        skip = Int(tskip[q])
        for c in 1:3^D
            node = Int(colleagues[c, i])
            node == 0 && continue
            acc = _device_near(K, sources, τ, gplo, gphi, node, x, Int(sg), skip, acc, output, 1)
        end
        out[oidx[q]] = acc
    end
end

"""
    evaluate!(out, dev::DeviceEvaluator, ds, targets, nodes; output=Potential(), workgroupsize=256)

Device evaluation at `targets` (a device vector of `SVector`s) located in the leaves
`nodes` (a device `Int32` vector, from `locate` on the CPU), from the device state
`ds`; the backend is that of `out`.
"""
function evaluate!(out::AbstractVector, dev::DeviceEvaluator{D}, ds, targets::AbstractVector, nodes::AbstractVector;
                   output=Potential(), workgroupsize=256) where {D}
    nt = length(targets)
    length(out) == nt == length(nodes) || throw(DimensionMismatch("out, targets and nodes differ in length"))
    backend = get_backend(out)
    ident = similar(nodes)
    copyto!(ident, Int32.(1:nt))
    skip = similar(nodes)
    fill!(skip, Int32(0))
    _launch!(backend, workgroupsize, out, ident, targets, nodes, skip, 0, dev, ds, output)
    return out
end

"""
    evaluate_at_sources!(out, dev::DeviceEvaluator, ds, g=1; output=Potential(), workgroupsize=256)

`out[i]` = `output` at point `i` of group `g` (the group's original order) from all
other sources of all groups, on the device.
"""
function evaluate_at_sources!(out::AbstractVector, dev::DeviceEvaluator{D}, ds, g::Int=1; output=Potential(), workgroupsize=256) where {D}
    1 <= g <= ngroups(dev) || throw(ArgumentError("group $g of $(ngroups(dev))"))
    sources = _group_tuple(ds.sources)
    n = length(sources[g])
    length(out) == n || throw(DimensionMismatch("expected $n outputs for group $g"))
    backend = get_backend(out)
    skip = similar(dev.gperm[g])
    copyto!(skip, Int32.(1:n))
    _launch!(backend, workgroupsize, out, dev.gperm[g], sources[g], dev.gpoint_node[g], skip, g, dev, ds, output)
    return out
end

function _launch!(backend, workgroupsize, out, oidx, targets, nodes, skip, sg::Int, dev::DeviceEvaluator{D}, ds, output) where {D}
    dsg = (locals=ds.locals, sources=_group_tuple(ds.sources), τ=_group_tuple(ds.τ))
    _launch_far!(dev.far, backend, workgroupsize, out, oidx, targets, nodes, skip, sg, dev, dsg, output)
    synchronize(backend)
    return out
end
function _launch_far!(far::DeviceSurface, backend, workgroupsize, out, oidx, targets, nodes, skip, sg, dev::DeviceEvaluator{D}, ds, output) where {D}
    nc = ncomp(dev.kernel, Val(D))
    kern = _evaluate_kernel!(backend, workgroupsize)
    kern(out, oidx, targets, nodes, skip, ds.locals, far.ysoa, dev.level, dev.centers, dev.colleagues, dev.gplo, dev.gphi,
         ds.sources, ds.τ, Int32(sg), dev.kernel, output, Val(D), Val(nc); ndrange=length(targets))
    return out
end
function _launch_far!(far::DeviceHarmonic, backend, workgroupsize, out, oidx, targets, nodes, skip, sg, dev::DeviceEvaluator{3}, ds, output)
    kern = _evaluate_harmonic_kernel!(backend, workgroupsize)
    kern(out, oidx, targets, nodes, skip, ds.locals, far, dev.level, dev.centers, dev.colleagues, dev.gplo, dev.gphi,
         ds.sources, ds.τ, Int32(sg), dev.kernel, output, Val(far.nb); ndrange=length(targets))
    return out
end

# the harmonic methods: the local expansion through the folds of harmonics.jl
@inline _harmonic_far(L, x̂, s, p, tb, ::Potential, ::Val{1}) = _harmonic_potential(L, x̂, p, tb)
@inline _harmonic_far(L, x̂, s, p, tb, ::Gradient, ::Val{1}) = _harmonic_gradient(L, x̂, p, tb) * s
@inline _harmonic_far(L, x̂, s, p, tb, output, ::Val{4}) = _stokes_output(L, (p + 1)^2, x̂, s, p, tb, output)
@kernel function _evaluate_harmonic_kernel!(out, @Const(oidx), @Const(targets), @Const(tnode), @Const(tskip), @Const(locals),
                                            far, @Const(level), @Const(centers), @Const(colleagues), @Const(gplo),
                                            @Const(gphi), sources, τ, sg::Int32, K, output, ::Val{nb}) where {nb}
    q = @index(Global, Linear)
    @inbounds begin
        x = targets[q]
        i = Int(tnode[q])
        s = far.scales[Int(level[i])]
        acc = convert(eltype(out), _harmonic_far(view(locals, :, i), (x - centers[i]) * s, s, far.p, far.tb, output, Val(nb)))
        skip = Int(tskip[q])
        for c in 1:27
            node = Int(colleagues[c, i])
            node == 0 && continue
            acc = _device_near(K, sources, τ, gplo, gphi, node, x, Int(sg), skip, acc, output, 1)
        end
        out[oidx[q]] = acc
    end
end
