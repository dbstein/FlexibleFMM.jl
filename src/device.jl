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
with the `FastMath` policy if `fast` (the device's reciprocal square root; Metal,
Float32). `to` maps a CPU array to the device's array type.
"""
struct DeviceEvaluator{D,K,AI,VI,VP,F}
    kernel::K
    colleagues::AI        # 3^D × nnodes, Int32
    plo::VI               # first and last source (sorted order) of each node, Int32
    phi::VI
    level::VI
    centers::VP           # SVector{D,T} per node
    far::F                # DeviceSurface or DeviceHarmonic
    point_node::VI        # leaf of each source (sorted order)
    perm::VI              # original index of each sorted source
end

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

function _device_far(f::FMM{D}, ::AbstractKernelIndependent, to) where {D}
    L = nlevels(f.tree)
    ys = f.ops[1].extra.ysoa
    ysoa = Array{eltype(ys),3}(undef, size(ys, 1), D, L)
    for l in 1:L
        ysoa[:, :, l] = f.ops[l].extra.ysoa
    end
    return DeviceSurface(to(ysoa))
end
function _device_far(f::FMM, m::HarmonicMethod, to)
    L = nlevels(f.tree)
    tb = f.ops[1].extra.tb
    return DeviceHarmonic(m.p, _nblocks(m), to([f.ops[l].extra.scale for l in 1:L]),
                          HarmonicTables(tb.p, to(tb.a), to(tb.b), to(tb.e), to(tb.cz), to(tb.cm), to(tb.cp)))
end
_device_far(f::FMM, m, to) = throw(ArgumentError("the device evaluation is written for the kernel-independent and harmonic methods"))
_device_far(f::FMM, m::SphericalHarmonics{<:Yukawa3D}, to) = throw(ArgumentError("the Yukawa harmonic method is not on the device yet"))

function DeviceEvaluator(f::FMM{D}, to=identity; fast::Bool=false) where {D}
    tree = f.tree
    n = nnodes(tree)
    far = _device_far(f, f.method, to)
    K = fast ? fastmath(near_kernel(f.method)) : near_kernel(f.method)
    return DeviceEvaluator{D,typeof(K),typeof(to(tree.colleagues)),typeof(to(Int32[])),
                           typeof(to(SVector{D,eltype(tree.origin)}[])),typeof(far)}(
        K, to(Int32.(tree.colleagues)),
        to(Int32[first(r) for r in tree.points]), to(Int32[last(r) for r in tree.points]),
        to(Int32.(tree.level)), to([center(tree, i) for i in 1:n]), far,
        to(Int32.(tree.point_node)), to(Int32.(tree.perm)))
end

"""
    device_state(state, to=identity)

The per-evaluation data: the locals, sorted sources and densities of a built state,
moved with `to`.
"""
device_state(state::FMMState, to=identity) = (locals=to(state.locals), sources=to(state.sources), τ=to(state.τ))

@inline _load_col(L, k::Int, i::Int, n::Int, ::Val{1}) = @inbounds L[k, i]
@inline _load_col(L, k::Int, i::Int, n::Int, ::Val{nc}) where {nc} = SVector(ntuple(a -> (@inbounds L[(a - 1) * n + k, i]), Val(nc)))

@kernel function _evaluate_kernel!(out, @Const(oidx), @Const(targets), @Const(tnode), @Const(tskip), @Const(locals),
                                   @Const(ysoa), @Const(level), @Const(centers), @Const(colleagues), @Const(plo),
                                   @Const(phi), @Const(sources), @Const(τ), K, output, ::Val{D}, ::Val{nc}) where {D,nc}
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
            for j in Int(plo[node]):Int(phi[node])
                j == skip && continue
                acc += target_field(K, sources[j], x, τ[j], output)
            end
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
    _launch!(backend, workgroupsize, out, ident, targets, nodes, skip, dev, ds, output)
    return out
end

"""
    evaluate_at_sources!(out, dev::DeviceEvaluator, ds; output=Potential(), workgroupsize=256)

Device evaluation at the sources themselves (the `i = j` term skipped), `out` in
the original order.
"""
function evaluate_at_sources!(out::AbstractVector, dev::DeviceEvaluator{D}, ds; output=Potential(), workgroupsize=256) where {D}
    n = length(ds.sources)
    length(out) == n || throw(DimensionMismatch("expected $n outputs"))
    backend = get_backend(out)
    skip = similar(dev.perm)
    copyto!(skip, Int32.(1:n))
    _launch!(backend, workgroupsize, out, dev.perm, ds.sources, dev.point_node, skip, dev, ds, output)
    return out
end

function _launch!(backend, workgroupsize, out, oidx, targets, nodes, skip, dev::DeviceEvaluator{D}, ds, output) where {D}
    _launch_far!(dev.far, backend, workgroupsize, out, oidx, targets, nodes, skip, dev, ds, output)
    synchronize(backend)
    return out
end
function _launch_far!(far::DeviceSurface, backend, workgroupsize, out, oidx, targets, nodes, skip, dev::DeviceEvaluator{D}, ds, output) where {D}
    nc = ncomp(dev.kernel, Val(D))
    kern = _evaluate_kernel!(backend, workgroupsize)
    kern(out, oidx, targets, nodes, skip, ds.locals, far.ysoa, dev.level, dev.centers, dev.colleagues, dev.plo, dev.phi,
         ds.sources, ds.τ, dev.kernel, output, Val(D), Val(nc); ndrange=length(targets))
    return out
end
function _launch_far!(far::DeviceHarmonic, backend, workgroupsize, out, oidx, targets, nodes, skip, dev::DeviceEvaluator{3}, ds, output)
    kern = _evaluate_harmonic_kernel!(backend, workgroupsize)
    kern(out, oidx, targets, nodes, skip, ds.locals, far, dev.level, dev.centers, dev.colleagues, dev.plo, dev.phi,
         ds.sources, ds.τ, dev.kernel, output, Val(far.nb); ndrange=length(targets))
    return out
end

# the harmonic methods: the local expansion through the folds of harmonics.jl
@inline _harmonic_far(L, x̂, s, p, tb, ::Potential, ::Val{1}) = _harmonic_potential(L, x̂, p, tb)
@inline _harmonic_far(L, x̂, s, p, tb, ::Gradient, ::Val{1}) = _harmonic_gradient(L, x̂, p, tb) * s
@inline _harmonic_far(L, x̂, s, p, tb, output, ::Val{4}) = _stokes_output(L, (p + 1)^2, x̂, s, p, tb, output)
@kernel function _evaluate_harmonic_kernel!(out, @Const(oidx), @Const(targets), @Const(tnode), @Const(tskip), @Const(locals),
                                            far, @Const(level), @Const(centers), @Const(colleagues), @Const(plo),
                                            @Const(phi), @Const(sources), @Const(τ), K, output, ::Val{nb}) where {nb}
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
            for j in Int(plo[node]):Int(phi[node])
                j == skip && continue
                acc += target_field(K, sources[j], x, τ[j], output)
            end
        end
        out[oidx[q]] = acc
    end
end
