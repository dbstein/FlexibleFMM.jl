# Device evaluation with KernelAbstractions: one work-item per target computes the
# local expansion of its leaf (the density on the large surface, as `l2t`) plus the
# direct sum over the leaf's colleagues (as `evaluate_target`), on the flat arrays
# of the tree. The caller chooses the device through a function that moves arrays
# (`identity` for the CPU backend, `MtlArray`, `CuArray`, …), so backends and their
# array types are not dependencies of the package. The same body runs on every
# backend; on the CPU backend it reproduces the threaded evaluation to rounding.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

"""
    DeviceEvaluator(fmm, to=identity)

The tree data the device evaluation needs, moved once with `to`: the colleagues,
source ranges, levels and centers of the nodes, the large surface of every level
(`nlarge × D × nlevels`), the leaf and the output slot of every source, and the
kernel. `to` maps a CPU array to the device's array type.
"""
struct DeviceEvaluator{D,K,AI,VI,VP,AR}
    kernel::K
    colleagues::AI        # 3^D × nnodes, Int32
    plo::VI               # first and last source (sorted order) of each node, Int32
    phi::VI
    level::VI
    centers::VP           # SVector{D,T} per node
    ysoa::AR              # large surface per level, nlarge × D × nlevels
    point_node::VI        # leaf of each source (sorted order)
    perm::VI              # original index of each sorted source
end

function DeviceEvaluator(f::FMM{D}, to=identity) where {D}
    tree = f.tree
    n = nnodes(tree)
    L = nlevels(tree)
    ys = f.ops[1].extra.ysoa
    ysoa = Array{eltype(ys),3}(undef, size(ys, 1), D, L)
    for l in 1:L
        ysoa[:, :, l] = f.ops[l].extra.ysoa
    end
    return DeviceEvaluator{D,typeof(near_kernel(f.method)),typeof(to(tree.colleagues)),typeof(to(Int32[])),
                           typeof(to(SVector{D,eltype(tree.origin)}[])),typeof(to(ysoa))}(
        near_kernel(f.method), to(Int32.(tree.colleagues)),
        to(Int32[first(r) for r in tree.points]), to(Int32[last(r) for r in tree.points]),
        to(Int32.(tree.level)), to([center(tree, i) for i in 1:n]), to(ysoa),
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
    nc = ncomp(dev.kernel, Val(D))
    kern = _evaluate_kernel!(backend, workgroupsize)
    kern(out, oidx, targets, nodes, skip, ds.locals, dev.ysoa, dev.level, dev.centers, dev.colleagues, dev.plo, dev.phi,
         ds.sources, ds.τ, dev.kernel, output, Val(D), Val(nc); ndrange=length(targets))
    synchronize(backend)
    return out
end
