# Spectral (FFT) translation operators for square/cube surfaces.
#
# When the upward-equivalent points of the source box and the downward-check points
# of the target box lie on lattices with the same spacing h (p points per edge), the
# kernel values between them depend only on the lattice index difference J - I (and
# the box offset), so the check potentials are a discrete convolution of the density
# embedded in the p^D lattice with a kernel table on (2p-1)^D differences. Zero
# padding to (2p)^D makes it a circular convolution computed by FFTs:
#   û = κ̂_o ⊙ σ̂,   σ embedded at its lattice indices, κ_o(m) = K(o·w, h·m).
# One forward FFT per source box and one inverse FFT per target box serve all
# offsets; per pair only a Hadamard product remains, (2p)^D flops instead of N²
# (13× fewer at p = 6 in 3D, 20× at p = 8). Vector kernels use one table per
# component pair. The Stokes augmentations (pressure row, point-source column) are
# small dense corrections applied per pair.
#
# Expansions whose eltype is a dual number or complex are split into their real or
# complex parts (the transforms are linear), so the path is differentiable and
# would run on a GPU's real/complex FFTs.

"""
    SpectralM2L{R,D}

Translation operator of one far offset: FFTs of the kernel tables (`nc²` of them,
index `(a-1)·nc + b`, each of length `(2p)^D`), their real and imaginary parts as
separate vectors for the level-wise path, the lattice data (`p`, the linear index
of each surface point in the padded lattice), and the dense pressure row and
point-source column of the Stokes augmentations (empty otherwise).
"""
struct SpectralM2L{R,D}
    tables::Vector{Vector{Complex{R}}}
    tr::Vector{Vector{R}}
    ti::Vector{Vector{R}}
    nc::Int
    p::Int
    embed::Vector{Int}
    pin_row::Vector{R}       # length multipole_length, or empty
    src_col::Vector{R}       # length nc·N, or empty
end
function SpectralM2L{R,D}(tables::Vector{Vector{Complex{R}}}, nc::Int, p::Int, embed::Vector{Int},
                          pin_row::Vector{R}, src_col::Vector{R}) where {R,D}
    return SpectralM2L{R,D}(tables, [real.(t) for t in tables], [imag.(t) for t in tables], nc, p, embed, pin_row, src_col)
end

# Frequency blocks of the level-wise path: a block's data for a box is 2B·ncp reals,
# and a thread's working set over its pairs is that times the boxes it touches, so
# the block shrinks with the number of columns per box (nc components × np parts)
# to keep the working set in cache: the largest power of two up to 64/ncp (at least
# 8) that divides (2p)^D, so that the blocked layout is exactly the size of the spectra.
function _block_size(nfreq::Int, ncp::Int)
    B = 64
    while B > 8 && B * ncp > 64
        B >>= 1
    end
    while nfreq % B != 0
        B >>= 1
    end
    return B
end
Base.size(op::SpectralM2L) = (op.nc * length(op.embed) + (isempty(op.pin_row) ? 0 : 1),
                              op.nc * length(op.embed) + (isempty(op.src_col) ? 0 : 1))

# ---- parts: an expansion scalar as a tuple of complex numbers the FFT can take ----
_nparts(::Type{R}) where {R<:Real} = 1
_nparts(::Type{Complex{R}}) where {R<:Real} = 1
_nparts(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V<:Real,N} = N + 1
_nparts(::Type{Complex{ForwardDiff.Dual{T,V,N}}}) where {T,V<:Real,N} = N + 1
_basereal(::Type{R}) where {R<:Real} = R
_basereal(::Type{Complex{R}}) where {R} = _basereal(R)
_basereal(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = _basereal(V)
@inline _rpart(x::Real, k::Int) = k == 1 ? x : zero(x)
@inline _rpart(x::ForwardDiff.Dual, k::Int) = k == 1 ? ForwardDiff.value(x) : ForwardDiff.partials(x, k - 1)
@inline _cpart(x::Real, k::Int) = Complex(_rpart(x, k), zero(_rpart(x, k)))
@inline _cpart(x::ForwardDiff.Dual, k::Int) = Complex(_rpart(x, k), zero(_rpart(x, k)))
@inline _cpart(x::Complex, k::Int) = Complex(_rpart(real(x), k), _rpart(imag(x), k))
@inline _cassemble(::Type{R}, parts::Tuple) where {R<:Real} = R(real(parts[1]))
@inline _cassemble(::Type{Complex{R}}, parts::Tuple) where {R<:Real} = Complex{R}(parts[1])
@inline function _cassemble(::Type{ForwardDiff.Dual{T,V,N}}, parts::Tuple) where {T,V<:Real,N}
    return ForwardDiff.Dual{T,V,N}(V(real(parts[1])), ForwardDiff.Partials{N,V}(ntuple(i -> V(real(parts[i+1])), Val(N))))
end
@inline function _cassemble(::Type{Complex{ForwardDiff.Dual{T,V,N}}}, parts::Tuple) where {T,V<:Real,N}
    re = ForwardDiff.Dual{T,V,N}(V(real(parts[1])), ForwardDiff.Partials{N,V}(ntuple(i -> V(real(parts[i+1])), Val(N))))
    im = ForwardDiff.Dual{T,V,N}(V(imag(parts[1])), ForwardDiff.Partials{N,V}(ntuple(i -> V(imag(parts[i+1])), Val(N))))
    return Complex(re, im)
end

"""
    cube_boundary_points(::Val{D}, p, a) -> (points, embed)

The boundary points of the `p^D` lattice spanning `[-a, a]^D` (coordinate 1 fastest)
and the linear index of each in the zero-padded `(2p)^D` lattice.
"""
function cube_boundary_points(::Val{D}, p::Int, a::R) where {D,R}
    p >= 2 || throw(ArgumentError("p must be at least 2"))
    coords = [-a + 2a * R(i - 1) / (p - 1) for i in 1:p]
    pts = SVector{D,R}[]
    embed = Int[]
    lin = LinearIndices(ntuple(_ -> 2p, Val(D)))
    for I in CartesianIndices(ntuple(_ -> p, Val(D)))
        any(i -> i == 1 || i == p, Tuple(I)) || continue
        push!(pts, SVector{D,R}(ntuple(d -> coords[I[d]], Val(D))))
        push!(embed, lin[I])
    end
    return pts, embed
end

# FFT of the circulant kernel table for offset `off` (units of w), lattice spacing h:
# entry m ∈ [-(p-1), p-1]^D at wrapped index mod(m, 2p) + 1.
function _kernel_tables(K, ::Val{D}, p::Int, h::R, off::SVector{D,R}, w::R) where {D,R}
    nc = ncomp(K, Val(D))
    n = 2p
    dims = ntuple(_ -> n, Val(D))
    T = eltype(kernel_eltype(K, SVector{D,R}, SVector{D,R}))
    tabs = [zeros(Complex{R}, dims) for _ in 1:nc * nc]
    src = off * w
    for m in CartesianIndices(ntuple(_ -> -(p - 1):(p - 1), Val(D)))
        t = SVector{D,R}(ntuple(d -> h * m[d], Val(D)))
        k = K(src, t)
        idx = CartesianIndex(ntuple(d -> mod(m[d], n) + 1, Val(D)))
        for b in 1:nc, a in 1:nc
            tabs[(a - 1) * nc + b][idx] = nc == 1 ? k : k[a, b]
        end
    end
    return [vec(fft(t)) for t in tabs]
end

"""
    apply!(out, op::SpectralM2L, x, α=1)

Single-vector application (tests and references): embed, transform, multiply,
transform back, extract, and add the dense corrections. Allocates; the batched
level-wise path is `m2l_level!`.
"""
function apply!(out::AbstractVector{S}, op::SpectralM2L{R,D}, x::AbstractVector, α=true) where {R,D,S}
    nc = op.nc
    N = length(op.embed)
    n = 2 * op.p
    dims = ntuple(_ -> n, Val(D))
    nfreq = n^D
    np = _nparts(S)      # parts of the output scalar; a real input has zero partials beyond its value
    σ̂ = [zeros(Complex{R}, dims) for _ in 1:nc]
    û = [zeros(Complex{R}, dims) for _ in 1:nc]
    parts = [Vector{Complex{R}}(undef, nfreq) for _ in 1:np, _ in 1:nc]
    for k in 1:np
        for b in 1:nc
            fill!(σ̂[b], 0)
            for i in 1:N
                σ̂[b][op.embed[i]] = _cpart(x[(b - 1) * N + i], k)
            end
            fft!(σ̂[b])
        end
        for a in 1:nc
            fill!(û[a], 0)
            for b in 1:nc
                û[a] .+= reshape(op.tables[(a - 1) * nc + b], dims) .* σ̂[b]
            end
            ifft!(û[a])
            copyto!(parts[k, a], vec(û[a]))
        end
    end
    for a in 1:nc, i in 1:N
        out[(a - 1) * N + i] += α * _cassemble(S, ntuple(k -> parts[k, a][op.embed[i]], np))
    end
    if !isempty(op.pin_row)
        out[nc * N + 1] += α * sum(op.pin_row[j] * x[j] for j in eachindex(op.pin_row))
    end
    if !isempty(op.src_col)
        q = x[end]
        for j in 1:nc * N
            out[j] += α * op.src_col[j] * q
        end
    end
    return out
end

# Workspace of the level-wise path: spectra of all boxes of a level, one complex
# (2p)^D lattice per (component, part), and the in-place FFT plans.
# The Hadamard stage runs in a frequency-blocked layout, `(2B, nc·np·nboxes, nblocks)`
# reals with the real parts of a block's B frequencies followed by the imaginary
# parts: a block's data for every box is contiguous and small enough to stay in
# cache while all pairs of the level are processed, the inner loop is a plain real
# multiply-add over B entries, and blocks are independent, so threads own disjoint
# blocks with no synchronisation. `Sb` is `Û`'s memory and `Ub` is `Σ̂`'s: each
# transpose writes into the buffer whose contents are no longer needed.
struct SpectralWorkspace{R,D,P1,P2}
    Σ̂::Array{Complex{R},3}        # nfreq × (nc·np) × nboxes
    Û::Array{Complex{R},3}
    Sb::Array{R,3}                # blocked spectra of the sources (aliases Û)
    Ub::Array{R,3}                # blocked accumulated spectra of the targets (aliases Σ̂)
    B::Int
    fwd::P1
    bwd::P2
end
function SpectralWorkspace(::Type{R}, ::Val{D}, p::Int, ncp::Int, nboxes::Int) where {R,D}
    n = 2p
    nfreq = n^D
    B = _block_size(nfreq, ncp)
    Σ̂ = zeros(Complex{R}, nfreq, ncp, nboxes)
    Û = zeros(Complex{R}, nfreq, ncp, nboxes)
    bdims = (2B, ncp * nboxes, nfreq ÷ B)
    Sb = unsafe_wrap(Array{R,3}, Ptr{R}(pointer(Û)), bdims)
    Ub = unsafe_wrap(Array{R,3}, Ptr{R}(pointer(Σ̂)), bdims)
    dims = (ntuple(_ -> n, Val(D))..., ncp, nboxes)
    fwd = plan_fft!(reshape(Σ̂, dims), 1:D; num_threads=Threads.nthreads())
    bwd = plan_bfft!(reshape(Û, dims), 1:D; num_threads=Threads.nthreads())
    return SpectralWorkspace{R,D,typeof(fwd),typeof(bwd)}(Σ̂, Û, Sb, Ub, B, fwd, bwd)
end

# One block of one pair for all component pairs at once, on linear indices:
# Ub[a] += Σ_b T_ab ⊙ Sb[b] for a, b in 1:nc, where column a of the target starts at
# ut + (a-1)·stride (real parts of the B entries, then imaginary parts), column b of
# the source at us + (b-1)·stride, and the tables' parts are the separate vectors
# tr[(a-1)nc+b], ti[...] read at tt+q. Generated per nc so that the component loops
# unroll into one vectorised pass: each source column is loaded once and each
# target column updated once per block (`ivdep`: iteration q touches only entry q
# of every stream).
@generated function _hadamard_block!(Ub::Array{R,3}, ut::Int, Sb::Array{R,3}, us::Int,
                                     tr::Vector{Vector{R}}, ti::Vector{Vector{R}}, tt::Int,
                                     B::Int, stride::Int, ::Val{nc}) where {R,nc}
    n2 = nc * nc
    quote
        Base.Cartesian.@nexprs $n2 i -> (tr_i = tr[i]; ti_i = ti[i])
        @inbounds @simd ivdep for q in 1:B
            Base.Cartesian.@nexprs $nc b -> (sr_b = Sb[us + (b - 1) * stride + q]; si_b = Sb[us + (b - 1) * stride + B + q])
            Base.Cartesian.@nexprs $nc a -> begin
                ar = zero(R)
                ai = zero(R)
                Base.Cartesian.@nexprs $nc b -> begin
                    ar += tr_{(a - 1) * $nc + b}[tt + q] * sr_b - ti_{(a - 1) * $nc + b}[tt + q] * si_b
                    ai += tr_{(a - 1) * $nc + b}[tt + q] * si_b + ti_{(a - 1) * $nc + b}[tt + q] * sr_b
                end
                Ub[ut + (a - 1) * stride + q] += ar
                Ub[ut + (a - 1) * stride + B + q] += ai
            end
        end
        return nothing
    end
end

# spectra (nfreq × ncols) -> blocked real layout, and back
function _to_blocked!(Sb::Array{R,3}, Z::AbstractMatrix{Complex{R}}, B::Int) where {R}
    nb = size(Sb, 3)
    ncols = size(Z, 2)
    Threads.@threads :dynamic for blk in 1:nb
        q0 = (blk - 1) * B
        @inbounds for c in 1:ncols, q in 1:B
            z = Z[q0 + q, c]
            Sb[q, c, blk] = real(z)
            Sb[B + q, c, blk] = imag(z)
        end
    end
    return Sb
end
function _from_blocked!(Z::AbstractMatrix{Complex{R}}, Ub::Array{R,3}, B::Int) where {R}
    nb = size(Ub, 3)
    ncols = size(Z, 2)
    Threads.@threads :dynamic for blk in 1:nb
        q0 = (blk - 1) * B
        @inbounds for c in 1:ncols, q in 1:B
            Z[q0 + q, c] = Complex(Ub[q, c, blk], Ub[B + q, c, blk])
        end
    end
    return Z
end

"""
    spectral_m2l_level!(pL, ops, Mp, il, off, ws, α, ::Val{D}, N, nc)

The level-wise spectral translation: embed and transform every box's multipole,
accumulate the Hadamard products over the pairs of each far offset into the
targets' spectra, transform back, extract, and add the dense Stokes corrections.
"""
function spectral_m2l_level!(pL::AbstractMatrix, ops::LevelOperators, Mp::AbstractMatrix{S}, il::InteractionList,
                             off::Int, ws::SpectralWorkspace{R,D}, α, N::Int, nc::Int) where {S,R,D}
    return _spectral_m2l_level!(pL, ops, Mp, il, off, ws, α, N, Val(nc))
end
function _spectral_m2l_level!(pL::AbstractMatrix, ops::LevelOperators, Mp::AbstractMatrix{S}, il::InteractionList,
                              off::Int, ws::SpectralWorkspace{R,D}, α, N::Int, NC::Val{nc}) where {S,R,D,nc}
    np = _nparts(S)
    embed = ops.extra.embed
    nl = size(pL, 2)
    nfreq = size(ws.Σ̂, 1)
    Σ̂, Û = ws.Σ̂, ws.Û
    # 1. embed the multipoles, one lattice per (component, part)
    fill!(Σ̂, zero(Complex{R}))
    Threads.@threads :dynamic for j in 1:nl
        @inbounds for b in 1:nc, k in 1:np
            col = (b - 1) * np + k
            for i in 1:N
                Σ̂[embed[i], col, j] = _cpart(Mp[(b - 1) * N + i, off + j], k)
            end
        end
    end
    dims = (ntuple(_ -> 2 * ops.extra.p, Val(D))..., nc * np, nl)
    ws.fwd * reshape(Σ̂, dims)
    # 2. Hadamard products in the blocked layout: every pair of the level per block,
    #    threads over blocks (Sb aliases Û, Ub aliases Σ̂; see SpectralWorkspace)
    B = ws.B
    ncp = nc * np
    Sb, Ub = ws.Sb, ws.Ub
    _to_blocked!(Sb, reshape(Σ̂, nfreq, ncp * nl), B)
    ncols = ncp * nl
    Threads.@threads :dynamic for blk in 1:size(Ub, 3)
        @inbounds begin
            cbase = (blk - 1) * ncols * 2B          # linear offset of this block's columns
            for i in cbase + 1:cbase + ncols * 2B
                Ub[i] = zero(R)
            end
            tbase = (blk - 1) * B
            # targets in Morton order, each with all its pairs: the target's block stays in
            # cache across its pairs and neighbouring targets share sources
            for q in eachindex(il.torder)
                t = (il.torder[q] - off - 1) * ncp
                for kp in il.tptr[q]:il.tptr[q+1]-1
                    s = (il.tsrc[kp] - off - 1) * ncp
                    op = ops.M2L[il.tfar[kp]]
                    for k in 1:np
                        _hadamard_block!(Ub, cbase + (t + k - 1) * 2B, Sb, cbase + (s + k - 1) * 2B,
                                         op.tr, op.ti, tbase, B, np * 2B, NC)
                    end
                end
            end
        end
    end
    _from_blocked!(reshape(Û, nfreq, ncp * nl), Ub, B)
    # dense corrections per pair: the pinned pressure and the point-source column
    # (targets are unique within an offset)
    for f in eachindex(ops.M2L)
        pr = pair_range(il, f)
        op = ops.M2L[f]
        (isempty(pr) || (isempty(op.pin_row) && isempty(op.src_col))) && continue
        Threads.@threads :dynamic for idx in pr
            @inbounds begin
                t = il.tgt[idx] - off
                if !isempty(op.pin_row)
                    acc = zero(S)
                    for j in eachindex(op.pin_row)
                        acc += op.pin_row[j] * Mp[j, il.src[idx]]
                    end
                    pL[nc * N + 1, t] += α * acc
                end
                if !isempty(op.src_col)
                    q = Mp[end, il.src[idx]]
                    for j in 1:nc * N
                        pL[j, t] += α * op.src_col[j] * q
                    end
                end
            end
        end
    end
    # 3. transform back and extract (unnormalised backward transform: divide by nfreq)
    ws.bwd * reshape(Û, dims)
    β = α / nfreq
    Threads.@threads :dynamic for j in 1:nl
        @inbounds for a in 1:nc, i in 1:N
            pL[(a - 1) * N + i, j] += β * _cassemble(S, ntuple(k -> Û[embed[i], (a - 1) * np + k, j], np))
        end
    end
    return pL
end
