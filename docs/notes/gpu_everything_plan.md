# Everything on the device: the tree, the lists and the plan from positions on the GPU

Scope: a JellySim step on a node with one GPU and a few CPU cores, N = 10⁶–10⁷ points
whose positions change every step, positions and densities already on the device,
several `device_matvec!` per step. Today the tree, the interaction lists, the
`DevicePlan` and the permutation of the inputs are host work; this note inventories
that work, designs a device-resident pipeline from positions to a plan the existing
passes can run, reports what was measured locally on Metal to calibrate it, and
proposes stages. Companion to `docs/notes/gpu_plan.md` and `gpu_plan_delta.md` (the
matvec on the device, implemented) and to the concurrent CPU tree work (threading, a
Morton-key construction, reuse across steps), from which this note assumes only one
thing: **the key-based construction (sort the points once by a Morton key computed
with the builder's own `child_slot` rule, then every level's point ranges are prefix
runs of that order) becomes the shared building block of both builders.** Section 3
says what that implies here.

All timings below are measured on the M3 Max (Metal, Float32) unless marked "est."
or "H100" (the user's numbers from the cluster node: 8 Icelake threads, Stokes cube
method `p = 4`, `ppl = 800`, Float32, 10⁷ points on a sphere: tree 2.3 s, plan
0.23 s, matvec 0.28 s; at 10⁶: 0.15, 0.03, 0.04 s). The local configuration is the
same (sphere, Stokes cube `p = 4`, `ppl = 800`, Float32): 10⁷ points give 407 761
nodes on 8 levels (255 744 ghosts, 31 968 xlist leaves), 5.68 M pairs; 10⁶ give
21 521 nodes on 6 levels, 346 k pairs. Scripts and raw outputs are in the session
scratchpad (`gpu_explore/host_phases.jl`, `metal_primitives.jl`, `sort_checks.jl`,
`native_scan.jl`, `device_lists.jl`, `plan_calib.jl`, `gemm_views.jl`, with `.txt`
outputs); nothing under `src/`, `test/`, `bench/` was changed.

## 1. Inventory: what runs on the host today

### 1.1 Per tree (`Tree`, `FMM`, `DevicePlan`)

`Tree(points; ppl, bbox)` (`src/tree/builder.jl`, single-threaded), 10⁷ sphere
points, measured phase by phase (1.07 s total here; 2.3 s on the H100 node):

| phase | function | work | 10⁷ | 10⁶ |
|---|---|---|---|---|
| containment check, `collect`, `perm = 1:N` | `Tree` | 2 passes over the points | 0.026 s | 0.006 |
| refinement, 7 levels | `build_levels` → `split_nodes!` | per split node a counting sort of its points by `child_slot` (two passes over the node's points through `spts/sperm`); each level is two passes over all 10⁷ points | 0.225 s | 0.015 |
| colleagues, all levels | `tag_colleagues!` | `sortperm` of the level's cells as tuples, 27 `searchsortedfirst` per node | 0.089 s | 0.005 |
| 2:1 restriction | `level_restrict!` | rounds of `compute_depths!`, bad-leaf scan, `split_nodes!`, `tag_colleagues!` of the next level | 0.125 s | 0.003 |
| ghosts | `add_ghosts!` | xlist marking on all levels, then per level `split_nodes!(ghost=true)` of every xlist leaf (re-sorting their points: on a surface the xlist leaves hold most of the points) and `tag_colleagues!` of the next level (a `sortperm` over up to 253 k nodes) | **0.484 s** | 0.019 |
| flatten | `flatten` | global indices, `point_node`, `group_parent` checks | 0.028 s | 0.012 |
| interaction lists | `build_interaction_lists` (called by `flatten`) | per level: enumerate 27 × 8 candidates per non-ghost node, counting sort by far offset (CSR), `sortperm` of the targets' Morton keys for the by-target view (`torder/tptr/tsrc/tfar`) | 0.153 s | 0.009 |
| `FMM` | `fit_box!`, `get_operators!` (cached), `s2m_nodes`, `leaf_nodes` | lists of childless-with-sources and real-leaf nodes per level | 0.003 s | 0.000 |

The refinement and the ghost split are the only phases that move points; the rest
works on nodes (≈ 150 B per node). With a Morton sort the point-moving work
disappears entirely (section 3), which is the point of the shared building block.

`DevicePlan(f, to; …)` (`src/device_passes.jl`), 10⁷, measured with `verbose=true`
(1.15 s here; 0.23 s on the H100 node):

| stage | what | 10⁷ | 10⁶ |
|---|---|---|---|
| evaluator | `DeviceEvaluator`: `Int32` copies of `colleagues` (27 × n), `plo/phi`, `level`, `point_node`, `perm`; `center(tree, i)` for every node; the far data (`ysoa` per level, or the harmonic tables) | 0.079 s | 0.009 |
| operators | the class operators (`class_operators`/`class_factors`, built at one width and scaled for homogeneous kernels; `_stokes_dense_operators` for the Stokes harmonics), the signed permutations `idx/inv/sgn/sgn_t` (Ns × 316), uploads | 0.011 s warm (0.5 s cold at `p = 4`; the cache keeps them per method and width) | 0.012 |
| pair lists | `_class_chunks_host` per level (threaded over levels): `class_lists` (one pass over the by-target view), then chunks of ≤ 4096 pairs, each padded to the power of two above its size with **four host arrays per chunk** (`src`, `f`, `tgt`, `ptr`) uploaded one by one (≈ 1 400 chunks at 10⁷, 5 600 uploads) | **0.973 s** | 0.092 |
| buffers | `mk` (zero-filled device arrays) per level, sized by `nl`, `length(s2m_nodes[l])`, `length(leaf_nodes[l])`; `Gc/Yc/Zc` per chunk length; `s2m_nodes`, `leaf_nodes`, `leaf_cols`, `group_parent` uploaded as `Int32` | 0.084 s | 0.017 |

So at 10⁷ the plan is as expensive as the tree on this machine, and its pair-list
stage is 85 % of it. (On the H100 node the plan is 0.23 s; the pair-list stage there
is presumably 0.15–0.2 s. The difference to the 0.97 s here is the single-threaded
`class_lists` pass and 5 600 small `MtlArray` uploads; it is not worth optimising on
the host since section 4 removes it.)

### 1.2 Per matvec

`device_matvec!(out, plan, (sources, τ))` expects `sources` and `τ` **in tree
order**: the caller does `pts[perm]`, `τ[perm]` on the host and uploads (bench:
`ds = (sources=to(pts[perm]), τ=to(τ[perm]))`). Measured at 10⁷: `pts[perm]`
0.060 s, `τ[perm]` 0.093 s (24-byte stokeslet + doublet tuples), uploads 0.034 s;
on the device the same gathers take 2.4 and 3.1 ms. The output needs no host
un-permutation: `evaluate_at_sources!` writes `out[oidx[q]]` with `oidx = dev.perm`,
so `out` is in the original order on the device already. In a Krylov solve the
positions are fixed within a step and only `τ` changes, so today every matvec pays
`τ[perm]` + upload (0.13 s at 10⁷ against a 0.28 s matvec on the H100).

Inside `device_matvec!` the host only drives launches: per level the S2M kernel, the
pseudo-inverse GEMMs, the M2M/L2L GEMMs and scatters; per chunk three launches
(`_class_gather!`, GEMM, `_class_scatter!`): ≈ 1 400 chunks × 3 ≈ 4 200 launches at
10⁷ (10 µs each on Metal: ≈ 40 ms of host time inside the 1.1 s Metal matvec; CUDA
launches are a few µs). `synchronize` is called after the upward pass, after the
downward pass and after the evaluation.

Local matvec calibration (Metal, Float32, sphere, `p = 4`, `ppl = 800`): 10⁶: 0.131 s
(expansions 0.027, self evaluation 0.105); 10⁷: 1.10 s (expansions 0.33, self 0.77).
The near field dominates at `ppl = 800` on this GPU; the H100 has ≈ 10× the Float32
throughput, consistent with its 0.28 s.

### 1.3 What the device code assumes from the host-built structures

- **Levels as contiguous global ranges**: `lo[l]`, `nl[l]` are host integers used as
  launch sizes; `copyto!(vec(B), 1, vec(Mp), (lo[l+1]-1)·Nm + 1, …)` and the M2M
  reshape assume the children level is a contiguous block of whole sibling groups in
  `child_first` order; `group_parent[l][g]` is the parent of group `g`.
- **Node lists**: `s2m_nodes[l]` (childless nodes with sources, global ids),
  `leaf_nodes[l]`/`leaf_cols[l]` (real leaves; the `P_down` transform and the
  evaluation read only those columns); `pM[l]`, `Mup[l]`, `tup[l]`, `pLq[l]`,
  `tdn[l]`, `Ldn[l]` are allocated to their lengths.
- **Colleague tables** `27 × n` `Int32` with 0 for absent, used by the evaluation
  kernel (near field) only; the lists are consumed through the chunks.
- **`ClassChunk`**: per chunk `src[q]`, `f[q]` for `q ≤ len` (padding columns point
  at node 1 and offset 1 and are never scattered), `tgt` (level-local target
  indices) and `ptr` (CSR over the chunk's columns per target). The gather kernel
  reads `src/f`, the scatter kernel loops `ptr[t]:ptr[t+1]-1`. Nothing in the
  kernels depends on the order of the targets or of the pairs within a target, only
  on each target's pairs of a class being consecutive in the chunk.
- **Invariants relied on** (`docs/tree.md`): a target appears at most once per
  offset (the per-offset CPU scatter; the class scatter is target-owned anyway),
  ghosts are never targets, only nodes with sources are listed (`has_source`),
  the points of every node are contiguous in the sorted order (`plo/phi`), and
  `point_node[j]` is the real leaf of sorted point `j`.

## 2. Primitives available on the device (measured)

The pipeline needs: a key kernel, a sort (or sortperm) of 10⁷ 64-bit keys, scans
(prefix sums) of 10⁷ and of 4 × 10⁵ `Int32`, run-boundary flags, binary searches,
gathers, and small readbacks. All exist without a new dependency:

| primitive | Metal.jl 1.11.1 (`MtlArray`) | CUDA.jl 5.x (`CuArray`) | Array (tests, `to = identity`) |
|---|---|---|---|
| `sort!`, `sortperm!` (Base API) | MPSGraph sort/argsort, key types incl. `UInt64`/`Int64`, index `Int32`/`Int64` (`src/sorting.jl`) | bitonic by default, quicksort option (`src/sorting.jl`) | Base |
| `accumulate!(+, dst, src)` (scan) | MPSGraph scan (`src/accumulate.jl`) | `scan!` (`src/accumulate.jl`, `Base._accumulate!` methods) | Base |
| kernels with `@index(Global)`, binary searches, gathers | KernelAbstractions | KernelAbstractions | KA CPU backend |

Measured on 10⁷ elements (M3 Max, best of 3–5):

| operation | time |
|---|---|
| Morton keys, the builder's recursive `child_slot` rule on exact centers, 21 levels, `UInt64` | 6.1 ms |
| Morton keys by `floor((x − o)·2²¹)` | 1.0 ms, but differs from the recursive rule at 31 % of the points (points on level-21 box edges: `x > c` sends an edge point down, `floor` up) — **use the recursive rule**, it is what `locate` and the host builder do and it costs nothing |
| `sortperm!` `UInt64 → Int32` (MPSGraph) | 24.5 ms (`UInt32` keys: 8.4 ms; `sort!` `UInt64`: 9.9 ms) |
| host `Base.sortperm` of the same keys, one thread | 0.67 s |
| gather `SVector{3,Float32}[perm]` / 24-byte tuple `[perm]` | 2.4 / 3.1 ms |
| run-boundary flags at one level (`key >> shift` differs from the predecessor's) | 1.1 ms |
| `accumulate!` of 10⁷ `Int32` (MPSGraph) | 0.8 ms (`Int64`: 2.5 ms) |
| flags + scan for 7 levels | 17 ms |
| colleagues of all 8 levels of the real 10⁷ tree: node keys, per-level `sortperm!`, 27 binary searches per node | 18 ms (10 ms at 10⁶), **equal to `tree.colleagues`** |
| interaction lists of all levels: count per target, scan, one readback per level, fill | 11 ms (8 ms at 10⁶), **the (target, source, offset) triples equal to `tree.ilists`** |
| class lists (16 classes): per-(target, class) counts, class-major scan, fill | 13 ms (6 ms at 10⁶), per-class counts equal to `class_lists` |
| upload / download 10⁷ `SVector{3,Float32}` | 8.6 / 11.8 ms |
| one kernel launch (no sync) / `synchronize` / launch + 1-element readback | 10 µs / 150 µs / 305 µs |
| GEMM `889 × 889` by `889 × 4096` on a whole `MtlMatrix` / on `view(B, :, 1:4096)` of a larger one / on `reshape(view(vec(B), 1:n))` | 1.6 / 1.8 / 1.2 ms (4.1 TFlop/s): contiguous views of an `MtlArray` are `MtlArray`s in Metal.jl 1.11 (`GPUArrays.derive`), so **capacity buffers with exact-size views run the native GEMM at full speed**; the "no views on Metal" rule of `gpu_plan.md` §2.3 applied to the MPS path of older versions |

Stability and determinism of the sorts on ties (10⁷ keys with 2²⁰ distinct values):
MPSGraph `sortperm!` stable, AcceleratedKernels' `sortperm!` and
`merge_sort_by_key!` stable (merge sort is stable by construction); CUDA's bitonic
`sortperm!` is deterministic (a fixed network) but not stable. The pipeline below
makes the keys unique enough that this does not matter (section 3.1).

**AcceleratedKernels.jl** (0.4.3 in the depot, not in any environment): one
implementation of sort / sort-by-key / sortperm / accumulate / reduce /
`searchsortedfirst` for every KA backend. Measured: `sort!` `UInt64` 26 ms,
`sortperm!` 35 ms, `merge_sort_by_key!` 32 ms, inclusive scan 2.5 ms — all slower
than the native Metal routines; its **exclusive scan returned wrong values on Metal
at 10⁷** (the inclusive one is right; its documentation itself excludes the
decoupled-lookback algorithm on Metal for lack of device-scope memory ordering), and
one `sort!` of the Morton keys came back unsorted once in the first script (not
reproduced in six later runs). It pins `KernelAbstractions` to 0.9 (`compat
"0.9.34"`; the package allows 0.9 and 0.10) and adds 1.3 s of precompilation.
**Recommendation: do not add it.** The Base API (`sortperm!`, `accumulate!`) on the
device array type is dispatched by the array the user passes, exactly as `mul!` is
today, so backends stay out of the dependencies; the CPU backend of the tests gets
Base's implementations for free; and the few kernels the pipeline needs (keys,
flags, searches, counts, fills, gathers) are 10–30 lines each with
`@index(Global)`. The one thing lost is a guaranteed-stable sort-by-key; section 3.1
does without it.

**Metal-specific limits versus CUDA** (for the record): Float32 only (no Float64
anywhere, so a Float64 tree type never reaches the device: the key kernel is
generic in `T`); no complex GEMM (`DevicePlan(…; eltype=ComplexF64)` is CUDA only);
32-bit atomics only (the pipeline uses none); no device-scope memory ordering
(single-pass scans with lookback are out; MPSGraph's scan is what to call);
64-bit integer arithmetic is slow but the key kernel (63-bit keys) still takes 6 ms;
launch/sync/readback latencies of 10/150/305 µs against a few µs / ≈ 10–20 µs on
CUDA, which decides how many host round trips the level loop may have (section
3.4); and a stale manifest entry — `bench/gpu/Manifest.toml` records an extension
`FlexibleFMMMetalExt` with `weakdeps = ["Metal"]` for the package, which no longer
has an `ext/` directory, so `using Metal` in that environment prints a harmless
extension-loading error (`Pkg.resolve()` in `bench/gpu` would clear it; not done
here).

## 3. The device-resident pipeline

From `xs` on the device (any order; the user's arrays) to a `DevicePlan` the
existing passes run unchanged, as the sequence of steps the host builder performs
today, each as kernels over nodes or points plus scans. Node arrays are stored per
level with a global numbering by level (as `flatten` produces), so the passes' `lo[l]`
and the contiguous children blocks hold as before.

### 3.1 Keys and the sort

- Key kernel, one work-item per point: 21 steps of `child_slot(x, center)` with
  `center = origin + (cell + ½)·w`, `w` halved each step (exact), interleaved into a
  `UInt64` with coordinate 1 in the lowest bit of each 3-bit group (the layout of
  `morton_key` in `src/util.jl`, so the `k`-th 3-bit group is the child slot at level
  `k+1`). Identical to the host rule, hence to `locate`. 6 ms at 10⁷.
- `sortperm!(perm, keys)` (25 ms), then gather `keys[perm]` and `xs[perm]` (6 ms).
  `perm[j]` is the original index of sorted point `j`, exactly `tree.perm`'s meaning.
- Within a leaf the host keeps the original order (its counting sorts are stable
  and start from the identity); a full-key sort orders a leaf's points by their
  deeper Morton bits. Both are valid; the device order is canonical and
  deterministic for any deterministic sort, and backend-independent except among
  points coincident at the resolution of level 21 (`W0/2²⁰`; 21 of 10⁷ sphere
  points here). Decided: do not try to reproduce the host's within-leaf order on
  the device; the structural arrays (cells, flags, colleagues, ranges, lists) are
  made identical to the host's (3.2–3.3) and the tests compare those plus results to
  tolerance. If the CPU work adopts the same full-key sort with ties by original
  index, the two `perm`s become identical and a stable sort-by-key (`(key, index)`
  pairs, or a stable backend sort) is the only requirement.

### 3.2 Refinement, colleagues, restriction, ghosts

Parallel-trivial (one kernel over nodes or points, no sort):

- **Refinement, level `l → l+1`**: `split[i] = count[i] > ppl` for the nodes of
  level `l`; scan over `split` → the group index of each split node; the children's
  cells `2·cell + offs[k]`, parents, and point ranges by two binary searches per child
  in the sorted keys (the range of keys with the child's `3·l`-bit prefix; 24 steps
  on 10⁷ keys). Stop when no node splits (one count readback per level). `has_source
  = hi ≥ lo`. The host builder's `max_levels` guard transfers (the key has 21 levels;
  deeper than that the keys of coincident points tie and no split can separate
  them, so the loop stops at level 22 with the same error).
- **Colleagues of a level**: node keys from cells, `sortperm!` of the level's nodes,
  the sorted keys, 27 binary searches per node for `cell + offset` (out-of-lattice
  offsets give 0). Measured 18 ms for all 8 levels at 10⁷ with exact agreement.
  Cells are unique per level by construction (prefix runs), which the host asserts;
  keep the assertion as a duplicate check on the sorted node keys (one flag kernel
  + reduce).
- **Depths** (bottom-up, one kernel per level: `depth[i] = 1 + max depth of the 8
  children`), **bad-leaf flags** (per leaf, 27 colleague compares), **xlist flags**
  (per leaf: a colleague with children), **ghost children** (8 per xlist leaf,
  ranges by binary search in the keys at the next level's prefix — the xlist leaf's
  points are already sorted, so no point moves; this removes the host's 0.48 s),
  **`point_node`** (per leaf a loop over its range), **node lists** (`s2m_nodes`,
  `leaf_nodes`, `leaf_cols` by flag + scan + compaction per level; `group_parent` by
  a strided gather of `parent`).

Iterative / segmented:

- **2:1 restriction** is a fixed-point iteration as on the host: depths → bad-leaf
  flags → readback of the count → if nonzero, split them (append children to the
  next level, re-tag that level's colleagues, recompute depths), repeat. One round is
  ≈ 8 small kernels + a scan + a readback + a colleague pass of one level ≈ 5–10 ms
  (est. from the measured parts); rounds are few (`docs/tree.md`: leaves on the two
  deepest levels are never bad, no level is ever added).
- **Ghosts change node counts**: appending the ghost groups per level and re-tagging
  colleagues of levels 3..L is one pass per level (xlist count readback, fill,
  colleague pass), ≈ 8 × 3 ms (est.).
- Node arrays are appended to, so they are allocated with capacity (a bound: the
  final node count ≤ 8 × (nodes before ghosting), and the host builder's counts of
  the previous step are the natural estimate; grow and redo a level on overflow,
  which a readback detects).

The node order within a level follows the host's (refinement children in parent
order, restriction children appended per round, ghosts appended last), so **every
structural array can be compared with the host `Tree` bitwise**, which is the test
of stage 3 (section 6).

### 3.3 Lists, class lists, chunks

Measured on the real 10⁷ tree (its flat arrays uploaded) with exact agreement:

- **Interaction lists** (11 ms): per non-ghost node of level `l` one work-item
  enumerates the 27 colleagues of its parent × 8 children, tests `has_source` and
  the far offset (the loop of `src/tree/lists.jl`), and counts; scan → CSR pointer;
  one readback of the level's pair count; a fill kernel writes `(tgt, src, far)` in
  the same enumeration order. This is the by-target view directly (targets in node
  order rather than Morton order; the kernels do not care, and node order is Morton
  order for the refinement nodes). The by-offset CSR of `InteractionList` is not
  needed by the device passes.
- **Class lists** (13 ms): per target a 16-entry (or 34 for the harmonic classes;
  `class[f]` is an array argument) count of its pairs by class, a scan over the
  class-major flattening (`nl × nclass` counts), and a fill: the pairs of class `c`
  in target order, each target's pairs consecutive — exactly `ClassList.src/f`, with
  the target runs given by the nonzero counts (`tgt`, `ptr` by one more compaction).
- **Chunks**: a chunk is `(class, a, npk, len, r1, r2)` — the first pair `a` of the
  chunk in its class list, its pair count, the padded length (`max(64,
  nextpow(2, npk))`), and its run range from two binary searches in the class's
  `ptr`. One kernel over the ≈ 1 400 chunks of all levels and one readback of six
  `Int32` per chunk; the chunk boundaries themselves follow from the 16 per-class
  counts per level (one readback per level). The four per-chunk arrays disappear:
  `_class_gather!` reads `src[a + q − 1]`, `f[a + q − 1]` for `q ≤ npk` (node 1,
  offset 1 beyond, as the padding does today) and `_class_scatter!` reads
  `tgt[r]`, `max(ptr[r], a) − a + 1` for `r` in `r1:r2`, i.e. the `_run_range`
  arithmetic of `_class_chunks_host` moves into the kernels. `Gc/Yc/Zc` are needed
  for the seven lengths 64..4096 only, allocated once.

### 3.4 Host round trips versus fully device-side

KernelAbstractions needs `ndrange` on the host, so every pass whose size depends
on data (the next level's node count, a restriction round's split count, a level's
pair count, the chunk count) costs one small readback: ≈ 8 (levels) + 2–3 × 8
(restriction rounds) + 8 (ghosts) + 8 (lists) + 8 (chunks) ≈ 50–60 round trips ≈
18 ms on Metal (305 µs each) and ≈ 1 ms on CUDA. A fully device-side variant would
launch every kernel at capacity and mask, saving those 18 ms at the price of
capacity-sized launches everywhere and device-side counters; not worth it: the
readbacks are a few per cent of a device build and the level-loop control stays
readable Julia. Decided: the level loop stays on the host, with one readback per
level per pass, and all data on the device.

Estimated device build at 10⁷ (measured parts in bold): **keys 6** + **sort 25** +
**gathers 6** + refinement ≈ 15 + **colleagues 18** + restriction 10–25 + ghosts ≈ 30
+ `point_node`/node lists ≈ 10 + **lists 11** + **class lists 13** + chunks ≈ 2 +
round trips ≈ 18 ⇒ **≈ 0.15–0.2 s on the M3 Max** against 1.07 (tree) + 1.15 (plan)
+ 0.13 (permutation) s of host time measured on the same machine. The unmeasured
parts are kernels of the kinds measured (flags, scans, binary searches, fills over
≤ 4 × 10⁵ nodes or 10⁷ points), so the estimate's uncertainty is the number of
restriction rounds and launch overhead, not a new kind of cost. On an H100 the
memory-bound kernels run ≈ 5–8× faster and the round trips are negligible:
**est. 0.04–0.08 s** against 2.3 + 0.23 + 0.15 s today.

### 3.5 A fixed-depth octree instead

For a volume-filling distribution a complete octree of depth `L` needs no
refinement decisions, no restriction (all leaves on one level), no ghosts,
colleagues and lists by lattice arithmetic (static, built once), and per step only
the sort and the leaf ranges (run boundaries at one prefix: 2 ms). It is the right
tree for uniform data and would be a second tree type behind the same flat arrays.
It is the wrong tree for JellySim: its distributions are surfaces and fibres (the
10⁷ sphere has 63 % ghosts and 8 levels where uniform data would have 6), and a
complete octree at the depth of the finest leaves has 8⁷ = 2 M boxes on the last
level, mostly empty; with the `has_source` filter the lists change with the points
anyway, and without it the pair count explodes. Decided against for the production
path; noted as a later option for uniform benchmarks.

## 4. A JellySim step

Per step, all on the device:

1. positions `x` (the solver's arrays, original order) → keys → `sortperm!` →
   sorted positions; the root box is the simulation domain (fixed; a `mapreduce`
   containment check, 2 ms; a point outside is an error as today).
2. the tree, lists, class lists, chunks and node lists of section 3 into
   capacity-sized device buffers; the plan's per-level buffers are views of
   capacity buffers at the step's sizes (`view(buf, :, 1:n)`: an `MtlArray`/a
   `StridedCuMatrix`, full-speed GEMMs on both; measured on Metal).
3. per matvec: `τ` in original order → one gather by `perm` (3 ms at 10⁷) into the
   plan's `τ` buffer; `device_build_expansions!`; `evaluate_at_sources!` writes
   `out[perm[j]]`, so `out` is in the solver's order with no copy. Dual positions
   (Newton–Krylov) are gathered the same way.

**Rebuild versus validate-and-reuse.** The FMM is exact for any assignment of
points to the leaves of a valid tree; `ppl` only sets the cost. Reusing last step's
tree therefore needs: (i) every point inside the box, (ii) every childless node that
now has sources was a source node when the lists were built (`has_source` true then;
a node that gained its first source is in no list and no `s2m_nodes`, and a node
that lost all sources is harmless), (iii) a cost bound on the leaf counts (e.g. the
largest leaf ≤ 2·ppl, else the near field drifts up). The check costs the key kernel,
a `sortperm!` by leaf (the key at the leaf's level, found by a walk from the root,
≈ `locate`), the run boundaries (new `plo/phi`), and three reductions: ≈ 40 ms on
Metal / ≈ 10 ms on the H100 (est.), against a full rebuild of ≈ 0.15 / 0.05 s. The
rebuild is already a fraction of one matvec (0.28 s on the H100) and the solve has
several, so **rebuild every step first**; reuse is an optimisation to add once the
build is measured on the H100, and its condition (ii) must be checked, never
assumed (a surface moving through an empty sibling box fails it). Building the
lists without the `has_source` filter would make reuse unconditional at ≈ 2× the
pairs on a surface (half the children of a split box are empty); not by default.

**Memory** (10⁷, Stokes cube `p = 4`, Float32; `Nm = pll = 169`, `pml = Nl = 888`):
points, keys, perm, `point_node`, densities (24–48 B), output ≈ 0.6 GB; nodes
≈ 160 B each ≈ 65 MB; lists 5.7 M × 3 `Int32` + class copies ≈ 115 MB; expansions
`multipoles` + `plocals` 0.55 GB and `locals` 1.45 GB (a leaves-only `locals`,
0.47 GB, is a separate change: the evaluation kernel indexes by node); per-level
work (`pM` 888 × 253 k at level 8, 0.9 GB, is the largest) ≈ 2 GB; the class
operators 1.8 MB, the folded M2M and L2L (169 × 1352) 0.9 MB each, the pseudo-inverse
factors ≈ 1 MB each. ≈ 4.5 GB, ≈ 6 GB with 25 %
capacity margins; the operators and pseudo-inverse factors are shared by all steps
and all plans of the same method and width (the `OperatorCache` as today; a step
that adds a level needs the operators of a new width, which for the homogeneous
Stokes and Laplace kernels is a scale factor and for Yukawa a build and upload).
At `p = 8` Stokes multiply the expansion and work rows by 5.3 (`Nm = 889`,
`Nl = 4056`) ⇒ ≈ 15 GB: fine on an 80 GB H100, and `p = 4` is the configuration in
use.

## 5. Staged plan

Stage 1 is the largest reduction of host time at 10⁷ with the least new machinery,
judged against the three candidates named in the brief:

- *Threading the host build*: the CPU work's subject; it reduces the 2.3 s tree
  but leaves the plan's 0.23 s and the per-matvec 0.15 s.
- *A GPU Morton sort plus a device version of the builder's level loop*: removes
  the tree (2.3 s) but needs the restriction and ghost passes on the device before
  it produces a valid tree; the most machinery.
- *Keeping the host tree but building the lists and chunks on the device, and
  moving the permutation to the device*: four kernels and two scans, all measured
  and validated here against the host's tables; it removes the whole plan's
  pair-list stage and the per-matvec host work, and it is the half of the device
  pipeline that the later stages need anyway.

### Stage 1: lists, class lists, chunks and the permutation on the device

- `src/device_passes.jl`: a `device_lists!(tree arrays)` building the by-target
  lists, the class lists and the chunk descriptors from `DeviceEvaluator`'s arrays
  plus `parent`, `child_first`, `cell` (as `SVector{3,Int32}`), `ghost`,
  `has_source` (uploaded once, 65 MB at 10⁷); `ClassChunk` becomes the six-integer
  descriptor; `_class_gather!`/`_class_scatter!` take the class list arrays, the
  chunk offset and the run range. `DevicePlan` drops `_class_chunks_host` (and the
  `tree.ilists` dependency); `device_matvec!`/`device_build_expansions!` accept
  `(sources, τ)` in the original order and gather by `plan.dev.perm` into plan-owned
  buffers (a keyword to keep the present tree-order interface for the bench).
- `src/tree/builder.jl`/`tree.jl`: an option to skip `build_interaction_lists` in
  `flatten` when only the device passes will run (0.15 s at 10⁷; the CPU passes need
  the lists, the device passes will not).
- `src/operators/symmetry.jl`: `class_lists` stays for the CPU class-wise
  translation; the device kernel takes `sc.class`.
- Tests: `test/test_device_passes.jl` runs the new path on the CPU backend
  (`to = identity`: Base `sortperm!`/`accumulate!` on `Array`s, the kernels on KA's
  CPU backend) with its existing comparisons of multipoles, locals and the matvec
  against the CPU passes (tolerances unchanged: the arithmetic per pair is the
  same, only the chunking of the GEMMs changes); add to `test/test_coverage.jl`
  (which holds the list invariants; there is no `test_lists.jl`) the equality of the
  device-built `(target, source, offset)` triples per level with `tree.ilists` and
  of the per-class counts and the per-target runs with `class_lists`, on every
  distribution of `DISTRIBUTIONS` and the 3D sets; `test/test_device_harmonics.jl`
  covers the 34-class path.
- Expected (H100, 10⁷): plan 0.23 → ≈ 0.03 s (evaluator + buffers) plus ≈ 10 ms of
  device work; per matvec 0.15 s of host permutation → 3 ms on the device; the tree
  2.3 s unchanged. On the M3 Max: plan 1.15 → ≈ 0.12 s.

### Stage 2: the Morton sort on the device, the key-based host builder

Shared with the CPU work: once the host builder refines, restricts and ghosts on a
key-sorted order (ranges by binary search in the keys, no point moves), it takes the
device's `keys[perm]` as input (one download of 80 MB: ≈ 12 ms on Metal, ≈ 3 ms over
PCIe). The host then does node work only (colleagues, depths, flags, lists: ≈ 0.3–0.4
s single-threaded at 10⁷ by the phase table of 1.1, less threaded), and the sorted
positions never leave the device. Tests: `test/test_tree.jl`'s
`check_tree_invariants` and `test_coverage.jl` as they are; a determinism test that
the device keys equal `morton_key` of the host's cells at every level.

### Stage 3: the tree on the device

The level loop of 3.2 with its restriction and ghost passes, producing the same flat
arrays; the host keeps the loop control and the readbacks. Files: a new
`src/tree/device_builder.jl` (kernels + the loop), `src/device.jl` (`DeviceEvaluator`
from device arrays), `src/device_passes.jl` (`DevicePlan` from a device tree).
Tests: assemble a host `Tree` from the downloaded device arrays and run
`check_tree_invariants` (all invariants except the within-leaf order of `perm`,
compared as sets per leaf) and `check_coverage` on every distribution; bitwise
equality of `cell`, `level`, `parent`, `child_first`, `points`, flags, `colleagues`
and the lists with the host `Tree` of the same points (the node order is made the
host's, 3.2); two device builds bitwise equal; the full matvec against the CPU as in
`test_device_passes.jl`. Expected: tree + plan ≈ 0.15–0.2 s on Metal, est. 0.05–0.1
s on the H100, host ≈ 0 (loop control).

### Stage 4: the device-resident step

Capacity buffers and views (no per-step allocation), the per-step flow of section
4, the reuse check as an option, the launch count per matvec (chunks of 16 384 pairs
cut the ≈ 4 200 launches at 10⁷ by 4× at 11 MB per `Gc/Yc` buffer at `p = 4`; a
parameter to measure), a leaves-only `locals`. Expected step at 10⁷ on the H100 with
k matvecs: today 2.7 + k·(0.28 + 0.15) s; after stage 1: 2.3 + k·0.28; after stage
3: ≈ 0.1 + k·0.28.

## 6. Open questions and risks

- **The H100 numbers are extrapolated.** Everything was measured on Metal; the
  CUDA path (bitonic `sortperm!`, `scan!`, launch latencies) is untested here. The
  first thing to measure on the cluster is `metal_primitives.jl` with `CuArray`
  (the script is KA-generic except for the array constructor).
- **Restriction rounds** are the one data-dependent loop; on pathological inputs
  (a tight cluster next to empty space) the host takes several rounds and so will
  the device. Bounded by the level count as on the host.
- **Capacity bounds** for the appended node arrays (restriction and ghosts): the
  previous step's counts plus a margin, with an overflow readback and a retry;
  the first step of a run has no previous count and uses the host builder's bound.
- **Determinism**: all kernels are one-writer-per-element, scans and sorts are
  deterministic per backend; the only backend-dependent order is among points
  coincident at level 21, which changes the near-field summation order of those
  points only.
- **The within-leaf order** differs from the host's (3.1); tests compare structures
  and results, not `perm`, unless the CPU work adopts the same full-key sort.
- **Float32 keys**: 21 levels at Float32 resolution; a domain whose points cluster
  below `W0/2²¹` apart would exceed `ppl` without separable keys and stop at level 22
  with the `max_levels` error, as the host does with coincident points.
- **AcceleratedKernels** is left out; if a backend's Base `sortperm!` turns out to
  be unstable or slow on the cluster, a `(key, index)`-packed `sort!` of unique keys
  (13.5 ms on Metal for 10⁷ `UInt64`) is the dependency-free fallback.
- **Metal views**: contiguous column ranges only (`view(B, :, 1:n)`); a row-range
  view is a `SubArray` and would fall back. A test asserting `typeof` of every
  GEMM operand on Metal, as `gpu_plan.md` risk 5 suggested, still applies.
- **Compile time**: the plan's device kernels compile per element and density
  type (0.5 s plan + 3.5 s first matvec at warm-up size here); the new kernels add a
  few seconds once per session, as today.
