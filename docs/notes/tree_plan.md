# Tree construction: where the time goes and how to make it fast

Scope: the construction cost of `FMM(m, pts; ppl, bbox, cache)` with warm operators,
i.e. `Tree` (refinement, colleagues, level restriction, ghosts, flattening, the
interaction lists) plus the constructor's bookkeeping, for the JellySim regime:
10⁶–10⁷ points on a surface, every position changing every step, the data on a GPU
with a few CPU cores beside it. Measurements are from this machine (M3 Max, 12
performance cores, Julia 1.12.5, Float64 points, `-t auto` = 12 threads; the builder
itself is single-threaded, so the thread count does not affect the first table).
Scripts: `/private/tmp/claude-504/-Users-dstein-Documents-Repos-flexmm/56594ad4-d0ed-4daa-b0c3-59387940395c/scratchpad/tree_explore/`
(`phases.jl` times the package's own functions
phase by phase; `morton_proto.jl` is the redesigned builder, verified field by field
against `Tree`; `resort_proto.jl` the reuse experiment; `sort_ref.jl`,
`structural_pairs.jl` reference numbers). Nothing under `src/` was changed.

The companion GPU exploration (a device-resident tree) is a separate note; §3.4 and
§3.5 say which pieces here are its building blocks.

## 1. Measurements of the current builder

### 1.1 Phase times, single-threaded (ms)

"tag" is `tag_colleagues!` (a `sortperm` of the cell tuples of a level plus 27 binary
searches per node); "restrict tag" and "ghost tag" are the same function re-run on
whole levels after the restriction splits and the ghost splits; "lists" is
`build_interaction_lists` (enumeration by `push!`, CSR by offset, the by-target
Morton view). "FMM extra" is everything `FMM` adds to `Tree` with a warm cache:
operator lookup, `s2m_nodes`, `leaf_nodes`.

| case | levels | nodes (ghosts) | pairs | **Tree** | refine | tag | restrict tag | ghost split | ghost tag | flatten | lists | FMM extra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10⁶ uniform, ppl 200 | 6 | 37 449 (72) | 6.03 M | **252** | 17 | 36 | 0 | 0 | 31 | 8 | 172 | 0.1 |
| 10⁶ uniform, ppl 800 | 5 | 4 681 (0) | 0.64 M | **39** | 12 | 3 | 0 | 0 | 0 | 1 | 21 | 0.0 |
| 10⁶ sphere, ppl 200 | 7 | 98 785 (59 056) | 1.53 M | **220** | 20 | 21 | 24 (3 it.) | 1 | 95 | 9 | 42 | 0.2 |
| 10⁶ sphere, ppl 800 | 6 | 21 521 (12 864) | 0.35 M | **65** | 15 | 4 | 3 (2 it.) | 1 | 18 | 1 | 15 | 0.0 |
| 10⁷ uniform, ppl 200 | 7 | 299 593 (0) | 52.3 M | **1 541** | 205 | 390 | 0 | 0 | 0 | 27 | 1 203 | 0.6 |
| 10⁷ uniform, ppl 800 | 6 | 37 449 (0) | 6.04 M | **340** | 152 | 38 | 0 | 0 | 0 | 10 | 108 | 0.0 |
| 10⁷ sphere, ppl 200 | 9 | 1 656 593 (1 045 840) | 22.0 M | **4 182** | 268 | 388 | 611 (4 it.) | 40 | 2 122 | 68 | 568 | 3.1 |
| 10⁷ sphere, ppl 800 | 8 | 407 761 (255 744) | 5.68 M | **1 104** | 226 | 87 | 124 (3 it.) | 17 | 455 | 31 | 147 | 0.8 |

The `FMM(...)` time with a warm cache equals the `Tree` time within noise in every
case (e.g. 1.103 s against 1.104 s for 10⁷ sphere 800), so the constructor's
bookkeeping and the operator lookup are not a factor. The "lists" column is the
in-place measurement; the sub-phases measured separately sum to less (123 against
172 ms for 10⁶ uniform 200), the difference being the growth of the three `push!`
vectors and first-touch page faults on the fresh output arrays (see §3.6).

The 10⁷ sphere 800 case is the production configuration of the brief (2.3 s on the
Icelake node with 8 threads, which is the same single-threaded code on a slower core:
1.1 s here).

### 1.2 Reading

1. **Colleague tagging is the cost on surfaces: 0.67 s of 1.10 s at 10⁷ sphere 800
   and 3.1 s of 4.2 s at ppl 200.** `tag_colleagues!` sorts the cells of a whole
   level as tuples and does 27 binary searches per node, and it is re-run on the
   *whole* next level after every restriction split (per iteration, per level) and
   after every ghost split. On the sphere the ghost levels are the large ones (63 %
   of all nodes are ghosts at ppl 200), so the ghost re-tag alone is 2.1 s.
2. **The interaction lists are the cost on uniform data** (1.2 s of 1.5 s at 10⁷,
   ppl 200: 52 M pairs). Half of it is the `push!` enumeration, the rest the two
   copies (CSR by offset, by-target view), all single-threaded and allocating about
   3 GB of `Int64` for the seven arrays.
3. **Refinement is cheap**: 0.15–0.27 s at 10⁷, three passes over the points per
   level (count, scatter, copy back) with the exact `child_slot` against the node's
   center. It is bandwidth-bound and parallel per node.
4. Everything else (restriction scans, depths, flattening, bookkeeping) is below
   0.1 s at 10⁷.
5. **Nothing in the builder is threaded.** The only threaded tree code is
   `locate!` (targets); `_class_chunks_host` in the device plan is threaded over
   levels. `Tree` runs the same on 1 and 12 threads.

## 2. The invariants any redesign must keep

From `docs/tree.md`, checked by `test/test_tree.jl` (`check_tree_invariants`),
`test/test_coverage.jl` (`check_coverage`) and `test/test_3d.jl` (the same checks on
3D distributions: uniform, clustered, sphere, helix, one octant, sphere with a tight
cluster): `perm` a permutation with childless ranges partitioning the points and
children's ranges partitioning the parent's in slot order; `cell(child) = 2·cell +
bits(slot)`; every level ≥ 2 whole sibling groups; cells unique per level;
colleagues equal to the brute-force 3^D block and symmetric; 2:1 balance on
colleagues and adjacency balance between real leaves; the ghost/xlist flags
(`ghost ⇒ !has_children`; `leaf ∧ has_children ⇒ xlist` with ghost children; every
colleague of a real non-leaf a non-ghost with children; a real leaf `xlist` iff it
has a real non-leaf colleague); `locate` returning `point_node` for every sorted
source; the lists exactly `V` restricted to sourced sources and non-ghost targets, a
target at most once per offset, the by-target view in Morton order; determinism of
two builds. There is no `test/test_lists.jl`; the list checks live in
`test_coverage.jl`.

Every option below produces the *same* tree, including `perm` (the prototype is
checked for equality of every field of `Tree` and of every `InteractionList` against
the package on all eight cases of §1.1, with 1 and 12 threads), so the existing tests
protect it unchanged. A reused tree (§3.5) is a different object and needs its own
tests (listed there).

## 3. Options

### 3.1 Colleagues by the parent rule (removes the sorts and searches)

**Claim.** For a node `i` on level `l ≥ 2` with cell `c` and parent `p`, and any
offset `o ∈ {−1,0,1}^D`: the node with cell `c' = c + o` exists on level `l` iff the
colleague of `p` at the parent-level offset `d = (c' >> 1) − (c >> 1)` exists and has
children, and then it is that colleague's child in slot `1 + Σ_k (c'_k & 1)·2^(k−1)`.

**Proof.** Every level ≥ 2 consists of whole sibling groups, so a cell exists on
level `l` iff its parent cell `c' >> 1` exists on level `l−1` and was split (real or
ghost children; `child_first > 0` in both cases). Since `|c'_k − c_k| ≤ 1`,
`⌊c'_k/2⌋ − ⌊c_k/2⌋ ∈ {−1, 0, 1}` (arithmetic shift gives the floor for `c'_k = −1`),
so the parent cell of `c'` is at a colleague offset of `p` and is read from `p`'s
colleague row; cells outside `[0, 2^(l−1))` have parent cells outside
`[0, 2^(l−2))`, absent by induction down to the root, whose row holds only itself.
The slot is the child-offset bit pattern by `child_offsets` (coordinate 1 least
significant). Uniqueness of cells per level, which `tag_colleagues!` checks by the
sort, holds by construction (children of distinct parents have distinct prefixes,
siblings distinct low bits) and stays asserted in `check_tree_invariants`
(`allunique(tree.cell[r])`). ∎

So a level is tagged top-down from its parent level in `27` table lookups per node,
no sort, no search, no allocation beyond the matrix, and the loop over nodes is
embarrassingly parallel. The rule is dimension-generic (`_offset_tuple` index
`1 + Σ (d_k+1)·3^(k−1)`, slot `1 + Σ (c'_k & 1)·2^(k−1)`). Ghosts need nothing
special: a ghost parent has `child_first = 0`, so nothing is found through it, and a
ghost child is tagged like any node of its level (as today).

After a restriction or ghost split the appended level is simply re-tagged by the rule
(the parent level's colleagues are unchanged by a split below it); an incremental
update of only the new nodes and their symmetric slots would be cheaper still but is
not needed: the full re-tag of the 1.66 M-node levels of 10⁷ sphere 200 costs 143 ms
single-threaded against 2 122 ms today.

| measured (single thread) | today | parent rule |
|---|---|---|
| 10⁶ uniform 200: tag + ghost tag | 36 + 31 ms | 1.9 + 1.6 ms |
| 10⁶ sphere 200: tag + restrict tag + ghost tag | 21 + 24 + 95 ms | 1.9 + 1.5 + 4.8 ms |
| 10⁷ sphere 800: same | 87 + 124 + 455 ms | 5 + 6 + 20 ms |
| 10⁷ sphere 200: same | 388 + 611 + 2 122 ms | 19 + 30 + 143 ms |

Complexity: small. `tag_colleagues!(lev, par)` replaces `tag_colleagues!(lev)` in
`src/tree/builder.jl` (three call sites: `build_levels`, `level_restrict!`,
`add_ghosts!`), plus a `tag_root!`. Nothing else changes.

### 3.2 Two-pass interaction lists

Replace the `push!` enumeration by a count pass (pairs per non-ghost target, from the
parent's colleagues' children, as now), a prefix sum, and a fill pass into exact-size
arrays; both passes are per target and threaded. The per-target ranges of the fill
are the groups the by-target view needs, so the `starts` scan disappears; the Morton
`sortperm` over targets stays (1.5–26 ms at 10⁷) and the by-offset CSR stays a
counting sort. Same pairs in the same order, hence identical `InteractionList`s.

Measured single-threaded at 10⁷ uniform 200 (52 M pairs): enumerate 523 → count 71 +
fill 369 ms; the whole lists 1 203 → 802 ms (the CSR and by-target copies, 180 +
174 ms, are now the larger part). On 12 threads the lists of 10⁶ uniform 200 go
172 → 41–53 ms. What remains is memory traffic: seven `Int64` arrays of 52 M entries.
Two further reductions, each a separate decision:

- `Int32` indices in `InteractionList` (nodes < 2³¹; the device plan converts to
  `Int32` anyway): halves the traffic and the memory (3 GB → 1.5 GB at 10⁷ uniform 200).
- Build the by-offset CSR lazily: only the dense per-offset M2L of `m2l_level!` (the
  circle method and non-covariant kernels) reads `offset_ptr/tgt/src`; the spectral
  path and the device plan read the by-target view. The CSR could be built on first
  use by a method that needs it (a field with a sentinel, or a method trait).

### 3.3 Threading the existing builder

Which phases parallelise, and how (measured in the prototype):

| phase | parallel over | note |
|---|---|---|
| refinement `split_nodes!` | nodes of a level (disjoint ranges) | the first levels have few nodes with huge ranges (level 1: one node with all N points, level 2: 8, level 3: 64); those are split chunk-parallel inside the node: per-chunk 8-bin histograms, one exclusive scan, per-chunk stable scatter, then the copy back. The result is the same stable counting sort whatever the chunking, so it does not depend on the thread count. |
| `tag_colleagues!` (parent rule) | nodes | trivially |
| restriction: bad-leaf scan, `compute_depths!` | nodes (levels in order) | the outer iteration is sequential (1–4 iterations observed) |
| ghosts: mark, split | nodes / xlist leaves | ghost ranges are ≤ ppl points, no inner parallelism needed |
| flatten | nodes per level; `point_node` per leaf | |
| lists | targets (count, fill) | the CSR counting sort and the by-target copy could be chunk-parallel like the splits; serial in the prototype |

Rules learnt the hard way in the prototype, both documented in `docs/framework.md`
(Threading): a variable assigned inside a `@threads` body *and* in the enclosing
function is one shared boxed variable (the first version's `pos` made the lists both
wrong on 12 threads and 10× slower on one); threaded bodies belong in their own
functions. And `@threads :static` cannot nest: the per-node loop and the chunk-parallel
inner sort are alternatives chosen per level (few large nodes → inner, many → outer).
Thread regions cost 20 µs here and 40 µs on the Icelake node but 30 ms on the Genoa
node (`bench/threads.jl`); a build has roughly levels × phases ≈ 50–100 regions,
which is 3 s there — on such a node the builder would need the same region economy as
the matvec (one region per level, tasks taking node blocks from a counter), or to run
serially. This is a configuration to measure on the target node, not a default.

Measured, the prototype of §3.1–3.3 against the package (`Tree` time; the package
time varies ±30 % between runs with the GC state, the ranges are given):

| case | package | prototype, 1 thread | prototype, 12 threads |
|---|---|---|---|
| 10⁶ uniform 200 | 0.25–0.38 s | 0.107 s | 0.077 s |
| 10⁶ uniform 800 | 0.03–0.04 s | 0.021 s | 0.013 s |
| 10⁶ sphere 200 | 0.20–0.26 s | 0.065 s | 0.027 s |
| 10⁶ sphere 800 | 0.055–0.065 s | 0.028 s | 0.011 s |
| 10⁷ uniform 200 | 1.5–2.3 s | 1.03 s | 0.49 s |
| 10⁷ uniform 800 | 0.34–0.48 s | 0.23 s | 0.10 s |
| 10⁷ sphere 200 | 4.2–4.7 s | 1.18 s | 0.43 s |
| 10⁷ sphere 800 | 1.10–1.34 s | 0.48 s | 0.18 s |

Single-threaded remainder at 10⁷ sphere 800 (0.48 s): refinement 230 ms, lists 119
(count 29, fill 66, CSR 9, by-target 15), ghosts 31, flatten 10, restriction 9,
tagging 5. On 12 threads (0.18 s): refinement 70 ms, lists 67 (fill 19, CSR 11,
by-target 24), the rest below 10 ms each. Two things scale poorly. The refinement
only gains 3.3× (230 → 70 ms; 205 → 60 ms on uniform 200): it is memory-bound (24 +
8 bytes per point, three passes per level) and its copy-back pass is serial per
node; a ping-pong of the two point buffers would remove that pass (points of a
split node move to the other buffer; a final gather assembles the leaves), and
Float32 points halve the bytes. And the two list copies are serial: at 10⁷ uniform
200 they are 137 + 157 ms of the 487 ms (the fill itself scales 11×, 369 → 32 ms);
a chunk-parallel counting sort, as for the splits, applies to both.

### 3.4 A Morton-key construction (and what the CPU and GPU versions share)

**The key.** One `UInt64` per point holding its child slot on levels 2..22 (3 bits
per level, 63 bits; `max_levels` becomes 22 on this path, against 30 today — more
than `ppl` points within `2⁻²¹` of the box width throw as they do now). The slot of
level `l+1` is computed exactly as the builder and `locate` compute it: `child_slot(x,
_center(origin, cell_l, ldexp(W0, 1−l)))`, accumulating `cell_{l+1} = 2·cell_l +
bits`. This matters: a key by `floor((x − origin)·2^(L−1)/W0)` rounds, and would put
a point within an ulp of a box face into a different box from the one the comparison
walk of `locate` (and the device evaluator, which takes `point_node`) finds; the
level-wise comparison never rounds a coordinate, which is the design rule of
`docs/tree.md` ("nothing in the builder or the locator ever rounds a float to recover
a lattice position"). A device kernel computing the key does the same IEEE arithmetic
in the same order (`origin + (cell + 0.5)·w`, `w = ldexp(W0, 1−l)`), so the keys agree
bitwise as long as the FastMath policy is kept away from that kernel.

**Construction from sorted keys.** With the points sorted by key, every box of every
level is a run of equal prefixes inside its parent's run: a split is eight binary
searches (or one pass) over the parent's range, the refinement decision is `count >
ppl` per run, and the restriction and ghost splits never move a point again. The
colleagues come from the parent rule (§3.1), the lists from the two passes (§3.2), the
restriction and ghost phases are the same per-node predicates followed by appends
(stream compactions). The flattening numbers each level's nodes in Morton order
(sibling groups stay contiguous in slot order since the slot bits are the low bits
of the key), which makes the by-target Morton view the identity on the targets.

**Measured on the CPU, this buys nothing for the build.** The key computation costs
76 ms per 10⁶ points single-threaded (3.5 ns per level per point, 22 levels; 31 ms
for 12 levels) and 7 ms on 12 threads; at 10⁷, 0.76–0.82 s single-threaded, more
than the whole redesigned build (§3.3), and the per-level splits it speeds up only go
from 230 to 170 ms. On 12 threads it is a wash: keys 30–45 ms plus a 0.04 s radix
sort against splits of 53–76 ms, for the same total build time (0.42 against 0.49 s
at uniform 200, 0.20 against 0.18 s at sphere 800). The direct-slot splits (the exact `child_slot` on the fly) are a
stable MSD radix sort by 3 bits per level, which is already the right CPU algorithm:
one level per pass, no full-key sort, and — a property worth keeping — the points of
each childless node stay in their original order, so `perm` is bitwise what it is
today. A full LSD radix `sortperm` of 10⁷ 63-bit keys (11-bit digits, 6 passes,
chunk-parallel, `sort_ref.jl`) costs 0.18 s single-threaded and 0.04 s on 12 threads
(Base `sort` of the keys alone, a radix sort, 0.08 s; Base `sortperm`, a comparison
sort, 0.63 s), and would order the points of a leaf by their deeper key bits, a
different `perm` (the same tree otherwise). Recommendation: the CPU builder keeps
the comparison-based per-level splits; keys are not added to the CPU path.

**Where the keys are the right tool: the device.** On a GPU the key kernel is one
pass over the points (microseconds at 10⁷), the sort is the backend's radix sort
(CUDA.jl's `sort!`/`sortperm!` through CUB; Metal has no library sort, GPUArrays' bitonic
sort or a KA radix kernel), and every later phase is a per-node kernel plus a scan:
runs of prefixes (segmented counts), the refinement predicate, the parent rule, the
bad-leaf and xlist predicates with compaction of the appended groups, the two-pass
lists. The per-level stable split of the CPU would instead cost a kernel launch and a
scan per level with little work each. What the CPU and the GPU constructions share,
and should share as code or at least as definitions: (1) the slot/key definition
above; (2) the level derivation (a box is a run of equal prefix, split iff `count >
ppl`, children in slot order); (3) the parent rule for colleagues; (4) restriction and
ghosting as per-node predicates on `depth`/`child_first` of the colleagues followed
by appends; (5) the two-pass lists; (6) the flattened `Tree` layout (`Int32` on the
device as the plan already does). What differs: the sort (LSD on the full key versus
per-level stable counting sorts; if bitwise agreement of `perm` with the CPU tree is
wanted, a segmented sort by original index within each childless range restores it,
one extra pass), and the handling of the few huge nodes of the first levels. The
device build also removes the upload of a rebuilt tree, which at 10⁷ sphere 200 is
180 MB of colleagues and 350 MB of lists in `Int32` per rebuild.

### 3.5 Reuse across time steps

**What depends on the points.** The ghost-cell structure — cells, levels, children,
colleagues, flags, 2:1 balance, the interaction lists as a relation between nodes —
is valid for *any* assignment of points to its childless nodes: `ppl` is a cost
parameter, not a correctness one (the coverage argument of `docs/tree.md` never uses
counts). The points enter only through `perm`, `points` (ranges), `point_node`,
`has_source` and the two derived filters: the lists keep pairs whose source
`has_source`, and `s2m_nodes` are the childless nodes with sources.

**Resort.** Given moved positions, the childless node of each point is its real leaf
from `locate` (threaded already) or, for an xlist leaf, the ghost child by
`child_slot` against the leaf's center; a counting sort by childless node in the
depth-first slot order the ranges follow gives the new `perm`, the childless ranges,
the parents' ranges bottom-up, and `point_node`. On the unmoved points this
reproduces `perm`, `points` and `point_node` of the build exactly (checked). Cost at
10⁶ on 12 threads: 6–11 ms (locate 3–6 ms, the serial counting sort and range
assembly 4–11 ms), against 11–27 ms for the redesigned build and 60–250 ms for the
present one; at 10⁷ about 0.1 s by extrapolation against 0.18–0.43 s.

**Validity per step.** Three checks, all O(N) or O(nodes), reducing to flags:
(1) every point inside the root box (else rebuild; a box with a margin makes this
rare); (2) `max count ≤ tol·ppl` over childless nodes, a cost guard: `tol = 1`
rebuilds as soon as any leaf exceeds today's rule, `tol = 2` accepts a near field up
to twice as expensive for the targets of that leaf; (3) the `has_source` pattern:
the filtered lists are stale as soon as a node that had no sources receives one.
Check (3) is the delicate one. Dropping the filter is not an option on surfaces:
the structural lists (every pair of existing nodes) have 4.6× the pairs of the
filtered ones at 10⁶ sphere 200 and 4.0× at sphere 800 (6–7× more childless nodes
than sourced ones: the restriction refines the empty space around the shell), and
1.0× on uniform data — so unfiltered lists would cost the M2L stage 4× per matvec.
The right design is therefore: the tree keeps the *structural* pair relation (or
simply the node structure), and the *active* lists and `s2m_nodes` are recomputed
per step by a compaction over `has_source` — on the CPU this is the two-pass list
build of §3.2 on the fixed structure (15–30 ms at 10⁶ on 12 threads), on a device a
count/scan/fill over the structural pairs. The device plan's class chunks are cut
from the by-target view (`_class_chunks_host`, 0.03 s at 10⁶ on the CPU backend; the
plan build is 0.23 s at 10⁷ on the H100 node), so with the present plan a change of
`has_source` costs a plan rebuild; a device-side compaction of the chunk lists is
the piece the GPU note has to provide for the reuse to be fully device-resident.

**How often does the check pass?** Simulated at 10⁶ on the sphere (ppl 800: finest
leaf width w = 22 h with h the point spacing; ppl 200: w = 11 h) and on uniform points,
150 steps each, three motions: a random walk of δ per point per step, a rigid
translation by δ per step, and a radial pulsation of 5 % of the radius over 100
steps (a bell-like deformation; the shell moves by up to 14 h per half cycle). The
initial maximum counts are 716 (ppl 800) and 195 (ppl 200). First step at which a
childless count exceeds tol·ppl, and first step with a newly sourced node
("never" = not within 150 steps):

| motion (sphere, 10⁶) | ppl | > ppl | > 1.5 ppl | > 2 ppl | new sourced node | left the box |
|---|---|---|---|---|---|---|
| random walk, 0.1 h per step | 800 | never | never | never | 1 | never |
| random walk, 0.5 h | 800 | never | never | never | 1 | never |
| translation, 0.1 h | 800 | never | never | never | 1 | never |
| translation, 0.5 h | 800 | 123 | 126 | 128 | 1 | 150 |
| pulsation 5 % | 800 | never | never | never | 1 | never |
| random walk, 0.1 h | 200 | never | never | never | 1 | never |
| random walk, 0.5 h | 200 | never | never | never | 1 | never |
| translation, 0.1 h | 200 | 6 | never | never | 1 | never |
| translation, 0.5 h | 200 | 21 | 57 | 57 | 1 | 150 |
| pulsation 5 % | 200 | 53 | never | never | 1 | never |

The uniform cases (points filling the exact unit box, no margin) leave the box at
step 1 under every motion, which says only that the root box of a time-stepped run
needs a margin (§3.6).

Reading: random motion, even 0.5 h per step (75 h ≈ 3.4 w of cumulative displacement
at ppl 800), never pushes a count past ppl in 150 steps; a rigid drift passes the
`tol = 2` guard until the cumulative displacement reaches 2.5–3 finest leaf widths
(64 h at ppl 800, 28 h at ppl 200), i.e. for about 2.5·w/δ steps — the coarser leaves
bordering the refined region fill slowly because the shell has to travel across them;
the 5 % pulsation crosses ppl but never 1.5 ppl. And the `has_source` pattern changes
at the first step in every sphere case (a point crossing into an empty neighbour
box), which is why the active lists must be recomputed per step rather than checked.
Under JellySim-like motion (displacements per step a fraction of the point spacing,
bodies translating and deforming), the structure therefore lasts of the order of
2·w/δ steps at `tol = 2`: a hundred or more steps at 0.1 h per step, tens at 0.5 h.
The per-step cost is a resort plus a list compaction, with a rebuild every so many
steps. On the CPU the resort is 2–3× cheaper than the redesigned threaded build, a
modest margin; with the positions on the GPU the resort is three kernels and a scan
and avoids both the 120 MB download of the positions and the upload of a new tree,
which is where it pays.

### 3.6 Other findings

- **Workspace reuse.** Fresh allocations are paged in on first touch: the CSR copy of
  the 10⁶ uniform 200 lists took 64 ms in a fresh process and 10 ms when the GC had
  memory to reuse; at 10⁷ uniform 200 the lists allocate about 3 GB. A `TreeWorkspace`
  (the scratch point/perm buffers, the level arrays, the list arrays) kept across
  rebuilds in a time-stepping loop removes this; `Int32` lists halve it.
- **Restriction** needs no further work once tagging is cheap: 9 ms at 10⁷ sphere
  800, 4 iterations at most observed.
- **Lazy lists on the device** (count/scan/fill kernels from the device-resident
  `colleagues`, `child_first`, `cell`, `has_source`): the same two passes as §3.2;
  this is how a device-resident tree would produce its active lists per step (§3.5).
- **Float32 points** halve the bytes of the refinement passes (12 against 24 per
  point); the production runs are Float32.
- **Bounding box.** A fixed root box with a margin around the body (JellySim knows
  its domain) avoids the containment failure of §3.5 and the cache's width snapping
  per step.

## 4. Recommended path

Ordered by gain per effort; each step leaves the tree bitwise identical to today's
(the existing tests apply unchanged) until step 5.

1. **Parent-rule tagging** (§3.1). `src/tree/builder.jl`: `tag_colleagues!(lev,
   par)` and `tag_root!`, called top-down in `build_levels` and after the splits in
   `level_restrict!` and `add_ghosts!`; drop the sort and the duplicate check (the
   uniqueness is asserted in `check_tree_invariants`). Gain at 10⁷: sphere 800
   1.10 → ≈ 0.45 s, sphere 200 4.2 → ≈ 1.2 s, uniform 200 1.5 → ≈ 1.15 s
   (single-threaded). Record the proof of §3.1 in `docs/tree.md` (Construction, step 2).
2. **Two-pass lists** (§3.2). `src/tree/lists.jl`: count, prefix sum, fill, the
   by-target view from the per-target ranges; threaded over targets. Decide `Int32`
   and the lazy by-offset CSR separately (`InteractionList` is read by `m2l_level!`,
   `spectral_m2l.jl`, `symmetry.jl`, `device_passes.jl`, `test_coverage.jl`). Gain at
   10⁷ uniform 200: lists 1.2 → 0.8 s single-threaded, ≈ 0.2 s threaded.
3. **Threading** (§3.3). `split_nodes!` per node with the chunk-parallel inner sort
   for the large nodes, `tag_colleagues!`, the restriction scan and depths, the ghost
   mark, `flatten` per node, the list passes; bodies in their own functions. Keep
   the chunk decomposition independent of the thread count. Add to
   `test/test_threads.jl` that a tree built in the test process equals one built
   serially (a `threads::Bool` keyword, or a serial reference path kept for the
   test). Measured in the prototype: 10⁷ sphere 800 in 0.18 s on 12 threads here
   (sphere 200: 0.43 s, uniform 200: 0.49 s); on the 8 Icelake threads of the cluster
   node (whose core is 2.1× slower on this code) an estimated 0.3–0.5 s against
   2.3 s, the serial list copies and the memory-bound refinement scaling least;
   with the copies chunk-parallel and `Int32` lists, about 0.2 s (estimate).
4. **Workspace reuse** (§3.6), optional: a `TreeWorkspace` argument to `Tree`/`FMM`
   for time stepping; `Int32` lists.
5. **Resort** (§3.5). New `src/tree/resort.jl`: `resort(tree, points; tol) ->
   (Tree, ok)` sharing the structural arrays and carrying new `perm`, `points`,
   `point_node`, `has_source` and freshly compacted lists; `FMM(f::FMM, points)`
   reusing the cache and recomputing `s2m_nodes`/`leaf_nodes`. Tests: the resort of
   the unmoved points reproduces `perm`, `points`, `point_node`; after a move, every
   point inside its node's box, children's ranges partition the parents', the lists
   equal `build_interaction_lists` on the same structure, and the FMM on a reused tree
   agrees with direct sums to the method's accuracy (the reused and the fresh trees
   differ in structure, so direct sums are the reference, as in `test_fmm_laplace.jl`).
6. **Device building blocks** (with the GPU note): the key kernel (§3.4, the exact
   slot definition shared with `child_slot`), a device `locate` + counting sort (the
   resort on the device), a device compaction of the active lists and chunk lists,
   and, as the end state, the device construction of §3.4 on the sorted keys. Steps 1,
   2 and 5 are its prerequisites in the sense that the device code should implement
   the same parent rule, the same two-pass lists and the same structural/active split
   of the lists; step 3's chunked sort is the CPU stand-in for the device radix sort.

Expected end state: tree build of 10⁷ sphere points at ppl 800 in 0.18 s on 12
threads here (measured in the prototype; an estimated 0.3–0.5 s on the 8-thread
cluster node, ≈ 0.2 s with the list copies parallel and `Int32` lists) against
1.1 s / 2.3 s now, i.e. the build at the cost of one or two device matvecs instead
of eight; with the resort on the device, a per-step cost of a few kernels and a
rebuild every of the order of 2·w/δ steps.

## 5. Open questions

- The thread-region cost on the Genoa node (30 ms) makes a naively threaded builder
  slower than the serial one there; is that node a target for the build, or only
  the Icelake/H100 nodes? If it is, the builder needs the one-region-per-level
  structure of the matvec.
- `tol` for the count guard in the resort, and whether JellySim's step
  displacements are closer to 0.1 h or 0.5 h: the reuse interval scales as
  (finest leaf width / 2) / δ.
- `Int32` in `InteractionList` and the lazy by-offset CSR touch the CPU M2L
  paths; worth doing together with the list rewrite or kept apart?
- Whether a device-resident tree (the GPU note) makes steps 4 and 5 unnecessary for
  JellySim; steps 1–3 are worth having regardless for every CPU user of the package.
