# The ghost-cell quadtree

`FlexibleFMM` uses an adaptive quadtree in which, after the usual 2:1 level
restriction, every leaf that touches a finer neighbour is given a full set of
"ghost" children. The effect is that the FMM needs only two kinds of interaction:

- **near field**: a target in leaf `B` sums directly over the sources of the
  same-level *colleagues* of `B` (the 3×3 block of boxes around it, itself included);
- **far field**: a node receives multipole-to-local translations only from
  same-level, same-width boxes (the children of its parent's colleagues that are not
  its own colleagues).

There are no W or X lists, the only M2L operators ever needed are the
`7² − 3² = 40` same-width translations per level, and the tree can be stored as flat
per-node arrays. This file describes the construction, the price, and the invariants
the tests check. The code is in `src/tree/`.

## Storage

Nodes have global indices; `level_range[l]` is the index range of level `l`, the root
being level 1. Each node stores its integer lattice `cell` (coordinates in
`[0, 2^(l-1))²`), its `parent`, `child_first` (children are the four consecutive
nodes starting there, in `child_offsets` slot order: slot `k` has bit pattern `k-1`
with coordinate 1 as the least significant bit), its sorted-point range `points`,
and the colleague list `colleagues[:, i]` (one slot per offset in `(-1:1)²`, 0 if
absent). All geometry is derived from the cell: `center = origin + (cell + ½)·w_l`
with `w_l = ldexp(W0, 1-l)`, which is exact in floating point. Nothing in the
builder or the locator ever rounds a float to recover a lattice position.

Points are permuted so that every node owns a contiguous range; `perm[j]` is the
original index of the `j`-th sorted point and `point_node[j]` the real leaf holding
it. The tree stores no coordinates: positions are passed to the FMM at evaluation
time, which is what makes the FMM a generic (differentiable) function of them.

Three orthogonal flags describe a node:

| flag | meaning |
|---|---|
| `leaf` | terminal for target location; real leaves, including xlist leaves |
| `ghost` | a fake child of an xlist leaf: has sources and a multipole, is never a target, never has children |
| `xlist` | a real leaf with a non-leaf colleague; keeps its targets but has ghost children |

The passes themselves only need `has_children`: a childless node (real non-xlist
leaf or ghost) gets its multipole from its sources, a node with children (non-leaf
or xlist leaf) from its children. The Python `compute_upwards`/`compute_downwards`
flags reduce exactly to this.

## Construction

1. **Refinement.** Starting from the root box, every node with more than `ppl`
   points is split into four children; the node's points are counting-sorted in
   place by child slot. This fixes the number of levels: neither of the later steps
   ever adds one (a leaf on the two deepest levels can never be "bad", and ghosts are
   only added below levels `2..L−1`). More than `ppl` coincident points would refine
   forever; `max_levels` guards against that.
2. **Colleagues.** Per level, by lookup of `cell + offset` in a sorted key table.
   Cells must be unique on a level, which is also what guarantees that a target
   receives at most one source per far offset (so the M2L scatter-add has no write
   conflicts).
3. **Level restriction.** With `depth` the height of a node's subtree, a leaf is
   *bad* if a colleague's depth differs from its own by more than one. Bad leaves are
   split, colleagues of the new level are retagged, depths recomputed, until no bad
   leaf remains. Balance on colleagues implies the usual adjacency balance: two
   touching real leaves differ by at most one level (if `A` at level `l` touched `B`
   at level `l+2` or deeper, `B`'s ancestor at level `l` would be a colleague of `A`
   with depth ≥ 2).
4. **Ghosts, in two phases.** (a) Every real leaf with a colleague that has (real)
   children is marked `xlist`, on all levels, before anything is added. (b) Every
   xlist leaf receives four ghost children covering its point range; the leaf stays a
   leaf. Phase (a) must be complete before phase (b) starts: if "has children" were
   evaluated live, a freshly ghosted leaf would make its neighbours xlist and the
   whole level would ghost itself. (The Python code achieved the same by letting its
   depth arrays go stale and skipping ghosts with `cj < n`.)

The builder works on growable per-level arrays and is flattened to global indices at
the end; sibling groups of an appended level are contiguous in level order but not
in parent order, so `group_parent[l][g]` records the parent of each group and the
driver gathers by it (the Python used an `argsort` instead).

## Why it works

After step 4, every colleague of a non-leaf node has children, and none of them is a
ghost (a ghost's parent is a leaf of depth 0 next to the non-leaf's parent of depth
≥ 2, which balance forbids). Hence the interaction list of any real node consists
of actual nodes with multipoles, all at its own level. Ghost multipoles are formed
from their sources; the xlist leaf's own multipole is the M2M of its ghosts; a real
leaf next to a ghosted leaf sees the ghosts in its colleague list (near field) and
the other ghosts of that neighbour in its interaction list (far field).

The combinatorial statement that makes this a correct FMM, checked by
`test/test_coverage.jl` on every distribution: for each childless node `T` with
sources and each real leaf `B`, exactly one of

- some colleague of `B` is an ancestor-or-self of `T` (near field), or
- exactly one pair `(X, Y)` with `X` an ancestor-or-self of `T`, `Y` an
  ancestor-or-self of `B`, and `X` in the interaction list of `Y`

holds. The second case is the standard argument: walking down from the root, the
first level at which the ancestors of `T` and `B` stop being colleagues is the one
and only level where they interact; a childless `T` coarser than `B` is never a
colleague of `B`'s ancestor at that level because that ancestor has children.

## The price

- A target in a coarse leaf sums directly over the *whole* source range of each
  non-leaf colleague (up to `4·ppl` sources each, since balance makes those
  colleagues' children leaves). The classical adaptive FMM would handle the far half
  of those by multipole evaluation (W list).
- Ghosts add one multipole, one `s2m!` and a handful of M2L pairs per xlist leaf
  quadrant.
- Colleague-based restriction refines empty space next to deep clusters down to
  within one level of the cluster (the three empty quadrants of a one-quadrant
  distribution are refined along their border with the populated one).

For uniform data none of this costs anything measurable; for strongly adaptive data
the near field grows by a bounded factor.

## Invariants (all asserted in `test/test_tree.jl`)

- `perm` is a permutation; childless ranges partition the points; children's ranges
  partition the parent's in slot order; `cell(child) = 2·cell(parent) + bits(slot)`.
- Every level ≥ 2 consists of whole sibling groups; cells are unique per level.
- Colleagues equal the brute-force 3×3 lattice block and are symmetric.
- 2:1 balance on colleagues; adjacency balance between real leaves.
- `ghost ⇒ !has_children`; `leaf ∧ has_children ⇒ xlist` with ghost children;
  every colleague of a real non-leaf is a non-ghost with children;
  a real leaf is `xlist` iff it has a real non-leaf colleague.
- The locator (`locate`), which walks from the root with the same `child_slot`
  predicate the builder used (strictly greater than the center goes up), returns
  `point_node[j]` for every sorted source `j`.

## What changes in three dimensions

Nothing in this file except the constants: 8 children, 27 colleagues,
`7³ − 3³ = 316` far offsets, and `child_offsets`/`colleague_offsets`/`far_offsets`
enumerate them generically from the dimension parameter `D`. The same builder,
invariants and coverage check run on 3D point sets (uniform, clustered, a sphere, a
helix, one octant, and a sphere with a tight cluster) in `test/test_3d.jl`. What
does not carry over for free is the operator side: the cube surfaces with spectral
translations (`docs/kifmm.md`) are the 3D method, since 316 dense M2L operators per
level would not fit; circles and the Chebyshev black box remain 2D.
