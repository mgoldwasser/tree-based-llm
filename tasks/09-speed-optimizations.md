# Task 09: Speed Optimizations — Closing the Gap with Standard Attention

**Status:** Pending
**Priority:** High — tree models are 4-14x slower than standard
**Depends on:** None
**Files:** `main.py`

## Current Speed Profile (CPU, 1000 steps)
| Model | ms/step | Slowdown vs Standard |
|-------|---------|---------------------|
| Standard Transformer | 26.6 | 1.0x |
| Batched Forest (attn) | 107.2 | 4.0x |
| Boosted Forest (attn) | 217.0 | 8.2x |
| Boosted Forest (full) | 361.2 | 13.6x |

## Where Time Goes

### 1. The depth loop (can't parallelize across levels, CAN parallelize within)
`_compute_leaf_probs` iterates `depth` times. Each iteration depends on the previous. At depth 3, that's 3 sequential tensor ops. The operations within each level are already batched across trees.

**Optimization:** Precompute a tree structure matrix at init time. Replace the depth loop with a single batched gather operation:
```python
# Precompute: which decisions affect which leaves, and left vs right
# tree_paths: (n_leaves, depth) — indices into decisions
# tree_signs: (n_leaves, depth) — 1 for left, 0 for right
# Then: single gather + product instead of loop
```

### 2. 4 separate Q/K/V/O projections
Each TreeAttention layer calls `make_projection` 4 times → 4 separate `BatchedTreeForest.forward()` calls. For boosted, that's 4 × 3 stages = 12 forest forward passes per layer.

**Optimization:** Fuse Q/K/V into a single larger forest:
```python
# Instead of 3 separate forests for Q, K, V:
qkv_forest = BatchedTreeForest(d_model, d_model * 3, ...)
qkv = qkv_forest(x)
Q, K, V = qkv.chunk(3, dim=-1)
```
This reduces 3 forests to 1 with 3x wider leaf outputs. The routing computation (the expensive part) is shared.

### 3. Redundant LayerNorm
Each `BatchedTreeForest` has its own LayerNorm. When inside `BoostedForest` (which also has LayerNorm), and inside `TreeTransformerBlock` (which also has LayerNorm), we're running 3+ LayerNorms per projection. Each is a full tensor operation.

**Optimization:** Add a `use_norm=False` parameter to `BatchedTreeForest`. Disable internal norm when the forest is wrapped by BoostedForest or used inside a transformer block.

### 4. The einsum itself
`einsum('bsd,tnd->bstn')` is the routing computation. For B=32, S=64, D=64, T=12, N=7 (depth 3):
- Input: 32×64×64 and 12×7×64
- Output: 32×64×12×7
- FLOPs: 32 × 64 × 12 × 7 × 64 = 10.8M

vs `nn.Linear` for Q projection: 32 × 64 × 64 × 64 = 8.4M FLOPs

The einsums are comparable in FLOPs but harder to optimize (4D output vs 3D). On GPU, both would be highly memory-bound.

### 5. QKV fusion (biggest single win)
Fusing Q/K/V into a single forest reduces:
- 3 routing einsums → 1 (3x speedup on routing)
- 3 leaf prob computations → 1 (3x speedup on depth loop)
- 3 output einsums → 1 (slightly larger but still 1 op)

Expected: ~2.5x speedup on attention projection, bringing Batched Forest from 4x → ~2x slowdown.

## Implementation Priority
1. QKV fusion (biggest win, moderate complexity)
2. Remove redundant LayerNorms (easy, small win)
3. Precomputed tree structure (moderate complexity, helps deeper trees)
4. Profile with `torch.profiler` to find remaining bottlenecks

## Target
Get Batched Forest (attn) from 107ms to ~50ms/step (2x vs standard instead of 4x).
