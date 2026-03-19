# Task 13: Scaling Analysis — Time & Compute Complexity

**Status:** Pending
**Priority:** Medium — informational, guides future architecture decisions
**Depends on:** None
**Files:** new `scaling_analysis.py`

## Current Complexity

### Standard Linear Projection
- Forward: O(B × S × D²) — matrix multiply
- Backward: O(B × S × D²) — same
- Parameters: O(D²)
- GPU: single cuBLAS GEMM, highly optimized

### BatchedTreeForest (current)
- Routing einsum: O(B × S × T × N_internal × D) where N_internal = 2^depth - 1
- Depth loop: O(depth) sequential steps, each O(B × S × T × 2^level)
- Leaf aggregation: O(B × S × T × N_leaves × D_out)
- Total: O(B × S × T × 2^depth × D)
- Parameters: O(T × 2^depth × D)
- GPU: multiple kernel launches (einsum + depth loop + aggregation)

### Scaling Comparison
| Operation | Standard | Tree (T=12, D=3) | Tree (T=12, D=5) |
|-----------|----------|-------------------|-------------------|
| FLOPs per token | D² = 4096 | T×2^D×D = 6144 | T×2^D×D = 24576 |
| Parameters | D² = 4096 | T×(2^D-1+2^D)×D = ~11K | T×(2^D-1+2^D)×D = ~47K |
| GPU kernels | 1 | 3 + depth | 3 + depth |

### Key Scaling Laws
- **Depth:** Exponential in compute (2^depth), but only linear in expressivity gains
- **Number of trees:** Linear in both compute and expressivity
- **Input dimension:** Linear in compute for both standard and tree

**Implication:** Scaling via more shallow trees is better than fewer deep trees.
- 24 trees × depth 3 = 24 × 8 = 192 leaves, O(24 × 8 × D) compute
- 3 trees × depth 6 = 3 × 64 = 192 leaves, O(3 × 64 × D) compute
- Same leaf count, but the shallow version is better parallelized (more trees = more batching)

## Profiling Script
Create `scaling_analysis.py` that:
1. Measures forward pass time for varying: n_trees, depth, d_model, seq_len
2. Compares with equivalent nn.Linear at each config
3. Estimates GPU utilization via theoretical FLOPs / measured time
4. Identifies the crossover point where trees become competitive

## GPU vs CPU Scaling Differences
- **CPU:** Memory-bound. Tree's scattered memory access (depth loop) is worse than Linear's contiguous GEMM. Trees lose more on CPU.
- **GPU:** Compute-bound at large batch sizes. Tree's batched einsums can saturate GPU compute units. Gap narrows significantly.
- **Expected:** Trees are ~4x slower on CPU but ~1.5-2x slower on GPU (at sufficient batch size).
