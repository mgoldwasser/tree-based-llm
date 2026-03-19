# OPT-01: Vectorized Leaf Probabilities (Eliminate Depth Loop)

- **Category:** math-reformulation
- **Priority:** 7.8/10
- **Impact:** 6/10
- **Feasibility:** 9/10
- **Confidence:** 9/10

## Summary

Replace the depth-3 Python `for` loop in `_compute_leaf_probs` with a single vectorized operation using precomputed path index tensors. Currently 24 sequential kernel launches → 3 fused ops.

## Implementation

Precompute a `(n_leaves, depth)` path index tensor and a `(n_leaves, depth)` direction mask at `__init__` time. Register as buffers. Then:

```python
# decisions: (B, S, T, n_internal)
# path_indices: (n_leaves, depth) — which node on path to each leaf
# path_dirs: (n_leaves, depth) — 1=left, 0=right
relevant = decisions[..., self.path_indices]  # (B, S, T, n_leaves, depth)
probs_per_level = self.path_dirs * relevant + (1 - self.path_dirs) * (1 - relevant)
leaf_probs = probs_per_level.prod(dim=-1)  # (B, S, T, n_leaves)
```

## Expected Effect

- **Speedup:** 1.3-1.5x overall on BatchedTreeForest forward
- **Accuracy delta:** 0.0pp (mathematically exact)
- **Kernel launches:** 24 → 3 for leaf prob computation

## Risks

- Marginal if ObliviousTreeForest becomes the default (it already avoids this loop)
- Advanced indexing may not be as efficient as expected on some backends

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `BatchedTreeForest.__init__` (add buffers), `_compute_leaf_probs` (replace loop)

## Dependencies

None. Can be implemented independently.
