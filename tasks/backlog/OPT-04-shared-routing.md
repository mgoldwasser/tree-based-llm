# OPT-04: Shared Routing / Fused QKV Forest

- **Category:** architecture
- **Priority:** 7.5/10
- **Impact:** 9/10
- **Feasibility:** 6/10
- **Confidence:** 7/10

## Summary

Compute routing decisions and leaf probabilities ONCE per attention block, then apply 4 different leaf output matrices for Q, K, V, O. Currently routing + leaf probs (the expensive part: ~60% of forest time) runs 4 times independently.

## Implementation

New `SharedRoutingAttention` class:

```python
class SharedRoutingAttention(nn.Module):
    def __init__(self, d_model, n_heads, ...):
        self.routing_forest = BatchedTreeForest(d_model, ...)  # routing only
        self.leaf_q = nn.Parameter(...)  # (T, L, d_model)
        self.leaf_k = nn.Parameter(...)
        self.leaf_v = nn.Parameter(...)
        self.leaf_o = nn.Parameter(...)
        self.base_qkv = nn.Linear(d_model, 3*d_model)  # linear base
        self.base_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Shared routing (runs ONCE)
        leaf_probs, gate_weights = self.routing_forest.compute_routing(x)
        # Four cheap leaf-output einsums
        Q = self.base_qkv_q(x) + apply_leaves(leaf_probs, gate_weights, self.leaf_q)
        K = self.base_qkv_k(x) + apply_leaves(leaf_probs, gate_weights, self.leaf_k)
        V = self.base_qkv_v(x) + apply_leaves(leaf_probs, gate_weights, self.leaf_v)
        # ... attention math ...
        O = self.base_o(context) + apply_leaves(leaf_probs, gate_weights, self.leaf_o)
```

## Expected Effect

- **Speedup:** 2-3x on attention projection time
- **Accuracy delta:** -0.0 to -0.5pp (shared routing constraint)

## Risks

- Q and K may genuinely benefit from different routing. Mitigation: share routing for QKV but keep O separate.
- Requires refactoring `BatchedTreeForest` to expose intermediate routing results.

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — new `SharedRoutingAttention` class; refactor `BatchedTreeForest` to expose `compute_routing()`; update `make_projection` factory
- `/Users/goldy/tree-based-llm/train.py` — new model config entry

## Dependencies

Synergistic with OPT-03 (torch.compile). Alternative to OPT-05 (selective trees).
