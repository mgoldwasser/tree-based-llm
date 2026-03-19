# Task 12: Simultaneous Accuracy + Speed Improvements

**Status:** Pending
**Priority:** High — the core question: how do we win on BOTH axes?
**Depends on:** Task 07, 08, 09
**Files:** `main.py`

## The Fundamental Tension
More trees = more expressivity = better accuracy, but also more compute = slower.
We need approaches that improve accuracy WITHOUT proportionally increasing compute.

## Strategies for Better Accuracy at Same/Lower Cost

### 1. Shared Routing, Separate Leaves
All trees in a forest share the SAME routing decisions but have DIFFERENT leaf outputs.
- Routing cost: O(1) instead of O(n_trees) — the expensive einsum runs once
- Leaf cost: still O(n_trees) but leaf aggregation is cheap (just a weighted sum)
- Speedup: ~2-3x (routing is the bottleneck)
- Accuracy: each tree specializes its outputs while sharing the same input partitioning

```python
# One set of routing decisions
decisions = einsum('bsd,nd->bsn', x, shared_decision_weights)
leaf_probs = compute_leaf_probs(decisions)  # (B, S, n_leaves)

# Multiple sets of leaf outputs
per_tree = einsum('bsl,tlo->bsto', leaf_probs, all_leaf_outputs)
output = weighted_sum(per_tree)
```

This is analogous to multi-head attention: shared QK computation, different V projections.

### 2. Deeper Trees with Top-K Pruning
Instead of shallow trees (depth 2-3) visiting all leaves, use deeper trees (depth 5-6) but only compute the top-K most likely leaves.

- Depth 5 = 32 leaves, but top-4 routing = O(depth) compute
- Much more expressive (finer partitioning) at similar cost to depth-3 full routing
- Requires straight-through estimator for gradients

### 3. Conditional Tree Selection (Mixture of Tree Experts)
Not all trees need to fire for every input. Route each input to its top-K most relevant trees:

```python
# Lightweight gating network picks top-K trees
gate_logits = self.gate(x)  # (B, S, n_trees)
top_k_indices = gate_logits.topk(k=4).indices  # only compute 4 of 12 trees
```

This is the MoE principle applied to trees. 12 trees but only 4 active per token.
- 3x cheaper than running all 12
- Accuracy maintained because different tokens use different trees

### 4. Progressive Tree Growing
Start training with depth-1 trees (2 leaves), then grow to depth-2 (4 leaves), then depth-3 (8 leaves) as training progresses. Each growth step initializes new nodes from their parent.

- Early training: fast convergence on coarse patterns
- Late training: fine-grained splits for detail
- Inspired by: progressive training in GANs, curriculum learning

### 5. Distillation from Standard → Tree
Train a standard transformer first (fast), then distill its knowledge into a tree transformer:
- Teacher: standard transformer (already trained)
- Student: tree transformer
- Loss: KL divergence between teacher and student outputs
- Trees learn to match the teacher's behavior using their piecewise structure
- Often converges faster than training from scratch

## Priority Order
1. **Shared routing** (biggest speed win with no accuracy loss)
2. **Conditional tree selection** (MoE-style, proven at scale)
3. **Top-K pruning** (enables deeper trees)
4. **Progressive growing** (nice-to-have, helps convergence)
5. **Distillation** (if direct training plateaus)
