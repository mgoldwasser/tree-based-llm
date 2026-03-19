# OPT-06: Materialization-Free Tree Output

- **Category:** kernel-fusion
- **Priority:** 5.5/10
- **Impact:** 4/10
- **Feasibility:** 6/10
- **Confidence:** 7/10

## Summary

Fuse the leaf-output einsum and tree-weight reduction into a single operation to avoid materializing the (B,S,T,D) intermediate tensor. At B=32, S=128, T=24, D=64, this saves 12.5MB per forest call.

## Implementation

Replace two einsums with one fused operation:

```python
# Before:
per_tree = einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)  # materializes (B,S,T,D)
output = einsum('bsto,bst->bso', per_tree, weights)

# After: weight leaf_probs by gate, then contract T dimension first
weighted_probs = leaf_probs * weights.unsqueeze(-1)  # (B,S,T,L)
summed_probs = weighted_probs.sum(dim=2)  # (B,S,L) — sum over T... but leaves are per-tree
# Need: einsum('bstl,tlo,bst->bso', leaf_probs, leaf_outputs, weights) — 3-operand einsum
```

## Expected Effect

- **Speedup:** 1.1-1.2x overall
- **Accuracy delta:** 0.0pp (exact)

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `BatchedTreeForest.forward`, `ObliviousTreeForest.forward`
