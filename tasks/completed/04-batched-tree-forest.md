# Task 04: BatchedTreeForest

**Status:** Completed

## What was done
Replaced `SoftDecisionTree` + `TreeEnsemble` (Python loop over individual trees) with a single `BatchedTreeForest` class that stores all tree parameters in stacked tensors and computes everything via batched einsum.

## Key changes
- `decision_weights: (n_trees, n_internal, input_dim)` — single stacked tensor
- `einsum('bsd,tnd->bstn', x, decision_weights)` — all trees in one op
- Leaf prob computation batched across trees in the depth loop
- `einsum('bstl,tlo->bsto', leaf_probs, leaf_outputs)` — batched output

## Result
12 batched trees (107ms/step) vs old 3 sequential trees (56-95ms/step). ~4x more trees at similar speed.
