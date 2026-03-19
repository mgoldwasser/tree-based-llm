# OPT-07: Low-Rank Leaf Factorization

- **Category:** architecture
- **Priority:** 6.5/10
- **Impact:** 5/10
- **Feasibility:** 8/10
- **Confidence:** 7/10

## Summary

Factor leaf outputs from (T, L, D) into (T, L, R) @ (R, D) where R << D. Reduces leaf computation and intermediate tensor sizes. More impactful at larger d_model.

## Expected Effect

- **Speedup:** 1.2-1.5x on leaf output computation
- **Accuracy delta:** -0.0 to -0.2pp (for R=16 with D=64)

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `BatchedTreeForest.__init__` and `forward`; `ObliviousTreeForest.__init__` and `forward`

## Dependencies

Becomes more valuable with OPT-08 (deeper trees have more leaves).
