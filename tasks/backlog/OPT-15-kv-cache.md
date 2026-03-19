# OPT-15: KV-Cache Analog for Inference

- **Category:** inference
- **Priority:** 6.2/10
- **Impact:** 6/10
- **Feasibility:** 5/10
- **Confidence:** 8/10

## Summary

Cache routing decisions and leaf probabilities per position during autoregressive generation. Currently `generate()` recomputes the entire sequence each step. With caching, only the new token runs through the forest.

## Expected Effect

- **Speedup:** O(S)x for autoregressive generation (from quadratic to linear per step)
- **Accuracy delta:** 0.0pp (exact)

## Risks

- Only matters for inference/generation, not training
- Moderate refactor to thread cache through all forest + attention layers

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `BatchedTreeForest.forward`, `ObliviousTreeForest.forward`, `TreeAttention.forward`, `TreeTransformer.forward`
- `/Users/goldy/tree-based-llm/data.py` — `ShakespeareDataset.generate`
