# OPT-03: torch.compile Wrapping

- **Category:** compilation
- **Priority:** 7.1/10
- **Impact:** 8/10
- **Feasibility:** 7/10
- **Confidence:** 6/10

## Summary

Wrap forest/model forward passes with `torch.compile(mode="reduce-overhead")`. Fuses elementwise ops, batches kernel launches, unrolls depth loops at trace time. Addresses the primary bottleneck (438 kernel launches → ~50-100).

## Implementation

```python
# In train.py, after model creation:
model = torch.compile(model, mode="reduce-overhead")
```

May need to:
1. Remove dynamic `if x.dim() == 2` branches (use fixed input shapes)
2. Gate `_cached_decisions` / `_cached_leaf_probs` behind a flag to avoid graph breaks
3. Ensure OPT-02 (unrolled loops) is applied first for best fusion

## Expected Effect

- **Speedup:** 1.5-3x (CUDA); 1.2-1.5x (MPS); 1.0x if graph breaks
- **Accuracy delta:** 0.0pp (exact)

## Risks

- Graph breaks on MPS backend (project currently runs on Mac)
- Dynamic shapes from variable sequence lengths may cause recompilation
- `_cached_decisions` side-effect may break tracing

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — remove dynamic branches, add compile-friendly caching
- `/Users/goldy/tree-based-llm/train.py` — add `torch.compile()` wrapper

## Dependencies

Benefits greatly from OPT-02 (unrolled loops enable full fusion).
