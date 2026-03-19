# OPT-11: Depth-Alternating Trees

- **Category:** architecture
- **Priority:** 7.8/10
- **Impact:** 6/10
- **Feasibility:** 10/10
- **Confidence:** 8/10

## Summary

Use tree projections in odd-numbered layers only, standard linear in even layers. Exact 2x speedup with ~5 lines changed.

## Implementation

```python
# In TreeTransformer.__init__:
for i in range(n_layers):
    layer_proj = proj_type if i % 2 == 1 else "linear"
    self.layers.append(TreeTransformerBlock(..., proj_type=layer_proj, ...))
```

## Expected Effect

- **Speedup:** 2.0x
- **Accuracy delta:** -0.2 to -0.5pp

## Risks

- May lose the accuracy advantage that makes trees worthwhile
- Which layers benefit most from trees is an empirical question

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `TreeTransformer.__init__` (lines 389-394)

## Dependencies

None. Simplest possible speedup.
