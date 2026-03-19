# OPT-05: Selective Tree Augmentation (V+O Only)

- **Category:** architecture
- **Priority:** 7.5/10
- **Impact:** 7/10
- **Feasibility:** 9/10
- **Confidence:** 7/10

## Summary

Use `nn.Linear` for Q and K projections, keep `LinearPlusForest` for V and O only. Q and K primarily compute geometric similarity (dot-product); V and O are where nonlinear expressiveness matters most.

## Implementation

~5 lines in `TreeAttention.__init__`:

```python
self.q_proj = nn.Linear(d_model, d_model)  # standard linear
self.k_proj = nn.Linear(d_model, d_model)  # standard linear
self.v_proj = make_projection(d_model, d_model, proj_type, **proj_kwargs)  # tree
self.o_proj = make_projection(d_model, d_model, proj_type, **proj_kwargs)  # tree
```

## Expected Effect

- **Speedup:** ~2x on attention projection time (2 forests instead of 4)
- **Accuracy delta:** -0.0 to -0.3pp

## Risks

- May lose accuracy advantage that justifies tree-based attention. Need to validate empirically.

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `TreeAttention.__init__` (lines 291-294)

## Dependencies

None. Alternative to OPT-04 (mutually exclusive approaches).
