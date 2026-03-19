# OPT-02: Unrolled Oblivious Outer Product

- **Category:** math-reformulation
- **Priority:** 6.9/10
- **Impact:** 3/10
- **Feasibility:** 10/10
- **Confidence:** 9/10

## Summary

Unroll the depth-3 outer product loop in `ObliviousTreeForest._compute_leaf_probs_outer` into explicit element-wise products. Eliminates dynamic reshaping and makes the code trivially fusible by `torch.compile`.

## Implementation

Replace the loop with explicit depth-3 computation:

```python
d0, d1, d2 = decisions[..., 0], decisions[..., 1], decisions[..., 2]
p0 = torch.stack([d0, 1-d0], dim=-1)       # (B,S,T,2)
p1 = torch.stack([d1, 1-d1], dim=-1)       # (B,S,T,2)
p2 = torch.stack([d2, 1-d2], dim=-1)       # (B,S,T,2)
leaf_probs = (p0.unsqueeze(-1).unsqueeze(-1)
            * p1.unsqueeze(-2).unsqueeze(-1)
            * p2.unsqueeze(-2).unsqueeze(-2)).reshape(B, S, T, 8)
```

## Expected Effect

- **Speedup:** 1.05x standalone; 1.2x when combined with torch.compile
- **Accuracy delta:** 0.0pp (exact)

## Risks

- Almost no standalone value; primary value is as prerequisite for OPT-03

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `ObliviousTreeForest._compute_leaf_probs_outer`

## Dependencies

Prerequisite for OPT-03 (torch.compile) to achieve full fusion.
