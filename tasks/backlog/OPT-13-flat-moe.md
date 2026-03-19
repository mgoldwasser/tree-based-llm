# OPT-13: Flat MoE Approximation

- **Category:** architecture
- **Priority:** 6.9/10
- **Impact:** 7/10
- **Feasibility:** 8/10
- **Confidence:** 6/10

## Summary

Replace tree routing with direct softmax gating over K=8 experts (matching 8 leaves). Answers the fundamental question: "does the tree's hierarchical structure add value, or is it just a structured MoE?"

## Implementation

New `FlatMoEProjection` class:

```python
class FlatMoEProjection(nn.Module):
    def __init__(self, input_dim, output_dim, n_experts=8):
        self.gate = nn.Linear(input_dim, n_experts)
        self.experts = nn.Parameter(torch.empty(n_experts, input_dim, output_dim))
        # ... init ...

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=-1)  # (B, S, K)
        expert_out = torch.einsum('bsd,kdo->bsko', x, self.experts)  # (B, S, K, O)
        return torch.einsum('bsko,bsk->bso', expert_out, weights)
```

## Expected Effect

- **Speedup:** 2-3x (simpler gating, no tree traversal)
- **Accuracy delta:** -0.5 to +0.5pp (genuinely uncertain)

## Risks

- If flat MoE matches tree accuracy, it invalidates the tree premise
- If tree beats flat MoE, it validates hierarchical routing

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — new class; update `make_projection` factory
- `/Users/goldy/tree-based-llm/train.py` — new model config

## Dependencies

None. Important ablation experiment.
