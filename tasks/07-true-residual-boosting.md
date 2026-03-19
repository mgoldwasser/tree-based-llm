# Task 07: True Residual Boosting

**Status:** Pending
**Priority:** High — addresses the core "are trees fitting to residuals?" question
**Depends on:** None
**Files:** `main.py` (BoostedForest class)

## Problem
Current BoostedForest stages all see the original input `x`. This means every stage independently tries to solve the same problem — they're parallel ensembles, not true boosted stages. In gradient boosting, each stage fits to the residual error of all previous stages, which is fundamentally more powerful.

## Design

### Option A: True Residual Boosting (recommended)
Each stage receives the residual between the target projection and what previous stages produced:

```python
def forward(self, x):
    output = self.base_proj(x)                    # Stage 0: linear base
    for i, stage in enumerate(self.stages):
        # Each stage sees original features but learns to correct
        # the current output's shortcomings
        residual_input = torch.cat([x, output.detach()], dim=-1)  # or just x
        correction = stage(x)                      # fit to implicit residual
        output = output + self.shrinkage[i] * correction
    return self.norm(output)
```

### Option B: Explicit Residual Targets (gradient-boosting style)
During training, compute pseudo-residuals and train each stage separately:
- Too complex for differentiable end-to-end training
- Better suited for a separate boosting framework, not within attention

### Option C: Feature Pass-Through (hybrid)
Concatenate original input with running output at each stage, giving later stages access to both raw features and what's been predicted so far:

```python
def forward(self, x):
    output = self.base_proj(x)
    for i, stage in enumerate(self.stages):
        # Concatenate original features with current output
        stage_input = torch.cat([x, output], dim=-1)
        correction = stage(stage_input)  # stage has (input_dim + output_dim) features
        output = output + self.shrinkage[i] * correction
    return self.norm(output)
```

This requires each stage's `BatchedTreeForest` to accept `input_dim + output_dim` inputs instead of `input_dim`. More parameters but gives each stage the "residual" implicitly through the current output.

## Recommendation
Start with Option C (feature pass-through). It's the simplest change that gives stages access to the residual information, and the tree routing can naturally learn to focus on the output dimensions where correction is needed.

## Complexity Impact
- Option A: No parameter change, ~same speed
- Option C: Each stage gets `input_dim + output_dim` features → ~2x routing weights per stage
- No change to depth loop or leaf computation — only the input einsum is wider

## Verification
- Compare accuracy of residual-boosted vs current parallel-boosted on linear task
- Residual version should converge faster (each stage has a clearer objective)
