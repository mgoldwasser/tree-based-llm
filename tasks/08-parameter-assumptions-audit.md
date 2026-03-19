# Task 08: Parameter & Assumption Audit for Tree-Based Models

**Status:** Pending
**Priority:** High — foundational for all future work
**Depends on:** None
**Files:** `main.py`, `benchmark.py`

## Problem
We're using hyperparameters designed for standard transformers on a fundamentally different architecture. Trees have different gradient dynamics, different capacity scaling, and different optimization landscapes. We need to audit every assumption.

## Assumptions That May Not Hold

### 1. Xavier Initialization
- **Current:** `xavier_normal_` on decision_weights and leaf_outputs
- **Problem:** Xavier assumes linear layers with specific fan-in/fan-out. Trees have multiplicative path probabilities and sigmoid nonlinearities at every node.
- **Impact:** Decision weights initialized too large → sigmoid saturation from step 1. Too small → all paths equally likely (no routing signal).
- **Fix:** Initialize decision_weights with smaller magnitude (e.g., `normal_(0, 0.01)`) so early routing is near 0.5 (exploratory). Initialize leaf_outputs with Xavier (these act more like linear projections).

### 2. AdamW Optimizer
- **Current:** AdamW with lr=3e-4, weight_decay=0.01
- **Problem:** Adam's adaptive learning rate divides by the running variance of gradients. Tree routing gradients have very different variance characteristics than matrix multiply gradients — they flow through sigmoid chains and probability products.
- **Impact:** Decision weights and leaf weights may need different learning rates.
- **Fix:** Try parameter group-specific learning rates:
  ```python
  optimizer = AdamW([
      {'params': decision_params, 'lr': 1e-3},   # routing needs faster learning
      {'params': leaf_params, 'lr': 3e-4},        # leaves are more like linear weights
      {'params': other_params, 'lr': 3e-4},       # embeddings, norms, etc.
  ])
  ```

### 3. Learning Rate
- **Current:** 3e-4 (standard for small transformers)
- **Problem:** Trees with temperature annealing have a non-stationary optimization landscape. Early training (high temp) needs different LR than late training (low temp).
- **Fix:** Warmup + cosine decay, or separate LR scheduler that accounts for temperature phase.

### 4. Gradient Clipping at 1.0
- **Current:** Global norm clipping at 1.0
- **Problem:** Tree gradients flow through depth products. At depth 3, gradient magnitude is ~(0.25)^3 = 0.016 of linear gradient. Global clipping may be too aggressive for tree params while too loose for others.
- **Fix:** Per-parameter-group clipping, or no clipping on tree params (they're already small).

### 5. Weight Decay
- **Current:** 0.01 on all parameters
- **Problem:** Weight decay on decision_weights directly conflicts with the entropy regularization. Entropy reg wants large |w| (for crisp sigmoid), weight decay wants small |w|.
- **Fix:** Exclude decision_weights from weight decay, or use very small weight decay on them.

### 6. LayerNorm Placement
- **Current:** LayerNorm after each forest output, plus pre-norm in transformer blocks
- **Problem:** Double normalization. Forest output → LayerNorm → residual → LayerNorm. This can suppress the tree correction signal.
- **Fix:** Remove LayerNorm from BatchedTreeForest/BoostedForest when used inside a transformer block that already has its own norms.

## Implementation Plan
1. Add parameter groups to optimizer (decision weights vs leaf weights vs other)
2. Test different init scales for decision_weights (0.01, 0.05, 0.1 vs Xavier)
3. Disable weight decay on decision_weights
4. Remove redundant LayerNorm from forest when inside transformer blocks
5. Benchmark each change independently on the linear task (500 steps, fast iteration)

## Expected Impact
These are individually small changes but collectively could be significant. The init + optimizer fixes are most likely to help — wrong initialization can waste hundreds of steps.
