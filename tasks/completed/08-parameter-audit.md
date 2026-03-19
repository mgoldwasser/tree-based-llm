# Task 08: Parameter & Assumption Audit

**Status:** Completed

## What was done
Fixed 4 key assumptions that don't hold for tree-based models:

1. **Init:** Changed decision_weights from xavier_normal_ to normal_(0, 0.02) for exploratory early routing
2. **Optimizer:** Added `make_optimizer()` with 3 param groups — decision weights get 3x LR and no weight decay
3. **Weight decay:** Disabled on decision_weights (conflicts with entropy regularization)
4. **Redundant LayerNorm:** Added `use_norm=False` flag, disabled inside BoostedForest and when used in transformer blocks

## Result
Demo accuracy jumped from ~50% to 93.8% at step 10 — the small init + higher routing LR makes a dramatic difference.
