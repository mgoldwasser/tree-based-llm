# Task 05: BoostedForest

**Status:** Completed

## What was done
Implemented multi-stage ensemble: `output = base_linear(x) + sum(shrinkage[i] * forest_i(x))`. Linear base + tree corrections.

## Key changes
- `BoostedForest` with `base_proj` (nn.Linear) + N `BatchedTreeForest` stages
- Learned `shrinkage` factors initialized at 0.1
- All stages see original input (not residuals — see Task 07 for true residual boosting)

## Result
Boosted Forest (attn) won the linear task at 20.8% vs Standard's 19.4%. The linear base handles the core pattern, trees add corrections.

## Limitation
Stages don't fit to residuals — they all see the same input `x`. True gradient boosting would be more effective (see Task 07).
