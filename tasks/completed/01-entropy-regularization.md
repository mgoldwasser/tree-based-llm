# Task 01: Entropy-Based Regularization

**Status:** Completed

## What was done
Replaced the original weight-magnitude proxy (`-lambda * |W|.mean()`) with a direct entropy penalty on actual routing decisions cached during the forward pass.

## Key changes
- `SoftDecisionTree._cached_decisions` stores sigmoid outputs during forward
- `tree_regularization_loss()` computes binary entropy `H(p) = -(p*log(p) + (1-p)*log(1-p))` on cached decisions
- Minimizing entropy pushes routing toward 0 or 1 (crisp splits)

## Result
Entropy drops from ~0.54 → ~0.06 over training, confirming crisp split emergence.
