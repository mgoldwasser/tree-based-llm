# Task 03: Log-Space Leaf Probabilities

**Status:** Completed

## What was done
Switched `_compute_leaf_probabilities` to accumulate in log-space, preventing vanishing products through deep trees.

## Key changes
- `_compute_leaf_log_probabilities()` accumulates `log(p)` instead of multiplying `p`
- Final `exp()` converts back to probabilities for the leaf output einsum

## Result
Numerically stable through arbitrary depth. Enables deeper trees without gradient vanishing.
