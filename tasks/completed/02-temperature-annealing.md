# Task 02: Temperature Annealing

**Status:** Completed

## What was done
Added `set_temperature(model, t)` utility and cosine annealing schedule in training loop: 1.0 → 0.1 over training.

## Key changes
- `set_temperature()` iterates over all `BatchedTreeForest` modules
- `get_routing_entropy()` monitors decision crispness
- Cosine schedule: `temp = 0.1 + 0.9 * (1 + cos(π * progress)) / 2`

## Result
Smooth gradient flow early, crisp tree-like behavior at convergence.
