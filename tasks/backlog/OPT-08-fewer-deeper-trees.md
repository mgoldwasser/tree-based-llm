# OPT-08: Fewer Trees + Deeper + Low-Rank Leaves

- **Category:** architecture
- **Priority:** 6.9/10
- **Impact:** 7/10
- **Feasibility:** 8/10
- **Confidence:** 6/10

## Summary

Replace 12 trees depth-3 (96 leaves, 12 kernel groups) with 4 trees depth-4 (64 leaves, 4 kernel groups). 1/3 the kernel launches with similar expressiveness. Combined with OPT-07 (low-rank leaves) to control parameter explosion at depth 4.

## Expected Effect

- **Speedup:** 2-3x (from reduced kernel launches)
- **Accuracy delta:** -0.0 to -0.5pp (uncertain — fewer trees = less ensemble diversity)

## Risks

- 4 trees may underfit the gating mechanism
- Depth 4 increases the leaf prob loop by 1 iteration (mitigated by OPT-01 or OPT-02)

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — hyperparameter defaults
- `/Users/goldy/tree-based-llm/train.py` — model configs

## Dependencies

OPT-07 (low-rank leaves) helps control parameter explosion at depth 4.
