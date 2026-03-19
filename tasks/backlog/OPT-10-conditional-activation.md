# OPT-10: Conditional Forest Activation (Token-Level Gating)

- **Category:** architecture
- **Priority:** 5.3/10
- **Impact:** 5/10
- **Feasibility:** 6/10
- **Confidence:** 5/10

## Summary

Per-token gate decides whether to run the forest or use only the linear base. "Easy" tokens (~70%) skip the forest entirely. Requires LinearPlusForest mode.

## Expected Effect

- **Speedup:** 1.5-2x if 70% tokens skip; 1.0x with soft gating
- **Accuracy delta:** -0.1 to -0.3pp

## Risks

- Hard gating requires sparse ops or dynamic shapes (GPU unfriendly)
- Soft gating (always compute but scale by gate) provides no speed benefit
- Unclear what fraction of tokens are truly "easy"

## Files to Modify

- `/Users/goldy/tree-based-llm/main.py` — `LinearPlusForest` class
