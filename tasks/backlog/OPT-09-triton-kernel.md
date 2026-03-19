# OPT-09: Custom Triton Kernel for Forest Forward

- **Category:** kernel-fusion
- **Priority:** 6.5/10
- **Impact:** 10/10
- **Feasibility:** 3/10
- **Confidence:** 8/10

## Summary

Fuse the entire forest computation (routing → sigmoid → leaf probs → output einsum → gating → reduction) into a single GPU kernel. Eliminates ALL intermediate memory traffic. 438 kernel launches → 1.

## Expected Effect

- **Speedup:** 5-10x on forest forward (approaching standard transformer speed)
- **Accuracy delta:** 0.0pp (exact computation, just fused)

## Risks

- Requires Triton expertise (~100-200 lines of Triton code)
- CUDA only — does NOT work on Mac MPS
- Backward pass needs separate kernel or autograd.Function
- High maintenance burden

## Files to Modify

- New file: `triton_forest.py`
- `/Users/goldy/tree-based-llm/main.py` — integrate as optional backend

## Dependencies

Requires CUDA hardware. Conceptually similar to Flash Attention's approach.
