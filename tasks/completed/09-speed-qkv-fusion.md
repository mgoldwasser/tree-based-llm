# Task 09: Speed — QKV Fusion

**Status:** Completed

## What was done
Fused Q/K/V into a single forest that outputs 3*d_model, split into Q, K, V. Shares routing decisions across all three projections.

## Key changes
- `TreeAttention.qkv_proj`: single forest outputting `d_model * 3`
- `Q, K, V = qkv.chunk(3, dim=-1)` after the forest
- Reduces 3 routing einsums → 1 (the expensive part)
- Separate `o_proj` for output projection (different input = post-attention context)

## Expected speedup
~2.5x on attention projection computation. The routing einsum (the bottleneck) runs once instead of 3 times.
