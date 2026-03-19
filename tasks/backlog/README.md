# Scaling Optimization Backlog

Prioritized backlog of optimizations to close the speed gap between tree-based and standard transformers while preserving the accuracy advantage.

## Current State (Shakespeare, d_model=64, 2 layers, 2000 steps)

| Model | Val Acc | ms/step | Slowdown |
|-------|---------|---------|----------|
| Standard Transformer | 28.6% | 35ms | 1.0x |
| Batched Forest | 27.7% | 223ms | 6.4x |
| Linear+Forest | **29.7%** | 385ms | 11.0x |
| Oblivious Forest | 27.8% | 174ms | 5.0x |
| Oblivious Linear+Forest | 29.4% | 343ms | 9.8x |

## Root Cause

The 5-11x slowdown is **not** from increased FLOPs (only 1.2-1.5x more). It's from:
- **438 kernel launches per block** vs 50 for standard (Python loops, unfused ops)
- **Memory bandwidth**: materializing (B,S,T,D) per-tree intermediates
- **4 separate forest calls** for Q/K/V/O vs 1 fused GEMM

## Ranked Backlog

| Rank | ID | Name | Priority | Speedup | Acc Delta | Effort |
|------|--------|------|----------|---------|-----------|--------|
| 1 | OPT-01 | Vectorized Leaf Probs | 7.8 | 1.3-1.5x | 0.0pp | Low |
| 2 | OPT-11 | Depth-Alternating Trees | 7.8 | 2.0x | -0.2-0.5pp | Very Low |
| 3 | OPT-04 | Shared Routing / Fused QKV | 7.5 | 2-3x | -0.0-0.5pp | Medium |
| 4 | OPT-05 | Selective Trees (V+O only) | 7.5 | 2.0x | -0.0-0.3pp | Low |
| 5 | OPT-03 | torch.compile | 7.1 | 1.5-3x | 0.0pp | Low |
| 6 | OPT-02 | Unrolled Oblivious Outer Product | 6.9 | 1.05x | 0.0pp | Very Low |
| 7 | OPT-08 | Fewer Trees + Deeper + Low-Rank | 6.9 | 2-3x | -0.0-0.5pp | Medium |
| 8 | OPT-13 | Flat MoE Comparison | 6.9 | 2-3x | +/-0.5pp | Medium |
| 9 | OPT-09 | Custom Triton Kernel | 6.5 | 5-10x | 0.0pp | High |
| 10 | OPT-07 | Low-Rank Leaf Factorization | 6.5 | 1.2-1.5x | -0.0-0.2pp | Low |
| 11 | OPT-15 | KV-Cache for Inference | 6.2 | O(S)x gen | 0.0pp | Medium |
| 12 | OPT-06 | Materialization-Free Output | 5.5 | 1.1-1.2x | 0.0pp | Medium |
| 13 | OPT-10 | Conditional Forest Activation | 5.3 | 1.5-2x | -0.1-0.3pp | Medium |
| 14 | OPT-14 | Forest-as-Adapter | 5.2 | 1x pretrain | Unknown | Medium |
| 15 | OPT-16 | Distilled Tree Knowledge | 4.6 | Near 1x | -0.2-0.5pp | High |
| 16 | OPT-17 | Hadamard-Structured Leaves | 4.5 | 1.0x | -0.1-0.3pp | Medium |
| 17 | OPT-12 | Polynomial/Walsh-Hadamard View | 3.6 | 1.0-1.5x | -0.0-1.0pp | High |

## Synergistic Combos

- **Combo A "Low-Hanging Fruit"**: OPT-01 + OPT-02 + OPT-03 + OPT-11 → ~4x speedup, ~96ms/step
- **Combo B "Architecture-First"**: OPT-05 + OPT-03 + OPT-08 → ~5x speedup, ~77ms/step
- **Combo C "Shared Everything"**: OPT-04 + OPT-03 + OPT-07 → ~6x speedup, ~64ms/step
- **Combo D "Nuclear Option"**: OPT-09 + OPT-04 → ~10x speedup, ~39ms/step (CUDA only)

## Individual backlog items in this directory

Each OPT-XX.md file contains full details: category, scoring, implementation notes, risks, and file paths.
