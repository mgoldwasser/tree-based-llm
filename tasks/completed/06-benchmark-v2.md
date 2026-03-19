# Task 06: Benchmark v2 — Accuracy, 1000 Steps, Multi-Model

**Status:** Completed

## What was done
Rewrote benchmark with accuracy as primary metric, 1000 training steps, 4 model configs, 2 data tasks (linear and non-linear patterns).

## Results (Linear Task)
| Model | Accuracy | Time | Params |
|-------|----------|------|--------|
| Standard Transformer | 19.4% | 27s | 137K |
| Batched Forest (attn) | 18.0% | 107s | 198K |
| Boosted Forest (attn) | **20.8%** | 217s | 272K |
| Boosted Forest (full) | 19.0% | 361s | 439K |

## Results (Non-Linear XOR Task)
All models ~2% accuracy (near random). Task too hard for 64-dim models.

## Key Insight
Total runtime ~20 minutes on CPU. Boosted Forest (full) alone takes 6+ minutes per task. Need GPU or simpler benchmark configs to iterate faster.
