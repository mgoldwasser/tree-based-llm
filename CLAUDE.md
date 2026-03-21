# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch research implementation of Tree-Based Attention for transformers. Replaces dense linear Q/K/V projections with differentiable soft decision trees, trained end-to-end via backpropagation. Two files: `main.py` (model) and `benchmark.py` (comparison vs standard attention).

## Commands

```bash
# Install dependencies (only PyTorch required)
pip install torch

# Run the demo (trains a classification model on synthetic data)
python main.py

# Run the full benchmark (4 models × 2 tasks × 1000 steps, ~5-10 min on CPU)
python benchmark.py

# Train best speed/accuracy tree model (~49ms/step, 29.5% val acc)
python train.py --fast --model boosted_alt

# Train best accuracy tree model (~108ms/step, 29.8% val acc)
python train.py --fast --model boosted

# Train all models on Shakespeare (fast config)
python train.py --fast --model all

# Disable torch.compile if needed
python train.py --fast --model boosted_alt --no-compile

# Run compound optimization experiment
python run_compound_experiment.py

# Run speed/accuracy check on fast configs
python run_speed_accuracy_check.py

# Phase A experiments (core question: can trees win at matched params?)
python run_matched_params.py          # NEW-02: Matched-parameter comparison
python run_depth_ablation.py          # NEW-06: Depth ablation study
python run_micro_tree.py              # NEW-01: Micro-tree experiment

# Phase B experiments (fundamental improvements)
python run_contextual_routing.py      # NEW-05: Context-aware routing
python run_bpe_experiment.py          # NEW-08: BPE vs char-level tokenization
python run_scaling_grid.py            # NEW-07: Scaling study (where trees win)

# Phase C experiments (practical deployment)
python run_hard_routing.py            # NEW-03: Hard routing at inference
python run_adapters.py                # NEW-04: Trees-as-adapters

# All experiment scripts support: --full (full config), --no-compile
```

There is no test suite, linter, or build system configured.

## Architecture

All model components live in `main.py`, layered bottom-up:

1. **BatchedTreeForest** — Core primitive replacing `nn.Linear`. Stores ALL tree parameters in stacked tensors `(n_trees, n_internal, input_dim)` and computes all trees in a single batched einsum pass. Features: per-node learnable temperature (modulates global schedule), input-dependent tree gating via `gate_proj` (MoE-style, replaces fixed tree weights). Caches routing decisions and leaf probs for regularization losses.

2. **ObliviousTreeForest** — NODE-style oblivious trees: all nodes at the same depth share one hyperplane. Decision weights: `(n_trees, depth, input_dim)` — fewer params than standard trees. Leaf probs computed via outer product (eliminates depth-dependent gradient vanishing). Same features as BatchedTreeForest (per-node temp, input-dependent gating).

3. **LinearPlusForest / ObliviousLinearPlusForest** — Linear base projection + single wide forest for nonlinear correction: `output = base_linear(x) + shrinkage * forest(x)`. The linear base provides stable gradients and preserves residual stream structure; the forest adds input-adaptive nonlinear refinement.

3b. **MicroTreeForest** — Minimal-overhead trees: depth 1-2, few trees (2-8), with low-rank leaf factorization (`leaf_down @ leaf_up`, rank << output_dim). Designed for <10% param overhead over nn.Linear. Uses oblivious-style routing. **LinearPlusMicroTree** combines a linear base with micro-tree correction.

3c. **ContextualRoutingForest** — Oblivious forest with context-aware routing: `routing_input = x + context_proj(ema_context)` where EMA context captures recent hidden state history. Addresses limitation that standard trees route on single tokens without context. **LinearPlusContextual** adds a linear base.

3d. **Speed-optimized projections** — Alternative approaches to input-dependent computation with lower overhead than trees:
   - **GatedProjection** — GLU-style: `W(x) * σ(V(x))`. Equivalent to depth-1 tree as parallel matmuls. Stack multiple gates for more "depth".
   - **DynamicLinear** — Base + per-token low-rank modulation: `W(x) + shrinkage * (x @ W_down) @ W_up`. What trees reduce to without routing. ~1.3x overhead.
   - **LowRankRoutingForest** — Oblivious forest with low-rank routing projection (r=16 vs d=128). Reduces routing einsum cost by d/r factor.
   - **RecursiveProjection** — Apply `gate * W_L(h) + (1-gate) * W_R(h)` K times, reusing parameters. Depth-K nonlinearity with depth-1 param count.
   - **ChunkedRoutingForest** — Share routing decisions across token chunks. Routes on chunk means, broadcasts to all tokens in chunk.
   - **ProductKeyProjection** — Hash-based leaf selection via split codebooks. O(sqrt(n_leaves)) routing cost.
   All have `LinearPlus*` variants (linear base + correction).

4. **`make_projection()` factory** — Creates projection layers by type string. Supports: `"linear"`, `"batched"`, `"boosted"`, `"oblivious"`, `"oblivious_boosted"`, `"micro_tree"`, `"micro_boosted"`, `"contextual"`, `"contextual_boosted"`, `"gated"`, `"gated_boosted"`, `"dynamic"`, `"dynamic_boosted"`, `"lowrank_routing"`, `"lowrank_boosted"`, `"recursive"`, `"recursive_boosted"`, `"chunked"`, `"chunked_boosted"`, `"product_key"`, `"product_key_boosted"`. All higher-level components use this factory via `proj_type` parameter.

5. **TreeAttention** — Multi-head attention with **separate** Q, K, V forests (unfused routing — each learns independent routing patterns). QK-norm (LayerNorm on Q and K after projection) stabilizes attention logits despite routing shifts during training. Output projection is also a tree forest.

6. **TreeTransformerBlock** — Pre-norm transformer block with TreeAttention. Optional `use_tree_ffn` flag to also replace FFN linear layers with tree projections.

7. **TreeTransformer** — Full model with embedding, positional encoding, stacked blocks, and a classification or language modeling head. `proj_type` parameter selects the projection type throughout.

8. **Utilities**:
   - `tree_regularization_loss()` — Entropy-based regularization with depth-decay weighting (2^-d). Positive lambda sharpens routing. Default: lambda=0.005.
   - `leaf_balancing_loss()` — MoE-style load-balancing loss on leaf utilization. Penalizes leaf probability concentration: `L = alpha * n_leaves * mean(sum(p_l^2))`.
   - `set_temperature()` / `get_routing_entropy()` — Temperature annealing and monitoring.
   - `make_optimizer()` — Tree-aware param groups: decision weights + node temps at 3x LR with no weight decay; gate/leaf/other params at standard LR.
   - `count_parameters()` — Reports total vs tree-specific parameter counts.
   - `freeze_non_tree_params()` / `unfreeze_all_params()` — For adapter-style fine-tuning: freeze everything except tree parameters.
   - `set_hard_routing(model, enabled, top_k)` — Toggle hard routing (top-k leaf selection) on all tree modules. For inference-time sparsification.

## Design Decisions

**Why this works with gradient descent:** The sigmoid routing at each tree node is the key. In a hard decision tree, you take a binary left/right path — that's non-differentiable. Here, every input "flows" through all paths simultaneously with soft probabilities, so the gradient is well-defined everywhere.

**Soft routing is the feature, not a compromise:** With soft routing, the forest computes an input-adaptive linear projection — a different effective W for every input, constructed as a weighted combination of leaf matrices. This is strictly more expressive than a single linear layer. Hard routing (entropy collapse) reduces trees to piecewise-constant lookup tables with 8x fewer effective parameters. Temperature annealing should be conservative (floor ~0.7) to preserve this advantage.

**Oblivious trees (NODE-style):** All nodes at the same depth share one hyperplane split. Leaf probs are computed as outer products of per-depth choices, eliminating the multiplicative gradient chain through sigmoid derivatives. Each depth level gets independent gradients. Fewer parameters: `(n_trees, depth, input_dim)` vs `(n_trees, 2^depth-1, input_dim)`.

**Unfused QKV routing:** Q ("what am I looking for?"), K ("what do I contain?"), and V (content) are fundamentally different functions. Shared routing forces identical tree paths for all three, limiting expressiveness. Separate forests allow each to specialize. The Linear+Forest gap (27.6% vs 26.7% batched) was partly explained by fused routing.

**QK-norm:** Tree projections have unpredictable output scale during training (routing shifts change which leaves dominate). LayerNorm on Q and K after projection stabilizes attention logit magnitudes, following Gemma 2 / ViT-22B practice.

**Input-dependent tree gating:** Replaces fixed `softmax(tree_weights)` with `softmax(gate_proj(x))`. Each token routes to different tree mixtures, analogous to MoE expert gating. Trees already provide input-dependence through internal routing; gating adds tree-level specialization.

**Per-node learnable temperature:** Each internal node learns its own temperature factor via `softplus(logit + 0.5413)` (initializes to 1.0). Root might want soft routing (explore both subtrees) while leaf-adjacent nodes want sharp (crisp final selection). Modulates the global temperature schedule.

**Leaf balancing loss (MoE analog):** `L = alpha * n_leaves * mean(sum(mean_prob_l^2))`. Prevents routing collapse where 1-2 leaves get all mass. Uniform leaves: loss = alpha * 1.0. Collapsed to 1 leaf: loss = alpha * n_leaves.

**Init std = 0.1 (up from 0.02):** With d_model=64, routing logits have std ~0.8, sigmoid range [0.31, 0.69]. Trees differentiate routing from step 1 instead of spending hundreds of steps learning to differentiate.

**Temperature annealing:** Controls the sharpness of routing decisions. Held at 1.0 (fully soft) for the first 50% of training, then cosine annealed from 1.0 → 0.7 over the second half. Use `set_temperature(model, t)` in the training loop.

**LR warmup + cosine decay:** 100-step linear warmup stabilizes Adam's variance estimates (critical with 3x LR multiplier for decision params). Cosine decay to 10% of peak lets the model settle into sharper minima. Applied via LambdaLR scheduler in train.py.

**Dropout = 0.0:** Model is underfitting (train/val gap ≈ 0), so dropout is pure noise. Removed by default.

**Entropy regularization:** Default lambda=0.005 (mild sharpening pressure). Uses depth-decay weighting (2^-d) so deeper nodes aren't over-regularized.

**Batched computation:** All trees in a forest are computed via a single `einsum('bsd,tnd->bstn', x, decision_weights)` — no Python loop. Oblivious trees further reduce this to `(n_trees, depth, input_dim)`.

**LinearPlusForest:** Linear base preserves residual stream structure; forest adds nonlinear correction. This explains the boosted-vs-batched accuracy gap.

**Practical tradeoffs:** Parameter count scales as `O(n_trees × 2^depth × (input_dim + output_dim))` for standard trees, `O(n_trees × (depth × input_dim + 2^depth × output_dim))` for oblivious. Recommended configs: depth 3 with 12 trees per forest for batched/oblivious, or 24 trees × depth 3 for boosted.

**Micro-trees (low-rank leaves):** Replace full leaf matrices `(n_leaves, output_dim)` with factored `leaf_down @ leaf_up` where `leaf_down: (n_leaves, rank)` and `leaf_up: (rank, output_dim)`. With depth=1, 4 trees, rank=8: ~10% param overhead over nn.Linear at d_model=128. The key insight: very shallow, very few trees can be parameter-efficient — the routing overhead drops from ~45% to ~10% of params.

**Contextual routing:** Standard trees route on the raw token embedding — "bank" routes identically in "river bank" vs "bank account". Contextual routing uses EMA of recent hidden states: `routing_input = x + context_proj(ema_context)`. Vectorized via power-series cumsum (no Python loop, torch.compile compatible).

**Hard routing at inference:** Train with full soft routing (all leaves get gradients), evaluate with top-k hard routing (only k leaves contribute). At depth 3, top-2 hard routing eliminates 75% of leaf computation. Key for making trees practical.

**Trees-as-adapters:** Pretrain a standard transformer (fast), freeze linear weights, add micro-tree corrections. Trees learn the *residual* — what linear projections miss. Much smaller trees suffice. This is LoRA with learned input-dependent routing instead of fixed low-rank decomposition.

## Key Hyperparameters

- `proj_type`: `"batched"`, `"boosted"`, `"oblivious"`, `"oblivious_boosted"`, `"micro_tree"`, `"micro_boosted"`, `"contextual"`, `"contextual_boosted"`
- `n_trees`: Trees per forest for batched/oblivious (default: 12), micro-tree (default: 4)
- `boosted_trees`: Trees in LinearPlusForest (default: 24)
- `leaf_rank`: Low-rank leaf factorization rank for MicroTreeForest (default: 8)
- `ema_decay`: EMA decay for contextual routing (default: 0.9)
- `tree_depth` / `boosted_depth`: Tree depth (default: 3)
- `temperature`: Routing softness — held at 1.0 for first 50%, then annealed 1.0 → 0.7
- `use_tree_ffn`: Whether FFN layers also use tree projections (default: True)
- `dropout`: Default 0.0 (model underfits)
- Entropy reg lambda: 0.005 (mild sharpening)
- Leaf balance alpha: 0.01

## Benchmark

`benchmark.py` compares 4 models on 2 synthetic tasks:

**Models:** Standard Transformer, Batched Forest (attn), Boosted Forest (attn), Boosted Forest (full)

**Tasks:**
- Linear: `next = (prev + offset) % vocab` — favors linear models
- Non-linear: `next = lookup_table[prev XOR prev_prev]` — tests non-linear capacity

**Metrics:** Next-token accuracy (primary), loss, routing entropy

## Train

`train.py` compares 5 models on Shakespeare character-level LM:

**Models:** Standard Transformer, Batched Forest, Linear+Forest, Oblivious Forest, Oblivious Linear+Forest

**Training:** LR warmup (100 steps) + cosine decay to 10%. Temperature annealing (1.0 → 0.7 cosine). Entropy reg (lambda=0.005) + leaf balancing (alpha=0.01).
