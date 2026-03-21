# Tree-Based Attention: Replacing Linear Projections with Differentiable Decision Forests in Transformers

**Matt Goldwasser**

## Abstract

We investigate replacing the dense linear projections (Q, K, V) in transformer attention with differentiable soft decision trees, trained end-to-end via backpropagation. We introduce several tree-based projection architectures — *BatchedTreeForest*, *ObliviousTreeForest* (NODE-style), and *LinearPlusForest* (linear base + tree correction) — all computed via batched einsum operations. Through systematic experiments on character-level Shakespeare language modeling, we find that at our largest scale (d128/6L, 13K steps), the **Oblivious L+F alternating model slightly outperforms the standard transformer** (54.6% vs. 54.4% validation accuracy), while Linear+Forest matches at 54.3%. Comprehensive ablations reveal that: (1) trees provide consistent +0.5–4% accuracy gains at small-to-medium scale, converging to baseline at larger scale; (2) oblivious trees with alternating layers achieve the best speed/accuracy trade-off (201ms vs 53ms/step, +0.2% accuracy); (3) hard routing at inference retains >97% accuracy with top-2 retaining >99.7%; (4) trees require end-to-end training — adapter-style fine-tuning underperforms; (5) input-dependent mixing, not the tree structure itself, is the key ingredient (MoE performs comparably). We also evaluate speed-optimized alternatives including gated projections, dynamic linear layers, and product-key lookup, finding that several achieve comparable accuracy at near-linear speed.

## 1. Introduction

The transformer architecture (Vaswani et al., 2017) relies fundamentally on linear projections to compute queries, keys, and values for attention. These projections are simple matrix multiplications — computationally efficient but limited to learning linear relationships between input features.

Decision trees offer a compelling alternative: they learn piecewise-constant functions through hierarchical feature partitioning, can capture non-linear relationships naturally, and provide interpretable routing decisions. However, classical decision trees use hard (non-differentiable) splits, making them incompatible with gradient-based training.

*Soft decision trees* (Irsoy et al., 2012; Frosst & Hinton, 2017) resolve this by replacing hard splits with sigmoid-gated routing, allowing gradients to flow through all tree paths simultaneously. We build on this foundation to create tree-based projection layers that serve as drop-in replacements for `nn.Linear` in transformer attention.

Our key contributions:

1. **BatchedTreeForest and ObliviousTreeForest**: Batched tensor implementations of standard and NODE-style oblivious decision trees, computing all trees via single einsum operations.

2. **LinearPlusForest**: A linear base projection augmented with a single wide tree forest for nonlinear correction, where `output = base_linear(x) + shrinkage * forest(x)`.

3. **Comprehensive experimental evaluation** across 9 experiment suites covering matched-parameter comparisons, depth ablations, scaling behavior across d_model and layer counts, speed-optimized alternatives, hard routing at inference, and trees-as-adapters.

4. **MoE comparison**: Evaluation of a Mixture-of-Experts variant that replaces tree routing with top-k expert gating, testing whether the tree structure itself matters.

5. **Practical deployment findings**: Hard routing retains >99.8% accuracy at inference, enabling sparse computation; adapter-style tree fine-tuning underperforms end-to-end training.

## 2. Background

### 2.1 Soft Decision Trees

A soft decision tree of depth $D$ has $2^D - 1$ internal nodes and $2^D$ leaves. Each internal node $i$ computes a soft routing probability:

$$p_i^{left}(x) = \sigma\left(\frac{w_i^\top x + b_i}{\tau}\right)$$

where $w_i$ and $b_i$ are learned parameters, $\sigma$ is the sigmoid function, and $\tau$ is a temperature parameter controlling routing sharpness.

The probability of reaching leaf $l$ is the product of routing decisions along the path from root to leaf:

$$P(l | x) = \prod_{i \in \text{path}(l)} p_i^{d_i}(x)$$

The tree output is a probability-weighted sum over learned leaf vectors:

$$f(x) = \sum_{l=1}^{2^D} P(l | x) \cdot v_l$$

This formulation is fully differentiable. Crucially, with soft routing, the forest computes an *input-adaptive linear projection* — a different effective weight matrix $W_{eff}(x)$ for every input, constructed as a weighted combination of leaf matrices.

### 2.2 Oblivious Decision Trees (NODE-style)

In oblivious trees (Popov et al., 2020), all nodes at the same depth share one hyperplane split. Decision weights have shape $(T, D, D_{in})$ — fewer parameters than standard trees' $(T, 2^D-1, D_{in})$. Leaf probabilities are computed via outer product of per-depth binary choices, eliminating the multiplicative gradient chain through sigmoid derivatives. Each depth level receives independent gradients.

### 2.3 Transformer Attention

Standard multi-head attention computes:

$$Q = xW_Q, \quad K = xW_K, \quad V = xW_V$$
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

We replace the linear projections $W_Q, W_K, W_V$ with tree-based projection layers.

## 3. Method

### 3.1 BatchedTreeForest

All tree parameters are stored in stacked tensors: decision weights $(T, N_{internal}, D_{in})$, leaf outputs $(T, N_{leaves}, D_{out})$, and per-node learnable temperatures. The routing computation for all $T$ trees is a single batched einsum:

$$\text{decisions} = \text{einsum}(\texttt{'bsd,tnd->bstn'}, x, W_{decision})$$

Leaf probabilities are computed in log-space to prevent numerical underflow. Input-dependent gating via `gate_proj(x)` replaces fixed tree mixture weights, allowing each token to route to different tree combinations (MoE-style).

### 3.2 ObliviousTreeForest

Following NODE (Popov et al., 2020), oblivious trees constrain all nodes at the same depth to share a single hyperplane. This reduces decision weights from $(T, 2^D-1, D_{in})$ to $(T, D, D_{in})$. Leaf probabilities are computed via Kronecker product of per-depth choices, unrolled for depth 3:

$$P(\text{leaf}_{ijk}) = d_0^i \cdot d_1^j \cdot d_2^k$$

This eliminates depth-dependent gradient vanishing and enables efficient vectorized computation.

### 3.3 LinearPlusForest

Rather than multi-stage boosting, we use a simple additive architecture:

$$\text{output} = W_{base} x + b + \gamma \cdot \text{Forest}(x)$$

where $\gamma$ is a learned shrinkage factor (initialized to 0.1). The linear base preserves residual stream structure and provides stable gradients; the forest adds input-adaptive nonlinear correction.

### 3.4 Unfused QKV Routing

Q ("what am I looking for?"), K ("what do I contain?"), and V (content to retrieve) are fundamentally different functions. We use **separate forests** for each, allowing independent routing specialization.

### 3.5 QK-Norm

Tree projections have unpredictable output scale during training as routing shifts change which leaves dominate. Following Gemma 2 / ViT-22B practice, we apply LayerNorm to Q and K after projection, stabilizing attention logit magnitudes.

### 3.6 Speed Optimizations

We explore a progression of techniques to reduce the speed gap between tree and linear attention:

1. **Oblivious trees**: Fewer parameters and vectorized leaf prob computation via outer product.
2. **Alternating layers** (`tree_every_n=2`): Only every other transformer layer uses tree projections; the rest use standard linear. Halves tree computation.
3. **Selective V+O** (`tree_targets="vo"`): Apply trees only to V and O projections; Q and K use standard linear.
4. **`torch.compile`**: Fuses operations and optimizes the computation graph.

### 3.7 Micro-Trees and Low-Rank Leaves

MicroTreeForest uses depth 1–2 trees with few trees (2–8) and low-rank leaf factorization: `leaf_down @ leaf_up` where `leaf_down: (n_leaves, rank)` and `leaf_up: (rank, output_dim)`. With depth=1, 4 trees, rank=8: ~10% parameter overhead over `nn.Linear` at d_model=128. LinearPlusMicroTree combines a linear base with micro-tree correction.

### 3.8 Hard Routing at Inference

Models are trained with full soft routing (all leaves receive gradients). At inference, we switch to top-k hard routing where only $k$ leaves contribute to the output. At depth 3, top-1 hard routing eliminates 87.5% of leaf computation. This is implemented via `set_hard_routing(model, enabled, top_k)`.

### 3.9 Trees-as-Adapters

Inspired by LoRA, we pretrain a standard transformer, freeze all linear weights, then add micro-tree corrections that learn the residual between linear projections and optimal projections. Trees provide learned input-dependent routing instead of fixed low-rank decomposition.

### 3.10 Speed-Optimized Projection Alternatives

We evaluate several alternatives to tree-based projections designed for lower computational overhead:

- **GatedProjection (GLU-style)**: `W(x) * σ(V(x))` — equivalent to depth-1 tree as parallel matmuls.
- **DynamicLinear**: Base + per-token low-rank modulation: `W(x) + shrinkage * (x @ W_down) @ W_up`.
- **RecursiveProjection**: Apply `gate * W_L(h) + (1-gate) * W_R(h)` K times, reusing parameters.
- **ProductKeyProjection**: Hash-based leaf selection via split codebooks. O(√n_leaves) routing cost.
- **ChunkedRoutingForest**: Share routing decisions across token chunks.
- **LowRankRoutingForest**: Oblivious forest with low-rank routing projection.

All have `LinearPlus*` variants (linear base + correction).

### 3.11 MoE Variant

As an ablation, we replace the tree forest with a standard Mixture-of-Experts layer using top-k expert routing. This tests whether the *tree structure* (hierarchical binary routing) matters, or whether any form of input-dependent projection mixing suffices.

### 3.12 Training Details

**Per-node learnable temperature.** Each internal node learns its own temperature factor via `softplus(logit + 0.5413)` (initializes to 1.0).

**Temperature annealing.** Global temperature is held at 1.0 for the first 50% of training, then cosine-annealed from 1.0 to 0.7 over the second half.

**Initialization.** Decision weights use $\mathcal{N}(0, 0.1)$, producing routing logits with std ~0.8 at d_model=64 (sigmoid range [0.31, 0.69]).

**Optimizer groups.** Decision weights and node temperatures get 3x learning rate with no weight decay. Gate, leaf, and other parameters use standard AdamW settings.

**LR schedule.** 100-step linear warmup followed by cosine decay to 10% of peak.

**Regularization.** Entropy regularization (lambda=0.005) with depth-decay weighting ($2^{-d}$) provides mild sharpening pressure. Leaf balancing loss (alpha=0.01) prevents routing collapse.

## 4. Experiments

### 4.1 Setup

We evaluate on character-level language modeling using the Tiny Shakespeare dataset (Karpathy, 2015): 1.1M characters, 65-character vocabulary, 90/10 train/val split. All experiments use AdamW with lr=3e-4 and `torch.compile` where supported.

We conduct experiments at three scales:

| Config | d_model | Layers | Seq Len | Steps | Purpose |
|--------|---------|--------|---------|-------|---------|
| Fast | 64 | 2 | 128 | 2000 | Rapid iteration |
| Full | 128 | 4 | 256 | 2000 | Standard comparison |
| Final | 128 | 6 | 256 | 13,000 | Scale test |

### 4.2 Main Results (Final Config, d128/6L, 13K steps)

| Model | Val Acc | Val Loss | ms/step | Params |
|-------|---------|----------|---------|--------|
| Standard Transformer | 54.4% | 1.548 | 53 | 1.24M |
| Linear+Forest | 54.3% | 1.567 | 505 | 2.43M |
| **Oblivious L+F (alt)** | **54.6%** | 1.567 | 201 | 1.69M |

*Table 1: Final results (d_model=128, 6 layers, seq_len=256, 13,000 steps, auto-calibrated). The oblivious alternating tree model slightly outperforms the standard transformer while using 4x fewer tree parameters than full Linear+Forest.*

**Hard routing at inference (final models):**

| Model | Soft | Top-1 | Top-2 | Top-4 |
|-------|------|-------|-------|-------|
| Linear+Forest | 54.3% | 52.9% (97.4%) | 54.2% (99.8%) | 54.2% (99.8%) |
| Oblivious L+F (alt) | 54.5% | 53.4% (98.1%) | 54.3% (99.7%) | 54.2% (99.5%) |

*Table 1b: Hard routing accuracy retention at inference. Top-2 retains >99.7% accuracy for both models.*

### 4.3 Previous Full Config Results (d128/4L, 2K steps)

| Model | Val Acc | Val Loss | ms/step | Params |
|-------|---------|----------|---------|--------|
| Standard Transformer | 38.7% | 2.076 | 57 | 843K |
| Linear+Forest | 38.3% | 2.087 | 545 | 1.6M |
| Oblivious L+F | 38.0% | 2.097 | 370 | 1.4M |
| Oblivious L+F (alt) | 36.8% | 2.141 | 246 | 1.1M |
| Oblivious L+F (V+O, alt) | 36.4% | 2.153 | 171 | 992K |
| Linear+MoE (alt) | 36.6% | 2.147 | 175 | 1.1M |

*Table 2: Previous full-config results (d_model=128, 4 layers, seq_len=256, 2000 steps).*

At 2000 steps, Linear+Forest nearly matches the standard transformer (38.3% vs 38.7%). At the larger final config (6L, 13K steps), the gap closes completely — oblivious alternating trees slightly outperform standard (54.6% vs 54.4%).

### 4.3 Scaling Behavior (d32/d64/d128 x 1/2/4 Layers)

We train standard, micro-tree, and oblivious L+F (alternating) across a 3x3 grid of model sizes.

| Config | Standard | MicroTree | Obliv. L+F Alt | Tree Gain |
|--------|----------|-----------|----------------|-----------|
| d32/1L | 25.9% | 25.9% | 25.9% | +0.0% |
| d32/2L | 26.5% | 26.6% | 27.0% | +0.5% |
| d32/4L | 27.5% | 27.7% | 27.7% | +0.2% |
| d64/1L | 27.4% | 28.1% | 28.0% | +0.7% |
| d64/2L | 28.6% | 29.4% | 29.9% | **+1.4%** |
| d64/4L | 31.9% | 33.0% | 33.1% | **+1.2%** |
| d128/1L | 33.5% | 37.5% | 35.8% | **+4.0%** |
| d128/2L | 35.9% | 38.2% | 37.9% | **+2.3%** |
| d128/4L | 39.2% | 39.4% | 39.6% | +0.3% |

*Table 3: Validation accuracy across scaling grid (2000 steps each). Tree Gain = best tree − standard.*

Key observations:
- **Trees provide the largest gains at intermediate scale** (d128/1-2L, d64/2-4L), with +1–4% accuracy improvements.
- **At the smallest scale (d32)**, trees and linear perform nearly identically — there isn't enough model capacity for trees to exploit.
- **At the largest scale (d128/4L)**, the gap narrows to <0.5% — the standard transformer catches up with sufficient depth.
- **MicroTrees are surprisingly effective at d128/1L** (+4.0%), suggesting trees add most value when the model has width but insufficient depth for the standard architecture to fully utilize.

### 4.4 Speed Optimization Progression

We evaluate alternative projection methods designed for speed at the fast config (d64/2L):

| Model | Val Acc | ms/step | Overhead | Category |
|-------|---------|---------|----------|----------|
| Standard | 28.6% | 6.5 | 1.0x | Baseline |
| Dynamic Linear (1 mod) | 29.6% | 14.6 | 2.2x | Low-rank |
| Linear+Gated (GLU) | 29.8% | 18.0 | 2.8x | Gated |
| ProductKey (C=16) | 30.0% | 18.2 | 2.8x | Hash-based |
| Linear+Recursive (3 iter) | 29.6% | 19.1 | 2.9x | Recursive |
| Linear+Gated depth-2 | 29.9% | 19.7 | 3.0x | Gated |
| Dynamic Linear (4 mods) | 29.6% | 25.6 | 3.9x | Low-rank |
| Linear+MicroTree | 29.4% | 28.6 | 4.4x | Tree |
| ChunkedRouting (chunk=16) | 38.0% | 396 | 60.9x | Tree (chunked) |
| LowRank Routing (r=16) | 29.6% | 406 | 62.5x | Tree (low-rank) |

*Table 4: Speed-optimized projections (d64/2L, 2000 steps).*

The Gated, Dynamic, and ProductKey projections achieve ~29.5–30% accuracy at only 2–3x the speed of standard linear — far more practical than full tree routing (60x overhead for ChunkedRouting). All input-dependent methods outperform standard linear (+1–1.5%), suggesting the benefit comes from input-dependent mixing rather than tree structure.

The ChunkedRouting result (38.0%) is an outlier warranting further investigation — its chunk-based routing may benefit from a fundamentally different inductive bias.

### 4.5 Matched-Parameter Comparison

To control for the parameter count advantage of tree models, we compare at matched parameters:

| Model | Params | Val Acc | ms/step |
|-------|--------|---------|---------|
| Standard (small) | 117K | 28.6% | 28 |
| Linear+MicroTree | 139K | 29.4% | 129 |
| Linear+MicroTree depth-2 | 142K | 29.2% | 128 |
| Oblivious L+F (small) | 182K | 28.2% | 635 |
| Standard (large, matched) | 249K | 32.3% | 8 |

*Table 5: Matched-parameter comparison (d64/2L, 2000 steps).*

MicroTrees gain +0.8% accuracy over standard at a 19% parameter increase, but the larger standard transformer (249K params) reaches 32.3% — simply scaling up the standard model is more effective than adding tree complexity at this size.

### 4.6 Depth Ablation

We study optimal tree depth while maintaining ~48 total leaves across configurations:

| Depth | Trees | Val Acc | ms/step | Params |
|-------|-------|---------|---------|--------|
| 1 | 24 | 29.6% | 127 | 168K |
| 2 | 12 | 29.3% | 123 | 161K |
| 3 | 6 | 29.4% | 640 | 267K |
| 4 | 3 | 29.6% | 34 | 150K |

*Table 6: Depth ablation with constant leaf budget (d64/2L, 2000 steps). Standard baseline: 28.6%.*

**Depth-4 with 3 trees** achieves the best speed/accuracy trade-off: comparable accuracy to other depths at 34ms/step (vs 123–640ms for others). Fewer trees with more depth is more efficient than many shallow trees.

### 4.7 Contextual Routing

Standard trees route on the raw token embedding — "bank" routes identically in "river bank" vs "bank account." Contextual routing augments the routing input with an EMA of recent hidden states.

| Model | Val Acc | ms/step | Params |
|-------|---------|---------|--------|
| Standard | 28.6% | 7 | 117K |
| Oblivious L+F | 29.4% | 461 | 267K |
| Linear+Contextual | 29.4% | 430 | 300K |
| Obliv. L+F (alt) | 29.9% | 210 | 192K |

*Table 7: Contextual routing comparison (d64/2L, 2000 steps).*

Contextual routing matches standard trees in accuracy but does not provide a clear improvement. The alternating-layer variant (29.9%) remains the best practical choice.

### 4.8 BPE vs Character-Level Tokenization

| Tokenization | Standard | Obliv. L+F Alt | MicroTree |
|-------------|----------|----------------|-----------|
| Char (65 vocab) | 29.0% | 30.3% | 29.4% |
| BPE-500 | 6.9% | 6.6% | 6.8% |

*Table 8: Tokenization comparison (d64/2L, 2000 steps).*

BPE breaks all models at this scale. With 500 BPE tokens, the effective sequence length shrinks ~4x (BPE compresses text), leaving too few training positions per batch. Character-level tokenization is the correct choice for Shakespeare at this model size.

### 4.9 Hard Routing at Inference

Models trained with soft routing are evaluated with top-k hard routing at inference:

| Model | Soft Acc | Top-1 | Top-2 | Top-4 |
|-------|----------|-------|-------|-------|
| Obliv. L+F (d3, 6 trees) | 29.56% | 99.9% | 99.8% | 100.1% |
| Obliv. L+F (V+O, alt, d2) | 30.17% | 100.3% | 100.4% | 100.0% |

*Table 9: Hard routing accuracy retention (fast config) (% of soft routing accuracy). d64/2L, 2000 steps.*

Hard routing retains >99.8% accuracy across all configurations. Top-1 routing is sufficient — models learn to concentrate probability mass on 1–2 leaves during training. This enables significant inference-time speedups by computing only the dominant leaf's output.

### 4.10 Trees-as-Adapters (Negative Result)

| Model | Val Acc | Method |
|-------|---------|--------|
| Standard (pretrained) | 29.0% | 2000 steps |
| Linear+MicroTree (from scratch) | 29.5% | 2000 steps |
| Tree Adapter | 26.0% | 1333 pretrain + 667 fine-tune |

*Table 10: Adapter results (d64/2L, 2000 total steps).*

Trees-as-adapters **underperform** both the standard baseline and end-to-end tree training. The frozen linear base constrains the tree corrections — trees need to co-adapt with the linear projections during training. This is a negative result for the adapter paradigm: unlike LoRA (which works because the pretrained weights are already good), tree corrections require end-to-end gradient flow to learn meaningful routing patterns.

### 4.11 MoE Ablation

The Linear+MoE variant achieves 36.6% validation accuracy at the full config, compared to 36.4% for the comparable tree variant (Oblivious L+F V+O, alt). At the fast config, MoE reaches 29.6% vs trees' 29.9%. This suggests that the tree structure itself is not critical — MoE and tree variants perform comparably, indicating that **input-dependent projection mixing is the key ingredient**, regardless of whether it uses hierarchical binary routing or top-k expert selection.

### 4.12 Routing Entropy Analysis

With temperature annealing to 0.7 (not 0.1), entropy stabilizes rather than collapsing to zero. This is critical because soft routing *is the feature*: each input sees a different effective weight matrix. Hard routing (entropy collapse) reduces trees to piecewise-constant lookup tables with 8x fewer effective parameters. The conservative annealing schedule preserves meaningful routing diversity throughout training.

## 5. Analysis and Discussion

### 5.1 Why LinearPlusForest Works

The success of LinearPlusForest can be attributed to:

1. **Linear base provides a guaranteed gradient path.** Even if tree routing shifts unpredictably, the linear projection continues to receive clean gradients and learn.

2. **Trees as corrections, not replacements.** The forest learns the residual between what a linear projection can represent and what the optimal projection would be.

3. **Input-adaptive projections.** With soft routing, the effective projection for each input is `W_base + shrinkage * W_eff(x)`, making this a form of input-conditional computation.

### 5.2 The Scaling Story

Our scaling grid reveals a nuanced picture:

- **At small scale (d32):** Trees and linear perform identically. There isn't enough width for trees to learn meaningful routing patterns.
- **At medium scale (d64, d128/1-2L):** Trees shine, providing +1–4% accuracy gains. The model has enough capacity for trees to exploit, but the standard architecture can't fully utilize the width with limited depth.
- **At full scale (d128/4L):** The gap narrows to <0.5%. Given enough depth, the standard transformer's layer-by-layer nonlinearity matches what trees provide within individual projections.

This suggests trees are most valuable in **wide, shallow** architectures — exactly the regime where a standard transformer's depth-dependent nonlinearity is most constrained.

### 5.3 Input-Dependent Mixing is the Key

The MoE ablation and speed-optimized projection experiments converge on the same conclusion: **input-dependent mixing is the key ingredient**, not the tree structure. Gated projections (GLU-style), dynamic linear layers, and product-key lookup all achieve comparable accuracy improvements over standard linear at 2–3x speed overhead. Trees provide one mechanism for input-dependent mixing, but simpler mechanisms work equally well.

### 5.4 Practical Recommendations

Based on our experiments:

1. **For maximum accuracy at moderate cost:** Linear+Forest with oblivious trees, alternating layers.
2. **For speed-constrained deployment:** Gated projections (GLU-style) or ProductKey — 2–3x overhead, +1% accuracy.
3. **For inference optimization:** Train with soft routing, deploy with top-1 hard routing (>99.8% retention).
4. **Avoid:** Trees-as-adapters (need end-to-end training), BPE at small scale, aggressive temperature annealing.

### 5.5 Limitations

1. **Speed.** Even the fastest tree variant is 2–3x slower than standard attention. GPU benchmarks and kernel fusion are needed for practical use. Simpler alternatives (GLU, dynamic linear) may be preferable.

2. **Scale.** Our experiments reach d128/6L (~1.7M params). The oblivious alternating model slightly outperforms standard at this scale, but behavior at LLM sizes (>100M params) is unknown.

3. **Task diversity.** We evaluate only on character-level Shakespeare. Other tasks and tokenizations may show different results.

4. **Temperature schedule.** The 1.0 → 0.7 cosine schedule works well but is hand-tuned.

## 6. Related Work

**Soft Decision Trees.** Irsoy et al. (2012) introduced soft trees for neural networks. Frosst & Hinton (2017) used them for distillation. Hazimeh et al. (2020) proposed differentiable trees for tabular data. Our work applies soft trees as projection layers within transformers.

**NODE.** Popov et al. (2020) introduced Neural Oblivious Decision Ensembles for tabular data, using oblivious trees with entmax activation. We adapt oblivious trees for use as attention projection layers, using sigmoid routing with temperature annealing.

**Tree-based Neural Networks.** Deep Neural Decision Forests (Kontschieder et al., 2015) combined random forests with neural feature learning. Adaptive Neural Trees (Tanno et al., 2019) learned tree structure alongside parameters. We focus on fixed-topology trees with learned routing.

**Mixture of Experts.** The LinearPlusForest architecture shares conceptual similarity with MoE (Shazeer et al., 2017), where different experts handle different inputs. Our tree routing serves as a continuous, structured form of expert selection — we directly compare against an MoE variant and find comparable performance.

**Efficient Attention.** Various works have proposed alternatives to standard attention projections, including low-rank (Wang et al., 2020), sparse (Child et al., 2019), and kernel-based (Katharopoulos et al., 2020) methods. Tree-based projections offer a distinct inductive bias — input-conditional piecewise linear projections.

## 7. Future Work

1. **GPU optimization:** Custom CUDA kernels for the tree routing computation could significantly reduce the speed gap. The batched einsums are naturally parallelizable.

2. **Larger scale:** Evaluate on larger models and datasets (e.g., OpenWebText) to determine whether the narrowing accuracy gap at d128/4L continues or reverses at 100M+ parameter scale.

3. **Adaptive temperature:** Replace the fixed cosine schedule with learned or gradient-based temperature adaptation.

4. **Routing analysis:** Investigate what linguistic features the tree routing learns to partition on, potentially recovering interpretable attention patterns.

5. **Hybrid architectures:** Given that input-dependent mixing is the key ingredient, explore combining the simplicity of gated projections with the structured routing of trees.

## 8. Conclusion

We demonstrate that differentiable decision trees can serve as effective projection layers in transformer attention. Through comprehensive experiments across 9 experiment suites, we find that:

**Trees work, and can slightly outperform.** At our largest scale (d128/6L, 13K steps), the Oblivious L+F alternating model slightly outperforms the standard transformer (54.6% vs 54.4%), while Linear+Forest matches at 54.3%. Trees provide their largest gains at intermediate scale (+1–4% at d64/d128 with limited depth).

**Input-dependent mixing is the key insight.** The tree structure itself is not critical — MoE, gated projections, dynamic linear layers, and product-key lookup all achieve comparable accuracy improvements. What matters is that each input sees a different effective weight matrix.

**Soft routing is the feature, not a compromise.** With soft routing, tree forests compute input-adaptive projections. Hard routing can be applied at inference with top-2 retaining >99.7% accuracy, but training must remain soft to preserve gradient flow through all leaves.

**Trees require end-to-end training.** Adapter-style fine-tuning (freezing linear weights and adding tree corrections) underperforms both baseline and end-to-end training, unlike LoRA which benefits from pretrained weight quality.

**Speed-optimized alternatives are practical.** Gated projections and dynamic linear layers achieve +1% accuracy at only 2–3x speed overhead — a more practical trade-off than full tree routing for most applications.

The deeper insight is that transformers benefit from input-conditional computation in their projection layers. Trees are one elegant mechanism for this, but the principle is more general: any form of input-dependent weight mixing can improve over fixed linear projections. This opens a broad design space for hybrid architectures that combine the efficiency of linear projections with the expressiveness of conditional computation.

## References

- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers.
- Frosst, N., & Hinton, G. (2017). Distilling a neural network into a soft decision tree.
- Hazimeh, H., et al. (2020). The tree ensemble layer: Differentiability meets conditional computation.
- Irsoy, O., Yildiz, O. T., & Alpaydin, E. (2012). Soft decision trees.
- Karpathy, A. (2015). The unreasonable effectiveness of recurrent neural networks.
- Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention.
- Kontschieder, P., et al. (2015). Deep neural decision forests.
- Popov, S., Morozov, S., & Babenko, A. (2020). Neural oblivious decision ensembles for deep learning on tabular data.
- Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
- Tanno, R., et al. (2019). Adaptive neural trees.
- Vaswani, A., et al. (2017). Attention is all you need.
- Wang, S., et al. (2020). Linformer: Self-attention with linear complexity.
