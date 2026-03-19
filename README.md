# Tree-Based Attention for Transformers

A PyTorch research implementation that replaces dense linear projections (Q, K, V) in transformer attention with differentiable soft decision trees, trained end-to-end via backpropagation.

**Key result:** Tree-augmented transformers beat standard transformers on character-level language modeling (+0.9pp accuracy) while approaching competitive speed through a series of architectural and computational optimizations.

## Quick Start

```bash
pip install torch matplotlib  # matplotlib optional, for figures

# Run the demo
python main.py

# Train on Shakespeare (fast config, ~5 min)
python train.py --fast --model oblivious_boosted_vo_alt

# Train all models for comparison
python train.py --fast --model all

# Run synthetic benchmark
python benchmark.py
```

## Results

Shakespeare character-level LM (d_model=64, 2 layers, 2000 steps):

| Model | Val Acc | ms/step | vs Standard |
|-------|---------|---------|-------------|
| Standard Transformer | 28.8% | 20ms | 1.0x |
| Obliv L+F (V+O, alternating) | **29.4%** | 50ms compiled | 2.7x |
| Linear+MoE (alternating) | **29.7%** | 64ms compiled | 3.4x |

The best tree model exceeds standard accuracy by +0.9pp. Speed gap reduced from 11x to 2.7x through optimizations.

## Architecture

All model components live in `main.py`:

### Projection Types

| Type | Description | Use Case |
|------|-------------|----------|
| `batched` | Soft decision tree forest | Pure tree routing |
| `oblivious` | NODE-style oblivious trees | Faster, better gradients |
| `boosted` | Linear + BatchedTreeForest | Best accuracy |
| `oblivious_boosted` | Linear + ObliviousTreeForest | Best speed/accuracy |
| `moe` | Flat MoE (no tree structure) | Ablation baseline |
| `moe_boosted` | Linear + Flat MoE | Fastest nonlinear |

### Key Components

- **BatchedTreeForest** -- Replaces `nn.Linear`. All trees computed via batched einsum. Features per-node learnable temperature, input-dependent gating (MoE-style), vectorized leaf probability computation.

- **ObliviousTreeForest** -- NODE-style: one hyperplane per depth level. Leaf probs via outer product (no depth-dependent gradient vanishing). Fewer parameters, faster computation.

- **LinearPlusForest** -- `output = linear(x) + shrinkage * forest(x)`. Linear base preserves residual stream structure; forest adds nonlinear correction. This is what makes tree models beat standard transformers.

- **SharedRoutingForest** -- Computes routing once, applies separate leaf outputs for Q/K/V. Saves ~60% of attention projection time.

- **TreeAttention** -- Multi-head attention with separate Q/K/V/O projections. QK-norm stabilizes attention despite routing shifts. Supports `tree_targets="vo"` to use trees only on V and O (where nonlinearity helps most).

### Speed Optimizations

| Optimization | Speedup | Description |
|---|---|---|
| OPT-01: Vectorized leaf probs | 1.5x | Precomputed path indices, gather+prod (no Python loop) |
| OPT-02: Unrolled outer product | 1.3x | Depth-3 Kronecker product explicitly unrolled |
| OPT-03: torch.compile | 1.4x | Fused kernels via inductor backend |
| OPT-04: Shared routing | 2-3x | One routing pass for Q/K/V |
| OPT-05: Selective V+O trees | 2x | Linear Q/K, tree V/O only |
| OPT-11: Alternating layers | 2x | Trees in odd layers, linear in even |

## Training

```bash
# Fast config (d_model=64, 2 layers, 2000 steps)
python train.py --fast --model oblivious_boosted_vo_alt

# Full config (d_model=128, 4 layers, 2000 steps)
python train.py --model oblivious_boosted_vo_alt

# All available models
python train.py --fast --model all

# Disable torch.compile
python train.py --fast --model oblivious_boosted_alt --no-compile

# Custom config
python train.py --d_model 128 --n_layers 4 --steps 5000 --model boosted
```

### Training Features

- LR warmup (100 steps) + cosine decay to 10% of peak
- Temperature annealing: held at 1.0 for first 50%, cosine to 0.7 over second half
- Tree-aware optimizer: routing params get 3x LR, no weight decay
- Entropy regularization (lambda=0.005) + leaf balancing loss (alpha=0.01)

## Model Configs

Available via `--model`:

| Config | Description |
|--------|-------------|
| `standard` | Standard transformer baseline |
| `batched` | Batched tree forest (attn only) |
| `boosted` | Linear+Forest (attn only) |
| `oblivious` | Oblivious tree forest (attn only) |
| `oblivious_boosted` | Oblivious Linear+Forest (attn only) |
| `boosted_alt` | Linear+Forest (alternating layers) |
| `oblivious_boosted_alt` | Oblivious L+F (alternating layers) |
| `oblivious_boosted_vo` | Oblivious L+F (V+O only) |
| `oblivious_boosted_vo_alt` | Oblivious L+F (V+O, alternating) |
| `moe_boosted_alt` | Linear+MoE (alternating) |
| `*_shared` | Shared routing variants |

## How It Works

Standard transformers use dense linear projections: `Q = W_q @ x`. Tree-based attention replaces these with soft decision tree forests where every input "flows" through all paths simultaneously with soft probabilities:

1. **Routing**: Each tree node computes `sigmoid(w @ x / temperature)` to get left/right probabilities
2. **Leaf probabilities**: Product of routing decisions along each root-to-leaf path
3. **Output**: Weighted sum of leaf output vectors, mixed across trees via input-dependent gating
4. **LinearPlusForest**: A standard linear projection provides the base; the forest adds a nonlinear correction term

The key insight is that soft routing makes forests compute **input-adaptive linear projections** -- a different effective weight matrix for every input, constructed as a weighted combination of leaf matrices. This is strictly more expressive than a single linear layer.

## File Structure

```
main.py                  # All model components
train.py                 # Shakespeare training + model configs
benchmark.py             # Synthetic task benchmark
data.py                  # Shakespeare dataset loader
generate_figures.py      # Plot results from JSON
results/                 # Saved experiment results
tasks/backlog/           # Optimization backlog (17 scored items)
figures/                 # Generated plots
```

## Citation

If you use this code in your research:

```
@software{tree_based_attention,
  title={Tree-Based Attention for Transformers},
  url={https://github.com/mgoldwasser/tree-based-llm},
  year={2025}
}
```
