"""
Benchmark: Tree-Based Attention vs Standard Attention
=====================================================
Next-token prediction on two synthetic tasks, comparing:
  1. Standard Transformer (linear projections)
  2. Batched Forest Transformer (12 trees, depth 3)
  3. Boosted Forest Transformer (3 stages × 12 trees × depth 2 = 36 trees)
  4. Boosted Forest Full (boosted in attn + FFN)

Reports accuracy as primary metric. 1000 training steps.
Designed to run on a MacBook CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from main import (
    TreeTransformer, tree_regularization_loss, count_parameters,
    set_temperature, get_routing_entropy,
)


# =============================================================================
# Standard Transformer (baseline)
# =============================================================================

class StandardAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(context), attn_weights


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attention = StandardAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x, attn_weights


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, mask=None):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(seq_len, device=device).unsqueeze(0))
        x = self.emb_dropout(x)
        if mask is None:
            mask = self._causal_mask(seq_len, device)
        all_attn = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            all_attn.append(attn_w)
        return self.head(self.final_norm(x)), all_attn


# =============================================================================
# Synthetic data generators
# =============================================================================

def make_linear_batch(batch_size, seq_len, vocab_size, device, **_):
    """Linear pattern: next = (prev + offset) % vocab."""
    offsets = torch.randint(1, 6, (batch_size, 1), device=device)
    tokens = [torch.randint(0, vocab_size, (batch_size, 1), device=device)]
    for _ in range(seq_len - 1):
        next_tok = (tokens[-1] + offsets) % vocab_size
        noise = (torch.rand(batch_size, 1, device=device) < 0.15).long()
        next_tok = next_tok * (1 - noise) + torch.randint(0, vocab_size, (batch_size, 1), device=device) * noise
        tokens.append(next_tok)
    tokens = torch.cat(tokens, dim=1)
    return tokens[:, :-1], tokens[:, 1:]


def make_nonlinear_batch(batch_size, seq_len, vocab_size, device, lookup_table=None, **_):
    """Non-linear pattern: next = lookup_table[prev XOR prev_prev]."""
    if lookup_table is None:
        lookup_table = torch.randint(0, vocab_size, (vocab_size,), device=device)
    tokens = [torch.randint(0, vocab_size, (batch_size, 1), device=device) for _ in range(2)]
    for _ in range(seq_len - 2):
        xor_val = (tokens[-1] ^ tokens[-2]) % vocab_size
        next_tok = lookup_table[xor_val.squeeze(-1)].unsqueeze(-1)
        noise = (torch.rand(batch_size, 1, device=device) < 0.15).long()
        next_tok = next_tok * (1 - noise) + torch.randint(0, vocab_size, (batch_size, 1), device=device) * noise
        tokens.append(next_tok)
    tokens = torch.cat(tokens, dim=1)
    return tokens[:, :-1], tokens[:, 1:]


# =============================================================================
# Training loop
# =============================================================================

def train_loop(model, n_steps, batch_size, seq_len, vocab_size, lr, device,
               use_tree_reg=False, anneal_temp=False, data_fn=None, **data_kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    losses, accuracies, entropies = [], [], []
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        if anneal_temp:
            progress = step / n_steps
            temp = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            set_temperature(model, temp)

        inputs, targets = data_fn(batch_size, seq_len, vocab_size, device, **data_kwargs)
        logits, _ = model(inputs)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        if use_tree_reg:
            loss = loss + tree_regularization_loss(model, 0.01)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)
        if anneal_temp:
            entropies.append(get_routing_entropy(model))

    elapsed = time.perf_counter() - start
    return losses, accuracies, elapsed, entropies


# =============================================================================
# Output
# =============================================================================

def print_results(name, params, losses, accuracies, elapsed, n_steps, entropies=None):
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  Parameters:      {params['total']:>10,}")
    if params.get('tree', 0) > 0:
        print(f"  Tree params:     {params['tree']:>10,}  ({params['tree_pct']:.1f}%)")
    print(f"  Total time:      {elapsed:>10.2f}s  ({elapsed/n_steps*1000:.1f} ms/step)")

    n = 20
    avg_loss = sum(losses[-n:]) / n
    avg_acc = sum(accuracies[-n:]) / n
    print(f"  Final loss:      {avg_loss:>10.4f}  (avg last {n})")
    print(f"  Final accuracy:  {avg_acc:>10.1%}  (avg last {n})")
    if entropies:
        print(f"  Routing entropy: {entropies[0]:.4f} → {entropies[-1]:.4f}")

    interval = max(1, n_steps // 10)
    header = f"  {'Step':>6}  {'Loss':>8}  {'Accuracy':>9}"
    if entropies:
        header += f"  {'Entropy':>8}"
    print(f"\n{header}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}" + (f"  {'-'*8}" if entropies else ""))
    for i in range(interval - 1, n_steps, interval):
        line = f"  {i+1:>6}  {losses[i]:>8.4f}  {accuracies[i]:>9.1%}"
        if entropies:
            line += f"  {entropies[i]:>8.4f}"
        print(line)


# =============================================================================
# Model definitions
# =============================================================================

MODEL_CONFIGS = {
    "Standard Transformer": {
        "is_standard": True,
    },
    "Batched Forest (attn)": {
        "proj_type": "batched", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3,
    },
    "Boosted Forest (attn)": {
        "proj_type": "boosted", "use_tree_ffn": False,
        "n_stages": 3, "trees_per_stage": 12, "boosted_depth": 2,
    },
    "Boosted Forest (full)": {
        "proj_type": "boosted", "use_tree_ffn": True,
        "n_stages": 3, "trees_per_stage": 12, "boosted_depth": 2,
    },
}


def create_model(name, config, vocab_size, d_model, n_layers, n_heads, seq_len, device):
    mc = MODEL_CONFIGS[name]
    if mc.get("is_standard"):
        return StandardTransformer(vocab_size, d_model, n_layers, n_heads, seq_len).to(device)
    proj_kwargs = {k: v for k, v in mc.items() if k not in ("is_standard", "use_tree_ffn", "proj_type")}
    return TreeTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_seq_len=seq_len, num_classes=vocab_size,
        use_tree_ffn=mc.get("use_tree_ffn", False), task="lm",
        proj_type=mc["proj_type"], **proj_kwargs,
    ).to(device)


# =============================================================================
# Main
# =============================================================================

def run_task(task_name, data_fn, device, cfg, **data_kwargs):
    print(f"\n{'#' * 65}")
    print(f"  TASK: {task_name}")
    print(f"{'#' * 65}")

    results = []
    n = 20

    for model_name in MODEL_CONFIGS:
        is_tree = not MODEL_CONFIGS[model_name].get("is_standard")
        print(f"\n  Training {model_name}...")

        torch.manual_seed(42)
        model = create_model(model_name, cfg, cfg['vocab'], cfg['d_model'],
                             cfg['n_layers'], cfg['n_heads'], cfg['seq_len'], device)
        params = count_parameters(model)

        torch.manual_seed(42)
        losses, accs, elapsed, ents = train_loop(
            model, cfg['n_steps'], cfg['batch'], cfg['seq_len'],
            cfg['vocab'], cfg['lr'], device,
            use_tree_reg=is_tree, anneal_temp=is_tree, data_fn=data_fn,
            **data_kwargs,
        )
        print_results(model_name, params, losses, accs, elapsed, cfg['n_steps'],
                      ents if is_tree else None)

        avg_acc = sum(accs[-n:]) / n
        results.append((model_name, avg_acc, elapsed, params['total']))

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  {task_name} — SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Model':<28s}  {'Accuracy':>9}  {'Time':>7}  {'Params':>8}")
    print(f"  {'-'*28}  {'-'*9}  {'-'*7}  {'-'*8}")
    for name, acc, t, p in results:
        print(f"  {name:<28s}  {acc:>9.1%}  {t:>6.1f}s  {p:>8,}")
    best = max(results, key=lambda r: r[1])
    print(f"  Winner: {best[0]} ({best[1]:.1%})")

    return results


def main():
    DEVICE = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'

    cfg = {
        'vocab': 256, 'd_model': 64, 'n_layers': 2, 'n_heads': 4,
        'seq_len': 64, 'batch': 32, 'n_steps': 1000, 'lr': 3e-4,
    }

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")

    # Task 1: Linear pattern
    linear_results = run_task(
        "Linear: next = (prev + offset) % vocab",
        make_linear_batch, DEVICE, cfg,
    )

    # Task 2: Non-linear pattern
    torch.manual_seed(0)
    lut = torch.randint(0, cfg['vocab'], (cfg['vocab'],), device=DEVICE)

    nonlinear_results = run_task(
        "Non-Linear: next = table[prev XOR prev_prev]",
        make_nonlinear_batch, DEVICE, cfg, lookup_table=lut,
    )

    # Cross-task comparison
    print(f"\n{'#' * 65}")
    print(f"  CROSS-TASK COMPARISON")
    print(f"{'#' * 65}")
    names = [r[0] for r in linear_results]
    print(f"\n  {'Model':<28s}  {'Linear Acc':>10}  {'NonLinear Acc':>13}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*13}")
    for i, name in enumerate(names):
        la = linear_results[i][1]
        na = nonlinear_results[i][1]
        print(f"  {name:<28s}  {la:>10.1%}  {na:>13.1%}")


if __name__ == "__main__":
    main()
