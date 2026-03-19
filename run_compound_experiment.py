"""
Compound optimization experiment: confirm that individual gains stack.

Tests the BEST tree architecture (Linear+Forest / boosted) with cumulative
optimizations applied one at a time, to measure compounding speedup:

  1. boosted (baseline tree model — already has OPT-01, OPT-02 baked in)
  2. boosted + torch.compile
  3. boosted + shared_routing
  4. boosted + shared_routing + torch.compile  (full compound)

Also runs standard transformer as the speed/accuracy reference.

Results saved to results/compound_experiment_{timestamp}.json

Usage:
    python run_compound_experiment.py
"""

import json
import math
import os
import time
import torch
import torch.nn.functional as F

from main import (
    TreeTransformer, tree_regularization_loss, leaf_balancing_loss,
    count_parameters, set_temperature, get_routing_entropy, make_optimizer,
)
from data import ShakespeareDataset

# Inline standard transformer (same as train.py)
class StandardAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.W_qkv = torch.nn.Linear(d_model, d_model * 3)
        self.W_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        qkv = self.W_qkv(x).reshape(B, S, 3, self.n_heads, self.d_k)
        Q, K, V = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.W_o(out), attn


class StandardBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attn = StandardAttention(d_model, n_heads, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ff_dim), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(ff_dim, d_model))
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, w = self.attn(x, mask)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, w


class StandardTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab_size, d_model)
        self.pos = torch.nn.Embedding(max_seq_len, d_model)
        self.drop = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [StandardBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.pos_emb = self.pos

    def _mask(self, s, dev):
        return torch.tril(torch.ones(s, s, device=dev)).unsqueeze(0).unsqueeze(0)

    def forward(self, ids, mask=None):
        B, S = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(S, device=ids.device))
        x = self.drop(x)
        if mask is None:
            mask = self._mask(S, ids.device)
        attn = []
        for layer in self.layers:
            x, w = layer(x, mask)
            attn.append(w)
        return self.head(self.norm(x)), attn


# =============================================================================
# Experiment configs — each builds on the previous
# =============================================================================

CONFIGS = [
    {
        "name": "standard",
        "description": "Standard Transformer (reference)",
        "is_standard": True,
        "compile": False,
    },
    {
        "name": "boosted",
        "description": "Linear+Forest (baseline tree)",
        "proj_type": "boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": False,
        "compile": False,
    },
    {
        "name": "boosted_compiled",
        "description": "Linear+Forest + torch.compile",
        "proj_type": "boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": False,
        "compile": True,
    },
    {
        "name": "boosted_shared",
        "description": "Linear+Forest + shared routing",
        "proj_type": "boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": True,
        "compile": False,
    },
    {
        "name": "boosted_shared_compiled",
        "description": "Linear+Forest + shared routing + compile",
        "proj_type": "boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": True,
        "compile": True,
    },
]


# =============================================================================
# Training loop (self-contained)
# =============================================================================

def train_and_eval(config, dataset, cfg, device):
    desc = config["description"]
    use_compile = config.get("compile", False)
    is_tree = not config.get("is_standard", False)

    # Create model
    torch.manual_seed(42)
    if config.get("is_standard"):
        model = StandardTransformer(
            dataset.vocab_size, cfg['d_model'], cfg['n_layers'],
            cfg['n_heads'], cfg['seq_len'], cfg['dropout'],
        ).to(device)
    else:
        proj_kwargs = {k: v for k, v in config.items()
                       if k not in ("name", "description", "is_standard",
                                    "compile", "proj_type")}
        model = TreeTransformer(
            vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
            n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
            max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
            dropout=cfg['dropout'], use_tree_ffn=False, task="lm",
            proj_type=config["proj_type"], **proj_kwargs,
        ).to(device)

    raw_model = model  # for utility functions
    params = count_parameters(raw_model)

    if use_compile:
        model = torch.compile(model, backend='inductor')
        raw_model = model._orig_mod

    print(f"\n{'=' * 65}")
    print(f"  {desc}")
    print(f"  Params: {params['total']:,}" +
          (f" (tree: {params['tree_pct']:.1f}%)" if params.get('tree', 0) > 0 else ""))
    if use_compile:
        print(f"  torch.compile: ON")
    print(f"{'=' * 65}")

    # Optimizer
    if is_tree:
        optimizer = make_optimizer(raw_model, lr=cfg['lr'])
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    # LR schedule
    n_steps = cfg['n_steps']
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    eval_interval = max(1, n_steps // 5)
    model.train()
    start = time.perf_counter()

    eval_log = {'steps': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'entropy': [], 'elapsed': [],
                'lr': [], 'temperature': []}

    temp = 1.0
    for step in range(1, n_steps + 1):
        if is_tree:
            progress = step / n_steps
            if progress < 0.5:
                temp = 1.0
            else:
                phase = (progress - 0.5) / 0.5
                temp = 0.7 + 0.3 * (1 + math.cos(math.pi * phase)) / 2
            set_temperature(raw_model, temp)

        x, y = dataset.get_batch(cfg['batch'], device, 'train')
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
        if is_tree:
            loss = loss + tree_regularization_loss(raw_model, 0.005)
            loss = loss + leaf_balancing_loss(raw_model, 0.01)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        acc = (logits.argmax(-1) == y).float().mean().item()

        if step % eval_interval == 0 or step == 1:
            eval_res = dataset.estimate_loss(model, cfg['batch'], device, n_batches=10)
            elapsed = time.perf_counter() - start
            ms_per_step = elapsed / step * 1000
            ent = get_routing_entropy(raw_model) if is_tree else 0.0
            current_lr = optimizer.param_groups[0]['lr']

            eval_log['steps'].append(step)
            eval_log['train_loss'].append(round(loss.item(), 4))
            eval_log['train_acc'].append(round(acc, 4))
            eval_log['val_loss'].append(round(eval_res['val']['loss'], 4))
            eval_log['val_acc'].append(round(eval_res['val']['acc'], 4))
            eval_log['entropy'].append(round(ent, 4))
            eval_log['elapsed'].append(round(elapsed, 1))
            eval_log['lr'].append(round(current_lr, 6))
            eval_log['temperature'].append(round(temp, 4))

            line = (f"  Step {step:>5d}/{n_steps} | "
                    f"loss={loss.item():.3f} acc={acc:.1%} | "
                    f"val_loss={eval_res['val']['loss']:.3f} "
                    f"val_acc={eval_res['val']['acc']:.1%} | "
                    f"{ms_per_step:.0f}ms/step")
            if is_tree:
                line += f" | ent={ent:.3f} | temp={temp:.3f}"
            line += f" | lr={current_lr:.1e}"
            print(line)

    elapsed = time.perf_counter() - start
    final = dataset.estimate_loss(model, cfg['batch'], device)

    print(f"\n  Final: train_loss={final['train']['loss']:.3f} "
          f"train_acc={final['train']['acc']:.1%} | "
          f"val_loss={final['val']['loss']:.3f} val_acc={final['val']['acc']:.1%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_steps*1000:.0f}ms/step)")

    return {
        'name': desc,
        'config_name': config['name'],
        'params': params['total'],
        'tree_pct': params.get('tree_pct', 0),
        'compiled': use_compile,
        'shared_routing': config.get('shared_routing', False),
        'final_train_loss': final['train']['loss'],
        'final_train_acc': final['train']['acc'],
        'final_val_loss': final['val']['loss'],
        'final_val_acc': final['val']['acc'],
        'time': elapsed,
        'ms_per_step': elapsed / n_steps * 1000,
        'eval_log': eval_log,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    cfg = {
        'd_model': 64, 'n_layers': 2, 'n_heads': 4,
        'seq_len': 128, 'batch': 32, 'n_steps': 2000,
        'lr': 3e-4, 'dropout': 0.0,
    }

    print(f"Compound Optimization Experiment")
    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")
    print(f"Testing {len(CONFIGS)} configurations")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])
    all_results = {}

    for config in CONFIGS:
        result = train_and_eval(config, dataset, cfg, DEVICE)
        all_results[config['name']] = result
        # Free memory
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Summary
    print(f"\n{'#' * 70}")
    print(f"  COMPOUND OPTIMIZATION RESULTS (2000 steps, --fast)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'ms/step':>8} {'Speedup':>8} {'Params':>8}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    base_ms = all_results.get('boosted', {}).get('ms_per_step', 1)
    for key in ['standard', 'boosted', 'boosted_compiled',
                'boosted_shared', 'boosted_shared_compiled']:
        if key in all_results:
            r = all_results[key]
            speedup = base_ms / r['ms_per_step'] if r['ms_per_step'] > 0 else 0
            marker = " <-- best" if key == 'boosted_shared_compiled' else ""
            print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
                  f"{r['ms_per_step']:>7.0f}ms {speedup:>7.1f}x "
                  f"{r['params']:>8,}{marker}")

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("results", f"compound_experiment_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    # Also save as latest
    with open(os.path.join("results", "compound_experiment_latest.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to {output_file}")

    # Recommendation
    best_tree = min(
        [(k, v) for k, v in all_results.items() if k != 'standard'],
        key=lambda kv: -kv[1]['final_val_acc']
    )
    fastest_tree = min(
        [(k, v) for k, v in all_results.items() if k != 'standard'],
        key=lambda kv: kv[1]['ms_per_step']
    )
    print(f"\n  RECOMMENDATION:")
    print(f"    Best accuracy:  {best_tree[0]} ({best_tree[1]['final_val_acc']:.1%})")
    print(f"    Fastest:        {fastest_tree[0]} ({fastest_tree[1]['ms_per_step']:.0f}ms/step)")


if __name__ == "__main__":
    main()
