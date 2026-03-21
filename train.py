"""
Training script for Tree-Based Attention on Shakespeare.
Compares Standard, Batched Forest, and Boosted Forest transformers.

Usage:
    python train.py --fast              # fast iteration (~16 min total)
    python train.py --fast --model batched  # single model, fast
    python train.py --model all         # full config (~100 min total)
    python train.py --steps 5000        # custom steps
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F

from main import (
    TreeTransformer, tree_regularization_loss, leaf_balancing_loss,
    count_parameters, set_temperature, get_routing_entropy, make_optimizer,
    freeze_non_tree_params, unfreeze_all_params, set_hard_routing,
)
from data import ShakespeareDataset


def unwrap(model):
    """Get underlying model from torch.compile wrapper."""
    return getattr(model, '_orig_mod', model)


# =============================================================================
# Standard Transformer (baseline — inline to avoid benchmark.py dependency)
# =============================================================================

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
        self.pos_emb = self.pos  # for generate()

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
# Model configs
# =============================================================================

MODEL_CONFIGS = {
    "standard": {
        "description": "Standard Transformer",
        "is_standard": True,
    },
    "batched": {
        "description": "Batched Forest (attn only)",
        "proj_type": "batched", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3,
    },
    "boosted": {
        "description": "Linear+Forest (attn only)",
        "proj_type": "boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
    },
    "oblivious": {
        "description": "Oblivious Forest (attn only)",
        "proj_type": "oblivious", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3,
    },
    "oblivious_boosted": {
        "description": "Oblivious Linear+Forest (attn only)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
    },
    "boosted_alt": {
        "description": "Linear+Forest (alternating layers)",
        "proj_type": "boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_every_n": 2,
    },
    "oblivious_boosted_alt": {
        "description": "Oblivious L+F (alternating layers)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_every_n": 2,
    },
    "oblivious_boosted_vo": {
        "description": "Oblivious L+F (V+O only, OPT-05)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_targets": "vo",
    },
    "oblivious_boosted_vo_alt": {
        "description": "Oblivious L+F (V+O, alternating)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_targets": "vo",
        "tree_every_n": 2,
    },
    "moe_boosted_alt": {
        "description": "Linear+MoE (alternating, OPT-13)",
        "proj_type": "moe_boosted", "use_tree_ffn": False,
        "n_trees": 12,
        "tree_every_n": 2,
    },
    # --- Shared routing variants (OPT-04) ---
    "batched_shared": {
        "description": "Batched Forest (shared routing)",
        "proj_type": "batched", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3,
        "shared_routing": True,
    },
    "boosted_shared": {
        "description": "Linear+Forest (shared routing)",
        "proj_type": "boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": True,
    },
    "oblivious_shared": {
        "description": "Oblivious Forest (shared routing)",
        "proj_type": "oblivious", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3,
        "shared_routing": True,
    },
    "oblivious_boosted_shared": {
        "description": "Oblivious L+F (shared routing)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "shared_routing": True,
    },
    # --- Micro-tree variants (NEW-01) ---
    "micro_tree": {
        "description": "Micro-Tree Forest (attn only)",
        "proj_type": "micro_tree", "use_tree_ffn": False,
        "n_trees": 4, "tree_depth": 1, "leaf_rank": 8,
    },
    "micro_boosted": {
        "description": "Linear+MicroTree (attn only)",
        "proj_type": "micro_boosted", "use_tree_ffn": False,
        "n_trees": 4, "tree_depth": 1, "leaf_rank": 8,
    },
    "micro_boosted_d2": {
        "description": "Linear+MicroTree depth-2 (attn only)",
        "proj_type": "micro_boosted", "use_tree_ffn": False,
        "n_trees": 4, "tree_depth": 2, "leaf_rank": 8,
    },
    # --- Contextual routing variants (NEW-05) ---
    "contextual": {
        "description": "Contextual Routing Forest (attn only)",
        "proj_type": "contextual", "use_tree_ffn": False,
        "n_trees": 12, "tree_depth": 3, "ema_decay": 0.9,
    },
    "contextual_boosted": {
        "description": "Linear+Contextual (attn only)",
        "proj_type": "contextual_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3, "ema_decay": 0.9,
    },
    # --- Depth ablation variants (NEW-06) ---
    "oblivious_boosted_d1": {
        "description": "Oblivious L+F depth-1 (24 trees, 48 leaves)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 1,
    },
    "oblivious_boosted_d2": {
        "description": "Oblivious L+F depth-2 (12 trees, 48 leaves)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 12, "boosted_depth": 2,
    },
    "oblivious_boosted_d4": {
        "description": "Oblivious L+F depth-4 (3 trees, 48 leaves)",
        "proj_type": "oblivious_boosted", "use_tree_ffn": False,
        "boosted_trees": 3, "boosted_depth": 4,
    },
    # --- Speed-optimized projections ---
    "gated_boosted": {
        "description": "Linear+Gated (GLU-style)",
        "proj_type": "gated_boosted", "use_tree_ffn": False,
        "n_gates": 1,
    },
    "gated_boosted_d2": {
        "description": "Linear+Gated depth-2 (2 gates)",
        "proj_type": "gated_boosted", "use_tree_ffn": False,
        "n_gates": 2,
    },
    "dynamic": {
        "description": "Dynamic Linear (rank-8, 1 mod)",
        "proj_type": "dynamic", "use_tree_ffn": False,
        "leaf_rank": 8,
        "n_modulations": 1,
    },
    "dynamic_boosted": {
        "description": "Dynamic Linear (rank-8, 4 mods)",
        "proj_type": "dynamic_boosted", "use_tree_ffn": False,
        "leaf_rank": 8,
        "n_modulations": 4,
    },
    "lowrank_boosted": {
        "description": "Linear+LowRankRouting (r=16)",
        "proj_type": "lowrank_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "routing_rank": 16,
    },
    "recursive_boosted": {
        "description": "Linear+Recursive (3 iterations)",
        "proj_type": "recursive_boosted", "use_tree_ffn": False,
        "n_iterations": 3,
    },
    "chunked_boosted": {
        "description": "Linear+ChunkedRouting (chunk=16)",
        "proj_type": "chunked_boosted", "use_tree_ffn": False,
        "boosted_trees": 24, "boosted_depth": 3,
        "chunk_size": 16,
    },
    "product_key_boosted": {
        "description": "Linear+ProductKey (C=16)",
        "proj_type": "product_key_boosted", "use_tree_ffn": False,
        "codebook_size": 16,
        "top_k": 4,
    },
}


def create_model(name, vocab_size, cfg, device):
    mc = MODEL_CONFIGS[name]
    if mc.get("is_standard"):
        return StandardTransformer(
            vocab_size, cfg['d_model'], cfg['n_layers'], cfg['n_heads'],
            cfg['seq_len'], cfg['dropout'],
        ).to(device)
    proj_kwargs = {k: v for k, v in mc.items()
                   if k not in ("is_standard", "use_tree_ffn", "proj_type", "description",
                                "tree_every_n")}
    return TreeTransformer(
        vocab_size=vocab_size, d_model=cfg['d_model'], n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'], max_seq_len=cfg['seq_len'],
        num_classes=vocab_size, dropout=cfg['dropout'],
        use_tree_ffn=mc.get("use_tree_ffn", False), task="lm",
        proj_type=mc["proj_type"],
        tree_every_n=mc.get("tree_every_n", 1),
        **proj_kwargs,
    ).to(device)


# =============================================================================
# Training
# =============================================================================

def train_model(model, dataset, name, cfg, device, is_tree=False):
    mc = MODEL_CONFIGS[name]
    desc = mc["description"]
    raw_model = unwrap(model)  # for utility functions that iterate modules
    params = count_parameters(raw_model)
    print(f"\n{'=' * 65}")
    print(f"  {desc}")
    print(f"  Params: {params['total']:,}" +
          (f" (tree: {params['tree_pct']:.1f}%)" if params.get('tree', 0) > 0 else ""))
    print(f"{'=' * 65}")

    if is_tree:
        optimizer = make_optimizer(raw_model, lr=cfg['lr'])
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    # LR schedule: linear warmup (100 steps) + cosine decay to 10% of peak
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

    best_val_loss = float('inf')
    results = {'losses': [], 'accs': [], 'entropies': [], 'val_losses': [], 'val_accs': []}
    # Structured log for figure generation (eval-step snapshots)
    eval_log = {'steps': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'entropy': [], 'elapsed': [],
                'lr': [], 'temperature': []}

    temp = 1.0
    for step in range(1, n_steps + 1):
        if is_tree:
            progress = step / n_steps
            if progress < 0.5:
                temp = 1.0  # fully soft for first half
            else:
                # cosine from 1.0 → 0.7 over second half
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
        results['losses'].append(loss.item())
        results['accs'].append(acc)
        if is_tree:
            results['entropies'].append(get_routing_entropy(raw_model))

        if step % eval_interval == 0 or step == 1:
            eval_res = dataset.estimate_loss(model, cfg['batch'], device, n_batches=10)
            results['val_losses'].append(eval_res['val']['loss'])
            results['val_accs'].append(eval_res['val']['acc'])

            elapsed = time.perf_counter() - start
            ms_per_step = elapsed / step * 1000

            # Log structured eval snapshot
            ent = results['entropies'][-1] if is_tree and results['entropies'] else 0.0
            eval_log['steps'].append(step)
            eval_log['train_loss'].append(round(loss.item(), 4))
            eval_log['train_acc'].append(round(acc, 4))
            eval_log['val_loss'].append(round(eval_res['val']['loss'], 4))
            eval_log['val_acc'].append(round(eval_res['val']['acc'], 4))
            eval_log['entropy'].append(round(ent, 4))
            eval_log['elapsed'].append(round(elapsed, 1))
            current_lr = optimizer.param_groups[0]['lr']
            current_temp = temp if is_tree else 1.0
            eval_log['lr'].append(round(current_lr, 6))
            eval_log['temperature'].append(round(current_temp, 4))

            line = (f"  Step {step:>5d}/{n_steps} | "
                    f"loss={loss.item():.3f} acc={acc:.1%} | "
                    f"val_loss={eval_res['val']['loss']:.3f} "
                    f"val_acc={eval_res['val']['acc']:.1%} | "
                    f"{ms_per_step:.0f}ms/step")
            if is_tree and results['entropies']:
                line += f" | ent={ent:.3f}"
            line += f" | lr={current_lr:.1e}"
            if is_tree:
                line += f" | temp={current_temp:.3f}"
            print(line)

            if eval_res['val']['loss'] < best_val_loss:
                best_val_loss = eval_res['val']['loss']

    elapsed = time.perf_counter() - start

    # Final eval
    final = dataset.estimate_loss(model, cfg['batch'], device)

    print(f"\n  Final: train_loss={final['train']['loss']:.3f} "
          f"train_acc={final['train']['acc']:.1%} | "
          f"val_loss={final['val']['loss']:.3f} val_acc={final['val']['acc']:.1%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_steps*1000:.0f}ms/step)")

    # Generate sample
    print(f"\n  --- Sample generation ---")
    sample = dataset.generate(model, prompt="ROMEO:\n", max_tokens=200,
                              device=device, temperature=0.8)
    for line in sample.split('\n')[:8]:
        print(f"  {line}")
    print(f"  ...")

    return {
        'name': desc,
        'params': params['total'],
        'tree_pct': params.get('tree_pct', 0),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=list(MODEL_CONFIGS.keys()) + ['all'])
    parser.add_argument('--fast', action='store_true',
                        help='Fast iteration config: d_model=64, 2 layers, seq_len=128, 1000 steps')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--compile', action='store_true', default=True,
                        help='Wrap model with torch.compile for speed (default: on)')
    parser.add_argument('--no-compile', dest='compile', action='store_false',
                        help='Disable torch.compile')
    args = parser.parse_args()

    # Apply fast or full defaults for unset params
    if args.fast:
        defaults = {'steps': 2000, 'd_model': 64, 'n_layers': 2, 'seq_len': 128}
    else:
        defaults = {'steps': 2000, 'd_model': 128, 'n_layers': 4, 'seq_len': 256}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    cfg = {
        'd_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': args.n_heads,
        'seq_len': args.seq_len, 'batch': args.batch, 'n_steps': args.steps,
        'lr': args.lr, 'dropout': args.dropout,
    }

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == 'all' else [args.model]
    all_results = []

    for name in models_to_run:
        is_tree = not MODEL_CONFIGS[name].get("is_standard")
        torch.manual_seed(42)
        model = create_model(name, dataset.vocab_size, cfg, DEVICE)
        if args.compile:
            print(f"  Compiling model with torch.compile(backend='inductor')...")
            model = torch.compile(model, backend='inductor')
        result = train_model(model, dataset, name, cfg, DEVICE, is_tree=is_tree)
        all_results.append(result)
        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Summary
    if len(all_results) > 1:
        print(f"\n{'#' * 65}")
        print(f"  SHAKESPEARE RESULTS SUMMARY")
        print(f"{'#' * 65}")
        print(f"  {'Model':<40s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
        for r in all_results:
            print(f"  {r['name']:<40s} {r['final_val_acc']:>8.1%} "
                  f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
                  f"{r['params']:>8,}")
        best = max(all_results, key=lambda r: r['final_val_acc'])
        print(f"\n  Best: {best['name']} ({best['final_val_acc']:.1%} val accuracy)")

    # Save results JSON for figure generation
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "shakespeare_results.json")
    save_data = {}
    for r in all_results:
        key = [k for k, v in MODEL_CONFIGS.items() if v.get('description', k) == r['name']]
        key = key[0] if key else r['name']
        save_data[key] = r
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print("Run `python generate_figures.py` to generate figures from these results.")


if __name__ == "__main__":
    main()
