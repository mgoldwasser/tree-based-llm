"""
3000-step comparison: boosted_alt vs obl_boost_vo_alt.

Tests whether the fastest config (22ms) catches up to the speed/accuracy
sweet spot (49ms) given more training steps.

Usage:
    python run_3000_step_check.py
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


CONFIGS = [
    {
        "name": "boosted_alt_compiled",
        "description": "Linear+Forest alternating + compile",
        "proj_type": "boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_every_n": 2,
    },
    {
        "name": "obl_boost_vo_alt_compiled",
        "description": "Oblivious L+F V+O alt + compile",
        "proj_type": "oblivious_boosted",
        "boosted_trees": 24, "boosted_depth": 3,
        "tree_targets": "vo",
        "tree_every_n": 2,
    },
]


def train_and_eval(config, dataset, cfg, device):
    desc = config["description"]

    torch.manual_seed(42)
    proj_kwargs = {k: v for k, v in config.items()
                   if k not in ("name", "description", "proj_type")}
    model = TreeTransformer(
        vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
        dropout=cfg['dropout'], use_tree_ffn=False, task="lm",
        proj_type=config["proj_type"], **proj_kwargs,
    ).to(device)

    raw_model = model
    params = count_parameters(raw_model)
    model = torch.compile(model, backend='inductor')
    raw_model = model._orig_mod

    print(f"\n{'=' * 65}")
    print(f"  {desc}")
    print(f"  Params: {params['total']:,} (tree: {params['tree_pct']:.1f}%)")
    print(f"{'=' * 65}")

    optimizer = make_optimizer(raw_model, lr=cfg['lr'])

    n_steps = cfg['n_steps']
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    eval_interval = max(1, n_steps // 10)
    model.train()
    start = time.perf_counter()

    eval_log = {'steps': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'entropy': [], 'elapsed': [],
                'lr': [], 'temperature': []}

    temp = 1.0
    for step in range(1, n_steps + 1):
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
            ent = get_routing_entropy(raw_model)
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
                    f"{ms_per_step:.0f}ms/step | ent={ent:.3f} | "
                    f"temp={temp:.3f} | lr={current_lr:.1e}")
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
        'final_train_loss': final['train']['loss'],
        'final_train_acc': final['train']['acc'],
        'final_val_loss': final['val']['loss'],
        'final_val_acc': final['val']['acc'],
        'time': elapsed,
        'ms_per_step': elapsed / n_steps * 1000,
        'eval_log': eval_log,
    }


def main():
    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    cfg = {
        'd_model': 64, 'n_layers': 2, 'n_heads': 4,
        'seq_len': 128, 'batch': 32, 'n_steps': 5000,
        'lr': 3e-4, 'dropout': 0.0,
    }

    print(f"3000-Step Comparison: boosted_alt vs obl_boost_vo_alt")
    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])
    all_results = {}

    for config in CONFIGS:
        result = train_and_eval(config, dataset, cfg, DEVICE)
        all_results[config['name']] = result

    # Summary
    print(f"\n{'#' * 75}")
    print(f"  3000-STEP COMPARISON")
    print(f"{'#' * 75}")
    print(f"  {'Config':<40s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8}")
    for r in all_results.values():
        print(f"  {r['name']:<40s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms")

    # Show learning curves side by side
    print(f"\n  Step-by-step val_acc comparison:")
    for name, r in all_results.items():
        log = r['eval_log']
        pairs = [f"{s}:{a:.1%}" for s, a in zip(log['steps'], log['val_acc'])]
        print(f"  {r['name'][:38]:<38s}  {', '.join(pairs)}")

    # Save
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("results", f"3000_step_check_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
