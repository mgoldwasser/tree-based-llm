"""
NEW-03: Progressive Sparsification — Hard routing at inference.

Train with soft routing (all leaves contribute), then evaluate with top-k
hard routing. Measures accuracy retention when only top-k leaves are active.

Tests: top-1, top-2, top-4 hard routing on depth-3 trees (8 leaves).
Success criterion: <0.5pp accuracy loss with top-2 hard routing.

Usage:
    python run_hard_routing.py              # fast config
    python run_hard_routing.py --full       # full config
    python run_hard_routing.py --no-compile
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
    set_hard_routing,
)
from data import ShakespeareDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_eval_hard_routing(args):
    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    if args.fast:
        cfg = {'d_model': 64, 'n_layers': 2, 'n_heads': 4,
               'seq_len': 128, 'batch': 32, 'n_steps': 2000,
               'lr': 3e-4, 'dropout': 0.0}
    else:
        cfg = {'d_model': 128, 'n_layers': 4, 'n_heads': 4,
               'seq_len': 256, 'batch': 32, 'n_steps': 2000,
               'lr': 3e-4, 'dropout': 0.0}

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])

    # Models to test hard routing on
    model_configs = [
        ("oblivious_boosted", {
            "proj_type": "oblivious_boosted", "use_tree_ffn": False,
            "boosted_trees": 24, "boosted_depth": 3,
        }),
        ("oblivious_boosted_alt", {
            "proj_type": "oblivious_boosted", "use_tree_ffn": False,
            "boosted_trees": 24, "boosted_depth": 3,
            "tree_every_n": 2,
        }),
    ]

    top_k_values = [1, 2, 4]
    all_results = {}

    for model_name, mc in model_configs:
        print(f"\n{'=' * 65}")
        print(f"  Training: {model_name} (soft routing)")
        print(f"{'=' * 65}")

        torch.manual_seed(42)
        model = TreeTransformer(
            vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
            n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
            max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
            dropout=cfg['dropout'], task="lm",
            use_tree_ffn=mc.get("use_tree_ffn", False),
            proj_type=mc["proj_type"],
            tree_every_n=mc.get("tree_every_n", 1),
            **{k: v for k, v in mc.items()
               if k not in ("proj_type", "use_tree_ffn", "tree_every_n")},
        ).to(DEVICE)

        if args.compile:
            model = torch.compile(model, backend='inductor')

        raw_model = getattr(model, '_orig_mod', model)
        params = count_parameters(raw_model)
        print(f"  Params: {params['total']:,} (tree: {params['tree_pct']:.1f}%)")

        optimizer = make_optimizer(raw_model, lr=cfg['lr'])
        n_steps = cfg['n_steps']
        warmup_steps = 100

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Train with soft routing
        model.train()
        start = time.perf_counter()
        eval_interval = max(1, n_steps // 5)

        for step in range(1, n_steps + 1):
            progress = step / n_steps
            if progress < 0.5:
                temp = 1.0
            else:
                phase = (progress - 0.5) / 0.5
                temp = 0.7 + 0.3 * (1 + math.cos(math.pi * phase)) / 2
            set_temperature(raw_model, temp)

            x, y = dataset.get_batch(cfg['batch'], DEVICE, 'train')
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
            loss = loss + tree_regularization_loss(raw_model, 0.005)
            loss = loss + leaf_balancing_loss(raw_model, 0.01)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_interval == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                elapsed = time.perf_counter() - start
                print(f"  Step {step:>5d}/{n_steps} | loss={loss.item():.3f} acc={acc:.1%} | "
                      f"{elapsed/step*1000:.0f}ms/step")

        elapsed = time.perf_counter() - start

        # Evaluate with soft routing (baseline)
        print(f"\n  Evaluating soft routing (baseline)...")
        set_hard_routing(raw_model, enabled=False)
        soft_eval = dataset.estimate_loss(model, cfg['batch'], DEVICE, n_batches=50)
        soft_acc = soft_eval['val']['acc']
        soft_loss = soft_eval['val']['loss']
        print(f"  Soft routing: val_acc={soft_acc:.1%}, val_loss={soft_loss:.3f}")

        model_results = {
            'model': model_name,
            'params': params['total'],
            'train_time': elapsed,
            'ms_per_step': elapsed / n_steps * 1000,
            'soft_val_acc': soft_acc,
            'soft_val_loss': soft_loss,
            'hard_routing': {},
        }

        # Evaluate with hard routing at different top-k values
        for k in top_k_values:
            print(f"  Evaluating hard routing (top-{k})...")
            set_hard_routing(raw_model, enabled=True, top_k=k)
            hard_eval = dataset.estimate_loss(model, cfg['batch'], DEVICE, n_batches=50)
            hard_acc = hard_eval['val']['acc']
            hard_loss = hard_eval['val']['loss']
            acc_delta = hard_acc - soft_acc
            print(f"  Top-{k}: val_acc={hard_acc:.1%} ({acc_delta:+.1%}), val_loss={hard_loss:.3f}")

            model_results['hard_routing'][f'top_{k}'] = {
                'val_acc': hard_acc,
                'val_loss': hard_loss,
                'acc_delta': acc_delta,
                'acc_retained_pct': hard_acc / soft_acc * 100 if soft_acc > 0 else 0,
            }

        # Reset to soft routing
        set_hard_routing(raw_model, enabled=False)
        all_results[model_name] = model_results

        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"hard_routing_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'#' * 65}")
    print(f"  HARD ROUTING RESULTS")
    print(f"{'#' * 65}")
    for name, r in all_results.items():
        print(f"\n  {name} (soft baseline: {r['soft_val_acc']:.1%})")
        for k_name, hr in r['hard_routing'].items():
            status = "PASS" if abs(hr['acc_delta']) < 0.005 else "FAIL"
            print(f"    {k_name}: {hr['val_acc']:.1%} "
                  f"({hr['acc_delta']:+.1%}, "
                  f"{hr['acc_retained_pct']:.1f}% retained) — {status}")

    print(f"\n  Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--full", dest="fast", action="store_false")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    args = parser.parse_args()
    train_and_eval_hard_routing(args)


if __name__ == "__main__":
    main()
