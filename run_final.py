"""
Final large-scale training: d128/6L comparison for paper.

Models: standard, boosted (Linear+Forest), oblivious_boosted_alt (Oblivious L+F alternating)
Includes calibration, hard routing eval, and text generation.

Usage:
    python run_final.py                    # default: 4h budget, auto-calibrate
    python run_final.py --steps 10000      # fixed step count
    python run_final.py --budget 2         # 2-hour budget
    python run_final.py --no-compile       # disable torch.compile
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from datetime import datetime

from main import (
    TreeTransformer, tree_regularization_loss, leaf_balancing_loss,
    count_parameters, set_temperature, get_routing_entropy, make_optimizer,
    set_hard_routing,
)
from train import StandardTransformer, MODEL_CONFIGS, unwrap
from data import ShakespeareDataset


MODELS_TO_RUN = ["standard", "boosted", "oblivious_boosted_alt"]
PROMPTS = [
    "ROMEO:\nO, ",
    "To be, or not to be",
    "KING HENRY:\nOnce more unto the breach, dear friends,\n",
]


def calibrate(model, dataset, cfg, device, n_steps=100):
    """Run n_steps to measure ms/step."""
    model.train()
    optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=cfg['lr'], weight_decay=0.01)
    start = time.perf_counter()
    for _ in range(n_steps):
        x, y = dataset.get_batch(cfg['batch'], device, 'train')
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    return elapsed / n_steps * 1000  # ms/step


def create_model(name, vocab_size, cfg, device):
    """Create model by name."""
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


def train_model(model, dataset, name, cfg, device, is_tree=False):
    """Full training with eval logging."""
    mc = MODEL_CONFIGS[name]
    desc = mc["description"]
    raw_model = unwrap(model)
    params = count_parameters(raw_model)
    n_steps = cfg['n_steps']

    print(f"\n{'=' * 65}")
    print(f"  {desc} (d={cfg['d_model']}, L={cfg['n_layers']})")
    print(f"  Params: {params['total']:,}" +
          (f" (tree: {params['tree_pct']:.1f}%)" if params.get('tree', 0) > 0 else ""))
    print(f"  Steps: {n_steps:,}")
    print(f"{'=' * 65}")

    if is_tree:
        optimizer = make_optimizer(raw_model, lr=cfg['lr'])
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    eval_interval = max(1, n_steps // 20)  # 20 eval points
    model.train()
    start = time.perf_counter()

    best_val_loss = float('inf')
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

        if step % eval_interval == 0 or step == 1 or step == n_steps:
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
            eval_log['temperature'].append(round(temp if is_tree else 1.0, 4))

            line = (f"  Step {step:>6d}/{n_steps} | "
                    f"loss={loss.item():.3f} acc={acc:.1%} | "
                    f"val_loss={eval_res['val']['loss']:.3f} "
                    f"val_acc={eval_res['val']['acc']:.1%} | "
                    f"{ms_per_step:.0f}ms/step")
            if is_tree:
                line += f" | ent={ent:.3f} temp={temp:.3f}"
            line += f" | lr={current_lr:.1e}"
            print(line)

            if eval_res['val']['loss'] < best_val_loss:
                best_val_loss = eval_res['val']['loss']

    elapsed = time.perf_counter() - start
    final = dataset.estimate_loss(model, cfg['batch'], device)

    print(f"\n  Final: train_loss={final['train']['loss']:.3f} "
          f"train_acc={final['train']['acc']:.1%} | "
          f"val_loss={final['val']['loss']:.3f} val_acc={final['val']['acc']:.1%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_steps*1000:.0f}ms/step)")

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


def eval_hard_routing(model, dataset, name, device, cfg):
    """Evaluate hard routing at top-1, top-2, top-4."""
    raw_model = unwrap(model)
    results = {}

    # Soft baseline
    set_hard_routing(raw_model, False)
    soft_eval = dataset.estimate_loss(model, cfg['batch'], device)
    soft_acc = soft_eval['val']['acc']
    results['soft'] = {'val_acc': soft_acc, 'val_loss': soft_eval['val']['loss']}
    print(f"    Soft routing: val_acc={soft_acc:.4f}")

    for top_k in [1, 2, 4]:
        set_hard_routing(raw_model, True, top_k=top_k)
        hard_eval = dataset.estimate_loss(model, cfg['batch'], device)
        hard_acc = hard_eval['val']['acc']
        retention = hard_acc / soft_acc * 100 if soft_acc > 0 else 0
        results[f'top_{top_k}'] = {
            'val_acc': hard_acc,
            'val_loss': hard_eval['val']['loss'],
            'retention_pct': round(retention, 2),
        }
        print(f"    Top-{top_k} hard: val_acc={hard_acc:.4f} ({retention:.1f}% retention)")

    # Restore soft routing
    set_hard_routing(raw_model, False)
    return results


def generate_samples(model, dataset, device):
    """Generate text samples from multiple prompts."""
    samples = {}
    for prompt in PROMPTS:
        text = dataset.generate(model, prompt=prompt, max_tokens=500,
                                device=device, temperature=0.8)
        samples[prompt.strip()] = text
        # Print first few lines
        lines = text.split('\n')[:6]
        print(f"    Prompt: {prompt.strip()[:40]}...")
        for line in lines:
            print(f"      {line}")
        print()
    return samples


def main():
    parser = argparse.ArgumentParser(description="Final large-scale training")
    parser.add_argument('--budget', type=float, default=4.0,
                        help='Time budget in hours (default: 4)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Fixed step count (overrides calibration)')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no-compile', dest='compile', action='store_false')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: standard,boosted,oblivious_boosted_alt)')
    args = parser.parse_args()

    models_to_run = args.models.split(',') if args.models else MODELS_TO_RUN

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    cfg = {
        'd_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': args.n_heads,
        'seq_len': args.seq_len, 'batch': args.batch, 'lr': args.lr,
        'dropout': args.dropout, 'n_steps': args.steps or 10000,
    }

    print(f"{'#' * 65}")
    print(f"  FINAL TRAINING: d={cfg['d_model']}, L={cfg['n_layers']}")
    print(f"  Device: {DEVICE}")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"  Budget: {args.budget}h")
    print(f"{'#' * 65}")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])

    # =========================================================================
    # Phase 1: Calibration
    # =========================================================================
    if args.steps is None:
        print(f"\n--- Phase 1: Calibration (100 steps each) ---")
        speeds = {}
        for name in models_to_run:
            torch.manual_seed(42)
            model = create_model(name, dataset.vocab_size, cfg, DEVICE)
            if args.compile:
                model = torch.compile(model, backend='inductor')
            ms = calibrate(model, dataset, cfg, DEVICE, n_steps=100)
            speeds[name] = ms
            print(f"  {name}: {ms:.1f} ms/step")
            del model

        # Calculate max steps fitting in budget
        budget_sec = args.budget * 3600
        # Account for eval overhead (~10% of training time)
        effective_budget = budget_sec * 0.9
        # Total time = sum(steps * ms/step) for all models
        total_ms_per_step = sum(speeds.values())
        max_steps = int(effective_budget * 1000 / total_ms_per_step)
        # Round to nearest 500
        max_steps = max(1000, (max_steps // 500) * 500)
        cfg['n_steps'] = max_steps

        total_time_est = max_steps * total_ms_per_step / 1000 / 3600
        print(f"\n  Calibrated step count: {max_steps:,}")
        print(f"  Estimated total time: {total_time_est:.1f}h")
        for name in models_to_run:
            est = max_steps * speeds[name] / 1000 / 60
            print(f"    {name}: ~{est:.0f} min")
    else:
        print(f"\n  Using fixed step count: {cfg['n_steps']:,}")

    # =========================================================================
    # Phase 2: Training
    # =========================================================================
    print(f"\n--- Phase 2: Training ({cfg['n_steps']:,} steps each) ---")
    all_results = {}
    trained_models = {}  # Keep references for hard routing eval

    for name in models_to_run:
        is_tree = not MODEL_CONFIGS[name].get("is_standard")
        torch.manual_seed(42)
        model = create_model(name, dataset.vocab_size, cfg, DEVICE)
        if args.compile:
            print(f"  Compiling {name}...")
            model = torch.compile(model, backend='inductor')

        result = train_model(model, dataset, name, cfg, DEVICE, is_tree=is_tree)
        result['d_model'] = cfg['d_model']
        result['n_layers'] = cfg['n_layers']
        result['n_steps'] = cfg['n_steps']
        all_results[name] = result

        trained_models[name] = model

        # Save incrementally
        save_results(all_results, cfg)

    # =========================================================================
    # Phase 3: Post-training evaluation
    # =========================================================================
    print(f"\n--- Phase 3: Hard routing evaluation ---")
    for name, model in trained_models.items():
        print(f"\n  {MODEL_CONFIGS[name]['description']}:")
        hard_results = eval_hard_routing(model, dataset, name, DEVICE, cfg)
        all_results[name]['hard_routing'] = hard_results

    print(f"\n--- Phase 3b: Text generation ---")
    for name in models_to_run:
        model = trained_models[name]
        print(f"\n  {MODEL_CONFIGS[name]['description']}:")
        samples = generate_samples(model, dataset, DEVICE)
        all_results[name]['samples'] = samples

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'#' * 65}")
    print(f"  FINAL RESULTS SUMMARY (d={cfg['d_model']}, L={cfg['n_layers']}, {cfg['n_steps']:,} steps)")
    print(f"{'#' * 65}")
    print(f"  {'Model':<40s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
    for name in models_to_run:
        r = all_results[name]
        print(f"  {r['name']:<40s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
              f"{r['params']:>8,}")

    best = max(all_results.values(), key=lambda r: r['final_val_acc'])
    print(f"\n  Best: {best['name']} ({best['final_val_acc']:.1%} val accuracy)")

    # Final save
    save_results(all_results, cfg)
    print(f"\nDone!")


def save_results(results, cfg):
    """Save results to timestamped JSON."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    save_data = {
        'config': cfg,
        'models': results,
        'timestamp': timestamp,
    }
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {filepath}")

    # Also save as latest
    latest = os.path.join(results_dir, "final_latest.json")
    with open(latest, 'w') as f:
        json.dump(save_data, f, indent=2)


if __name__ == "__main__":
    main()
