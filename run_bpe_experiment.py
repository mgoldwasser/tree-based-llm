"""
NEW-08: BPE Tokenization — Do trees prefer semantic tokens?

Compares character-level vs BPE tokenization for standard and tree models.
Trees should benefit more from BPE because routing decisions would correlate
with semantic categories (words/subwords) rather than arbitrary characters.

Usage:
    python run_bpe_experiment.py              # fast config
    python run_bpe_experiment.py --full
    python run_bpe_experiment.py --no-compile
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
)
from train import StandardTransformer
from data import ShakespeareDataset, BPEShakespeareDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_model(model, dataset, cfg, device, is_tree=False, label=""):
    """Train a model and return results."""
    raw_model = getattr(model, '_orig_mod', model)
    params = count_parameters(raw_model)

    if is_tree:
        optimizer = make_optimizer(raw_model, lr=cfg['lr'])
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    n_steps = cfg['n_steps']
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    start = time.perf_counter()
    eval_interval = max(1, n_steps // 5)

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

        if step % eval_interval == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            elapsed = time.perf_counter() - start
            print(f"  [{label}] Step {step:>5d}/{n_steps} | loss={loss.item():.3f} "
                  f"acc={acc:.1%} | {elapsed/step*1000:.0f}ms/step")

    elapsed = time.perf_counter() - start
    final = dataset.estimate_loss(model, cfg['batch'], device)

    return {
        'name': label,
        'params': params['total'],
        'final_val_acc': final['val']['acc'],
        'final_val_loss': final['val']['loss'],
        'final_train_acc': final['train']['acc'],
        'final_train_loss': final['train']['loss'],
        'time': elapsed,
        'ms_per_step': elapsed / n_steps * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--full", dest="fast", action="store_false")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--bpe-vocab", type=int, default=500,
                        help="BPE vocabulary size (default: 500)")
    args = parser.parse_args()

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

    # Load both datasets
    print("\n--- Character-level dataset ---")
    char_dataset = ShakespeareDataset(block_size=cfg['seq_len'])

    print("\n--- BPE dataset ---")
    bpe_dataset = BPEShakespeareDataset(block_size=cfg['seq_len'],
                                          bpe_vocab_size=args.bpe_vocab)

    # Model configurations to test
    model_specs = [
        ("standard", False),
        ("oblivious_boosted_alt", True),
        ("micro_boosted", True),
    ]

    all_results = {}

    for tokenization, dataset, tok_label in [
        ("char", char_dataset, "char"),
        ("bpe", bpe_dataset, f"bpe{args.bpe_vocab}"),
    ]:
        for model_name, is_tree in model_specs:
            label = f"{model_name}/{tok_label}"
            print(f"\n{'=' * 65}")
            print(f"  {label} (vocab={dataset.vocab_size})")
            print(f"{'=' * 65}")

            torch.manual_seed(42)

            if model_name == "standard":
                model = StandardTransformer(
                    dataset.vocab_size, cfg['d_model'], cfg['n_layers'],
                    cfg['n_heads'], cfg['seq_len'], cfg['dropout'],
                ).to(DEVICE)
            elif model_name == "oblivious_boosted_alt":
                model = TreeTransformer(
                    vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
                    n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
                    max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
                    dropout=cfg['dropout'], use_tree_ffn=False, task="lm",
                    proj_type="oblivious_boosted",
                    boosted_trees=24, boosted_depth=3, tree_every_n=2,
                ).to(DEVICE)
            elif model_name == "micro_boosted":
                model = TreeTransformer(
                    vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
                    n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
                    max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
                    dropout=cfg['dropout'], use_tree_ffn=False, task="lm",
                    proj_type="micro_boosted",
                    n_trees=4, tree_depth=1, leaf_rank=8,
                ).to(DEVICE)

            if args.compile:
                model = torch.compile(model, backend='inductor')

            result = train_model(model, dataset, cfg, DEVICE,
                                is_tree=is_tree, label=label)
            result['tokenization'] = tokenization
            result['vocab_size'] = dataset.vocab_size
            all_results[f"{model_name}_{tok_label}"] = result

            del model
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"bpe_experiment_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'#' * 70}")
    print(f"  BPE EXPERIMENT RESULTS")
    print(f"{'#' * 70}")
    print(f"  {'Config':<40s} {'Tok':>5} {'Val Acc':>8} {'Val Loss':>9} {'Params':>10}")
    print(f"  {'-'*40} {'-'*5} {'-'*8} {'-'*9} {'-'*10}")
    for key, r in all_results.items():
        print(f"  {r['name']:<40s} {r['tokenization']:>5} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['params']:>10,}")

    # Compare char vs BPE gap for each model
    print(f"\n  BPE benefit by model (val_acc delta):")
    for model_name, _ in model_specs:
        char_key = f"{model_name}_char"
        bpe_key = f"{model_name}_bpe{args.bpe_vocab}"
        if char_key in all_results and bpe_key in all_results:
            char_acc = all_results[char_key]['final_val_acc']
            bpe_acc = all_results[bpe_key]['final_val_acc']
            delta = (bpe_acc - char_acc) * 100
            print(f"  {model_name}: {delta:+.1f}pp (char={char_acc:.1%}, bpe={bpe_acc:.1%})")

    # Check if trees benefit more from BPE
    std_char = all_results.get("standard_char", {}).get('final_val_acc', 0)
    std_bpe = all_results.get(f"standard_bpe{args.bpe_vocab}", {}).get('final_val_acc', 0)
    std_delta = std_bpe - std_char

    for model_name, _ in model_specs:
        if model_name == "standard":
            continue
        tree_char = all_results.get(f"{model_name}_char", {}).get('final_val_acc', 0)
        tree_bpe = all_results.get(f"{model_name}_bpe{args.bpe_vocab}", {}).get('final_val_acc', 0)
        tree_delta = tree_bpe - tree_char
        relative = (tree_delta - std_delta) * 100
        print(f"\n  {model_name} BPE benefit vs standard: {relative:+.1f}pp relative")
        if relative > 0:
            print(f"    Trees benefit MORE from BPE than standard")
        else:
            print(f"    Trees benefit LESS from BPE than standard")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
