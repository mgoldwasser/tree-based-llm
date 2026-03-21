"""
NEW-01: Micro-Tree Experiment — Tests core thesis of minimal routing overhead.

Implements depth-1 and depth-2 trees with low-rank leaf factorization.
Compares against standard transformer at matched parameter counts.

Target: Val acc > standard baseline with < 5% parameter increase.

Configs:
- Standard Transformer (baseline)
- MicroTree Forest (depth-1, 4 trees, rank-8 leaves, attn only)
- Linear+MicroTree (depth-1, 4 trees, rank-8, attn only)
- Linear+MicroTree (depth-2, 4 trees, rank-8, attn only)
- Linear+MicroTree (depth-1, 8 trees, rank-4, attn only) — more trees, lower rank

Usage:
    python run_micro_tree.py              # fast config
    python run_micro_tree.py --full       # full config
    python run_micro_tree.py --no-compile
"""

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


CONFIGS = [
    {
        "name": "standard",
        "args": ["--model", "standard"],
    },
    {
        "name": "micro_tree_d1_4t_r8",
        "args": ["--model", "micro_tree"],
    },
    {
        "name": "micro_boosted_d1_4t_r8",
        "args": ["--model", "micro_boosted"],
    },
    {
        "name": "micro_boosted_d2_4t_r8",
        "args": ["--model", "micro_boosted_d2"],
    },
    {
        "name": "oblivious_boosted_ref",
        "args": ["--model", "oblivious_boosted"],
    },
]


def run_config(config, fast=True, no_compile=False):
    cmd = [sys.executable, "train.py"] + config["args"]
    if fast:
        cmd.append("--fast")
    if no_compile:
        cmd.append("--no-compile")

    label = config["name"]
    print(f"\n{'=' * 70}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True,
                           cwd=os.path.dirname(__file__) or ".")
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: {label} failed (exit code {result.returncode})")
        return None

    print(f"  Completed {label} in {elapsed:.0f}s")

    results_file = os.path.join(RESULTS_DIR, "shakespeare_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        model_name = config["args"][config["args"].index("--model") + 1]
        if model_name in data:
            return data[model_name]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    fast = not args.full
    config_label = "FAST" if fast else "FULL"

    print(f"\n{'#' * 70}")
    print(f"  NEW-01: MICRO-TREE EXPERIMENT ({config_label} CONFIG)")
    print(f"  Goal: Minimal routing overhead + low-rank leaves beats linear?")
    print(f"{'#' * 70}")

    all_results = {}
    for config in CONFIGS:
        result = run_config(config, fast=fast, no_compile=args.no_compile)
        if result:
            all_results[config["name"]] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"micro_tree_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'#' * 70}")
    print(f"  MICRO-TREE RESULTS ({config_label} CONFIG)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
              f"{r['params']:>10,}")

    # Analysis vs standard
    std = all_results.get("standard")
    if std:
        print(f"\n  Standard baseline: {std['final_val_acc']:.1%} at {std['params']:,} params")
        for name, r in all_results.items():
            if name != "standard":
                overhead = (r['params'] - std['params']) / std['params'] * 100
                acc_delta = (r['final_val_acc'] - std['final_val_acc']) * 100
                speed_ratio = r['ms_per_step'] / std['ms_per_step']
                status = "WIN" if r['final_val_acc'] > std['final_val_acc'] else "LOSE"
                print(f"  {r['name']}: {'+' if acc_delta >= 0 else ''}{acc_delta:.1f}pp acc, "
                      f"{overhead:+.1f}% params, {speed_ratio:.2f}x speed — {status}")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
