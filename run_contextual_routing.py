"""
NEW-05: Contextual Routing — Route on context, not just token.

Compares context-blind vs context-aware tree routing.
Tests whether routing on EMA of recent hidden states improves accuracy.

Usage:
    python run_contextual_routing.py              # fast config
    python run_contextual_routing.py --full
    python run_contextual_routing.py --no-compile
"""

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = [
    {"name": "standard", "args": ["--model", "standard"]},
    {"name": "oblivious_boosted", "args": ["--model", "oblivious_boosted"]},
    {"name": "contextual_boosted", "args": ["--model", "contextual_boosted"]},
    {"name": "oblivious_boosted_alt", "args": ["--model", "oblivious_boosted_alt"]},
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
    print(f"  NEW-05: CONTEXTUAL ROUTING EXPERIMENT ({config_label} CONFIG)")
    print(f"  Goal: Does context-aware routing improve accuracy?")
    print(f"{'#' * 70}")

    all_results = {}
    for config in CONFIGS:
        result = run_config(config, fast=fast, no_compile=args.no_compile)
        if result:
            all_results[config["name"]] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"contextual_routing_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'#' * 70}")
    print(f"  CONTEXTUAL ROUTING RESULTS ({config_label} CONFIG)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
              f"{r['params']:>10,}")

    # Compare contextual vs non-contextual
    ctx = all_results.get("contextual_boosted")
    obliv = all_results.get("oblivious_boosted")
    if ctx and obliv:
        delta = (ctx['final_val_acc'] - obliv['final_val_acc']) * 100
        print(f"\n  Contextual vs Oblivious L+F: {delta:+.1f}pp")
        print(f"  Param overhead: {ctx['params'] - obliv['params']:+,}")
        print(f"  Speed ratio: {ctx['ms_per_step'] / obliv['ms_per_step']:.2f}x")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
