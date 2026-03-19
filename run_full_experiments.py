"""
Run full-config experiments for paper: 6 models on Shakespeare.

Full config: d_model=128, n_layers=4, n_heads=4, seq_len=256, 2000 steps.
Also runs 3 fast-config models for comparison.

Usage:
    python run_full_experiments.py
    python run_full_experiments.py --fast-only    # just fast configs
    python run_full_experiments.py --full-only    # just full configs
"""

import json
import os
import subprocess
import sys
import time


FULL_MODELS = [
    "standard",
    "boosted",
    "oblivious_boosted",
    "oblivious_boosted_alt",
    "oblivious_boosted_vo_alt",
    "moe_boosted_alt",
]

FAST_MODELS = [
    "standard",
    "oblivious_boosted_vo_alt",
    "moe_boosted_alt",
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_model(model_name, fast=False, no_compile=False):
    """Run train.py for a single model and return parsed results."""
    cmd = [sys.executable, "train.py", "--model", model_name]
    if fast:
        cmd.append("--fast")
    if no_compile:
        cmd.append("--no-compile")
    # Full config is the default (d_model=128, n_layers=4, seq_len=256, 2000 steps)

    label = f"{'fast' if fast else 'full'}/{model_name}"
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

    # Read the results JSON that train.py saves
    results_file = os.path.join(RESULTS_DIR, "shakespeare_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        if model_name in data:
            return data[model_name]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (avoids long compile times)")
    args = parser.parse_args()

    run_full = not args.fast_only
    run_fast = not args.full_only

    all_results = {"full": {}, "fast": {}}

    # Run full-config models
    if run_full:
        print("\n" + "#" * 70)
        print("  FULL CONFIG: d_model=128, n_layers=4, seq_len=256, 2000 steps")
        print("#" * 70)
        for model_name in FULL_MODELS:
            result = run_model(model_name, fast=False, no_compile=args.no_compile)
            if result:
                all_results["full"][model_name] = result

    # Run fast-config models
    if run_fast:
        print("\n" + "#" * 70)
        print("  FAST CONFIG: d_model=64, n_layers=2, seq_len=128, 2000 steps")
        print("#" * 70)
        for model_name in FAST_MODELS:
            result = run_model(model_name, fast=True, no_compile=args.no_compile)
            if result:
                all_results["fast"][model_name] = result

    # Save combined results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"full_config_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    latest_file = os.path.join(RESULTS_DIR, "full_config_results.json")
    with open(latest_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'#' * 70}")
    print(f"  EXPERIMENT RESULTS SUMMARY")
    print(f"{'#' * 70}")

    for config_name in ["full", "fast"]:
        results = all_results[config_name]
        if not results:
            continue
        print(f"\n  {config_name.upper()} CONFIG:")
        print(f"  {'Model':<42s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>10}")
        print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
        for name, r in results.items():
            print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
                  f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
                  f"{r['params']:>10,}")

    print(f"\n  Results saved to {output_file}")
    print(f"  Latest results: {latest_file}")


if __name__ == "__main__":
    main()
