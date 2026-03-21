"""
NEW-06: Depth Ablation Study — How much depth do you actually need?

Systematic comparison of tree depths with matched leaf counts:
- Depth 1, 24 trees (48 leaves)
- Depth 2, 12 trees (48 leaves)
- Depth 3, 6 trees (48 leaves)  [closest: oblivious_boosted with boosted_trees=6]
- Depth 4, 3 trees (48 leaves)

All use Oblivious Linear+Forest (oblivious_boosted) on Shakespeare.
Runs both fast and full configs.

Usage:
    python run_depth_ablation.py              # fast config
    python run_depth_ablation.py --full       # full config
    python run_depth_ablation.py --no-compile # disable torch.compile
"""

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Depth configs with matched leaf counts (~48 leaves each)
DEPTH_CONFIGS = [
    {"name": "standard", "args": ["--model", "standard"]},
    {"name": "depth1_24t", "args": ["--model", "oblivious_boosted_d1"]},
    {"name": "depth2_12t", "args": ["--model", "oblivious_boosted_d2"]},
    {"name": "depth3_6t", "args": ["--model", "oblivious_boosted"],  # default has 24 trees
     "note": "Using boosted_trees=6 requires custom config — falling back to depth3 with 24 trees as reference"},
    {"name": "depth3_24t", "args": ["--model", "oblivious_boosted"]},
    {"name": "depth4_3t", "args": ["--model", "oblivious_boosted_d4"]},
]


def run_config(config, fast=True, no_compile=False):
    """Run a single depth configuration."""
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

    # Read results
    results_file = os.path.join(RESULTS_DIR, "shakespeare_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        # Return the most recently written model result
        model_name = config["args"][config["args"].index("--model") + 1]
        if model_name in data:
            return data[model_name]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full config instead of fast")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    fast = not args.full
    config_label = "FAST" if fast else "FULL"

    print(f"\n{'#' * 70}")
    print(f"  NEW-06: DEPTH ABLATION STUDY ({config_label} CONFIG)")
    print(f"  Goal: Find minimum useful tree depth")
    print(f"{'#' * 70}")

    all_results = {}
    for config in DEPTH_CONFIGS:
        result = run_config(config, fast=fast, no_compile=args.no_compile)
        if result:
            all_results[config["name"]] = result

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"depth_ablation_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'#' * 70}")
    print(f"  DEPTH ABLATION RESULTS ({config_label} CONFIG)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
              f"{r['params']:>10,}")

    if all_results:
        best = max(all_results.values(), key=lambda r: r['final_val_acc'])
        print(f"\n  Best: {best['name']} ({best['final_val_acc']:.1%} val accuracy)")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
