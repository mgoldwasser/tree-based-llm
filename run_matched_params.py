"""
NEW-02: Matched-Parameter Comparison — Honest benchmark at equal param counts.

Current comparisons are unfair — tree models have 1.1-1.6M params vs standard's 843K.
This experiment compares models at exactly matched parameter counts.

Configs (targeting ~850K params):
- Standard (baseline): d_model=128, 4 layers — ~843K params
- Micro-Boosted: d_model=128, 4 layers, micro_boosted — should be ~870K params
- Oblivious L+F (small): d_model=96, 4 layers, fewer trees — target ~850K
- Standard-Large: d_model=160, 4 layers — reference upper bound (~1.3M)

Also runs fast-config variants for quick iteration.

Usage:
    python run_matched_params.py              # fast config
    python run_matched_params.py --full       # full config
    python run_matched_params.py --no-compile
"""

import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Matched-parameter configurations
# Each config specifies explicit train.py arguments
MATCHED_CONFIGS = [
    {
        "name": "standard_baseline",
        "description": "Standard Transformer (baseline)",
        "args": ["--model", "standard"],
        # Uses default d_model/n_layers from --fast or full
    },
    {
        "name": "micro_boosted_matched",
        "description": "Linear+MicroTree (matched params)",
        "args": ["--model", "micro_boosted"],
        # Micro trees add minimal overhead (~5-13%), so should be close to standard
    },
    {
        "name": "micro_boosted_d2_matched",
        "description": "Linear+MicroTree depth-2 (matched params)",
        "args": ["--model", "micro_boosted_d2"],
    },
    {
        "name": "oblivious_boosted_small",
        "description": "Oblivious L+F (reduced d_model for matched params)",
        "args": ["--model", "oblivious_boosted", "--d_model", "96"],
        # Reducing d_model from 128 to 96 reduces params substantially
    },
    {
        "name": "standard_large",
        "description": "Standard Transformer (larger, reference)",
        "args": ["--model", "standard", "--d_model", "160"],
        # Upper bound reference — more params than tree models
    },
]

# Fast-config versions use smaller dims
MATCHED_CONFIGS_FAST = [
    {
        "name": "standard_baseline",
        "description": "Standard Transformer (baseline)",
        "args": ["--model", "standard"],
    },
    {
        "name": "micro_boosted_matched",
        "description": "Linear+MicroTree (matched params)",
        "args": ["--model", "micro_boosted"],
    },
    {
        "name": "micro_boosted_d2_matched",
        "description": "Linear+MicroTree depth-2 (matched params)",
        "args": ["--model", "micro_boosted_d2"],
    },
    {
        "name": "oblivious_boosted_small",
        "description": "Oblivious L+F (reduced d_model)",
        "args": ["--model", "oblivious_boosted", "--d_model", "48"],
    },
    {
        "name": "standard_large",
        "description": "Standard Transformer (larger)",
        "args": ["--model", "standard", "--d_model", "96"],
    },
]


def run_config(config, fast=True, no_compile=False):
    """Run a single configuration."""
    cmd = [sys.executable, "train.py"] + config["args"]
    if fast:
        cmd.append("--fast")
    if no_compile:
        cmd.append("--no-compile")

    label = config["name"]
    print(f"\n{'=' * 70}")
    print(f"  Running: {label} — {config['description']}")
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
        model_name = config["args"][config["args"].index("--model") + 1]
        if model_name in data:
            r = data[model_name]
            r['config_name'] = config['name']
            r['config_description'] = config['description']
            return r
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full config")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    fast = not args.full
    config_label = "FAST" if fast else "FULL"
    configs = MATCHED_CONFIGS_FAST if fast else MATCHED_CONFIGS

    print(f"\n{'#' * 70}")
    print(f"  NEW-02: MATCHED-PARAMETER COMPARISON ({config_label} CONFIG)")
    print(f"  Goal: Compare models at equal parameter counts")
    print(f"{'#' * 70}")

    all_results = {}
    for config in configs:
        result = run_config(config, fast=fast, no_compile=args.no_compile)
        if result:
            all_results[config["name"]] = result

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"matched_params_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary with param counts
    print(f"\n{'#' * 70}")
    print(f"  MATCHED-PARAMETER RESULTS ({config_label} CONFIG)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'Val Loss':>9} {'ms/step':>8} {'Params':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1].get('params', 0)):
        desc = r.get('config_description', r.get('name', name))
        print(f"  {desc:<42s} {r['final_val_acc']:>8.1%} "
              f"{r['final_val_loss']:>9.3f} {r['ms_per_step']:>7.0f}ms "
              f"{r['params']:>10,}")

    # Analysis
    if len(all_results) >= 2:
        sorted_by_acc = sorted(all_results.values(), key=lambda r: r['final_val_acc'], reverse=True)
        print(f"\n  Best accuracy: {sorted_by_acc[0].get('config_description', 'N/A')} "
              f"({sorted_by_acc[0]['final_val_acc']:.1%}, {sorted_by_acc[0]['params']:,} params)")

        # Check if any tree model beats standard at similar params
        std_result = all_results.get("standard_baseline")
        if std_result:
            std_params = std_result['params']
            std_acc = std_result['final_val_acc']
            print(f"\n  Standard baseline: {std_acc:.1%} at {std_params:,} params")
            for name, r in all_results.items():
                if name != "standard_baseline" and name != "standard_large":
                    param_ratio = r['params'] / std_params
                    acc_delta = r['final_val_acc'] - std_acc
                    print(f"  {r.get('config_description', name)}: "
                          f"{'+' if acc_delta >= 0 else ''}{acc_delta:.1%} acc, "
                          f"{param_ratio:.2f}x params")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
