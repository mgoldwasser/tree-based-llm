"""
Speed Enhancement Benchmark — Compare all new fast projection types.

Tests 6 new speed-optimized projections against baselines:
- Standard Transformer (baseline)
- Micro-Boosted (current best lightweight tree)
- Gated (GLU-style, 1 and 2 gates)
- Dynamic Linear (1 and 4 modulations)
- Low-rank Routing (r=16)
- Recursive (3 iterations)
- Chunked Routing (chunk=16)
- Product-Key (C=16)

Usage:
    python run_speed_enhancements.py              # fast config
    python run_speed_enhancements.py --full
    python run_speed_enhancements.py --no-compile
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
    {"name": "micro_boosted", "args": ["--model", "micro_boosted"]},
    {"name": "gated_boosted", "args": ["--model", "gated_boosted"]},
    {"name": "gated_boosted_d2", "args": ["--model", "gated_boosted_d2"]},
    {"name": "dynamic", "args": ["--model", "dynamic"]},
    {"name": "dynamic_boosted", "args": ["--model", "dynamic_boosted"]},
    {"name": "lowrank_boosted", "args": ["--model", "lowrank_boosted"]},
    {"name": "recursive_boosted", "args": ["--model", "recursive_boosted"]},
    {"name": "chunked_boosted", "args": ["--model", "chunked_boosted"]},
    {"name": "product_key_boosted", "args": ["--model", "product_key_boosted"]},
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
    print(f"  SPEED ENHANCEMENT BENCHMARK ({config_label} CONFIG)")
    print(f"  Goal: Find projections that match tree accuracy with less overhead")
    print(f"{'#' * 70}")

    all_results = {}
    for config in CONFIGS:
        result = run_config(config, fast=fast, no_compile=args.no_compile)
        if result:
            all_results[config["name"]] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"speed_enhancements_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary sorted by ms/step
    print(f"\n{'#' * 70}")
    print(f"  SPEED ENHANCEMENT RESULTS ({config_label} CONFIG)")
    print(f"{'#' * 70}")
    print(f"  {'Config':<42s} {'Val Acc':>8} {'ms/step':>8} {'Slowdown':>9} {'Params':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

    std = all_results.get("standard")
    std_ms = std['ms_per_step'] if std else 1

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['ms_per_step']):
        slowdown = r['ms_per_step'] / std_ms if std_ms > 0 else 0
        print(f"  {r['name']:<42s} {r['final_val_acc']:>8.1%} "
              f"{r['ms_per_step']:>7.0f}ms {slowdown:>8.1f}x "
              f"{r['params']:>10,}")

    # Efficiency analysis: accuracy per unit of speed cost
    if std:
        print(f"\n  Efficiency ranking (acc gain per unit slowdown):")
        rankings = []
        for name, r in all_results.items():
            if name == "standard":
                continue
            acc_gain = r['final_val_acc'] - std['final_val_acc']
            slowdown = r['ms_per_step'] / std_ms
            efficiency = acc_gain / max(slowdown - 1, 0.01) * 100  # pp per x slowdown
            rankings.append((name, r, acc_gain, slowdown, efficiency))

        rankings.sort(key=lambda x: x[4], reverse=True)
        for name, r, acc_gain, slowdown, efficiency in rankings:
            print(f"  {r['name']:<42s} {acc_gain:+.1%} acc / {slowdown:.1f}x speed = "
                  f"{efficiency:.1f} efficiency")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
