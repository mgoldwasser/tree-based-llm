"""
Speed benchmark: torch.compile + shared routing optimizations.

Compares key models with/without compile and with/without shared routing.
Results saved to results/speed_benchmark.json

Usage:
    python run_speed_benchmark.py              # full benchmark
    python run_speed_benchmark.py --quick      # 500 steps, fewer models
"""

import argparse
import json
import os
import subprocess
import sys
import time


# Models to benchmark, grouped by category
BENCHMARK_GROUPS = {
    "baseline": ["standard"],
    "independent_routing": ["batched", "boosted", "oblivious_boosted"],
    "shared_routing": ["batched_shared", "boosted_shared", "oblivious_boosted_shared"],
}

QUICK_GROUPS = {
    "baseline": ["standard"],
    "independent_routing": ["batched", "boosted"],
    "shared_routing": ["batched_shared", "boosted_shared"],
}


def run_single_model(model_name, steps, compile_flag, fast=True):
    """Run train.py for a single model and return parsed results."""
    cmd = [sys.executable, "train.py", "--fast" if fast else "", "--model", model_name,
           "--steps", str(steps)]
    if compile_flag:
        cmd.append("--compile")
    cmd = [c for c in cmd if c]  # remove empty strings

    print(f"\n{'='*60}")
    print(f"  Running: {model_name} {'(compiled)' if compile_flag else '(no compile)'}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall_time = time.time() - start

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:]}")
        return None

    # Parse results from the saved JSON
    results_file = os.path.join("results", "shakespeare_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        if model_name in data:
            entry = data[model_name]
            entry['wall_time'] = wall_time
            entry['compiled'] = compile_flag
            return entry

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick benchmark: 500 steps, fewer models')
    args = parser.parse_args()

    steps = 500 if args.quick else 2000
    groups = QUICK_GROUPS if args.quick else BENCHMARK_GROUPS

    os.makedirs("results", exist_ok=True)

    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Run each model without compile, then with compile
    for compile_flag in [False, True]:
        mode = "compiled" if compile_flag else "no_compile"
        for group_name, models in groups.items():
            for model_name in models:
                key = f"{model_name}__{mode}"
                result = run_single_model(model_name, steps, compile_flag)
                if result:
                    all_results[key] = result

    # Save structured results
    output_file = os.path.join("results", f"speed_benchmark_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Also save as latest
    latest_file = os.path.join("results", "speed_benchmark_latest.json")
    with open(latest_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'#'*70}")
    print(f"  SPEED BENCHMARK RESULTS ({steps} steps, --fast)")
    print(f"{'#'*70}")
    print(f"  {'Model':<35s} {'Compile':>8} {'Val Acc':>8} {'ms/step':>8} {'Params':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        mode = "yes" if r.get('compiled') else "no"
        print(f"  {r['name']:<35s} {mode:>8} {r['final_val_acc']:>8.1%} "
              f"{r['ms_per_step']:>7.0f}ms {r['params']:>8,}")

    print(f"\n  Results saved to {output_file}")
    print(f"  Latest results: {latest_file}")


if __name__ == "__main__":
    main()
