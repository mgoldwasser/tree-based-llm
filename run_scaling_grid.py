"""
NEW-07: Scaling Study — Where do trees win?

Runs a grid of model sizes for standard and tree models to find the crossover
point where trees stop outperforming standard transformers.

Grid: d_model ∈ {32, 64, 128, 256}, n_layers ∈ {1, 2, 4}
Models: standard, micro_boosted, oblivious_boosted_alt

Usage:
    python run_scaling_grid.py                  # default (2000 steps each)
    python run_scaling_grid.py --steps 1000     # faster iteration
    python run_scaling_grid.py --no-compile
"""

import argparse
import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

D_MODELS = [32, 64, 128]
N_LAYERS = [1, 2, 4]
MODELS = ["standard", "micro_boosted", "oblivious_boosted_alt"]


def run_config(model_name, d_model, n_layers, steps, no_compile=False):
    """Run a single grid point."""
    # Adjust n_heads to divide d_model evenly
    n_heads = min(4, d_model // 16) if d_model >= 32 else 1
    n_heads = max(1, n_heads)
    # Ensure d_model is divisible by n_heads
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    seq_len = 128  # fixed for comparability

    cmd = [sys.executable, "train.py",
           "--model", model_name,
           "--d_model", str(d_model),
           "--n_layers", str(n_layers),
           "--n_heads", str(n_heads),
           "--seq_len", str(seq_len),
           "--steps", str(steps),
           "--fast"]  # use fast flag for base settings
    if no_compile:
        cmd.append("--no-compile")

    label = f"{model_name}/d{d_model}_L{n_layers}"
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
        if model_name in data:
            r = data[model_name]
            r['d_model'] = d_model
            r['n_layers'] = n_layers
            r['grid_label'] = label
            return r
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to include in grid")
    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"  NEW-07: SCALING STUDY")
    print(f"  Grid: d_model={D_MODELS}, n_layers={N_LAYERS}")
    print(f"  Models: {args.models}")
    print(f"  Steps: {args.steps}")
    print(f"{'#' * 70}")

    all_results = {}

    for d_model in D_MODELS:
        for n_layers in N_LAYERS:
            for model_name in args.models:
                key = f"{model_name}_d{d_model}_L{n_layers}"
                result = run_config(model_name, d_model, n_layers,
                                   args.steps, args.no_compile)
                if result:
                    all_results[key] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"scaling_grid_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print grid summary
    print(f"\n{'#' * 70}")
    print(f"  SCALING GRID RESULTS")
    print(f"{'#' * 70}")

    # Print as grid: rows = (d_model, n_layers), columns = model
    print(f"\n  Val Accuracy Grid:")
    header = f"  {'d_model':>7} {'layers':>6}"
    for m in args.models:
        header += f" {m:>20}"
    print(header)
    print(f"  {'-'*7} {'-'*6}" + f" {'-'*20}" * len(args.models))

    for d_model in D_MODELS:
        for n_layers in N_LAYERS:
            line = f"  {d_model:>7} {n_layers:>6}"
            for model_name in args.models:
                key = f"{model_name}_d{d_model}_L{n_layers}"
                if key in all_results:
                    r = all_results[key]
                    line += f" {r['final_val_acc']:>8.1%} ({r['params']:>7,})"
                else:
                    line += f" {'N/A':>20}"
            print(line)

    # Find crossover points
    print(f"\n  Tree vs Standard delta (pp):")
    for d_model in D_MODELS:
        for n_layers in N_LAYERS:
            std_key = f"standard_d{d_model}_L{n_layers}"
            if std_key not in all_results:
                continue
            std_acc = all_results[std_key]['final_val_acc']
            for model_name in args.models:
                if model_name == "standard":
                    continue
                key = f"{model_name}_d{d_model}_L{n_layers}"
                if key in all_results:
                    delta = (all_results[key]['final_val_acc'] - std_acc) * 100
                    status = "WIN" if delta > 0 else "LOSE"
                    print(f"  d{d_model}_L{n_layers} {model_name}: {delta:+.1f}pp ({status})")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
