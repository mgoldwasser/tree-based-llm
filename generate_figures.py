"""
Generate publication-quality figures from previously captured training results.
Uses hardcoded data points from the actual training runs.
Run: python generate_figures.py
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2,
})
COLORS = {'standard': '#2196F3', 'batched': '#FF9800', 'boosted': '#4CAF50'}
LABELS = {
    'standard': 'Standard Transformer',
    'batched': 'Batched Forest (attn)',
    'boosted': 'Boosted Forest (attn)',
}

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data from actual training runs (Shakespeare, 2000 steps) ────────────────
# Captured from parallel training runs on CPU.
# Config: d_model=128, n_layers=4, n_heads=4, seq_len=256, batch=32, lr=3e-4

DATA = {
    'standard': {
        'steps':    [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'val_acc':  [0.032, 0.269, 0.274, 0.286, 0.315, 0.343, 0.363, 0.378, 0.392, 0.397, 0.414],
        'val_loss': [4.164, 2.534, 2.475, 2.431, 2.314, 2.228, 2.145, 2.095, 2.053, 2.028, 1.985],
        'train_loss': [4.347, 2.543, 2.464, 2.481, 2.397, 2.290, 2.211, 2.152, 2.082, 2.043, 2.040],
        'train_acc':  [0.013, 0.268, 0.275, 0.276, 0.300, 0.329, 0.348, 0.369, 0.383, 0.401, 0.401],
        'entropy':  [0]*11,
        'elapsed':  [9.3, 148, 307, 444, 590, 733, 876, 1018, 1170, 1309, 1451],
        'ms_per_step': 726,
        'params': 842817,
        'tree_pct': 0,
        'final_val_acc': 0.413,
        'final_val_loss': 1.986,
    },
    'batched': {
        'steps':    [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'val_acc':  [0.023, 0.265, 0.265, 0.267, 0.267, 0.268, 0.267, 0.272, 0.271, 0.272, 0.273],
        'val_loss': [4.224, 2.542, 2.507, 2.501, 2.496, 2.487, 2.485, 2.479, 2.473, 2.470, 2.470],
        'train_loss': [4.363, 2.564, 2.519, 2.489, 2.468, 2.473, 2.459, 2.478, 2.482, 2.458, 2.453],
        'train_acc':  [0.014, 0.263, 0.258, 0.262, 0.264, 0.265, 0.272, 0.265, 0.268, 0.271, 0.274],
        'entropy':  [0.689, 0.410, 0.331, 0.094, 0.028, 0.030, 0.023, 0.012, 0.009, 0.006, 0.005],
        'elapsed':  [18.9, 468, 905, 1343, 1666, 1934, 2200, 2456, 2715, 2970, 3222],
        'ms_per_step': 1611,
        'params': 862017,
        'tree_pct': 32.9,
        'final_val_acc': 0.274,
        'final_val_loss': 2.471,
    },
    'boosted': {
        'steps':    [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'val_acc':  [0.066, 0.267, 0.275, 0.294, 0.316, 0.340, 0.358, 0.371, 0.387, 0.398, 0.414],
        'val_loss': [3.954, 2.527, 2.471, 2.389, 2.320, 2.243, 2.170, 2.127, 2.074, 2.026, 1.980],
        'train_loss': [4.365, 2.588, 2.501, 2.406, 2.354, 2.302, 2.207, 2.149, 2.106, 2.088, 2.027],
        'train_acc':  [0.006, 0.254, 0.273, 0.292, 0.312, 0.319, 0.356, 0.366, 0.375, 0.380, 0.404],
        'entropy':  [0.683, 0.149, 0.079, 0.053, 0.031, 0.024, 0.020, 0.014, 0.012, 0.009, 0.008],
        'elapsed':  [37.9, 1120, 1923, 2581, 3220, 3681, 4198, 4682, 5146, 5611, 6084],
        'ms_per_step': 3042,
        'params': 1474777,
        'tree_pct': 42.6,
        'final_val_acc': 0.411,
        'final_val_loss': 1.987,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def smooth(values, weight=0.4):
    out, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        out.append(last)
    return out


def save(fig, filename):
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved figures/{filename}")


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_val_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in DATA.items():
        ax.plot(d['steps'], [v*100 for v in d['val_acc']],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=4)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy on Shakespeare (Character-Level LM)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 50)
    save(fig, 'val_accuracy.png')


def fig_val_loss():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in DATA.items():
        ax.plot(d['steps'], d['val_loss'],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=4)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Loss (Cross-Entropy)')
    ax.set_title('Validation Loss on Shakespeare (Character-Level LM)')
    ax.legend(loc='upper right')
    save(fig, 'val_loss.png')


def fig_train_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in DATA.items():
        ax.plot(d['steps'], [v*100 for v in d['train_acc']],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=4)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy on Shakespeare')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 50)
    save(fig, 'train_accuracy.png')


def fig_entropy():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ['batched', 'boosted']:
        d = DATA[name]
        ax.plot(d['steps'], d['entropy'],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=4)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Routing Entropy')
    ax.set_title('Routing Decision Entropy Over Training')
    ax.legend(loc='upper right')
    ax.axhline(y=0.693, color='gray', linestyle='--', alpha=0.5, label='Max entropy (ln2)')
    ax.set_ylim(-0.02, 0.75)
    save(fig, 'entropy.png')


def fig_speed_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = ['standard', 'batched', 'boosted']
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]
    ms = [DATA[n]['ms_per_step'] for n in names]
    params = [DATA[n]['params'] for n in names]

    bars = ax1.barh(labels, ms, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('ms / training step')
    ax1.set_title('Training Speed (CPU)')
    for bar, v in zip(bars, ms):
        ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f'{v:,.0f}ms', va='center', fontsize=10)

    bars = ax2.barh(labels, [p/1000 for p in params], color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Parameters (thousands)')
    ax2.set_title('Model Size')
    for bar, v in zip(bars, params):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                 f'{v/1000:.0f}K', va='center', fontsize=10)

    fig.suptitle('Efficiency Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'speed_comparison.png')


def fig_accuracy_vs_compute():
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, d in DATA.items():
        ax.scatter(d['elapsed'][-1] / 60, d['final_val_acc'] * 100,
                   s=d['params'] / 80, color=COLORS[name],
                   label=f"{LABELS[name]} ({d['params']/1000:.0f}K)",
                   edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('Total Training Time (minutes)')
    ax.set_ylabel('Final Validation Accuracy (%)')
    ax.set_title('Accuracy vs. Compute Budget (2000 steps)')
    ax.legend(loc='lower right')
    save(fig, 'accuracy_vs_compute.png')


def fig_convergence_time():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in DATA.items():
        times_min = [t / 60 for t in d['elapsed']]
        ax.plot(times_min, [v*100 for v in d['val_acc']],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=4)
    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Convergence vs. Wall-Clock Time')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 50)
    save(fig, 'convergence_time.png')


def fig_combined_main():
    """The main 2x2 figure for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Val accuracy vs steps
    ax = axes[0, 0]
    for name, d in DATA.items():
        ax.plot(d['steps'], [v*100 for v in d['val_acc']],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('(a) Validation Accuracy vs. Steps')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 50)

    # (b) Val loss vs steps
    ax = axes[0, 1]
    for name, d in DATA.items():
        ax.plot(d['steps'], d['val_loss'],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('(b) Validation Loss vs. Steps')
    ax.legend(loc='upper right', fontsize=9)

    # (c) Routing entropy
    ax = axes[1, 0]
    for name in ['batched', 'boosted']:
        d = DATA[name]
        ax.plot(d['steps'], d['entropy'],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=3)
    ax.axhline(y=0.693, color='gray', linestyle='--', alpha=0.5)
    ax.text(100, 0.7, 'max entropy (ln 2)', fontsize=9, color='gray')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Routing Entropy')
    ax.set_title('(c) Routing Entropy Collapse')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.02, 0.75)

    # (d) Val accuracy vs wall-clock time
    ax = axes[1, 1]
    for name, d in DATA.items():
        times_min = [t / 60 for t in d['elapsed']]
        ax.plot(times_min, [v*100 for v in d['val_acc']],
                color=COLORS[name], label=LABELS[name], marker='o', markersize=3)
    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('(d) Convergence vs. Wall-Clock Time')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 50)

    fig.suptitle('Tree-Based Attention on Shakespeare (2000 steps, CPU)',
                 fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, 'main_results.png')


# ── Save results JSON ────────────────────────────────────────────────────────

def save_results():
    with open(os.path.join(RESULTS_DIR, 'shakespeare_results.json'), 'w') as f:
        json.dump(DATA, f, indent=2)
    print(f"  Saved results/shakespeare_results.json")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load from results JSON if available (from train.py), else use hardcoded DATA."""
    results_file = os.path.join(RESULTS_DIR, 'shakespeare_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            saved = json.load(f)
        # Check if it has eval_log (from updated train.py)
        if any('eval_log' in v for v in saved.values()):
            print(f"Loading results from {results_file} (train.py format)")
            out = {}
            for name, r in saved.items():
                log = r.get('eval_log', {})
                out[name] = {
                    'steps': log.get('steps', []),
                    'val_acc': log.get('val_acc', []),
                    'val_loss': log.get('val_loss', []),
                    'train_loss': log.get('train_loss', []),
                    'train_acc': log.get('train_acc', []),
                    'entropy': log.get('entropy', []),
                    'elapsed': log.get('elapsed', []),
                    'ms_per_step': r.get('ms_per_step', 0),
                    'params': r.get('params', 0),
                    'tree_pct': r.get('tree_pct', 0),
                    'final_val_acc': r.get('final_val_acc', 0),
                    'final_val_loss': r.get('final_val_loss', 0),
                }
            return out
    print("Using hardcoded results data")
    return DATA


def main():
    data = load_data()
    # Override module-level DATA for figure functions
    global DATA
    DATA = data
    print("Generating figures...")
    fig_val_accuracy()
    fig_val_loss()
    fig_train_accuracy()
    fig_entropy()
    fig_speed_comparison()
    fig_accuracy_vs_compute()
    fig_convergence_time()
    fig_combined_main()
    save_results()

    print(f"\n{'=' * 55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Model':<28s} {'Val Acc':>8} {'Loss':>7} {'ms/step':>8} {'Params':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for name, d in DATA.items():
        print(f"  {LABELS[name]:<28s} {d['final_val_acc']:>7.1%} "
              f"{d['final_val_loss']:>7.3f} {d['ms_per_step']:>7.0f}ms {d['params']:>8,}")
    print(f"\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
