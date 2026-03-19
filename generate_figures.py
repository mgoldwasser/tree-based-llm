"""
Generate publication-quality figures from training results.
Loads data from results/full_config_results.json (preferred) or
results/shakespeare_results.json, with hardcoded fallback.

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
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

# 6-model color palette
COLORS = {
    'standard':                 '#2196F3',  # blue
    'boosted':                  '#4CAF50',  # green
    'oblivious_boosted':        '#FF9800',  # orange
    'oblivious_boosted_alt':    '#9C27B0',  # purple
    'oblivious_boosted_vo_alt': '#E91E63',  # pink
    'moe_boosted_alt':          '#795548',  # brown
    # Extras for fast-config or other models
    'batched':                  '#607D8B',  # grey
    'oblivious':                '#00BCD4',  # cyan
    'boosted_alt':              '#CDDC39',  # lime
}

LABELS = {
    'standard':                 'Standard Transformer',
    'boosted':                  'Linear+Forest',
    'oblivious_boosted':        'Oblivious L+F',
    'oblivious_boosted_alt':    'Oblivious L+F (alternating)',
    'oblivious_boosted_vo_alt': 'Oblivious L+F (V+O, alt)',
    'moe_boosted_alt':          'Linear+MoE (alternating)',
    'batched':                  'Batched Forest',
    'oblivious':                'Oblivious Forest',
    'boosted_alt':              'Linear+Forest (alternating)',
}

# Preferred display order
MODEL_ORDER = [
    'standard', 'boosted', 'oblivious_boosted', 'oblivious_boosted_alt',
    'oblivious_boosted_vo_alt', 'moe_boosted_alt',
]

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_train_results(saved):
    """Convert train.py output format to figure-friendly format."""
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
            'temperature': log.get('temperature', []),
            'ms_per_step': r.get('ms_per_step', 0),
            'params': r.get('params', 0),
            'tree_pct': r.get('tree_pct', 0),
            'final_val_acc': r.get('final_val_acc', 0),
            'final_val_loss': r.get('final_val_loss', 0),
            'name': r.get('name', name),
        }
    return out


def load_data():
    """Load results data. Priority: full_config_results.json > shakespeare_results.json."""
    # Try full config results first
    full_file = os.path.join(RESULTS_DIR, 'full_config_results.json')
    if os.path.exists(full_file):
        with open(full_file) as f:
            raw = json.load(f)
        if 'full' in raw and raw['full']:
            print(f"Loading from {full_file} (full config)")
            full_data = _parse_train_results(raw['full'])
            fast_data = _parse_train_results(raw.get('fast', {}))
            return full_data, fast_data

    # Fall back to shakespeare_results.json
    results_file = os.path.join(RESULTS_DIR, 'shakespeare_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            saved = json.load(f)
        if any('eval_log' in v for v in saved.values()):
            print(f"Loading from {results_file}")
            return _parse_train_results(saved), {}

    print("WARNING: No results files found. Run experiments first.")
    return {}, {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_color(name):
    return COLORS.get(name, '#999999')

def _get_label(name):
    return LABELS.get(name, name)

def _ordered_models(data):
    """Return model names in preferred display order."""
    return [m for m in MODEL_ORDER if m in data] + \
           [m for m in data if m not in MODEL_ORDER]

def save(fig, filename):
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved figures/{filename}")


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_val_accuracy(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in _ordered_models(data):
        d = data[name]
        if not d['steps']:
            continue
        ax.plot(d['steps'], [v*100 for v in d['val_acc']],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy on Shakespeare (Character-Level LM)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 55)
    save(fig, 'val_accuracy.png')


def fig_val_loss(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in _ordered_models(data):
        d = data[name]
        if not d['steps']:
            continue
        ax.plot(d['steps'], d['val_loss'],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Loss (Cross-Entropy)')
    ax.set_title('Validation Loss on Shakespeare (Character-Level LM)')
    ax.legend(loc='upper right', fontsize=8)
    save(fig, 'val_loss.png')


def fig_train_accuracy(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in _ordered_models(data):
        d = data[name]
        if not d['steps']:
            continue
        ax.plot(d['steps'], [v*100 for v in d['train_acc']],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy on Shakespeare')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 55)
    save(fig, 'train_accuracy.png')


def fig_entropy(data):
    """Routing entropy for tree models only."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tree_models = [m for m in _ordered_models(data)
                   if m != 'standard' and any(e != 0 for e in data[m].get('entropy', []))]
    for name in tree_models:
        d = data[name]
        ax.plot(d['steps'], d['entropy'],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Routing Entropy')
    ax.set_title('Routing Decision Entropy Over Training')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0.693, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[0] + 50, 0.7, 'max entropy (ln 2)', fontsize=9, color='gray')
    ax.set_ylim(-0.02, 0.75)
    save(fig, 'entropy.png')


def fig_speed_comparison(data):
    """Horizontal bar chart: speed and params."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models = _ordered_models(data)
    labels = [_get_label(n) for n in models]
    colors = [_get_color(n) for n in models]
    ms = [data[n]['ms_per_step'] for n in models]
    params = [data[n]['params'] for n in models]

    y_pos = range(len(models))
    bars = ax1.barh(y_pos, ms, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('ms / training step')
    ax1.set_title('Training Speed')
    ax1.invert_yaxis()
    for bar, v in zip(bars, ms):
        ax1.text(bar.get_width() + max(ms)*0.02, bar.get_y() + bar.get_height()/2,
                 f'{v:,.0f}ms', va='center', fontsize=9)

    bars = ax2.barh(y_pos, [p/1000 for p in params], color=colors,
                    edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Parameters (thousands)')
    ax2.set_title('Model Size')
    ax2.invert_yaxis()
    for bar, v in zip(bars, params):
        ax2.text(bar.get_width() + max(params)/1000*0.02,
                 bar.get_y() + bar.get_height()/2,
                 f'{v/1000:.0f}K', va='center', fontsize=9)

    fig.suptitle('Efficiency Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'speed_comparison.png')


def fig_speed_vs_accuracy(data):
    """Key result figure: scatter plot of speed vs accuracy."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in _ordered_models(data):
        d = data[name]
        ax.scatter(d['ms_per_step'], d['final_val_acc'] * 100,
                   s=max(80, d['params'] / 100),
                   color=_get_color(name),
                   label=f"{_get_label(name)} ({d['params']/1000:.0f}K)",
                   edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('Training Speed (ms/step)')
    ax.set_ylabel('Final Validation Accuracy (%)')
    ax.set_title('Speed vs. Accuracy Trade-off (2000 steps)')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    # Add Pareto frontier
    models = _ordered_models(data)
    points = sorted([(data[m]['ms_per_step'], data[m]['final_val_acc']*100) for m in models])
    pareto_x, pareto_y = [], []
    best_y = -1
    for x, y in points:
        if y > best_y:
            pareto_x.append(x)
            pareto_y.append(y)
            best_y = y
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.5, linewidth=1)

    save(fig, 'speed_vs_accuracy.png')


def fig_convergence_time(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in _ordered_models(data):
        d = data[name]
        if not d['elapsed']:
            continue
        times_min = [t / 60 for t in d['elapsed']]
        ax.plot(times_min, [v*100 for v in d['val_acc']],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=3)
    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Convergence vs. Wall-Clock Time')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 55)
    save(fig, 'convergence_time.png')


def fig_combined_main(data):
    """The main 2x2 figure for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Val accuracy vs steps
    ax = axes[0, 0]
    for name in _ordered_models(data):
        d = data[name]
        if not d['steps']:
            continue
        ax.plot(d['steps'], [v*100 for v in d['val_acc']],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('(a) Validation Accuracy vs. Steps')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_ylim(0, 55)

    # (b) Val loss vs steps
    ax = axes[0, 1]
    for name in _ordered_models(data):
        d = data[name]
        if not d['steps']:
            continue
        ax.plot(d['steps'], d['val_loss'],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('(b) Validation Loss vs. Steps')
    ax.legend(loc='upper right', fontsize=7)

    # (c) Routing entropy
    ax = axes[1, 0]
    tree_models = [m for m in _ordered_models(data)
                   if m != 'standard' and any(e != 0 for e in data[m].get('entropy', []))]
    for name in tree_models:
        d = data[name]
        ax.plot(d['steps'], d['entropy'],
                color=_get_color(name), label=_get_label(name),
                marker='o', markersize=2)
    ax.axhline(y=0.693, color='gray', linestyle='--', alpha=0.5)
    ax.text(100, 0.7, 'max entropy (ln 2)', fontsize=8, color='gray')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Routing Entropy')
    ax.set_title('(c) Routing Entropy Over Training')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-0.02, 0.75)

    # (d) Speed vs accuracy scatter
    ax = axes[1, 1]
    for name in _ordered_models(data):
        d = data[name]
        ax.scatter(d['ms_per_step'], d['final_val_acc'] * 100,
                   s=max(60, d['params'] / 120),
                   color=_get_color(name),
                   label=_get_label(name),
                   edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('Training Speed (ms/step)')
    ax.set_ylabel('Final Validation Accuracy (%)')
    ax.set_title('(d) Speed vs. Accuracy Trade-off')
    ax.legend(loc='best', fontsize=7, framealpha=0.9)

    fig.suptitle('Tree-Based Attention on Shakespeare (2000 steps)',
                 fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, 'main_results.png')


# ── Save results JSON ────────────────────────────────────────────────────────

def save_results_summary(data):
    summary = {}
    for name, d in data.items():
        summary[name] = {
            'label': _get_label(name),
            'final_val_acc': d['final_val_acc'],
            'final_val_loss': d['final_val_loss'],
            'ms_per_step': d['ms_per_step'],
            'params': d['params'],
            'tree_pct': d.get('tree_pct', 0),
        }
    with open(os.path.join(RESULTS_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results/results_summary.json")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    full_data, fast_data = load_data()

    if not full_data:
        print("No data to plot. Exiting.")
        return

    print("Generating figures...")
    fig_val_accuracy(full_data)
    fig_val_loss(full_data)
    fig_train_accuracy(full_data)
    fig_entropy(full_data)
    fig_speed_comparison(full_data)
    fig_speed_vs_accuracy(full_data)
    fig_convergence_time(full_data)
    fig_combined_main(full_data)
    save_results_summary(full_data)

    # Print summary table
    print(f"\n{'=' * 75}")
    print(f"  RESULTS SUMMARY (FULL CONFIG)")
    print(f"{'=' * 75}")
    print(f"  {'Model':<35s} {'Val Acc':>8} {'Loss':>7} {'ms/step':>8} {'Params':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
    for name in _ordered_models(full_data):
        d = full_data[name]
        print(f"  {_get_label(name):<35s} {d['final_val_acc']:>7.1%} "
              f"{d['final_val_loss']:>7.3f} {d['ms_per_step']:>7.0f}ms {d['params']:>10,}")

    if fast_data:
        print(f"\n  FAST CONFIG COMPARISON:")
        print(f"  {'Model':<35s} {'Val Acc':>8} {'Loss':>7} {'ms/step':>8} {'Params':>10}")
        print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
        for name in _ordered_models(fast_data):
            d = fast_data[name]
            print(f"  {_get_label(name):<35s} {d['final_val_acc']:>7.1%} "
                  f"{d['final_val_loss']:>7.3f} {d['ms_per_step']:>7.0f}ms {d['params']:>10,}")

    print(f"\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
