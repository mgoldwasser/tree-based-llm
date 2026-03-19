# TASK-17: Architecture Hyperparameter Search

- **Category:** hyperparameter-optimization
- **Priority:** 9.0/10
- **Impact:** 9/10
- **Feasibility:** 8/10
- **Confidence:** 8/10

## Problem Statement

**Current configs are educated guesses, not optimized.**

We're using:
- `n_trees=12` (why not 8 or 24?)
- `tree_depth=3` (why not 2 or 4?)
- `d_model=64/128` (scaling behavior unknown)
- `n_layers=2/4` (interaction with trees unclear)
- `boosted_trees=24` (double of n_trees, arbitrary)
- `shrinkage=0.1` (hardcoded in LinearPlusForest)

**No systematic exploration of:**
- Optimal tree depth for language modeling
- Number of trees vs depth tradeoff
- How model width/depth interacts with tree performance
- Whether trees benefit from over-parameterization

## Key Questions

### **Q1: Tree Depth**

**Hypothesis:** Deeper trees = more expressiveness, but:
- Depth 2 (4 leaves): Fast, low capacity
- Depth 3 (8 leaves): Current default
- Depth 4 (16 leaves): High capacity, gradient issues?

**Test:** Fix n_trees=12, vary depth ∈ {2, 3, 4, 5}

**Expected:** Depth 3-4 optimal (2 underfits, 5 overfits/vanishing gradients)

### **Q2: Number of Trees**

**Hypothesis:** More trees = ensemble diversity, but:
- 4 trees: Minimal ensemble
- 12 trees: Current default
- 24 trees: More capacity
- 48 trees: Overkill? Redundant?

**Test:** Fix depth=3, vary n_trees ∈ {4, 8, 12, 24, 48}

**Expected:** 12-24 optimal, diminishing returns beyond

### **Q3: Model Scale**

Standard transformer scaling laws: bigger is better (up to data limits).

**Do trees scale the same way?**

Test on grid:
```
d_model ∈ {64, 128, 256, 512}
n_layers ∈ {2, 4, 6, 8}
```

**Hypothesis:** Trees benefit MORE from width than depth
- Reason: Routing operates on d_model dims
- Wider input = richer routing decisions
- Deeper stacking = routing errors compound

### **Q4: Trees vs Params**

**Critical question:** Is tree performance parameter-limited?

Compare at matched param count (~1M params):
1. Standard: d_model=192, n_layers=4 (1.04M)
2. Tree: d_model=128, n_layers=4, trees=12, depth=3 (1.14M)
3. Tree-wide: d_model=160, n_layers=4, trees=8, depth=2 (1.02M)
4. Tree-deep: d_model=96, n_layers=6, trees=12, depth=3 (0.98M)

**Expected:** If trees still lose, it's not params—it's architecture.

### **Q5: Shrinkage in LinearPlusForest**

Current: `output = linear(x) + 0.1 * forest(x)`

**Question:** Is 0.1 optimal?

Test shrinkage ∈ {0.01, 0.05, 0.1, 0.2, 0.5}

**Hypothesis:** 0.05-0.1 optimal
- Too low → trees don't contribute
- Too high → destabilizes residual stream

### **Q6: Alternating Layer Patterns**

Current: `tree_every_n=2` (tree-linear-tree-linear)

**Alternatives:**
- `1` (all tree) - slow but max expressiveness
- `2` (alternating) - current best
- `3` (tree-linear-linear-tree) - cheaper, loss of expressiveness?
- `[0,1,1,0]` (tree in middle layers only) - like LoRA placement

**Test:** Sequence patterns for n_layers=4,6,8

## Proposed Search Strategy

### **Phase 1: Critical Hyperparameters (Priority)**

**Grid search (small models, fast iteration):**
```python
configs = {
    'tree_depth': [2, 3, 4],
    'n_trees': [8, 12, 24],
    'shrinkage': [0.05, 0.1, 0.2],
}

for depth in configs['tree_depth']:
    for n_trees in configs['n_trees']:
        for shrink in configs['shrinkage']:
            run_experiment(
                model='oblivious_boosted_vo_alt',
                d_model=64, n_layers=2, steps=2000,
                boosted_depth=depth, boosted_trees=n_trees,
                # Add shrinkage param to LinearPlusForest
            )
```

**Estimated time:** 27 configs × 5 min = 2.25 hours

### **Phase 2: Model Scale (Secondary)**

Fix best hyperparams from Phase 1, sweep scale:
```python
for d_model in [64, 128, 256]:
    for n_layers in [2, 4, 6]:
        run_experiment(
            model='oblivious_boosted_vo_alt',
            d_model=d_model, n_layers=n_layers,
            steps=2000,
            # Use Phase 1 optimal tree config
        )
```

**Estimated time:** 9 configs × 10-30 min = 2-5 hours

### **Phase 3: Matched Params (Final Validation)**

Best tree config vs matched-param standard transformer.

## Implementation

### **Step 1: Make shrinkage configurable**

```python
# main.py - LinearPlusForest.__init__
def __init__(self, d_input, d_output, forest, shrinkage=0.1):
    super().__init__()
    self.linear = nn.Linear(d_input, d_output)
    self.forest = forest
    self.shrinkage = nn.Parameter(torch.tensor(shrinkage))  # learnable!

def forward(self, x):
    return self.linear(x) + self.shrinkage * self.forest(x)
```

### **Step 2: Add search script**

```python
# run_hyperparam_search.py
import itertools
import subprocess
import json

grid = {
    'tree_depth': [2, 3, 4],
    'n_trees': [8, 12, 24],
    'shrinkage': [0.05, 0.1, 0.2],
}

results = []
for combo in itertools.product(*grid.values()):
    config = dict(zip(grid.keys(), combo))
    # Run training with this config
    result = subprocess.run([
        'python', 'train.py',
        '--model', 'oblivious_boosted_vo_alt',
        '--fast',
        '--depth', str(config['tree_depth']),
        '--trees', str(config['n_trees']),
        '--shrinkage', str(config['shrinkage']),
    ], capture_output=True, text=True)
    
    # Parse result, save to results
    results.append({'config': config, 'val_acc': parse_val_acc(result.stdout)})

# Save ranked results
with open('results/hyperparam_search.json', 'w') as f:
    json.dump(sorted(results, key=lambda r: r['val_acc'], reverse=True), f, indent=2)
```

### **Step 3: Visualization**

```python
# generate_figures.py - add heatmap
def plot_hyperparam_heatmap(results):
    """Heatmap: n_trees (x) vs depth (y), color=val_acc"""
    pivot = results.pivot_table(
        values='val_acc',
        index='tree_depth',
        columns='n_trees',
        aggfunc='mean'  # average over shrinkage
    )
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn')
    plt.title('Val Accuracy by Tree Config')
    plt.savefig('figures/hyperparam_search.png')
```

## Expected Outcomes

**Best case:** Find config that beats standard transformer
- Example: depth=2, trees=24, shrinkage=0.2 → 40.5% val acc

**Likely case:** Find Pareto frontier (speed vs accuracy)
- Depth=2, trees=16: Fast, 37% acc
- Depth=3, trees=12: Balanced, 38% acc
- Depth=4, trees=24: Slow, 38.5% acc (diminishing returns)

**Worst case:** No config beats standard
- Rules out hyperparams as the issue
- Points to fundamental architectural limitation

## Follow-Up Actions

**If trees win:**
- Scale up to d_model=512, larger dataset
- Publish optimal hyperparameters

**If trees still lose:**
- Investigate per-head tree configs (TASK-19)
- Try task-specific routing (TASK-20)
- Consider hybrid architectures (tree in FFN only, not attention)

## Dependencies

- Requires `shrinkage` to be configurable in LinearPlusForest
- Recommend running on GPU for Phase 2 (multi-hour experiments)

## Estimated Total Time

- Implementation: 2 hours
- Phase 1 search: 2-3 hours (CPU okay)
- Phase 2 search: 2-5 hours (GPU recommended)
- Analysis: 1 hour

**Total:** ~8-11 hours wall time, ~3-4 hours human time
