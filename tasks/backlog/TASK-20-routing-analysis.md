# TASK-20: Routing Analysis & Visualization

- **Category:** interpretability
- **Priority:** 9.5/10 ⭐ CRITICAL
- **Impact:** 9/10 (for understanding)
- **Feasibility:** 7/10
- **Confidence:** 8/10

## Problem Statement

**We have no visibility into what trees are learning.**

**Critical unknowns:**
- Which tokens route to which leaves?
- Do semantically similar tokens cluster?
- Are routing decisions random or structured?
- Do trees specialize by layer/head?
- Are all leaves being used, or are some dead?

**Without this visibility:**
- Can't diagnose why trees underperform
- Can't validate architectural improvements
- Can't understand the learned representations

## Proposed Visualizations

### **Analysis 1: Token-to-Leaf Mapping**

**Question:** Where do different tokens route?

```python
def analyze_token_routing(model, dataset, device, n_samples=1000):
    """Map each vocab token to its routing path."""
    model.eval()
    vocab_size = dataset.vocab_size
    
    routing_map = {}  # token_id -> (tree_id, leaf_id, probability)
    
    for token_id in range(vocab_size):
        # Create batch of this token
        x = torch.tensor([[token_id]], device=device)
        
        with torch.no_grad():
            # Extract routing decisions from model
            routing_decisions = extract_routing(model, x)
            
            # For each tree, find dominant leaf
            for tree_id, decisions in enumerate(routing_decisions):
                leaf_probs = decisions['leaf_probs'][0, 0]  # (n_leaves,)
                dominant_leaf = leaf_probs.argmax().item()
                max_prob = leaf_probs.max().item()
                
                routing_map[token_id] = routing_map.get(token_id, [])
                routing_map[token_id].append({
                    'tree': tree_id,
                    'leaf': dominant_leaf,
                    'prob': max_prob,
                })
    
    return routing_map

def extract_routing(model, x):
    """Hook into model to extract routing decisions."""
    routing_data = []
    
    def hook_fn(module, input, output):
        if hasattr(module, '_cached_decisions'):
            routing_data.append({
                'module_name': str(module),
                'decisions': module._cached_decisions,
                'leaf_probs': output,  # if module is TreeForest
            })
    
    # Register hooks on all tree modules
    hooks = []
    for name, module in model.named_modules():
        if 'Forest' in type(module).__name__:
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return routing_data
```

**Visualization:**
```python
def plot_token_leaf_heatmap(routing_map, dataset):
    """Heatmap: tokens (y-axis) vs leaves (x-axis), color=frequency."""
    # Convert to matrix
    n_tokens = len(routing_map)
    n_leaves = max(r['leaf'] for routes in routing_map.values() 
                   for r in routes if r['tree'] == 0) + 1
    
    heatmap = np.zeros((n_tokens, n_leaves))
    for token_id, routes in routing_map.items():
        for route in routes:
            if route['tree'] == 0:  # First tree only, for simplicity
                heatmap[token_id, route['leaf']] = route['prob']
    
    plt.figure(figsize=(12, 20))
    sns.heatmap(heatmap, cmap='viridis', 
                yticklabels=[dataset.decode([i]) for i in range(n_tokens)],
                xticklabels=range(n_leaves))
    plt.title('Token Routing Patterns (Tree 0)')
    plt.ylabel('Token')
    plt.xlabel('Leaf ID')
    plt.tight_layout()
    plt.savefig('figures/token_leaf_heatmap.png', dpi=150)
```

### **Analysis 2: Semantic Clustering**

**Question:** Do similar tokens route to similar leaves?

```python
def analyze_semantic_clustering(routing_map, dataset):
    """Do semantically similar tokens cluster?"""
    
    # Group tokens by category
    categories = {
        'punctuation': ['.', ',', '!', '?', ';', ':', '\n'],
        'common_words': ['the', 'and', 'to', 'a', 'of'],
        'names': ['ROMEO', 'JULIET', 'HAMLET', 'LADY'],
        'verbs': ['is', 'are', 'was', 'were', 'be'],
        'numbers': ['1', '2', '3', 'one', 'two'],
    }
    
    category_routing = {}
    for category, tokens in categories.items():
        token_ids = [dataset.encode(t)[0] for t in tokens if t in dataset.vocab]
        if token_ids:
            # Average routing for this category
            avg_routing = compute_average_routing(routing_map, token_ids)
            category_routing[category] = avg_routing
    
    # Compute pairwise similarity (cosine distance of routing vectors)
    similarity_matrix = compute_routing_similarity(category_routing)
    
    # Plot
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f',
                xticklabels=categories.keys(),
                yticklabels=categories.keys(),
                cmap='RdYlGn')
    plt.title('Routing Similarity by Token Category')
    plt.savefig('figures/semantic_clustering.png')
```

### **Analysis 3: Leaf Utilization**

**Question:** Are all leaves being used, or are some dead?

```python
def analyze_leaf_utilization(routing_map, n_trees, n_leaves):
    """Which leaves are used frequently vs never?"""
    
    leaf_counts = np.zeros((n_trees, n_leaves))
    
    for token_id, routes in routing_map.items():
        for route in routes:
            leaf_counts[route['tree'], route['leaf']] += route['prob']
    
    # Normalize to probabilities
    leaf_usage = leaf_counts / leaf_counts.sum(axis=1, keepdims=True)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for tree_id in range(min(n_trees, 12)):
        ax = axes.flatten()[tree_id]
        ax.bar(range(n_leaves), leaf_usage[tree_id])
        ax.set_title(f'Tree {tree_id} Leaf Usage')
        ax.set_xlabel('Leaf ID')
        ax.set_ylabel('Usage Probability')
        # Highlight dead leaves (usage < 1%)
        dead_leaves = np.where(leaf_usage[tree_id] < 0.01)[0]
        if len(dead_leaves) > 0:
            ax.axhline(0.01, color='red', linestyle='--', label='Dead threshold')
    
    plt.tight_layout()
    plt.savefig('figures/leaf_utilization.png')
    
    # Print statistics
    for tree_id in range(n_trees):
        dead = (leaf_usage[tree_id] < 0.01).sum()
        active = n_leaves - dead
        print(f"Tree {tree_id}: {active}/{n_leaves} leaves active ({dead} dead)")
```

### **Analysis 4: Routing Entropy Over Training**

**Question:** How does routing sharpness evolve during training?

```python
# During training, log routing entropy at each eval interval
# Already partially implemented via get_routing_entropy()

def plot_entropy_trajectory(results):
    """Plot routing entropy vs training step."""
    eval_log = results['eval_log']
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_log['steps'], eval_log['entropy'], marker='o')
    plt.axhline(0.69, color='gray', linestyle='--', label='Uniform (max entropy)')
    plt.axhline(0.3, color='orange', linestyle='--', label='Target range')
    plt.axhline(0.1, color='red', linestyle='--', label='Too sharp')
    plt.xlabel('Training Step')
    plt.ylabel('Routing Entropy')
    plt.title('Routing Entropy Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('figures/entropy_trajectory.png')
```

### **Analysis 5: Layer-by-Layer Routing**

**Question:** Do routing patterns differ by layer?

```python
def analyze_layer_specialization(model, dataset, device):
    """Compare routing across layers."""
    
    layer_routing = {}  # layer_id -> routing_map
    
    # Extract routing for each layer separately
    for layer_id in range(model.n_layers):
        # Hook only this layer
        routing_map = analyze_token_routing_single_layer(
            model, dataset, device, layer_id
        )
        layer_routing[layer_id] = routing_map
    
    # Compare: Do early layers route differently than late layers?
    for layer_id in range(model.n_layers):
        print(f"\n=== Layer {layer_id} ===")
        top_tokens_per_leaf = find_top_tokens_per_leaf(
            layer_routing[layer_id], dataset, k=5
        )
        for leaf_id, tokens in top_tokens_per_leaf.items():
            print(f"  Leaf {leaf_id}: {tokens}")
```

## Implementation Plan

### **Phase 1: Basic Infrastructure (2 hours)**
1. Implement `extract_routing()` with hooks
2. Implement `analyze_token_routing()`
3. Test on small model, verify output

### **Phase 2: Core Visualizations (3 hours)**
4. Token-to-leaf heatmap
5. Leaf utilization plots
6. Entropy trajectory plots

### **Phase 3: Advanced Analysis (3 hours)**
7. Semantic clustering analysis
8. Layer specialization comparison
9. Interactive visualization (optional: Plotly)

### **Phase 4: Integration (1 hour)**
10. Add `--analyze` flag to `train.py`
11. Auto-generate analysis after training
12. Save analysis results to `results/analysis/`

## Expected Insights

**If trees are working:**
- Similar tokens cluster in same leaves
- Leaves specialize (e.g., Leaf 3 = verbs, Leaf 5 = punctuation)
- Early layers: syntactic splits, late layers: semantic splits
- Entropy decreases smoothly during training

**If trees are failing:**
- Random routing (no semantic structure)
- Dead leaves (>30% leaves unused)
- Entropy collapses too early or doesn't decrease
- No layer specialization

## Files to Create

```
utils/routing_analysis.py   # Core analysis functions
scripts/analyze_routing.py  # Standalone analysis script
figures/                     # Output directory for plots
results/analysis/            # Saved analysis results
```

## Example Usage

```bash
# Train a model
python train.py --model oblivious_boosted_vo_alt --fast

# Analyze routing
python scripts/analyze_routing.py \
  --checkpoint checkpoints/oblivious_boosted_vo_alt_best.pt \
  --output results/analysis/

# View results
open figures/token_leaf_heatmap.png
open figures/semantic_clustering.png
open figures/leaf_utilization.png
```

## Success Metrics

- ✅ Can extract routing decisions from trained model
- ✅ Heatmap shows interpretable patterns (not random)
- ✅ Semantic categories cluster together (cosine sim > 0.7)
- ✅ <10% dead leaves (>90% utilization)
- ✅ Layer specialization evident (different patterns per layer)

## Follow-Up Tasks

- **TASK-21:** Routing Surgery - manually edit routing weights to test hypotheses
- **TASK-22:** Counterfactual Analysis - "What if token X routed to Leaf Y?"
- **TASK-23:** Routing Pruning - remove low-probability paths

## Dependencies

- Requires checkpointed models (TASK-19)
- Uses matplotlib, seaborn for plots
- Optional: Plotly for interactive visualizations

## Estimated Time

- Implementation: 6-8 hours
- Analysis of results: 2-3 hours
- **Total:** ~9-11 hours
