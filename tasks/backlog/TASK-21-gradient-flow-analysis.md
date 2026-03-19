# TASK-21: Gradient Flow Analysis & Training Diagnostics

- **Category:** training-diagnostics
- **Priority:** 9.0/10
- **Impact:** 9/10
- **Feasibility:** 8/10
- **Confidence:** 9/10

## Problem Statement

**We don't know if trees are learning properly.**

**Critical unknowns:**
- Are gradients flowing to leaf outputs? Or vanishing?
- Are decision weights updating? Or stuck?
- Do deeper tree layers learn slower than shallow?
- Is the linear base dominating the tree correction?
- Are some trees learning while others stay random?

**Symptoms that would indicate problems:**
- Gradient norms near zero (vanishing gradients)
- Gradient norms exploding (>100)
- Leaf outputs barely changing from initialization
- Routing decisions don't improve (random throughout training)

## Proposed Diagnostics

### **Diagnostic 1: Gradient Norm Tracking**

**Track gradient norms per component during training:**

```python
def log_gradient_norms(model, step):
    """Log gradient norms for each component."""
    grad_norms = {
        'decision_weights': [],
        'leaf_outputs': [],
        'linear_base': [],
        'embedding': [],
        'output_head': [],
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            # Categorize
            if 'decision_weight' in name:
                grad_norms['decision_weights'].append(grad_norm)
            elif 'leaf_output' in name:
                grad_norms['leaf_outputs'].append(grad_norm)
            elif 'linear' in name and 'forest' not in name:
                grad_norms['linear_base'].append(grad_norm)
            elif 'tok' in name or 'pos' in name:
                grad_norms['embedding'].append(grad_norm)
            elif 'head' in name:
                grad_norms['output_head'].append(grad_norm)
    
    # Average per category
    summary = {k: np.mean(v) if v else 0.0 for k, v in grad_norms.items()}
    
    return summary
```

**Visualization:**
```python
def plot_gradient_flow(gradient_log):
    """Plot gradient norms over training."""
    steps = sorted(gradient_log.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for component in ['decision_weights', 'leaf_outputs', 'linear_base']:
        values = [gradient_log[s][component] for s in steps]
        ax.plot(steps, values, label=component, marker='o')
    
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title('Gradient Flow by Component')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('figures/gradient_flow.png')
```

### **Diagnostic 2: Parameter Update Magnitude**

**Track how much parameters actually change:**

```python
class ParameterChangeTracker:
    def __init__(self, model):
        self.initial_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.history = []
    
    def log_changes(self, model, step):
        """Compute L2 norm of parameter changes since initialization."""
        changes = {}
        for name, param in model.named_parameters():
            if name in self.initial_params:
                delta = param.data - self.initial_params[name]
                changes[name] = delta.norm().item()
        
        self.history.append({'step': step, 'changes': changes})
        return changes
```

**Red flags:**
- Decision weights change < 0.01 after 1000 steps → Not learning
- Leaf outputs change < 0.1 after 1000 steps → Trees not updating
- Linear base changes > 10x tree changes → Trees irrelevant

### **Diagnostic 3: Layer-wise Learning Rate Analysis**

**Are all layers learning at similar rates?**

```python
def analyze_layer_learning_rates(model, gradient_log):
    """Compare learning across layers."""
    
    # Group gradients by layer
    layer_grads = {i: [] for i in range(model.n_layers)}
    
    for step, grads in gradient_log.items():
        for name, grad_norm in grads.items():
            # Extract layer number from parameter name
            if 'layers.' in name:
                layer_id = int(name.split('layers.')[1].split('.')[0])
                layer_grads[layer_id].append(grad_norm)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    layer_ids = sorted(layer_grads.keys())
    avg_grads = [np.mean(layer_grads[i]) for i in layer_ids]
    
    ax.bar(layer_ids, avg_grads)
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Average Gradient Norm')
    ax.set_title('Gradient Distribution Across Layers')
    plt.savefig('figures/layer_gradient_distribution.png')
```

### **Diagnostic 4: Dead Neuron Detection**

**Find neurons/leaves that never activate:**

```python
def detect_dead_neurons(model, dataset, device, n_batches=100):
    """Find neurons with zero activation across samples."""
    
    activations = {}  # module_name -> activation tensor
    
    def hook_fn(module, input, output):
        module_name = str(module)
        if module_name not in activations:
            activations[module_name] = []
        activations[module_name].append(output.detach())
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.GELU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Run forward passes
    model.eval()
    for _ in range(n_batches):
        x, y = dataset.get_batch(32, device, 'train')
        with torch.no_grad():
            model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze: neurons with activation always near zero
    dead_neurons = {}
    for module_name, acts in activations.items():
        acts_tensor = torch.cat(acts, dim=0)  # (total_samples, ...)
        # Neurons are dead if |activation| < threshold for all samples
        dead_mask = (acts_tensor.abs().max(dim=0)[0] < 1e-3)
        dead_count = dead_mask.sum().item()
        if dead_count > 0:
            dead_neurons[module_name] = dead_count
    
    return dead_neurons
```

### **Diagnostic 5: Loss Landscape Visualization**

**Explore loss surface near trained model:**

```python
def visualize_loss_landscape(model, dataset, device, checkpoint_path):
    """2D slice of loss landscape around checkpoint."""
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    
    # Generate two random directions
    direction1 = {name: torch.randn_like(param) 
                  for name, param in model.named_parameters()}
    direction2 = {name: torch.randn_like(param) 
                  for name, param in model.named_parameters()}
    
    # Normalize directions
    for d in [direction1, direction2]:
        norm = sum(p.norm()**2 for p in d.values()).sqrt()
        for name in d:
            d[name] /= norm
    
    # Evaluate loss on grid
    alpha_range = np.linspace(-1, 1, 21)
    beta_range = np.linspace(-1, 1, 21)
    loss_grid = np.zeros((len(alpha_range), len(beta_range)))
    
    original_state = {name: param.data.clone() 
                      for name, param in model.named_parameters()}
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Perturb parameters
            for name, param in model.named_parameters():
                param.data = (original_state[name] + 
                              alpha * direction1[name] + 
                              beta * direction2[name])
            
            # Compute loss
            x, y = dataset.get_batch(128, device, 'val')
            with torch.no_grad():
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, dataset.vocab_size),
                    y.reshape(-1)
                )
            loss_grid[i, j] = loss.item()
    
    # Restore original params
    for name, param in model.named_parameters():
        param.data = original_state[name]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(alpha_range, beta_range, loss_grid.T, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.scatter([0], [0], color='red', marker='*', s=200, label='Checkpoint')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape Around Trained Model')
    plt.legend()
    plt.savefig('figures/loss_landscape.png')
```

**Interpretation:**
- **Smooth bowl:** Good, model in good minimum
- **Steep cliff:** Model on edge of stability
- **Flat plateau:** Model stuck in bad region

## Implementation Plan

### **Phase 1: Gradient Tracking (2 hours)**
1. Add `log_gradient_norms()` to training loop
2. Store gradient log in results JSON
3. Plot gradient flow after training

### **Phase 2: Parameter Monitoring (1.5 hours)**
4. Implement `ParameterChangeTracker`
5. Log parameter updates every eval interval
6. Plot magnitude of changes over time

### **Phase 3: Advanced Diagnostics (2.5 hours)**
7. Layer-wise learning rate analysis
8. Dead neuron detection
9. Loss landscape visualization (optional)

### **Phase 4: Integration (1 hour)**
10. Add diagnostic plots to `generate_figures.py`
11. Create summary report: "Training Health Check"
12. Auto-run after each training session

## Expected Insights

**If gradient flow is healthy:**
- Decision weight gradients: 0.01 - 0.1
- Leaf output gradients: 0.1 - 1.0
- Gradients stable across layers (variance < 2x)
- Parameters change 1-10% from initialization
- <5% dead neurons

**If gradient flow is broken:**
- Gradients < 1e-6 (vanishing) or > 100 (exploding)
- Leaf outputs barely change (< 0.01 from init)
- Earlier layers have 10x smaller gradients (gradient vanishing)
- >20% dead neurons

## Fixes for Common Issues

### **Problem: Vanishing Gradients**
```python
# Solution: Skip connections in tree routing
class ResidualTreeForest(nn.Module):
    def forward(self, x):
        tree_output = self.forest(x)
        return x + 0.1 * tree_output  # Skip connection
```

### **Problem: Exploding Gradients**
```python
# Solution: Gradient clipping (already in train.py)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### **Problem: Dead Leaves**
```python
# Solution: Initialization with wider range
self.decision_weights = nn.Parameter(
    torch.randn(n_trees, n_internal, d_input) * 0.01  # Was 0.02
)
```

### **Problem: Linear Base Dominates**
```python
# Solution: Increase shrinkage or add warmup
shrinkage = min(0.5, 0.01 + 0.49 * (step / warmup_steps))
```

## Files to Create

```
utils/training_diagnostics.py   # Core diagnostic functions
scripts/analyze_training.py     # Standalone diagnostic script
figures/gradient_flow.png
figures/parameter_changes.png
figures/layer_gradient_distribution.png
figures/dead_neurons_report.txt
```

## Example Usage

```bash
# Train with diagnostics enabled
python train.py --model oblivious_boosted_vo_alt --fast --diagnostics

# Analyze existing training run
python scripts/analyze_training.py \
  --results results/shakespeare_results.json \
  --checkpoint checkpoints/oblivious_boosted_vo_alt_best.pt

# View diagnostic plots
open figures/gradient_flow.png
open figures/parameter_changes.png
```

## Success Metrics

- ✅ Gradient norms logged every eval interval
- ✅ Plots show healthy gradient flow (no vanishing/exploding)
- ✅ All layers learning (gradient variance < 3x)
- ✅ Parameters changing meaningfully (1-20% from init)
- ✅ <10% dead neurons

## Follow-Up Tasks

- **TASK-22:** Gradient Surgery - modify gradients to fix flow issues
- **TASK-23:** Adaptive Learning Rates - per-component LR based on gradient norms
- **TASK-24:** Weight Initialization Study - find optimal init for trees

## Dependencies

- Uses standard PyTorch hooks
- Matplotlib for visualizations
- NumPy for numerical analysis

## Estimated Time

- Implementation: 5-6 hours
- Testing & validation: 1-2 hours
- **Total:** ~7-8 hours
