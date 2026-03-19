# TASK-19: Checkpoint Infrastructure & Model Persistence

- **Category:** infrastructure
- **Priority:** 9.0/10
- **Impact:** 7/10
- **Feasibility:** 10/10
- **Confidence:** 10/10

## Problem Statement

**CRITICAL GAP:** Models are not being saved!

**Current state:**
- Training runs for hours
- Only saves JSON results (metrics)
- Model weights discarded after training
- **Cannot:**
  - Resume training if interrupted
  - Load best model for generation
  - Share trained models
  - Compare model snapshots over time
  - Debug what model learned at different checkpoints

## Proposed Solution

### **Feature 1: Best Model Checkpointing**

Save model when validation loss improves:

```python
# train.py - inside training loop
if eval_res['val']['loss'] < best_val_loss:
    best_val_loss = eval_res['val']['loss']
    
    checkpoint_path = os.path.join('checkpoints', f'{name}_best.pt')
    os.makedirs('checkpoints', exist_ok=True)
    
    torch.save({
        'step': step,
        'model_state_dict': unwrap(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': best_val_loss,
        'val_acc': eval_res['val']['acc'],
        'config': cfg,
        'model_config': MODEL_CONFIGS[name],
    }, checkpoint_path)
    
    print(f"  ✓ Saved best model to {checkpoint_path}")
```

### **Feature 2: Regular Interval Checkpoints**

Save every N steps (e.g., 500) to track training dynamics:

```python
checkpoint_interval = 500
if step % checkpoint_interval == 0:
    checkpoint_path = os.path.join(
        'checkpoints', f'{name}_step{step}.pt'
    )
    torch.save({...}, checkpoint_path)
```

### **Feature 3: Resume Training**

```python
# train.py - add argument
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')

# Before training loop
start_step = 1
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_step = checkpoint['step'] + 1
    print(f"Resuming from step {start_step}")

# Training loop
for step in range(start_step, n_steps + 1):
    # ...
```

### **Feature 4: Model Loading for Inference**

```python
# inference.py - new file
def load_trained_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model from config
    model_config = checkpoint['model_config']
    cfg = checkpoint['config']
    
    if model_config.get('is_standard'):
        model = StandardTransformer(...)
    else:
        model = TreeTransformer(...)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Usage
model = load_trained_model('checkpoints/oblivious_boosted_vo_alt_best.pt')
dataset = ShakespeareDataset()
sample = dataset.generate(model, prompt="ROMEO:\n", max_tokens=500)
print(sample)
```

### **Feature 5: Checkpoint Management**

```python
# utils/checkpoints.py - new file
import glob
import os

def keep_best_k_checkpoints(checkpoint_dir, k=5, metric='val_loss'):
    """Keep only the k best checkpoints by metric."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*_step*.pt'))
    
    # Load metadata
    ckpt_scores = []
    for ckpt in checkpoints:
        data = torch.load(ckpt, map_location='cpu')
        score = data.get(metric, float('inf'))
        ckpt_scores.append((ckpt, score))
    
    # Sort and keep top k
    ckpt_scores.sort(key=lambda x: x[1])
    to_keep = set(c[0] for c in ckpt_scores[:k])
    
    # Delete others
    for ckpt, _ in ckpt_scores[k:]:
        os.remove(ckpt)
        print(f"Deleted checkpoint: {ckpt}")
```

### **Feature 6: Checkpoint Comparison**

```python
# compare_checkpoints.py - new file
import torch
import matplotlib.pyplot as plt

def compare_checkpoints(checkpoint_paths):
    """Compare metrics across checkpoints."""
    metrics = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location='cpu')
        metrics.append({
            'step': ckpt['step'],
            'val_loss': ckpt['val_loss'],
            'val_acc': ckpt['val_acc'],
            'path': path,
        })
    
    # Plot
    steps = [m['step'] for m in metrics]
    val_accs = [m['val_acc'] for m in metrics]
    
    plt.plot(steps, val_accs, marker='o')
    plt.xlabel('Training Step')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Checkpoints Comparison')
    plt.savefig('figures/checkpoint_comparison.png')
```

## Implementation Plan

### **Phase 1: Basic Checkpointing (30 min)**
1. Add best model saving to `train.py`
2. Create `checkpoints/` directory
3. Test: Train for 100 steps, verify checkpoint created

### **Phase 2: Resume Training (30 min)**
4. Add `--resume` argument
5. Load state dicts before training
6. Test: Train 100 steps, resume from checkpoint, continue to 200

### **Phase 3: Utilities (1 hour)**
7. Create `inference.py` for loading and generating
8. Create `utils/checkpoints.py` for management
9. Create `compare_checkpoints.py` for analysis

### **Phase 4: Integration (30 min)**
10. Update README with checkpoint usage
11. Add checkpoint examples to documentation
12. Test full pipeline

## Expected Benefits

1. **Reliability:** Can recover from crashes/interruptions
2. **Experimentation:** Load best model, try different generation settings
3. **Debugging:** Compare model state at different training stages
4. **Reproducibility:** Share exact model weights with others
5. **Efficiency:** Don't re-train from scratch for every experiment

## Storage Considerations

**Checkpoint sizes (approximate):**
- Standard Transformer (843K params): ~3.4 MB
- Tree Model (1.4M params): ~5.6 MB

**Storage for 2000-step run with checkpoints every 500 steps:**
- 4 interval checkpoints × 5.6 MB = 22.4 MB
- 1 best checkpoint = 5.6 MB
- **Total:** ~28 MB per model

**Recommendation:** Keep last 10 checkpoints + best, ~60 MB per model

## Integration with Existing Code

**Minimal changes required:**
```python
# train.py - add ~20 lines for checkpoint saving
# inference.py - new file, ~50 lines
# utils/checkpoints.py - new file, ~40 lines
# compare_checkpoints.py - new file, ~30 lines
```

**No changes to:**
- `main.py` (model definitions)
- `data.py` (dataset)
- `benchmark.py` (synthetic tasks)

## Success Metrics

- ✅ Can save checkpoint during training
- ✅ Can resume training from checkpoint
- ✅ Can load model and generate text
- ✅ Checkpoint file size reasonable (<10 MB)
- ✅ Resume training continues from exact same state

## Follow-Up Tasks

After checkpointing works:
- **TASK-20:** Model Zoo - curate best checkpoints for each config
- **TASK-21:** Checkpoint Analyzer - visualize weight distributions, routing patterns
- **TASK-22:** Model Surgery - edit checkpoints (change temperature, prune trees, etc.)

## Dependencies

None. Uses only standard PyTorch functionality.

## Estimated Time

- Implementation: 2.5 hours
- Testing: 30 min
- Documentation: 30 min
- **Total:** ~3.5 hours
