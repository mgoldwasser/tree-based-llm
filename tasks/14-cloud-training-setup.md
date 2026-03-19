# Task 14: Cloud Training Setup (GCP)

**Status:** Pending
**Priority:** Medium — needed once architecture is stable
**Depends on:** Task 11 (Shakespeare), Task 09 (speed optimizations)
**Files:** new `train.py`, `requirements.txt`, optional `Dockerfile`

## Goal
Run Shakespeare training on a GPU instance in ~10 minutes instead of hours locally.

## Setup

### requirements.txt
```
torch>=2.0
wandb  # optional, for experiment tracking
```

### train.py
```python
# Key features:
# - Auto device detection (cuda > mps > cpu)
# - Mixed precision training (torch.amp)
# - Gradient accumulation
# - Checkpoint save/load
# - Shakespeare data loading
# - Multiple model configs via CLI args
# - Progress logging with loss, accuracy, entropy, ms/step
```

### GCP Quick Start
```bash
# Option 1: Colab (free/cheap, fastest to get started)
# Just upload main.py + train.py + data.py, select GPU runtime

# Option 2: GCP Compute Engine
gcloud compute instances create tree-llm-gpu \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --boot-disk-size=50GB \
  --preemptible

# SSH in, clone repo, run
gcloud compute ssh tree-llm-gpu
git clone <repo> && cd tree-based-llm
pip install -r requirements.txt
python train.py --model boosted --dataset shakespeare --steps 50000

# Option 3: Lambda Labs / RunPod (simplest GPU rental)
# Web UI, select A100, upload files, run
```

### Mixed Precision Training
```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
Expected: ~2x speedup + ~50% less GPU memory.

### torch.compile (PyTorch 2.0+)
```python
model = torch.compile(model)  # one-line graph optimization
```
Expected: 10-30% additional speedup from kernel fusion.

## Cost Estimate
- L4 GPU (GCP preemptible): ~$0.30/hr
- A100 GPU (Lambda): ~$1.10/hr
- Training 50K steps on Shakespeare: ~10-30 minutes on A100
- **Total cost: $0.20-0.50 per experiment**

## Experiment Matrix
Run on GPU with Shakespeare:
| Experiment | Model | Steps | Est. Time |
|-----------|-------|-------|-----------|
| Baseline | Standard Transformer | 50K | 5min |
| Batched Forest | 12 trees, depth 3 | 50K | 8min |
| Boosted Forest | 3×12 trees, depth 2 | 50K | 12min |
| Shared Routing | 12 trees, shared routing | 50K | 6min |
| All configs | 5 models | 50K each | ~45min |

Total GPU cost for full experiment: ~$1
