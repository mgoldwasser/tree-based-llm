# Task 10: GPU Parallelization & Scaling Strategy

**Status:** Pending
**Priority:** Medium — needed for serious training, but architecture must be solid first
**Depends on:** Task 09 (speed optimizations)
**Files:** `main.py`, `benchmark.py`, new `train.py`

## What Can Be Parallelized on GPU

### Already parallelizable (just needs CUDA)
1. **All einsums** — `einsum('bsd,tnd->bstn')` maps directly to batched GEMM on GPU. PyTorch will use cuBLAS/cutlass automatically.
2. **Sigmoid, log, exp** — elementwise ops, trivially parallel.
3. **Softmax** — PyTorch's fused softmax kernel.
4. **The depth loop iterations** — each iteration's tensor ops are parallel. The sequential dependency is between levels only.

### Needs architectural changes
1. **Cross-tree parallelism:** Already handled by batched tensors — GPU processes all trees in one kernel launch.
2. **Cross-stage parallelism (Boosted):** Currently sequential. If stages don't see each other's residuals (Option A from Task 07), all stages can run in parallel via a single mega-forest of `n_stages × trees_per_stage` trees. With residual pass-through (Option C), stages must be sequential.
3. **Multi-GPU / Data Parallel:** Standard PyTorch DDP wraps the model. Tree forests are `nn.Module` with `nn.Parameter`s — fully compatible. No special handling needed.
4. **Tensor Parallel:** Could split the tree dimension across GPUs — each GPU handles a subset of trees. Requires a final all-reduce for the weighted sum.

## GPU Memory Considerations
With vocab_size=65 (Shakespeare), d_model=256, n_layers=6, n_heads=8, 12 trees depth 4:
- Model params: ~5M → ~20MB in fp32
- Activations per batch (batch=64, seq=256): ~200MB
- Total: well under 8GB, fits on any modern GPU

With larger models (d_model=512, n_layers=12):
- Model: ~50M params → 200MB
- Activations: ~2GB
- Fits on a single A100/V100

## Infrastructure Options

### Google Cloud (recommended for initial scaling)
```bash
# A100 spot instance (~$1/hr)
gcloud compute instances create tree-llm \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --preemptible  # spot pricing
```

### Alternatives
- **Colab Pro** ($12/mo): A100 access, good for prototyping
- **Lambda Labs**: ~$1.10/hr for A100
- **RunPod**: ~$0.80/hr for A100
- **Local Mac GPU (MPS)**: Free but ~5x slower than A100. Already supported via `torch.backends.mps`.

## Training Script for GPU (`train.py`)
New file that handles:
- Device auto-detection (CUDA > MPS > CPU)
- Mixed precision (fp16/bf16) for 2x speedup + less memory
- Gradient accumulation for effective larger batches
- Checkpointing every N steps
- WandB or TensorBoard logging
- Shakespeare dataset loading

## Scaling Estimates
| Config | CPU (M2 Pro) | A100 GPU | Speedup |
|--------|-------------|----------|---------|
| Standard, 1000 steps | 27s | ~3s | ~9x |
| Batched Forest, 1000 steps | 107s | ~10s | ~11x |
| Full training, 10K steps | ~18min | ~1.5min | ~12x |
| Shakespeare, 50K steps | ~hours | ~10min | massive |

GPU speedup is higher for tree models because the batched einsums benefit more from GPU parallelism than simple matrix multiplies (more independent work units).

## Implementation Plan
1. Create `train.py` with device auto-detect and mixed precision
2. Add Shakespeare dataset loading (download + char-level tokenization)
3. Add `torch.compile()` support for PyTorch 2.0+ graph optimization
4. Test on local MPS first, then deploy to cloud GPU
