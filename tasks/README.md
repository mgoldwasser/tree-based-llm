# Development Roadmap

## Execution Order

### Phase 1: Fix Foundations (do these first, locally)
1. **[Task 08] Parameter Audit** — Fix init, optimizer groups, weight decay conflicts
2. **[Task 09] Speed Optimizations** — QKV fusion, remove redundant norms, shared routing
3. **[Task 07] True Residual Boosting** — Feature pass-through so stages correct residuals

### Phase 2: Better Accuracy (locally, fast iteration)
4. **[Task 12] Accuracy + Speed** — Shared routing, conditional tree selection, top-K pruning

### Phase 3: Real Data (needs GPU or patience)
5. **[Task 11] Shakespeare Dataset** — Character-level LM on real text
6. **[Task 13] Scaling Analysis** — Profile and understand where time goes

### Phase 4: Scale Up
7. **[Task 14] Cloud Training** — GCP/Colab setup, mixed precision, torch.compile
8. Run full experiment matrix on GPU

## Completed Tasks
See `completed/` directory:
- 01: Entropy-based regularization
- 02: Temperature annealing
- 03: Log-space leaf probabilities
- 04: BatchedTreeForest (batched einsum)
- 05: BoostedForest (multi-stage ensemble)
- 06: Benchmark v2 (accuracy, 1000 steps, 4 models)

## Key Architectural Questions
- Can shared-routing trees match separate-routing accuracy? (Task 12.1)
- Does residual pass-through help boosted stages specialize? (Task 07)
- Do trees provide value on real language data? (Task 11)
- Where is the speed/accuracy Pareto frontier? (Task 13)
