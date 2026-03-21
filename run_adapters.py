"""
NEW-04: Trees-as-Adapters — Pretrain linear, fine-tune trees.

Two-phase training:
1. Pretrain standard transformer for N steps (fast, well-understood)
2. Convert linear projections to LinearPlusMicroTree, freeze linear bases
3. Fine-tune only tree parameters for M steps

Tests whether trees shine as adaptation mechanism rather than from-scratch.
Analogous to LoRA but with input-dependent routing.

Usage:
    python run_adapters.py              # fast config
    python run_adapters.py --full       # full config
    python run_adapters.py --no-compile
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import (
    TreeTransformer, tree_regularization_loss, leaf_balancing_loss,
    count_parameters, set_temperature, get_routing_entropy, make_optimizer,
    freeze_non_tree_params, unfreeze_all_params, LinearPlusMicroTree,
)
from train import StandardTransformer, create_model, MODEL_CONFIGS
from data import ShakespeareDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def add_tree_adapters(standard_model, tree_depth=1, n_trees=4, leaf_rank=8):
    """
    Replace nn.Linear projections in attention with LinearPlusMicroTree.
    Copies the original linear weights into the base_proj of the new module.
    Returns a new TreeTransformer-based model.
    """
    # Strategy: Rather than surgery on the standard model, we:
    # 1. Create a TreeTransformer with micro_boosted proj_type
    # 2. Copy matching weights from the standard model
    # This is cleaner and ensures all tree infrastructure is in place.

    d_model = standard_model.tok.weight.shape[1]
    vocab_size = standard_model.tok.weight.shape[0]
    n_layers = len(standard_model.layers)
    n_heads = standard_model.layers[0].attn.n_heads
    max_seq_len = standard_model.pos.weight.shape[0]

    # Create tree model with micro_boosted projections
    tree_model = TreeTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_seq_len=max_seq_len,
        num_classes=vocab_size, dropout=0.0,
        use_tree_ffn=False, task="lm",
        proj_type="micro_boosted",
        n_trees=n_trees, tree_depth=tree_depth, leaf_rank=leaf_rank,
    ).to(next(standard_model.parameters()).device)

    # Copy embeddings and head
    tree_model.token_emb.weight.data.copy_(standard_model.tok.weight.data)
    tree_model.pos_emb.weight.data.copy_(standard_model.pos.weight.data)
    tree_model.head.weight.data.copy_(standard_model.head.weight.data)
    tree_model.head.bias.data.copy_(standard_model.head.bias.data)
    tree_model.final_norm.weight.data.copy_(standard_model.norm.weight.data)
    tree_model.final_norm.bias.data.copy_(standard_model.norm.bias.data)

    # Copy layer weights
    for i, (tree_layer, std_layer) in enumerate(zip(tree_model.layers, standard_model.layers)):
        # Copy layer norms
        tree_layer.norm1.weight.data.copy_(std_layer.norm1.weight.data)
        tree_layer.norm1.bias.data.copy_(std_layer.norm1.bias.data)
        tree_layer.norm2.weight.data.copy_(std_layer.norm2.weight.data)
        tree_layer.norm2.bias.data.copy_(std_layer.norm2.bias.data)

        # Copy FFN weights (FFN uses standard linear in tree model with use_tree_ffn=False)
        for j in range(len(tree_layer.ffn)):
            if hasattr(tree_layer.ffn[j], 'weight'):
                tree_layer.ffn[j].weight.data.copy_(std_layer.ffn[j].weight.data)
                if hasattr(tree_layer.ffn[j], 'bias') and tree_layer.ffn[j].bias is not None:
                    tree_layer.ffn[j].bias.data.copy_(std_layer.ffn[j].bias.data)

        # Copy attention linear weights into LinearPlusMicroTree base_proj
        # Standard model has W_qkv (fused) and W_o
        # Tree model has separate q_proj, k_proj, v_proj, o_proj (each LinearPlusMicroTree)
        std_qkv_w = std_layer.attn.W_qkv.weight.data  # (3*d_model, d_model)
        std_qkv_b = std_layer.attn.W_qkv.bias.data      # (3*d_model,)

        # Split fused QKV into Q, K, V
        q_w, k_w, v_w = std_qkv_w.chunk(3, dim=0)
        q_b, k_b, v_b = std_qkv_b.chunk(3, dim=0)

        for proj_name, w, b in [('q_proj', q_w, q_b), ('k_proj', k_w, k_b),
                                  ('v_proj', v_w, v_b)]:
            proj = getattr(tree_layer.attention, proj_name)
            if hasattr(proj, 'base_proj'):
                proj.base_proj.weight.data.copy_(w)
                proj.base_proj.bias.data.copy_(b)
                # Initialize shrinkage to small value so trees start as small correction
                proj.shrinkage.data.fill_(0.01)

        # Copy output projection
        o_proj = tree_layer.attention.o_proj
        if hasattr(o_proj, 'base_proj'):
            o_proj.base_proj.weight.data.copy_(std_layer.attn.W_o.weight.data)
            o_proj.base_proj.bias.data.copy_(std_layer.attn.W_o.bias.data)
            o_proj.shrinkage.data.fill_(0.01)

    return tree_model


def train_phase(model, dataset, cfg, device, n_steps, is_tree=False, label=""):
    """Run a training phase and return results."""
    raw_model = getattr(model, '_orig_mod', model)

    if is_tree:
        # Only optimize unfrozen params
        trainable = [p for p in raw_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg['lr'], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    warmup_steps = min(100, n_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    start = time.perf_counter()
    eval_interval = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        if is_tree:
            progress = step / n_steps
            if progress < 0.5:
                temp = 1.0
            else:
                phase = (progress - 0.5) / 0.5
                temp = 0.7 + 0.3 * (1 + math.cos(math.pi * phase)) / 2
            set_temperature(raw_model, temp)

        x, y = dataset.get_batch(cfg['batch'], device, 'train')
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
        if is_tree:
            loss = loss + tree_regularization_loss(raw_model, 0.005)
            loss = loss + leaf_balancing_loss(raw_model, 0.01)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            elapsed = time.perf_counter() - start
            print(f"  [{label}] Step {step:>5d}/{n_steps} | loss={loss.item():.3f} "
                  f"acc={acc:.1%} | {elapsed/step*1000:.0f}ms/step")

    elapsed = time.perf_counter() - start
    final = dataset.estimate_loss(model, cfg['batch'], device)
    return {
        'val_acc': final['val']['acc'],
        'val_loss': final['val']['loss'],
        'train_acc': final['train']['acc'],
        'train_loss': final['train']['loss'],
        'time': elapsed,
        'ms_per_step': elapsed / n_steps * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--full", dest="fast", action="store_false")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--pretrain-steps", type=int, default=None,
                        help="Steps for phase 1 pretraining (default: 2/3 of total)")
    parser.add_argument("--finetune-steps", type=int, default=None,
                        help="Steps for phase 2 fine-tuning (default: 1/3 of total)")
    args = parser.parse_args()

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else 'cpu')

    if args.fast:
        cfg = {'d_model': 64, 'n_layers': 2, 'n_heads': 4,
               'seq_len': 128, 'batch': 32, 'n_steps': 2000,
               'lr': 3e-4, 'dropout': 0.0}
    else:
        cfg = {'d_model': 128, 'n_layers': 4, 'n_heads': 4,
               'seq_len': 256, 'batch': 32, 'n_steps': 2000,
               'lr': 3e-4, 'dropout': 0.0}

    pretrain_steps = args.pretrain_steps or (cfg['n_steps'] * 2 // 3)
    finetune_steps = args.finetune_steps or (cfg['n_steps'] * 1 // 3)

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")
    print(f"Phase 1 (pretrain): {pretrain_steps} steps")
    print(f"Phase 2 (finetune): {finetune_steps} steps")

    dataset = ShakespeareDataset(block_size=cfg['seq_len'])
    all_results = {}

    # === Baseline 1: Standard transformer (full training) ===
    print(f"\n{'=' * 65}")
    print(f"  Baseline: Standard Transformer ({cfg['n_steps']} steps)")
    print(f"{'=' * 65}")
    torch.manual_seed(42)
    std_model = StandardTransformer(
        dataset.vocab_size, cfg['d_model'], cfg['n_layers'], cfg['n_heads'],
        cfg['seq_len'], cfg['dropout'],
    ).to(DEVICE)
    std_params = count_parameters(std_model)
    print(f"  Params: {std_params['total']:,}")

    if args.compile:
        std_model = torch.compile(std_model, backend='inductor')

    std_result = train_phase(std_model, dataset, cfg, DEVICE, cfg['n_steps'],
                              is_tree=False, label="Standard")
    std_result['params'] = std_params['total']
    std_result['name'] = 'Standard Transformer'
    all_results['standard_full'] = std_result
    print(f"  Final: val_acc={std_result['val_acc']:.1%}")

    # === Baseline 2: Tree model from scratch ===
    print(f"\n{'=' * 65}")
    print(f"  Baseline: Linear+MicroTree from scratch ({cfg['n_steps']} steps)")
    print(f"{'=' * 65}")
    torch.manual_seed(42)
    tree_scratch = TreeTransformer(
        vocab_size=dataset.vocab_size, d_model=cfg['d_model'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        max_seq_len=cfg['seq_len'], num_classes=dataset.vocab_size,
        dropout=cfg['dropout'], use_tree_ffn=False, task="lm",
        proj_type="micro_boosted", n_trees=4, tree_depth=1, leaf_rank=8,
    ).to(DEVICE)
    scratch_params = count_parameters(tree_scratch)
    print(f"  Params: {scratch_params['total']:,}")

    if args.compile:
        tree_scratch = torch.compile(tree_scratch, backend='inductor')

    scratch_result = train_phase(tree_scratch, dataset, cfg, DEVICE, cfg['n_steps'],
                                  is_tree=True, label="MicroTree-scratch")
    scratch_result['params'] = scratch_params['total']
    scratch_result['name'] = 'Linear+MicroTree (from scratch)'
    all_results['micro_tree_scratch'] = scratch_result
    print(f"  Final: val_acc={scratch_result['val_acc']:.1%}")
    del tree_scratch

    # === Adapter approach: Pretrain standard, then add trees ===
    print(f"\n{'=' * 65}")
    print(f"  Phase 1: Pretrain Standard ({pretrain_steps} steps)")
    print(f"{'=' * 65}")
    torch.manual_seed(42)
    pretrain_model = StandardTransformer(
        dataset.vocab_size, cfg['d_model'], cfg['n_layers'], cfg['n_heads'],
        cfg['seq_len'], cfg['dropout'],
    ).to(DEVICE)

    # Don't compile for phase 1 since we need to extract weights
    pretrain_result = train_phase(pretrain_model, dataset, cfg, DEVICE, pretrain_steps,
                                   is_tree=False, label="Pretrain")
    print(f"  After pretrain: val_acc={pretrain_result['val_acc']:.1%}")

    # Convert to tree model with adapters
    print(f"\n{'=' * 65}")
    print(f"  Phase 2: Add tree adapters, freeze base, fine-tune ({finetune_steps} steps)")
    print(f"{'=' * 65}")

    adapter_model = add_tree_adapters(pretrain_model, tree_depth=1, n_trees=4, leaf_rank=8)
    del pretrain_model

    adapter_params = count_parameters(adapter_model)
    print(f"  Params: {adapter_params['total']:,} (tree: {adapter_params['tree_pct']:.1f}%)")

    # Freeze non-tree params
    freeze_non_tree_params(adapter_model)
    trainable = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in adapter_model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    if args.compile:
        adapter_model = torch.compile(adapter_model, backend='inductor')

    adapter_result = train_phase(adapter_model, dataset, cfg, DEVICE, finetune_steps,
                                  is_tree=True, label="Adapter-finetune")
    adapter_result['params'] = adapter_params['total']
    adapter_result['trainable_params'] = trainable
    adapter_result['pretrain_steps'] = pretrain_steps
    adapter_result['finetune_steps'] = finetune_steps
    adapter_result['pretrain_val_acc'] = pretrain_result['val_acc']
    adapter_result['name'] = 'Tree Adapter (pretrain+finetune)'
    all_results['tree_adapter'] = adapter_result
    print(f"  Final: val_acc={adapter_result['val_acc']:.1%}")

    del adapter_model

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"adapter_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'#' * 65}")
    print(f"  ADAPTER EXPERIMENT RESULTS")
    print(f"{'#' * 65}")
    print(f"  {'Model':<40s} {'Val Acc':>8} {'Val Loss':>9} {'Params':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {r['name']:<40s} {r['val_acc']:>8.1%} "
              f"{r['val_loss']:>9.3f} {r['params']:>10,}")

    std_acc = all_results['standard_full']['val_acc']
    adapter_acc = all_results['tree_adapter']['val_acc']
    scratch_acc = all_results['micro_tree_scratch']['val_acc']

    print(f"\n  Adapter vs Standard: {(adapter_acc - std_acc)*100:+.1f}pp")
    print(f"  Adapter vs From-scratch: {(adapter_acc - scratch_acc)*100:+.1f}pp")
    print(f"  Adapter trained only {all_results['tree_adapter']['trainable_params']:,} params "
          f"({all_results['tree_adapter']['trainable_params']/all_results['tree_adapter']['params']*100:.1f}%)")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
