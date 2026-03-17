"""
Tree-Based Attention Mechanism for Sequence Modeling
=====================================================
Replaces dense linear projections (Q, K, V) in transformer attention
with differentiable soft decision trees, trained end-to-end via backprop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# 1. DIFFERENTIABLE SOFT DECISION TREE
# =============================================================================

class SoftDecisionTree(nn.Module):
    """
    Each internal node computes soft routing: p_left = sigmoid(w @ x + b)
    Each leaf holds a learned output vector. Output = probability-weighted
    sum over all leaves.
    
    Depth D → 2^D - 1 internal nodes, 2^D leaves.
    """
    
    def __init__(self, input_dim: int, output_dim: int, depth: int = 4, temperature: float = 1.0):
        super().__init__()
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.temperature = temperature
        
        self.decision_weights = nn.Parameter(torch.empty(self.n_internal, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(self.n_internal))
        self.leaf_outputs = nn.Parameter(torch.empty(self.n_leaves, output_dim))
        
        nn.init.xavier_normal_(self.decision_weights)
        nn.init.xavier_normal_(self.leaf_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        
        batch, seq_len, _ = x.shape
        
        # All routing decisions at once: (batch, seq, n_internal)
        decisions = torch.einsum('bsi,ni->bsn', x, self.decision_weights)
        decisions = decisions + self.decision_biases
        decisions = torch.sigmoid(decisions / self.temperature)
        
        # Path probabilities for each leaf
        leaf_probs = self._compute_leaf_probabilities(decisions)
        
        # Weighted sum of leaf outputs
        output = torch.einsum('bsl,lo->bso', leaf_probs, self.leaf_outputs)
        
        if squeeze:
            output = output.squeeze(1)
        return output
    
    def _compute_leaf_probabilities(self, decisions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = decisions.shape
        device = decisions.device
        
        current_probs = torch.ones(batch, seq_len, 1, device=device)
        
        for d in range(self.depth):
            n_nodes_at_level = 2 ** d
            start_idx = 2 ** d - 1
            
            level_decisions = decisions[:, :, start_idx:start_idx + n_nodes_at_level]
            
            left_probs = current_probs * level_decisions
            right_probs = current_probs * (1 - level_decisions)
            
            current_probs = torch.stack([left_probs, right_probs], dim=-1)
            current_probs = current_probs.reshape(batch, seq_len, -1)
        
        return current_probs


# =============================================================================
# 2. TREE ENSEMBLE (replaces nn.Linear)
# =============================================================================

class TreeEnsemble(nn.Module):
    """
    Multiple soft trees with learned weights — a differentiable forest
    that acts as a projection layer.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 4,
                 tree_depth: int = 4, temperature: float = 1.0):
        super().__init__()
        self.trees = nn.ModuleList([
            SoftDecisionTree(input_dim, output_dim, depth=tree_depth, temperature=temperature)
            for _ in range(n_trees)
        ])
        self.tree_weights = nn.Parameter(torch.ones(n_trees) / n_trees)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.tree_weights, dim=0)
        output = torch.zeros_like(self.trees[0](x))
        for i, tree in enumerate(self.trees):
            output = output + weights[i] * tree(x)
        return self.norm(output)


# =============================================================================
# 3. TREE-BASED MULTI-HEAD ATTENTION
# =============================================================================

class TreeAttention(nn.Module):
    """
    Multi-head attention with TreeEnsemble projections for Q, K, V, O.
    
        Q = TreeEnsemble_Q(x)     # not W_Q @ x
        K = TreeEnsemble_K(x)
        V = TreeEnsemble_V(x)
        Attn = softmax(QK^T / sqrt(d_k)) @ V
        Out = TreeEnsemble_O(concat(heads))
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, n_trees: int = 4,
                 tree_depth: int = 4, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        self.q_trees = TreeEnsemble(d_model, d_model, n_trees, tree_depth, temperature)
        self.k_trees = TreeEnsemble(d_model, d_model, n_trees, tree_depth, temperature)
        self.v_trees = TreeEnsemble(d_model, d_model, n_trees, tree_depth, temperature)
        self.o_trees = TreeEnsemble(d_model, d_model, n_trees, tree_depth, temperature)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape
        
        Q = self.q_trees(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_trees(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_trees(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.o_trees(context)
        
        return output, attn_weights


# =============================================================================
# 4. TREE TRANSFORMER BLOCK
# =============================================================================

class TreeTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_trees: int = 4,
                 tree_depth: int = 4, ff_dim: int = None, dropout: float = 0.1,
                 temperature: float = 1.0, use_tree_ffn: bool = True):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        
        self.attention = TreeAttention(d_model, n_heads, n_trees, tree_depth, dropout, temperature)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        if use_tree_ffn:
            self.ffn = nn.Sequential(
                TreeEnsemble(d_model, ff_dim, n_trees, tree_depth, temperature),
                nn.GELU(),
                nn.Dropout(dropout),
                TreeEnsemble(ff_dim, d_model, n_trees, tree_depth, temperature),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
            )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


# =============================================================================
# 5. FULL TREE TRANSFORMER
# =============================================================================

class TreeTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, n_trees: int = 4, tree_depth: int = 4,
                 max_seq_len: int = 512, num_classes: int = 2, dropout: float = 0.1,
                 temperature: float = 1.0, use_tree_ffn: bool = True,
                 task: str = "classification"):
        super().__init__()
        self.task = task
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TreeTransformerBlock(d_model, n_heads, n_trees, tree_depth,
                                dropout=dropout, temperature=temperature,
                                use_tree_ffn=use_tree_ffn)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        if task == "classification":
            self.head = nn.Linear(d_model, num_classes)
        elif task == "lm":
            self.head = nn.Linear(d_model, vocab_size)
    
    def _causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids, mask=None):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        if self.task == "lm" and mask is None:
            mask = self._causal_mask(seq_len, device)
        
        all_attn = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            all_attn.append(attn_w)
        
        x = self.final_norm(x)
        
        if self.task == "classification":
            x = x.mean(dim=1)  # mean pool
        
        return self.head(x), all_attn


# =============================================================================
# 6. UTILITIES
# =============================================================================

def tree_regularization_loss(model, lambda_reg=0.01):
    """Penalize uncertain routing (near 0.5), encouraging crisp splits."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'decision_weights' in name:
            reg = reg - lambda_reg * param.abs().mean()
    return reg


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    tree_p = sum(p.numel() for n, p in model.named_parameters()
                 if any(k in n for k in ['tree', 'decision', 'leaf']))
    return {'total': total, 'tree': tree_p, 'tree_pct': tree_p/total*100}


# =============================================================================
# 7. DEMO
# =============================================================================

if __name__ == "__main__":
    import time
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TreeTransformer(
        vocab_size=1000, d_model=64, n_layers=2, n_heads=4,
        n_trees=3, tree_depth=3, max_seq_len=32, num_classes=2,
        use_tree_ffn=False, task="classification"
    ).to(DEVICE)
    
    params = count_parameters(model)
    print(f"Params: {params['total']:,} (tree: {params['tree_pct']:.1f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(1, 51):
        tokens = torch.randint(0, 1000, (16, 32), device=DEVICE)
        labels = (tokens % 2 == 0).float().sum(1).gt(16).long()
        
        logits, _ = model(tokens)
        loss = criterion(logits, labels) + tree_regularization_loss(model, 0.001)
        
        optimizer.zero_grad()
        loss.backward()  # Gradients flow through soft trees!
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            acc = (logits.argmax(-1) == labels).float().mean()
            print(f"Step {step:3d} | Loss: {loss:.4f} | Acc: {acc:.3f}")
    
    print("\nAll gradients flow through tree-based projections!")
