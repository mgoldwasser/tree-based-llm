"""
Tree-Based Attention Mechanism for Sequence Modeling
=====================================================
Replaces dense linear projections (Q, K, V) in transformer attention
with differentiable soft decision trees, trained end-to-end via backprop.

Features:
- Batched tensor operations (all trees in one einsum)
- Separate Q/K/V forests (unfused routing)
- LinearPlusForest (linear base + wide forest correction)
- Oblivious trees (NODE-style, independent depth-level decisions)
- Input-dependent tree gating (MoE-style)
- Per-node learnable temperature
- QK-norm for stable attention
- MoE-style load-balancing loss on leaf utilization
- Tree-aware optimizer with separate parameter groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


# =============================================================================
# 0. SHARED HELPERS
# =============================================================================

def _build_path_indices(n_leaves: int, tree_depth: int):
    """Precompute path indices for vectorized leaf prob computation.
    Returns (path_indices, path_dirs) tensors as registered buffers."""
    path_indices = torch.zeros(n_leaves, tree_depth, dtype=torch.long)
    path_dirs = torch.zeros(n_leaves, tree_depth)
    for leaf in range(n_leaves):
        node = 0
        for d in range(tree_depth):
            path_indices[leaf, d] = node
            bit = (leaf >> (tree_depth - 1 - d)) & 1
            path_dirs[leaf, d] = 1.0 - bit
            node = 2 * node + 1 + bit
    return path_indices, path_dirs


def _compute_oblivious_leaf_probs(decisions: torch.Tensor) -> torch.Tensor:
    """Compute leaf probs via Kronecker product. Unrolled for depth 3, general otherwise."""
    B, S, T, D = decisions.shape
    if D == 3:
        d0, d1, d2 = decisions[..., 0], decisions[..., 1], decisions[..., 2]
        p0 = torch.stack([d0, 1 - d0], dim=-1)
        p1 = torch.stack([d1, 1 - d1], dim=-1)
        p2 = torch.stack([d2, 1 - d2], dim=-1)
        return (p0.unsqueeze(-1).unsqueeze(-1)
                * p1.unsqueeze(-2).unsqueeze(-1)
                * p2.unsqueeze(-2).unsqueeze(-2)
                ).reshape(B, S, T, 8)
    else:
        d0 = decisions[..., 0:1]
        probs = torch.cat([d0, 1 - d0], dim=-1)
        for d in range(1, D):
            dd = decisions[..., d:d+1]
            level = torch.cat([dd, 1 - dd], dim=-1)
            probs = (probs.unsqueeze(-1) * level.unsqueeze(-2)).reshape(B, S, T, -1)
        return probs


def _init_forest_params(decision_weights: torch.Tensor, leaf_outputs: torch.Tensor):
    """Standard init for forest parameters: normal(0, 0.1) for routing, xavier for leaves."""
    nn.init.normal_(decision_weights, 0, 0.1)
    # Xavier init per tree (or per head×tree for shared routing)
    flat = leaf_outputs.view(-1, *leaf_outputs.shape[-2:])
    for i in range(flat.shape[0]):
        nn.init.xavier_normal_(flat[i])


# =============================================================================
# 1. BATCHED TREE FOREST (replaces nn.Linear)
# =============================================================================

class BatchedTreeForest(nn.Module):
    """
    A forest of soft decision trees with all parameters stored in stacked
    tensors. All trees computed in a single batched pass.

    Features:
    - Per-node learnable temperature factors
    - Input-dependent tree gating (MoE-style)
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 12,
                 tree_depth: int = 3, temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_internal = 2 ** tree_depth - 1
        self.n_leaves = 2 ** tree_depth
        # Buffer tensor so torch.compile doesn't recompile on temperature changes
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Stacked parameters for ALL trees
        self.decision_weights = nn.Parameter(torch.empty(n_trees, self.n_internal, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, self.n_internal))
        self.leaf_outputs = nn.Parameter(torch.empty(n_trees, self.n_leaves, output_dim))

        # Input-dependent tree gating (replaces fixed tree_weights)
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Per-node learnable temperature factors
        # softplus(x + 0.5413) ≈ 1.0 at init, so effective temp starts at self.temperature
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, self.n_internal))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        path_indices, path_dirs = _build_path_indices(self.n_leaves, tree_depth)
        self.register_buffer('path_indices', path_indices)
        self.register_buffer('path_dirs', path_dirs)

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Batched routing for ALL trees: (B, S, T, n_internal)
        decisions = torch.einsum('bsd,tnd->bstn', x, self.decision_weights)
        decisions = decisions + self.decision_biases

        # Per-node temperature: global schedule * learned per-node factor
        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        # Leaf probabilities in log-space: (B, S, T, n_leaves)
        leaf_probs = self._compute_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs

        # Per-tree output → input-dependent weighted mixture: (B, S, output_dim)
        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)
        weights = F.softmax(self.gate_proj(x), dim=-1)  # (B, S, n_trees)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)

    def _compute_leaf_probs(self, decisions: torch.Tensor) -> torch.Tensor:
        """Vectorized leaf probs: gather path decisions + product. No Python loop."""
        # decisions: (B, S, T, n_internal)
        # path_indices: (n_leaves, depth) — which node on path to each leaf
        # path_dirs: (n_leaves, depth) — 1=use decision (left), 0=use 1-decision (right)
        relevant = decisions[..., self.path_indices]  # (B, S, T, n_leaves, depth)
        # For left paths use decision value, for right use (1-decision)
        probs_per_level = self.path_dirs * relevant + (1 - self.path_dirs) * (1 - relevant)
        return probs_per_level.prod(dim=-1)  # (B, S, T, n_leaves)


# =============================================================================
# 1b. OBLIVIOUS TREE FOREST (NODE-style)
# =============================================================================

class ObliviousTreeForest(nn.Module):
    """
    Oblivious decision trees: all nodes at the same depth share one hyperplane.

    Key advantages over BatchedTreeForest:
    - Decision weights: (n_trees, depth, input_dim) — fewer params
    - Leaf probs via outer product — eliminates depth-dependent gradient vanishing
    - Each depth level gets independent gradients (no multiplicative chain)
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 12,
                 tree_depth: int = 3, temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # One hyperplane per depth level (shared across all nodes at that depth)
        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_outputs = nn.Parameter(torch.empty(n_trees, self.n_leaves, output_dim))

        # Input-dependent tree gating
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Per-depth learnable temperature factors
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        decisions = torch.einsum('bsd,tnd->bstn', x, self.decision_weights)
        decisions = decisions + self.decision_biases

        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        leaf_probs = _compute_oblivious_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs

        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)
        weights = F.softmax(self.gate_proj(x), dim=-1)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


# =============================================================================
# 1c. MICRO TREE FOREST (depth 1-2, low-rank leaf factorization)
# =============================================================================

class MicroTreeForest(nn.Module):
    """
    Minimal-overhead tree forest: depth 1-2, few trees, low-rank leaf outputs.
    Designed for <5% parameter overhead over nn.Linear.

    Uses oblivious-style routing (one hyperplane per depth level).
    Low-rank factorization: leaf_output = leaf_down @ leaf_up (rank << output_dim).
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 4,
                 tree_depth: int = 1, leaf_rank: int = 8, temperature: float = 1.0,
                 use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.leaf_rank = leaf_rank
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # One hyperplane per depth level (oblivious-style)
        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))

        # Low-rank leaf factorization: leaf_output = leaf_down @ leaf_up
        self.leaf_down = nn.Parameter(torch.empty(n_trees, self.n_leaves, leaf_rank))
        self.leaf_up = nn.Parameter(torch.empty(n_trees, leaf_rank, output_dim))

        # Input-dependent tree gating
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Per-depth learnable temperature factors
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        # Init
        nn.init.normal_(self.decision_weights, 0, 0.1)
        nn.init.xavier_normal_(self.leaf_down.view(-1, self.n_leaves, leaf_rank)[0].unsqueeze(0).squeeze(0))
        for t in range(n_trees):
            nn.init.xavier_normal_(self.leaf_down[t])
            nn.init.xavier_normal_(self.leaf_up[t])

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Oblivious routing: (B, S, T, depth)
        decisions = torch.einsum('bsd,tnd->bstn', x, self.decision_weights)
        decisions = decisions + self.decision_biases

        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        # Leaf probs via outer product
        leaf_probs = _compute_oblivious_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs

        # Low-rank leaf outputs: (n_trees, n_leaves, output_dim)
        leaf_outputs = torch.bmm(self.leaf_down, self.leaf_up)  # (T, L, O)

        # Per-tree output → gated mixture
        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, leaf_outputs)
        weights = F.softmax(self.gate_proj(x), dim=-1)  # (B, S, n_trees)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


# =============================================================================
# 1d. CONTEXTUAL ROUTING FOREST (context-aware routing via EMA)
# =============================================================================

class ContextualRoutingForest(nn.Module):
    """
    Oblivious forest with context-aware routing.
    routing_input = x + context_proj(ema_context)
    where ema_context is exponential moving average of recent hidden states.
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 12,
                 tree_depth: int = 3, temperature: float = 1.0,
                 context_dim: int = None, ema_decay: float = 0.9,
                 use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.ema_decay = ema_decay
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Context projection: maps EMA context to routing space
        self.context_proj = nn.Linear(input_dim, input_dim)

        # One hyperplane per depth level (oblivious-style)
        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_outputs = nn.Parameter(torch.empty(n_trees, self.n_leaves, output_dim))

        # Input-dependent tree gating
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Per-depth learnable temperature factors
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def _compute_ema_context(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal EMA context along sequence dimension (vectorized)."""
        B, S, D = x.shape
        decay = self.ema_decay

        # Vectorized EMA via power series:
        # ema[t] = (1-d) * sum_{k=0}^{t-1} d^k * x[t-1-k]
        # Shift x by 1 (first position gets zeros context)
        x_shifted = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
                                x[:, :-1]], dim=1)

        # Compute decay powers: d^0, d^1, ..., d^{S-1}
        powers = decay ** torch.arange(S, device=x.device, dtype=x.dtype)  # (S,)

        # Scale shifted input by decay powers (reversed for convolution)
        # x_scaled[t] = d^t * x_shifted[t]
        x_scaled = x_shifted * powers.unsqueeze(0).unsqueeze(-1)  # (B, S, D)

        # Cumulative sum gives sum_{k=0}^{t} d^k * x_shifted[k]
        cumsum = torch.cumsum(x_scaled, dim=1)  # (B, S, D)

        # Undo the scaling: ema[t] = (1-d) * cumsum[t] / d^t
        # But d^t can be very small for large t, so use: cumsum[t] / d^t
        inv_powers = (1.0 / (powers + 1e-8)).unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
        ema_context = (1 - decay) * cumsum * inv_powers

        return ema_context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Compute context-aware routing input
        ema_context = self._compute_ema_context(x)
        routing_input = x + self.context_proj(ema_context)

        # Oblivious routing on context-enhanced input: (B, S, T, depth)
        decisions = torch.einsum('bsd,tnd->bstn', routing_input, self.decision_weights)
        decisions = decisions + self.decision_biases

        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        leaf_probs = _compute_oblivious_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs

        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)
        weights = F.softmax(self.gate_proj(x), dim=-1)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


# =============================================================================
# 2. LINEAR + FOREST (linear base + single wide forest correction)
# =============================================================================

class LinearPlusForest(nn.Module):
    """
    Linear base projection + single wide forest for nonlinear correction.
    output = base_linear(x) + shrinkage * forest(x)
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 24,
                 tree_depth: int = 3, temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = BatchedTreeForest(input_dim, output_dim, n_trees,
                                        tree_depth, temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


class ObliviousLinearPlusForest(nn.Module):
    """LinearPlusForest variant using oblivious trees."""

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 24,
                 tree_depth: int = 3, temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = ObliviousTreeForest(input_dim, output_dim, n_trees,
                                          tree_depth, temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


class LinearPlusMicroTree(nn.Module):
    """Linear base + micro tree correction. Low-rank factorized leaves for minimal overhead."""

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 4,
                 tree_depth: int = 1, leaf_rank: int = 8, temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = MicroTreeForest(input_dim, output_dim, n_trees, tree_depth,
                                       leaf_rank, temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


class LinearPlusContextual(nn.Module):
    """Linear base + contextual routing forest correction."""

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 24,
                 tree_depth: int = 3, temperature: float = 1.0,
                 ema_decay: float = 0.9):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = ContextualRoutingForest(input_dim, output_dim, n_trees,
                                               tree_depth, temperature,
                                               ema_decay=ema_decay, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


# =============================================================================
# 2e. SPEED-OPTIMIZED PROJECTIONS
# =============================================================================

class GatedProjection(nn.Module):
    """
    GLU-style gated projection: output = W(x) * sigmoid(V(x))

    Equivalent to a depth-1 tree with 2 leaves but implemented as
    parallel matmuls — no routing infrastructure needed.
    Can stack multiple gates for more "depth".
    """

    def __init__(self, input_dim: int, output_dim: int, n_gates: int = 1,
                 temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_gates = n_gates
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        self.W = nn.Linear(input_dim, output_dim)
        # One gate projection per "depth level"
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(n_gates)
        ])
        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        # For compatibility with tree utilities
        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.depth = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.W(x)
        for gate in self.gates:
            output = output * torch.sigmoid(gate(x) / self.temperature)
        return self.norm(output)


class LinearPlusGated(nn.Module):
    """Linear base + gated correction."""

    def __init__(self, input_dim: int, output_dim: int, n_gates: int = 1,
                 temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.gated = GatedProjection(input_dim, output_dim, n_gates,
                                      temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.gated(x))


class DynamicLinear(nn.Module):
    """
    Dynamic linear projection: base + per-token low-rank correction.
    output = W(x) + shrinkage * (x @ W_down) @ W_up

    Each token gets a different effective projection matrix.
    This is what trees reduce to once you strip away routing infrastructure.
    Cost: 2 matmuls (vs 4+ for trees).
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8,
                 n_modulations: int = 1, temperature: float = 1.0,
                 use_norm: bool = True):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        self.base = nn.Linear(input_dim, output_dim)
        # Multiple modulation heads for more expressiveness
        self.n_modulations = n_modulations
        self.down_projs = nn.ModuleList([
            nn.Linear(input_dim, rank, bias=False) for _ in range(n_modulations)
        ])
        self.up_projs = nn.ParameterList([
            nn.Parameter(torch.randn(rank, output_dim) * 0.01) for _ in range(n_modulations)
        ])

        if n_modulations > 1:
            self.mod_gate = nn.Linear(input_dim, n_modulations)

        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        # For compatibility with tree utilities
        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.depth = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if self.n_modulations == 1:
            modulation = self.down_projs[0](x) @ self.up_projs[0]
        else:
            # Gated mixture of modulations
            gate = F.softmax(self.mod_gate(x) / self.temperature, dim=-1)  # (B, S, n_mod)
            modulation = torch.zeros_like(base_out)
            for i in range(self.n_modulations):
                mod_i = self.down_projs[i](x) @ self.up_projs[i]  # (B, S, output_dim)
                modulation = modulation + gate[..., i:i+1] * mod_i

        return self.norm(base_out + self.shrinkage * modulation)


class LowRankRoutingForest(nn.Module):
    """
    Oblivious forest with low-rank routing projection.
    Projects input to a small routing space (r << d_model) before computing
    tree decisions. Reduces routing einsum cost by d_model/r factor.

    routing: x @ W_down (B,S,d -> B,S,r), then einsum with (T, depth, r)
    leaves: full-rank (T, n_leaves, output_dim) as usual
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 12,
                 tree_depth: int = 3, routing_rank: int = 16,
                 temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.routing_rank = routing_rank
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Low-rank routing: project input to routing_rank dims first
        self.route_down = nn.Linear(input_dim, routing_rank, bias=False)
        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, routing_rank))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_outputs = nn.Parameter(torch.empty(n_trees, self.n_leaves, output_dim))

        # Input-dependent tree gating
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Per-depth learnable temperature factors
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        nn.init.normal_(self.decision_weights, 0, 0.1)
        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Low-rank routing: project input to small routing space
        x_route = self.route_down(x)  # (B, S, routing_rank)

        # Oblivious routing in low-rank space: (B, S, T, depth)
        decisions = torch.einsum('bsr,tnr->bstn', x_route, self.decision_weights)
        decisions = decisions + self.decision_biases

        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        # Leaf probs via outer product
        leaf_probs = _compute_oblivious_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs

        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)
        weights = F.softmax(self.gate_proj(x), dim=-1)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


class LinearPlusLowRankRouting(nn.Module):
    """Linear base + low-rank routing forest correction."""

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 24,
                 tree_depth: int = 3, routing_rank: int = 16,
                 temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = LowRankRoutingForest(input_dim, output_dim, n_trees,
                                            tree_depth, routing_rank, temperature,
                                            use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


class RecursiveProjection(nn.Module):
    """
    Recursive gated projection: apply gate * W_L(h) + (1-gate) * W_R(h)
    K times in sequence, reusing the same parameters.

    Gets depth-K nonlinearity with depth-1 parameter count.
    Each iteration is like one level of tree depth.
    """

    def __init__(self, input_dim: int, output_dim: int, n_iterations: int = 3,
                 temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_iterations = n_iterations
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Shared across iterations (parameter reuse)
        self.gate_proj = nn.Linear(input_dim, output_dim)
        self.W_left = nn.Linear(input_dim, output_dim, bias=False)
        self.W_right = nn.Linear(input_dim, output_dim, bias=False)

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        # For compatibility
        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.depth = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x) / self.temperature)
        left = self.W_left(x)
        right = self.W_right(x)

        h = gate * left + (1 - gate) * right
        for _ in range(self.n_iterations - 1):
            # Recompute gate based on evolving h but with original x dimensions
            # Use h as the gating signal for subsequent iterations
            gate = torch.sigmoid(self.gate_proj(x) / self.temperature * (1 + 0.1 * h.detach().norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)))
            h = gate * left + (1 - gate) * right

        return self.norm(h)


class LinearPlusRecursive(nn.Module):
    """Linear base + recursive gated correction."""

    def __init__(self, input_dim: int, output_dim: int, n_iterations: int = 3,
                 temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.recursive = RecursiveProjection(input_dim, output_dim, n_iterations,
                                              temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.recursive(x))


class ChunkedRoutingForest(nn.Module):
    """
    Oblivious forest with chunked/amortized routing.
    Groups tokens into chunks, routes based on chunk means,
    then broadcasts routing decisions to all tokens in the chunk.

    Reduces routing cost by chunk_size factor.
    Leaf outputs are still computed per-token (gating is per-token).
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 12,
                 tree_depth: int = 3, chunk_size: int = 16,
                 temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.chunk_size = chunk_size
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_outputs = nn.Parameter(torch.empty(n_trees, self.n_leaves, output_dim))

        self.gate_proj = nn.Linear(input_dim, n_trees)
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.hard_routing = False
        self.hard_routing_k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B, S, D = x.shape
        C = self.chunk_size

        # Pad sequence to multiple of chunk_size
        pad = (C - S % C) % C
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x
        S_padded = x_padded.shape[1]
        n_chunks = S_padded // C

        # Compute causal chunk means for routing (no future leakage)
        chunks = x_padded.reshape(B, n_chunks, C, D)
        cumsum = chunks.cumsum(dim=2)
        counts = torch.arange(1, C + 1, device=x.device, dtype=x.dtype).reshape(1, 1, C, 1)
        causal_means = cumsum / counts  # Each position only sees past within chunk
        chunk_context = causal_means.reshape(B, S_padded, D)

        # Route on per-token causal chunk context
        decisions = torch.einsum('bsd,tnd->bstn', chunk_context, self.decision_weights)
        decisions = decisions + self.decision_biases

        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)

        self._cached_decisions = decisions[:, :S]

        leaf_probs = _compute_oblivious_leaf_probs(decisions)

        if self.hard_routing:
            topk_vals, topk_idx = leaf_probs.topk(self.hard_routing_k, dim=-1)
            hard_probs = torch.zeros_like(leaf_probs)
            hard_probs.scatter_(-1, topk_idx, topk_vals)
            leaf_probs = hard_probs / (hard_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._cached_leaf_probs = leaf_probs[:, :S]

        # Per-token leaf outputs and gating (still on original x_padded)
        per_tree = torch.einsum('bstl,tlo->bsto', leaf_probs, self.leaf_outputs)
        weights = F.softmax(self.gate_proj(x_padded), dim=-1)
        output = torch.einsum('bsto,bst->bso', per_tree, weights)

        # Remove padding
        output = output[:, :S]

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


class LinearPlusChunkedRouting(nn.Module):
    """Linear base + chunked routing forest correction."""

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 24,
                 tree_depth: int = 3, chunk_size: int = 16,
                 temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.forest = ChunkedRoutingForest(input_dim, output_dim, n_trees,
                                            tree_depth, chunk_size, temperature,
                                            use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.forest(x))


class ProductKeyProjection(nn.Module):
    """
    Product-key memory projection: hash-based leaf selection.

    Splits input into two halves, each selects top-k from a codebook.
    Product of selections gives k² leaf candidates from C² possible leaves.
    O(sqrt(n_leaves)) routing cost instead of O(n_leaves).
    """

    def __init__(self, input_dim: int, output_dim: int, codebook_size: int = 16,
                 top_k: int = 4, temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.codebook_size = codebook_size
        self.top_k = top_k
        self.n_leaves = codebook_size * codebook_size  # C²
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        half_dim = input_dim // 2
        self.other_half = input_dim - half_dim  # handle odd dims

        # Two codebooks for the two halves
        self.codebook_1 = nn.Parameter(torch.randn(codebook_size, half_dim) * 0.1)
        self.codebook_2 = nn.Parameter(torch.randn(codebook_size, self.other_half) * 0.1)

        # Leaf outputs: (C², output_dim) — one output per product-key combination
        self.leaf_outputs = nn.Parameter(torch.empty(self.n_leaves, output_dim))
        nn.init.xavier_normal_(self.leaf_outputs)

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        # For compatibility
        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.depth = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B, S, D = x.shape
        half = D // 2

        # Split input into two halves
        x1 = x[..., :half]        # (B, S, half)
        x2 = x[..., half:]        # (B, S, other_half)

        # Compute scores against each codebook
        scores_1 = torch.matmul(x1, self.codebook_1.T) / self.temperature  # (B, S, C)
        scores_2 = torch.matmul(x2, self.codebook_2.T) / self.temperature  # (B, S, C)

        # Soft selection (softmax over codebook entries)
        weights_1 = F.softmax(scores_1, dim=-1)  # (B, S, C)
        weights_2 = F.softmax(scores_2, dim=-1)  # (B, S, C)

        # Product gives joint distribution over C² leaves
        # (B, S, C, 1) * (B, S, 1, C) -> (B, S, C, C) -> (B, S, C²)
        joint_weights = (weights_1.unsqueeze(-1) * weights_2.unsqueeze(-2)).reshape(B, S, -1)

        # Weighted sum of leaf outputs
        output = torch.matmul(joint_weights, self.leaf_outputs)  # (B, S, output_dim)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


class LinearPlusProductKey(nn.Module):
    """Linear base + product-key memory correction."""

    def __init__(self, input_dim: int, output_dim: int, codebook_size: int = 16,
                 top_k: int = 4, temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.pkm = ProductKeyProjection(input_dim, output_dim, codebook_size,
                                         top_k, temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.pkm(x))


# =============================================================================
# 2c. FLAT MoE PROJECTION (OPT-13 ablation: trees vs direct softmax gating)
# =============================================================================

class FlatMoEProjection(nn.Module):
    """
    Flat Mixture-of-Experts projection — replaces tree routing with direct
    softmax gating over K experts. Same function class as a forest with K leaves,
    but without hierarchical routing. Answers: does the tree structure help?

    output = sum_k(gate_k(x) * expert_k(x))  where expert_k is a constant vector
    Plus a linear base for the LinearPlusForest analog.
    """

    def __init__(self, input_dim: int, output_dim: int, n_experts: int = 8,
                 n_groups: int = 12, temperature: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.n_groups = n_groups  # analogous to n_trees

        # Expert outputs: constant vectors (like tree leaves)
        self.expert_outputs = nn.Parameter(torch.empty(n_groups, n_experts, output_dim))

        # Gating: input-dependent softmax over experts (replaces tree routing)
        self.gate = nn.Linear(input_dim, n_groups * n_experts)

        # Group weighting (like tree gating)
        self.group_gate = nn.Linear(input_dim, n_groups)

        self.norm = nn.LayerNorm(output_dim) if use_norm else nn.Identity()

        _init_forest_params(self.gate.weight, self.expert_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None
        self.depth = 0  # for compatibility with tree utilities
        self.register_buffer('temperature', torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B, S, D = x.shape
        G, K = self.n_groups, self.n_experts

        # Expert gating: (B, S, G*K) → (B, S, G, K) → softmax per group
        gate_logits = self.gate(x).view(B, S, G, K)
        expert_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        self._cached_leaf_probs = expert_weights  # for leaf balancing loss

        # Per-group output: weighted sum of expert outputs
        per_group = torch.einsum('bsgk,gko->bsgo', expert_weights, self.expert_outputs)

        # Group weighting
        group_weights = F.softmax(self.group_gate(x), dim=-1)  # (B, S, G)
        output = torch.einsum('bsgo,bsg->bso', per_group, group_weights)

        if squeeze:
            output = output.squeeze(1)
        return self.norm(output)


class LinearPlusMoE(nn.Module):
    """Linear base + flat MoE correction (analog of LinearPlusForest for OPT-13)."""

    def __init__(self, input_dim: int, output_dim: int, n_experts: int = 8,
                 n_groups: int = 12, temperature: float = 1.0):
        super().__init__()
        self.base_proj = nn.Linear(input_dim, output_dim)
        self.moe = FlatMoEProjection(input_dim, output_dim, n_experts, n_groups,
                                     temperature, use_norm=False)
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base_proj(x) + self.shrinkage * self.moe(x))


# =============================================================================
# 2d. SHARED ROUTING FORESTS (OPT-04: route once, apply to multiple outputs)
# =============================================================================

class SharedRoutingBatchedForest(nn.Module):
    """
    Batched forest with shared routing for multiple output heads (e.g. Q, K, V).
    Routing (the expensive einsum) runs once; each head has its own leaf outputs.
    """

    def __init__(self, input_dim: int, output_dim: int, n_heads_out: int = 3,
                 n_trees: int = 12, tree_depth: int = 3, temperature: float = 1.0):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_internal = 2 ** tree_depth - 1
        self.n_leaves = 2 ** tree_depth
        self.n_heads_out = n_heads_out
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Single set of routing parameters (shared across all output heads)
        self.decision_weights = nn.Parameter(torch.empty(n_trees, self.n_internal, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, self.n_internal))
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, self.n_internal))
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Separate leaf outputs per head: (n_heads_out, n_trees, n_leaves, output_dim)
        self.leaf_outputs = nn.Parameter(
            torch.empty(n_heads_out, n_trees, self.n_leaves, output_dim))

        path_indices, path_dirs = _build_path_indices(self.n_leaves, tree_depth)
        self.register_buffer('path_indices', path_indices)
        self.register_buffer('path_dirs', path_dirs)

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None

    def forward(self, x: torch.Tensor) -> list:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Routing: computed ONCE for all output heads
        decisions = torch.einsum('bsd,tnd->bstn', x, self.decision_weights)
        decisions = decisions + self.decision_biases
        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        # Leaf probabilities
        relevant = decisions[..., self.path_indices]
        probs_per_level = self.path_dirs * relevant + (1 - self.path_dirs) * (1 - relevant)
        leaf_probs = probs_per_level.prod(dim=-1)  # (B, S, T, n_leaves)
        self._cached_leaf_probs = leaf_probs

        # Gate weights (shared across heads)
        weights = F.softmax(self.gate_proj(x), dim=-1)  # (B, S, n_trees)

        # Apply to all heads at once: leaf_outputs is (H, T, L, D)
        # per_tree: (H, B, S, T, D)
        per_tree = torch.einsum('bstl,htld->hbstd', leaf_probs, self.leaf_outputs)
        outputs = torch.einsum('hbstd,bst->hbsd', per_tree, weights)

        results = []
        for i in range(self.n_heads_out):
            out = outputs[i]
            if squeeze:
                out = out.squeeze(1)
            results.append(out)
        return results


class SharedRoutingObliviousForest(nn.Module):
    """
    Oblivious forest with shared routing for multiple output heads.
    Same shared-routing principle as SharedRoutingBatchedForest but with
    oblivious trees (one hyperplane per depth level).
    """

    def __init__(self, input_dim: int, output_dim: int, n_heads_out: int = 3,
                 n_trees: int = 12, tree_depth: int = 3, temperature: float = 1.0):
        super().__init__()
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaves = 2 ** tree_depth
        self.n_heads_out = n_heads_out
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        # Shared routing: one hyperplane per depth level
        self.decision_weights = nn.Parameter(torch.empty(n_trees, tree_depth, input_dim))
        self.decision_biases = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.node_temperature_logits = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.gate_proj = nn.Linear(input_dim, n_trees)

        # Separate leaf outputs per head
        self.leaf_outputs = nn.Parameter(
            torch.empty(n_heads_out, n_trees, self.n_leaves, output_dim))

        _init_forest_params(self.decision_weights, self.leaf_outputs)

        self._cached_decisions = None
        self._cached_leaf_probs = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        decisions = torch.einsum('bsd,tnd->bstn', x, self.decision_weights)
        decisions = decisions + self.decision_biases
        node_temps = self.temperature * F.softplus(self.node_temperature_logits + 0.5413)
        decisions = torch.sigmoid(decisions / node_temps)
        self._cached_decisions = decisions

        leaf_probs = _compute_oblivious_leaf_probs(decisions)
        self._cached_leaf_probs = leaf_probs

        # Gate weights (shared)
        weights = F.softmax(self.gate_proj(x), dim=-1)

        # All heads at once
        per_tree = torch.einsum('bstl,htld->hbstd', leaf_probs, self.leaf_outputs)
        outputs = torch.einsum('hbstd,bst->hbsd', per_tree, weights)

        results = []
        for i in range(self.n_heads_out):
            out = outputs[i]
            if squeeze:
                out = out.squeeze(1)
            results.append(out)
        return results


# =============================================================================
# 3. PROJECTION FACTORY
# =============================================================================

def make_projection(input_dim: int, output_dim: int, proj_type: str = "batched",
                    n_trees: int = 12, tree_depth: int = 3, temperature: float = 1.0,
                    boosted_trees: int = 24, boosted_depth: int = 3,
                    leaf_rank: int = 8, ema_decay: float = 0.9,
                    **kwargs) -> nn.Module:
    """Create a projection layer: 'linear', 'batched', 'boosted', 'oblivious',
    'oblivious_boosted', 'micro_tree', 'micro_boosted', 'contextual', or 'contextual_boosted'."""
    if proj_type == "linear":
        return nn.Linear(input_dim, output_dim)
    elif proj_type == "batched":
        return BatchedTreeForest(input_dim, output_dim, n_trees, tree_depth,
                                 temperature, use_norm=False)
    elif proj_type == "boosted":
        return LinearPlusForest(input_dim, output_dim, boosted_trees,
                                boosted_depth, temperature)
    elif proj_type == "oblivious":
        return ObliviousTreeForest(input_dim, output_dim, n_trees, tree_depth,
                                   temperature, use_norm=False)
    elif proj_type == "oblivious_boosted":
        return ObliviousLinearPlusForest(input_dim, output_dim, boosted_trees,
                                         boosted_depth, temperature)
    elif proj_type == "micro_tree":
        return MicroTreeForest(input_dim, output_dim, n_trees, tree_depth,
                               leaf_rank, temperature, use_norm=False)
    elif proj_type == "micro_boosted":
        return LinearPlusMicroTree(input_dim, output_dim, n_trees, tree_depth,
                                    leaf_rank, temperature)
    elif proj_type == "contextual":
        return ContextualRoutingForest(input_dim, output_dim, n_trees, tree_depth,
                                        temperature, ema_decay=ema_decay,
                                        use_norm=False)
    elif proj_type == "contextual_boosted":
        return LinearPlusContextual(input_dim, output_dim, boosted_trees,
                                     boosted_depth, temperature,
                                     ema_decay=ema_decay)
    elif proj_type == "gated":
        return GatedProjection(input_dim, output_dim, n_gates=kwargs.get('n_gates', 1),
                               temperature=temperature, use_norm=False)
    elif proj_type == "gated_boosted":
        return LinearPlusGated(input_dim, output_dim, n_gates=kwargs.get('n_gates', 1),
                               temperature=temperature)
    elif proj_type == "dynamic":
        return DynamicLinear(input_dim, output_dim, rank=kwargs.get('leaf_rank', 8),
                             n_modulations=kwargs.get('n_modulations', 1),
                             temperature=temperature, use_norm=False)
    elif proj_type == "dynamic_boosted":
        return DynamicLinear(input_dim, output_dim, rank=kwargs.get('leaf_rank', 8),
                             n_modulations=kwargs.get('n_modulations', 4),
                             temperature=temperature)
    elif proj_type == "lowrank_routing":
        return LowRankRoutingForest(input_dim, output_dim, n_trees, tree_depth,
                                     routing_rank=kwargs.get('routing_rank', 16),
                                     temperature=temperature, use_norm=False)
    elif proj_type == "lowrank_boosted":
        return LinearPlusLowRankRouting(input_dim, output_dim, boosted_trees,
                                         boosted_depth,
                                         routing_rank=kwargs.get('routing_rank', 16),
                                         temperature=temperature)
    elif proj_type == "recursive":
        return RecursiveProjection(input_dim, output_dim,
                                    n_iterations=kwargs.get('n_iterations', 3),
                                    temperature=temperature, use_norm=False)
    elif proj_type == "recursive_boosted":
        return LinearPlusRecursive(input_dim, output_dim,
                                    n_iterations=kwargs.get('n_iterations', 3),
                                    temperature=temperature)
    elif proj_type == "chunked":
        return ChunkedRoutingForest(input_dim, output_dim, n_trees, tree_depth,
                                     chunk_size=kwargs.get('chunk_size', 16),
                                     temperature=temperature, use_norm=False)
    elif proj_type == "chunked_boosted":
        return LinearPlusChunkedRouting(input_dim, output_dim, boosted_trees,
                                         boosted_depth,
                                         chunk_size=kwargs.get('chunk_size', 16),
                                         temperature=temperature)
    elif proj_type == "product_key":
        return ProductKeyProjection(input_dim, output_dim,
                                     codebook_size=kwargs.get('codebook_size', 16),
                                     top_k=kwargs.get('top_k', 4),
                                     temperature=temperature, use_norm=False)
    elif proj_type == "product_key_boosted":
        return LinearPlusProductKey(input_dim, output_dim,
                                     codebook_size=kwargs.get('codebook_size', 16),
                                     top_k=kwargs.get('top_k', 4),
                                     temperature=temperature)
    elif proj_type == "moe":
        return FlatMoEProjection(input_dim, output_dim, n_experts=8,
                                 n_groups=n_trees, temperature=temperature,
                                 use_norm=False)
    elif proj_type == "moe_boosted":
        return LinearPlusMoE(input_dim, output_dim, n_experts=8,
                             n_groups=n_trees, temperature=temperature)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


# =============================================================================
# 4. TREE ATTENTION (unfused Q/K/V, QK-norm)
# =============================================================================

class TreeAttention(nn.Module):
    """
    Multi-head attention with tree-based projections.

    Supports two modes:
    - Independent routing (default): separate forests for Q, K, V, O
    - Shared routing (OPT-04): one shared routing for Q, K, V with separate
      leaf outputs per projection. O keeps its own routing (different input).

    tree_targets controls which projections use trees vs nn.Linear (OPT-05):
      "qkvo" (default) — all four use trees
      "vo"   — only V and O use trees; Q and K use nn.Linear
      "o"    — only O uses trees
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 proj_type: str = "batched", tree_targets: str = "qkvo",
                 shared_routing: bool = False, **proj_kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        assert set(tree_targets) <= set("qkvo"), f"Invalid tree_targets: {tree_targets!r}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.shared_routing = shared_routing
        self._boosted_shared = False

        if shared_routing and proj_type not in ("linear",):
            # OPT-04: Shared routing for Q, K, V — one routing, 3 leaf output sets
            # Count how many of Q, K, V use trees
            qkv_tree_targets = [c for c in "qkv" if c in tree_targets]
            n_shared = len(qkv_tree_targets)
            self._qkv_tree_targets = qkv_tree_targets

            if n_shared > 0:
                # Choose shared forest type based on proj_type
                forest_kwargs = {k: v for k, v in proj_kwargs.items()
                                 if k in ('n_trees', 'tree_depth', 'temperature',
                                          'boosted_trees', 'boosted_depth')}
                if proj_type in ("oblivious", "oblivious_boosted"):
                    n_trees = forest_kwargs.get('boosted_trees' if 'boosted' in proj_type else 'n_trees', 12)
                    depth = forest_kwargs.get('boosted_depth' if 'boosted' in proj_type else 'tree_depth', 3)
                    self.qkv_forest = SharedRoutingObliviousForest(
                        d_model, d_model, n_heads_out=n_shared,
                        n_trees=n_trees, tree_depth=depth)
                else:
                    n_trees = forest_kwargs.get('boosted_trees' if 'boosted' in proj_type else 'n_trees', 12)
                    depth = forest_kwargs.get('boosted_depth' if 'boosted' in proj_type else 'tree_depth', 3)
                    self.qkv_forest = SharedRoutingBatchedForest(
                        d_model, d_model, n_heads_out=n_shared,
                        n_trees=n_trees, tree_depth=depth)

                # For boosted variants: separate linear bases + shrinkage
                self._boosted_shared = 'boosted' in proj_type
                if self._boosted_shared:
                    self.qkv_bases = nn.ModuleList([
                        nn.Linear(d_model, d_model) for _ in range(n_shared)])
                    self.qkv_shrinkage = nn.Parameter(torch.tensor(0.1))
                self.qkv_norms = nn.ModuleList([
                    nn.LayerNorm(d_model) for _ in range(n_shared)])

            # Non-tree QKV projections
            self._qkv_linear = nn.ModuleDict()
            for c in "qkv":
                if c not in tree_targets:
                    self._qkv_linear[c] = nn.Linear(d_model, d_model)

            # O projection is always separate (different input)
            if 'o' in tree_targets:
                self.o_proj = make_projection(d_model, d_model, proj_type, **proj_kwargs)
            else:
                self.o_proj = nn.Linear(d_model, d_model)
        else:
            # Original behavior: independent routing per projection
            def _proj(target_char):
                if target_char in tree_targets:
                    return make_projection(d_model, d_model, proj_type, **proj_kwargs)
                return nn.Linear(d_model, d_model)

            self.q_proj = _proj('q')
            self.k_proj = _proj('k')
            self.v_proj = _proj('v')
            self.o_proj = _proj('o')

        # QK-norm for stable attention with tree projections
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)

        self.dropout = nn.Dropout(dropout)

    def _get_qkv_shared(self, x):
        """Shared routing path: one forest forward for Q, K, V."""
        forest_outputs = self.qkv_forest(x)  # list of n_shared tensors

        # Map forest outputs to Q, K, V based on tree_targets
        proj_map = {}
        for i, c in enumerate(self._qkv_tree_targets):
            out = forest_outputs[i]
            if self._boosted_shared:
                out = self.qkv_norms[i](self.qkv_bases[i](x) + self.qkv_shrinkage * out)
            else:
                out = self.qkv_norms[i](out)
            proj_map[c] = out

        # Fill in linear projections for non-tree targets
        for c in "qkv":
            if c not in proj_map:
                proj_map[c] = self._qkv_linear[c](x)

        return proj_map['q'], proj_map['k'], proj_map['v']

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        if self.shared_routing:
            Q, K, V = self._get_qkv_shared(x)
        else:
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # QK-norm: stabilize attention logits despite tree routing shifts
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.o_proj(context)
        return output, attn_weights


# =============================================================================
# 5. TREE TRANSFORMER BLOCK
# =============================================================================

class TreeTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ff_dim: int = None,
                 dropout: float = 0.1, use_tree_ffn: bool = True,
                 proj_type: str = "batched", **proj_kwargs):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.attention = TreeAttention(d_model, n_heads, dropout, proj_type, **proj_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        if use_tree_ffn:
            self.ffn = nn.Sequential(
                make_projection(d_model, ff_dim, proj_type, **proj_kwargs),
                nn.GELU(),
                nn.Dropout(dropout),
                make_projection(ff_dim, d_model, proj_type, **proj_kwargs),
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
# 6. FULL TREE TRANSFORMER
# =============================================================================

class TreeTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_seq_len: int = 512, num_classes: int = 2,
                 dropout: float = 0.1, use_tree_ffn: bool = True,
                 task: str = "classification", proj_type: str = "batched",
                 tree_every_n: int = 1,
                 **proj_kwargs):
        super().__init__()
        self.task = task
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # tree_every_n: use trees every N layers, linear otherwise (OPT-11).
        # tree_every_n=1 (default): all layers use trees
        # tree_every_n=2: alternating (odd layers=tree, even=linear)
        layers = []
        for i in range(n_layers):
            if tree_every_n > 1 and i % tree_every_n != (tree_every_n - 1):
                layer_proj = "linear"
                layer_tree_ffn = False
            else:
                layer_proj = proj_type
                layer_tree_ffn = use_tree_ffn
            layers.append(TreeTransformerBlock(
                d_model, n_heads, dropout=dropout,
                use_tree_ffn=layer_tree_ffn, proj_type=layer_proj,
                **proj_kwargs))
        self.layers = nn.ModuleList(layers)

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
            x = x.mean(dim=1)
        return self.head(x), all_attn


# =============================================================================
# 7. UTILITIES
# =============================================================================

_TREE_MODULES = (BatchedTreeForest, ObliviousTreeForest, FlatMoEProjection,
                 SharedRoutingBatchedForest, SharedRoutingObliviousForest,
                 MicroTreeForest, ContextualRoutingForest,
                 GatedProjection, DynamicLinear, LowRankRoutingForest,
                 RecursiveProjection, ChunkedRoutingForest, ProductKeyProjection)

# Oblivious-style modules have (B,S,T,depth) decisions; batched have (B,S,T,n_internal)
_OBLIVIOUS_MODULES = (ObliviousTreeForest, SharedRoutingObliviousForest,
                      MicroTreeForest, ContextualRoutingForest,
                      LowRankRoutingForest, ChunkedRoutingForest)
_BATCHED_MODULES = (BatchedTreeForest, SharedRoutingBatchedForest)


def tree_regularization_loss(model, lambda_reg=0.01):
    """Entropy regularization on routing decisions (vectorized, no Python loop)."""
    if lambda_reg == 0.0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    eps = 1e-7
    for module in model.modules():
        if isinstance(module, _TREE_MODULES) and module._cached_decisions is not None:
            p = module._cached_decisions
            depth = module.depth
            if depth == 0:
                continue  # FlatMoEProjection has depth=0, no tree routing to regularize
            ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

            if isinstance(module, _OBLIVIOUS_MODULES):
                # Oblivious: decisions (B,S,T,depth) — one weight per depth level
                depth_weights = (0.5 ** torch.arange(depth, device=p.device))
                reg = reg + (ent.mean(dim=(0, 1, 2)) * depth_weights).sum()
            elif isinstance(module, _BATCHED_MODULES):
                # Batched: decisions (B,S,T,n_internal) — build per-node depth weights
                n_internal = 2 ** depth - 1
                node_depth_weights = torch.empty(n_internal, device=p.device)
                for d in range(depth):
                    start = 2 ** d - 1
                    node_depth_weights[start:start + 2 ** d] = 0.5 ** d
                reg = reg + (ent.mean(dim=(0, 1, 2)) * node_depth_weights).sum()
            count += 1
    if count > 0:
        reg = reg / count
    return lambda_reg * reg


def leaf_balancing_loss(model, alpha=0.01):
    """
    MoE-style load-balancing loss on leaf utilization.
    L = alpha * n_leaves * mean_over_trees(sum_l(mean_prob_l^2))

    Uniform leaves: loss = alpha * 1.0. Collapsed to 1 leaf: loss = alpha * n_leaves.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for module in model.modules():
        if isinstance(module, _TREE_MODULES) and module._cached_leaf_probs is not None:
            probs = module._cached_leaf_probs  # (B, S, T, n_leaves)
            n_leaves = probs.shape[-1]
            mean_prob = probs.mean(dim=(0, 1))  # (T, n_leaves)
            balance = n_leaves * (mean_prob ** 2).sum(dim=-1).mean()
            loss = loss + balance
            count += 1
    if count > 0:
        loss = loss / count
    return alpha * loss


def set_temperature(model, temperature):
    """Set base temperature for all tree forest modules."""
    for module in model.modules():
        if isinstance(module, _TREE_MODULES):
            module.temperature.fill_(temperature)


def get_routing_entropy(model):
    """Return mean routing entropy across all forests."""
    eps = 1e-7
    entropies = []
    for module in model.modules():
        if isinstance(module, _TREE_MODULES) and module._cached_decisions is not None:
            p = module._cached_decisions
            entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            entropies.append(entropy.mean().item())
    return sum(entropies) / len(entropies) if entropies else 0.0


def make_optimizer(model, lr=3e-4, weight_decay=0.01):
    """
    Tree-aware optimizer with parameter groups:
    - Decision weights + node temperatures: higher LR, no weight decay
    - Gate projections: standard LR
    - Leaf outputs: standard LR, standard weight decay
    - Other params: standard LR, standard weight decay
    """
    # Routing params (3x LR, no weight decay): decision weights, biases, temperatures,
    # and MoE gate weights (analogous to routing)
    _ROUTING_KEYS = ('decision_weights', 'decision_biases', 'node_temperature',
                     'moe.gate.weight', 'moe.gate.bias',
                     'route_down', 'codebook_1', 'codebook_2')
    # Leaf/expert output params (standard LR, standard weight decay)
    _LEAF_KEYS = ('leaf_outputs', 'leaf_down', 'leaf_up', 'expert_outputs', 'shrinkage',
                  'up_projs', 'W_left', 'W_right')
    # Gate params (standard LR)
    _GATE_KEYS = ('gate_proj', 'group_gate', 'context_proj')

    decision_params, leaf_params, gate_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if any(k in name for k in _ROUTING_KEYS):
            decision_params.append(param)
        elif any(k in name for k in _LEAF_KEYS):
            leaf_params.append(param)
        elif any(k in name for k in _GATE_KEYS):
            gate_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW([
        {'params': decision_params, 'lr': lr * 3, 'weight_decay': 0.0},
        {'params': leaf_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': gate_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
    ])


def count_parameters(model):
    _TREE_KEYS = ('decision', 'leaf', 'leaf_down', 'leaf_up', 'tree_weights',
                  'shrinkage', 'gate_proj', 'node_temperature', 'expert_outputs',
                  'group_gate', 'context_proj', 'ema_decay',
                  'qkv_forest', 'qkv_bases', 'qkv_shrinkage', 'qkv_norms',
                  'route_down', 'codebook_1', 'codebook_2', 'down_projs',
                  'up_projs', 'mod_gate', 'W_left', 'W_right')
    total = sum(p.numel() for p in model.parameters())
    tree_p = sum(p.numel() for n, p in model.named_parameters()
                 if any(k in n for k in _TREE_KEYS))
    return {'total': total, 'tree': tree_p, 'tree_pct': tree_p / total * 100 if total > 0 else 0}


# =============================================================================
# 7b. ADAPTER UTILITIES
# =============================================================================

def freeze_non_tree_params(model):
    """Freeze all parameters except tree-specific ones. For adapter fine-tuning."""
    _TREE_PARAM_KEYS = ('decision', 'leaf', 'shrinkage', 'gate_proj', 'node_temperature')
    for name, param in model.named_parameters():
        if any(k in name for k in _TREE_PARAM_KEYS):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all_params(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def set_hard_routing(model, enabled=True, top_k=2):
    """Enable/disable hard routing on all tree modules."""
    for module in model.modules():
        if isinstance(module, _TREE_MODULES) and hasattr(module, 'hard_routing'):
            module.hard_routing = enabled
            module.hard_routing_k = top_k


# =============================================================================
# 8. DEMO
# =============================================================================

if __name__ == "__main__":
    import time

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TreeTransformer(
        vocab_size=1000, d_model=64, n_layers=2, n_heads=4,
        max_seq_len=32, num_classes=2, use_tree_ffn=False,
        task="classification", proj_type="batched",
        n_trees=12, tree_depth=3,
    ).to(DEVICE)

    params = count_parameters(model)
    print(f"Batched Forest — Params: {params['total']:,} (tree: {params['tree_pct']:.1f}%)")

    optimizer = make_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    N_STEPS = 50
    model.train()
    for step in range(1, N_STEPS + 1):
        temp = 1.0 - 0.9 * (step / N_STEPS)
        set_temperature(model, temp)

        tokens = torch.randint(0, 1000, (16, 32), device=DEVICE)
        labels = (tokens % 2 == 0).float().sum(1).gt(16).long()

        logits, _ = model(tokens)
        loss = (criterion(logits, labels)
                + tree_regularization_loss(model, 0.005)
                + leaf_balancing_loss(model, 0.01))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            acc = (logits.argmax(-1) == labels).float().mean()
            entropy = get_routing_entropy(model)
            print(f"Step {step:3d} | Loss: {loss:.4f} | Acc: {acc:.3f} | "
                  f"Entropy: {entropy:.4f} | Temp: {temp:.2f}")

    print("\nAll gradients flow through batched tree forests!")
