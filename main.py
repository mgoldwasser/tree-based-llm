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
                    **kwargs) -> nn.Module:
    """Create a projection layer: 'linear', 'batched', 'boosted', 'oblivious', or 'oblivious_boosted'."""
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
                 SharedRoutingBatchedForest, SharedRoutingObliviousForest)

# Oblivious-style modules have (B,S,T,depth) decisions; batched have (B,S,T,n_internal)
_OBLIVIOUS_MODULES = (ObliviousTreeForest, SharedRoutingObliviousForest)
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
                     'moe.gate.weight', 'moe.gate.bias')
    # Leaf/expert output params (standard LR, standard weight decay)
    _LEAF_KEYS = ('leaf_outputs', 'expert_outputs', 'shrinkage')
    # Gate params (standard LR)
    _GATE_KEYS = ('gate_proj', 'group_gate')

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
    _TREE_KEYS = ('decision', 'leaf', 'tree_weights', 'shrinkage', 'gate_proj',
                  'node_temperature', 'expert_outputs', 'group_gate',
                  'qkv_forest', 'qkv_bases', 'qkv_shrinkage', 'qkv_norms')
    total = sum(p.numel() for p in model.parameters())
    tree_p = sum(p.numel() for n, p in model.named_parameters()
                 if any(k in n for k in _TREE_KEYS))
    return {'total': total, 'tree': tree_p, 'tree_pct': tree_p / total * 100 if total > 0 else 0}


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
