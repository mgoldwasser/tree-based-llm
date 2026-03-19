# TASK-18: Language-Aware Tree Structure Design

- **Category:** architecture-research
- **Priority:** 9.5/10 ⭐ CRITICAL
- **Impact:** 10/10
- **Feasibility:** 6/10
- **Confidence:** 6/10

## Central Question

**Why might trees be ineffective for language modeling, and how can we fix it?**

## Problem Analysis

### **Hypothesis 1: Trees Are Too Rigid**

**Language is compositional and hierarchical, but in a different way than trees.**

**Tree limitations:**
1. **Binary splits:** Language isn't binary (syntax/semantics are multi-way)
2. **Fixed depth:** All inputs traverse same depth (no early exit for simple tokens)
3. **Independent routing:** Each token routes separately (no context awareness)
4. **Deterministic paths:** One path per token (language has ambiguity)

**Evidence:**
- Common words ("the", "and") waste tree capacity
- Rare words might not have learned paths
- Syntactic structure (subject-verb-object) not captured

### **Hypothesis 2: Trees Route on Wrong Features**

**Current routing:** `decision = sigmoid(w @ token_embedding)`

**Problem:** Routing on token embedding alone ignores:
- Position in sequence (is this the first word or last?)
- Attention context (what tokens surround this one?)
- Syntactic role (is "run" a noun or verb?)
- Long-range dependencies (anaphora resolution)

**Better routing inputs:**
- Contextualized representation (after first attention layer)
- Position encoding + token embedding
- Attention-weighted context vector

### **Hypothesis 3: Leaf Outputs Are Too Simple**

**Current:** Each leaf has fixed output vector

**Problem:** No way to:
- Combine information from multiple tokens
- Adapt output based on context
- Implement attention-like gating

**Example failure mode:**
- Token "bank" routes to Leaf 3
- But "river bank" vs "money bank" should produce different outputs
- Fixed leaf can't distinguish

## Proposed Solutions

### **Solution 1: Multi-Scale Trees (Per-Layer Specialization)**

**Idea:** Different tree depths for different layers

```python
# Layer 1 (early): Shallow trees (depth=2)
# - Fast routing, coarse features
# - Learn basic token properties (frequent vs rare, punctuation vs word)

# Layer 2-3 (middle): Deep trees (depth=4)
# - Rich routing, semantic features
# - Learn content transformations (nouns vs verbs, entities vs actions)

# Layer 4 (late): Shallow trees (depth=2) or no trees
# - Simple linear, preserve residual stream structure
# - Let attention handle final composition
```

**Implementation:**
```python
# train.py - MODEL_CONFIGS
"multi_scale_trees": {
    "description": "Multi-scale trees (depth varies by layer)",
    "proj_type": "oblivious_boosted",
    "use_tree_ffn": False,
    "layer_depths": [2, 4, 4, 2],  # per-layer tree depths
    "boosted_trees": 24,
}
```

### **Solution 2: Context-Aware Routing**

**Idea:** Route based on contextualized representation, not just token embedding

```python
class ContextAwareTreeForest(nn.Module):
    def __init__(self, d_model, n_trees, depth):
        super().__init__()
        # NEW: Learn context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
        )
        # Route on compressed context
        self.decision_weights = nn.Parameter(...)  # (n_trees, n_internal, d_model // 4)
    
    def forward(self, x, context_vector=None):
        # context_vector from previous layer's attention
        if context_vector is not None:
            routing_input = self.context_encoder(context_vector)
        else:
            routing_input = self.context_encoder(x)
        
        decisions = sigmoid(routing_weights @ routing_input / temp)
        # ... rest of tree forward pass
```

**Benefit:** Routing adapts to context, not just token identity

### **Solution 3: Soft Leaves (Mixture-of-Leaves)**

**Current:** Hard routing → one leaf output per token

**Proposed:** Soft routing → weighted mixture of leaf outputs

```python
# Instead of:
leaf_probs = [..., n_leaves]  # probabilities sum to 1
output = einsum('bsl,ld->bsd', leaf_probs, leaf_outputs)

# Do:
top_k_leaves, top_k_probs = leaf_probs.topk(k=4, dim=-1)
# Only use top-K leaves (sparse mixture)
output = weighted_sum(top_k_leaves, top_k_probs)
```

**Benefit:** Allows multiple interpretations (e.g., "bank" uses 2-3 leaves)

### **Solution 4: Hierarchical Position-Aware Trees**

**Idea:** Trees should know token position in sequence

```python
class PositionAwareForest(nn.Module):
    def __init__(self, d_model, n_trees, depth, max_seq_len):
        super().__init__()
        # Learn position-dependent routing
        self.pos_embed = nn.Embedding(max_seq_len, d_model // 8)
        self.decision_weights = nn.Parameter(...)  # route on [token; position]
    
    def forward(self, x, positions):
        pos_enc = self.pos_embed(positions)  # (B, S, d_model // 8)
        routing_input = torch.cat([x, pos_enc], dim=-1)
        # Route on concatenation of token + position
        decisions = sigmoid(routing_weights @ routing_input / temp)
        # ...
```

**Benefit:** Early tokens can route differently than late tokens

### **Solution 5: Task-Specific Tree Heads**

**Observation:** Multi-head attention works because heads specialize.

**Idea:** Multi-tree routing where each tree head specializes

```python
class SpecializedTreeHeads(nn.Module):
    """Like multi-head attention, but with trees."""
    def __init__(self, d_model, n_heads=4, trees_per_head=3):
        super().__init__()
        self.heads = nn.ModuleList([
            ObliviousTreeForest(d_model // n_heads, trees_per_head, depth=3)
            for _ in range(n_heads)
        ])
        # Encourage specialization with orthogonality loss
    
    def forward(self, x):
        head_outputs = []
        for i, head in enumerate(self.heads):
            x_head = x[..., i * (d_model // n_heads):(i+1) * (d_model // n_heads)]
            head_outputs.append(head(x_head))
        return torch.cat(head_outputs, dim=-1)
```

**Benefit:** Some heads learn syntax, others semantics, others position patterns

## Critical Experiment: Attention-Guided Routing

**The nuclear option:** Use attention scores to guide tree routing

```python
class AttentionGuidedForest(nn.Module):
    def __init__(self, d_model, n_trees, depth):
        super().__init__()
        self.forest = ObliviousTreeForest(d_model, n_trees, depth)
        # Learn how to use attention for routing
        self.attn_to_routing = nn.Linear(d_model, n_trees * depth)
    
    def forward(self, x, attention_scores):
        # attention_scores: (B, H, S, S) from previous layer
        # Average over heads, get context vector
        context = attention_scores.mean(dim=1) @ x  # (B, S, D)
        
        # Use context to bias routing decisions
        routing_bias = self.attn_to_routing(context)  # (B, S, n_trees * depth)
        
        # Add bias to standard routing
        base_decisions = self.forest._compute_decisions(x)
        biased_decisions = base_decisions + routing_bias.reshape(...).unsqueeze(2)
        # ... rest of forward pass
```

**Why this might work:**
- Attention finds relevant tokens
- Trees route based on what we're attending to
- Combines strength of both architectures

## Implementation Priority

**Phase 1: Quick wins (1-2 days)**
1. Multi-scale trees (TASK-18.1) - different depths per layer
2. Position-aware routing (TASK-18.4) - concat position to input
3. Top-K soft leaves (TASK-18.3) - sparse mixture

**Phase 2: Research (1 week)**
4. Context-aware routing (TASK-18.2) - use previous layer's output
5. Specialized tree heads (TASK-18.5) - multi-head but with trees
6. Attention-guided routing (TASK-18.6) - incorporate attention scores

## Expected Impact

**If Phase 1 works:**
- +1-2pp accuracy from better specialization
- Validates that tree structure was the issue

**If Phase 2 works:**
- Trees match or beat standard transformers
- Opens path to scaled-up experiments

**If nothing works:**
- Strong evidence trees aren't suited for language
- Pivot to other architectures (State Space Models, Linear Attention, etc.)

## Why This Is Critical

**This is the make-or-break task.** If we can't find a tree structure that works for language, the entire project should pivot.

**Current results suggest:** Trees are close (36-38% vs 38.7%), so small architectural changes might close the gap.

## Comparison to Existing Work

**Similar ideas in literature:**
- **Neural Decision Trees (2020):** Context-dependent routing
- **Routing Networks (2017):** Task-specific routing modules
- **Adaptive Computation Time (2016):** Variable depth per input
- **Universal Transformers:** Depth-wise recurrence + halting

**Our contribution:** Applying these ideas specifically to tree-based transformers

## Files to Create

```
main.py - Add:
  - ContextAwareTreeForest
  - PositionAwareForest
  - SpecializedTreeHeads
  - AttentionGuidedForest

train.py - Add configs:
  - multi_scale_trees
  - context_aware_trees
  - attention_guided_trees

tasks/completed/ - Move here when working
```

## Success Metrics

- Val accuracy > 39% (beat standard baseline)
- Routing analysis shows semantic clustering
- Gradient flow analysis shows all layers learning
- Ablation studies confirm each component helps

## Risk Assessment

**High risk, high reward.**

- **Risk:** None of these work → trees fundamentally limited
- **Mitigation:** Try all 6 variants before concluding
- **Fallback:** Document learnings, pivot to other architectures
