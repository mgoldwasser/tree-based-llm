# TASK-16: Tokenization Strategy Analysis

- **Category:** data-representation
- **Priority:** 8.0/10
- **Impact:** 8/10
- **Feasibility:** 7/10
- **Confidence:** 7/10

## Problem Statement

**Current limitation:** Character-level tokenization (vocab=65) may be hurting tree performance.

**Why tokenization matters for trees:**

1. **Sequence length:** Char-level = 4-10x longer sequences
   - More tokens to route through trees
   - Harder to capture long-range dependencies
   - More opportunities for routing errors to compound

2. **Input distribution:** Characters have different statistics than subwords
   - 'e', 't', 'a' dominate (Zipfian distribution)
   - Trees might waste capacity on common characters
   - Harder to learn semantic splits

3. **Semantic granularity:** Characters have no inherent meaning
   - Tree routing decisions on 'e' vs 't' are arbitrary
   - BPE tokens like "ing", "the", "tion" have structure
   - Trees might route better on meaningful units

## Hypothesis

**Trees will perform better with BPE tokenization because:**
- Shorter sequences → easier to model dependencies
- Semantic units → routing decisions can be more meaningful
- Richer input space → trees can learn content-based splits

## Proposed Experiments

### **Experiment A: BPE vs Character (Same Model)**

Train oblivious_boosted_vo_alt on Shakespeare with:
1. Character-level (current, vocab=65)
2. BPE-1000 (vocab=1000, ~4x compression)
3. BPE-5000 (vocab=5000, ~8x compression)

**Metric:** Val accuracy at 2000 steps
**Expected:** BPE > char by 1-3pp

### **Experiment B: Standard Transformer Comparison**

Does standard transformer also improve with BPE?
- If yes → BPE is universally better
- If no → Trees specifically benefit from BPE

### **Experiment C: Routing Analysis**

Visualize routing decisions:
- Character-level: Do trees route based on letter frequency?
- BPE: Do trees route based on semantic categories (verbs vs nouns)?

**Method:**
```python
def analyze_token_routing(model, tokenizer, dataset):
    """Which paths do different token types take?"""
    routing_by_token = {}
    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id])
        routing = get_routing_path(model, token_id)
        routing_by_token[token_text] = routing
    
    # Cluster tokens by routing similarity
    # Do semantic tokens cluster together?
    return cluster_tokens(routing_by_token)
```

## Implementation

### **Phase 1: Add BPE Tokenizer (2-3 hours)**

```python
# data.py - new class
from tokenizers import Tokenizer, models, trainers

class BPEShakespeareDataset:
    def __init__(self, vocab_size=1000, block_size=128):
        self.tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size)
        # Train on shakespeare.txt
        self.tokenizer.train(files=['data/shakespeare.txt'], trainer=trainer)
        # ... rest similar to ShakespeareDataset
```

### **Phase 2: Update Training Script**

```python
# train.py - add argument
parser.add_argument('--tokenizer', type=str, 
                    choices=['char', 'bpe-1000', 'bpe-5000'],
                    default='char')

# Create appropriate dataset
if args.tokenizer == 'char':
    dataset = ShakespeareDataset(block_size=cfg['seq_len'])
else:
    vocab = int(args.tokenizer.split('-')[1])
    dataset = BPEShakespeareDataset(vocab_size=vocab, block_size=cfg['seq_len'])
```

### **Phase 3: Fair Comparison**

**Key question:** How to compare fairly across different sequence lengths?

**Option A:** Match parameter count
- BPE needs larger embedding (vocab 5000 vs 65)
- Reduce d_model to keep total params same

**Option B:** Match sequence length (tokens per context)
- Char: 256 tokens = ~40 words
- BPE: 32 tokens = ~40 words
- Use proportionally shorter block_size for BPE

**Recommendation:** Option B (match context, not token count)

## Expected Impact

**If successful:**
- Trees could match or beat standard transformer
- Would validate that char-level was the bottleneck
- Opens path to larger datasets (WikiText, OpenWebText)

**If unsuccessful:**
- Rules out tokenization as the issue
- Suggests fundamental tree limitation for sequential data

## Risks

1. **BPE vocabulary size:** Larger embedding = more parameters
   - Solution: Reduce d_model to compensate
   
2. **Overfitting:** Shakespeare is tiny for BPE vocab
   - Solution: Start with BPE-500, not BPE-5000
   
3. **Incomparable metrics:** Different vocab sizes = different tasks
   - Solution: Track perplexity, not just accuracy

## Dependencies

- Requires `tokenizers` library: `pip install tokenizers`
- May need larger dataset if BPE overfits

## Follow-Up Questions

If BPE helps trees:
1. **Optimal vocab size?** 500 vs 1000 vs 5000
2. **Byte-level BPE?** More robust to rare words
3. **SentencePiece?** Better for multilingual

If BPE doesn't help:
1. **Why?** Analyze routing patterns
2. **Hybrid?** BPE + char fallback for rare tokens
3. **Structured tokens?** Phonemes, morphemes
