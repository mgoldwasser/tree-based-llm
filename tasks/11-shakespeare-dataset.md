# Task 11: Shakespeare Dataset — Real-World Character-Level LM

**Status:** Pending
**Priority:** High — synthetic data has proven insufficient for meaningful comparison
**Depends on:** Task 08 (parameter audit), Task 09 (speed optimizations)
**Files:** new `data.py`, `train.py`, `benchmark.py`

## Why Shakespeare
- ~1.1MB of text, ~1.1M characters, 65 unique characters
- Small enough to train in minutes on GPU, hours on CPU
- Well-understood baselines (Karpathy's char-RNN, nanoGPT)
- Character-level = small vocab (65) → manageable output layer
- Real natural language patterns: hierarchical structure, long-range dependencies, grammar rules
- Trees should shine on hierarchical patterns (if they're going to shine anywhere)

## Known Baselines (character-level)
- RNN (Karpathy, 2015): ~1.5 bits/char
- Small Transformer (nanoGPT): ~1.0-1.2 bits/char with 10M params
- Our models (~200K-400K params): expect worse, but relative comparison is what matters

## Implementation

### data.py
```python
def download_shakespeare():
    """Download tiny-shakespeare (~1.1MB) from Karpathy's repo."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # Download, create train/val split (90/10)
    # Return char-level tokenizer + data tensors

class ShakespeareDataset:
    def __init__(self, text, block_size=256):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)
```

### Model Config for Shakespeare
```python
config = {
    'vocab': 65,          # unique characters in Shakespeare
    'd_model': 128,       # small but meaningful
    'n_layers': 4,        # enough depth for real patterns
    'n_heads': 4,
    'seq_len': 256,       # character-level needs longer context
    'batch': 64,
    'n_steps': 5000,      # minimum for real convergence
    'lr': 3e-4,
}
```

### Evaluation
- **Loss:** cross-entropy (lower = better)
- **Accuracy:** next-character prediction accuracy
- **Perplexity:** exp(loss) — interpretable
- **Generation:** sample text from each model to qualitatively compare
- **Val loss:** track overfitting

## Why This Matters
The synthetic tasks proved that all models are roughly equivalent on trivial patterns, and the XOR task was too hard for anyone. Shakespeare is the "Goldilocks zone" — complex enough to differentiate architectures, simple enough to train on limited compute.

## Expected Outcome
If trees provide value, they should show it on Shakespeare's hierarchical structure:
- Word boundaries (trees can learn character→word routing)
- Grammar rules (trees can route based on syntactic features)
- Verse structure (trees can route based on position/rhythm)

If trees DON'T help on Shakespeare, the approach may need fundamental rethinking.
