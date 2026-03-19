"""
Shakespeare dataset for character-level language modeling.
Downloads tiny-shakespeare (~1.1MB) and provides batch iterators.
"""

import os
import torch
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")


def download_shakespeare():
    """Download tiny-shakespeare if not already cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SHAKESPEARE_PATH):
        print(f"Downloading tiny-shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
        print(f"Saved to {SHAKESPEARE_PATH}")
    with open(SHAKESPEARE_PATH, 'r') as f:
        return f.read()


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join(self.idx_to_char[i] for i in indices)


class ShakespeareDataset:
    """Character-level Shakespeare dataset with train/val split."""

    def __init__(self, block_size: int = 256, val_frac: float = 0.1):
        text = download_shakespeare()
        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(len(data) * (1 - val_frac))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.block_size = block_size

        print(f"Shakespeare: {len(data):,} chars, vocab={self.vocab_size}, "
              f"train={len(self.train_data):,}, val={len(self.val_data):,}")

    def get_batch(self, batch_size: int, device: str, split: str = "train"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model, batch_size: int, device: str, n_batches: int = 20):
        """Estimate train and val loss over n_batches."""
        model.eval()
        results = {}
        for split in ['train', 'val']:
            losses, accs = [], []
            for _ in range(n_batches):
                x, y = self.get_batch(batch_size, device, split)
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, self.vocab_size), y.reshape(-1))
                acc = (logits.argmax(-1) == y).float().mean()
                losses.append(loss.item())
                accs.append(acc.item())
            results[split] = {
                'loss': sum(losses) / len(losses),
                'acc': sum(accs) / len(accs),
            }
        model.train()
        return results

    @torch.no_grad()
    def generate(self, model, prompt: str = "\n", max_tokens: int = 200,
                 device: str = "cpu", temperature: float = 0.8):
        """Generate text from a trained model."""
        model.eval()
        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long,
                           device=device).unsqueeze(0)
        # Detect max sequence length from positional embedding
        max_seq = 256  # default
        for attr in ['pos_emb', 'pos']:
            emb = getattr(model, attr, None)
            if emb is not None and hasattr(emb, 'num_embeddings'):
                max_seq = emb.num_embeddings
                break

        for _ in range(max_tokens):
            idx_crop = idx[:, -max_seq:]
            logits, _ = model(idx_crop)
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        model.train()
        return self.tokenizer.decode(idx[0].tolist())
