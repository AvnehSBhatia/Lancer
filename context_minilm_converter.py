"""
context_minilm_converter.py

Bidirectional converter between our context embeddings (64-d) and MiniLM (384-d).
Loads context_embeddings/ and trains linear projections for both directions.
Saves to context_embeddings/converter_*.pt
"""

import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

MINILM_DIM = 384
CONTEXT_DIM = 64
CONTEXT_DIR = Path("context_embeddings")
CONVERTER_EPOCHS = 1800
CONVERTER_LR = 0.001


class ContextMiniLMConverter(nn.Module):
    """Bidirectional linear projection between our embeddings and MiniLM space."""

    def __init__(self) -> None:
        super().__init__()
        self.ours_to_minilm = nn.Linear(CONTEXT_DIM, MINILM_DIM)
        self.minilm_to_ours = nn.Linear(MINILM_DIM, CONTEXT_DIM)
        nn.init.xavier_uniform_(self.ours_to_minilm.weight)
        nn.init.xavier_uniform_(self.minilm_to_ours.weight)
        nn.init.zeros_(self.ours_to_minilm.bias)
        nn.init.zeros_(self.minilm_to_ours.bias)

    def ours_to_minilm_vec(self, x: torch.Tensor) -> torch.Tensor:
        return self.ours_to_minilm(x)

    def minilm_to_ours_vec(self, x: torch.Tensor) -> torch.Tensor:
        return self.minilm_to_ours(x)


def load_context_embeddings() -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Returns (string -> 64-d tensor), vocab."""
    with open(CONTEXT_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(CONTEXT_DIR / "context_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i] for i in range(len(vocab))}, vocab


def get_minilm_embeddings(texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """Returns (N, 384) MiniLM embeddings. Batched for long context strings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return torch.tensor(embs, dtype=torch.float32)


def train_converter(
    our_embs: dict[str, torch.Tensor],
    minilm_embs: torch.Tensor,
    names: list[str],
) -> ContextMiniLMConverter:
    """Train bidirectional projections to align our embeddings with MiniLM."""
    converter = ContextMiniLMConverter()
    opt = torch.optim.Adam(converter.parameters(), lr=CONVERTER_LR)

    X_ours = torch.stack([our_embs[n] for n in names])
    X_minilm = minilm_embs
    X_minilm = X_minilm / (X_minilm.norm(dim=1, keepdim=True) + 1e-8)
    X_ours = X_ours / (X_ours.norm(dim=1, keepdim=True) + 1e-8)

    for ep in range(CONVERTER_EPOCHS):
        opt.zero_grad()
        pred_minilm = converter.ours_to_minilm_vec(X_ours)
        pred_ours = converter.minilm_to_ours_vec(X_minilm)

        pred_minilm_n = pred_minilm / (pred_minilm.norm(dim=1, keepdim=True) + 1e-8)
        pred_ours_n = pred_ours / (pred_ours.norm(dim=1, keepdim=True) + 1e-8)
        loss_ours2m = 1 - (pred_minilm_n * X_minilm).sum(dim=1).mean()
        loss_m2ours = 1 - (pred_ours_n * X_ours).sum(dim=1).mean()
        loss = loss_ours2m + loss_m2ours
        loss.backward()
        opt.step()

        if (ep + 1) % 200 == 0:
            print(f"  epoch {ep+1}/{CONVERTER_EPOCHS}  loss={loss.item():.4f}")

    return converter


def main() -> None:
    print("Loading context embeddings...")
    our_embs, vocab = load_context_embeddings()
    names = list(our_embs.keys())
    print(f"  {len(names)} contexts")

    print("Computing MiniLM embeddings (batched)...")
    minilm_embs = get_minilm_embeddings(names)

    print("Training converter...")
    converter = train_converter(our_embs, minilm_embs, names)

    os.makedirs(CONTEXT_DIR, exist_ok=True)
    torch.save(converter.state_dict(), CONTEXT_DIR / "converter_ours_minilm.pt")
    print(f"Saved {CONTEXT_DIR / 'converter_ours_minilm.pt'}")

    with open(CONTEXT_DIR / "converter_config.json", "w") as f:
        json.dump(
            {"context_dim": CONTEXT_DIM, "minilm_dim": MINILM_DIM, "vocab_path": "vocab.json"},
            f,
            indent=2,
        )
    print("Done.")


def load_converter(
    context_dir: Union[Path, str] = CONTEXT_DIR,
) -> tuple[ContextMiniLMConverter, dict[str, torch.Tensor], dict[str, int]]:
    """Load trained converter + context embeddings + vocab for inference."""
    context_dir = Path(context_dir)
    with open(context_dir / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(context_dir / "context_embeddings.pt", weights_only=True)
    emb_weight = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    our_embs = {inv_vocab[i]: emb_weight[i] for i in range(len(vocab))}

    converter = ContextMiniLMConverter()
    converter.load_state_dict(torch.load(context_dir / "converter_ours_minilm.pt"))
    converter.eval()
    return converter, our_embs, vocab


if __name__ == "__main__":
    main()
