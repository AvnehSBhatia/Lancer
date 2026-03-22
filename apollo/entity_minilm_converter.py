"""
entity_minilm_converter.py

Bidirectional converter between our entity embeddings (64-d) and MiniLM (384-d).
Loads entity_embeddings/ and trains linear projections for both directions.
Saves to entity_embeddings/converter_*.pt
"""

import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from apollo.paths import ENTITY_EMBEDDINGS_DIR as ENTITY_DIR

# MiniLM-L6-v2 output dim
MINILM_DIM = 384
ENTITY_DIM = 64
CONVERTER_EPOCHS = 1800
CONVERTER_LR = 0.001


class EntityMiniLMConverter(nn.Module):
    """Bidirectional linear projection between our embeddings and MiniLM space."""

    def __init__(self) -> None:
        super().__init__()
        self.ours_to_minilm = nn.Linear(ENTITY_DIM, MINILM_DIM)
        self.minilm_to_ours = nn.Linear(MINILM_DIM, ENTITY_DIM)
        nn.init.xavier_uniform_(self.ours_to_minilm.weight)
        nn.init.xavier_uniform_(self.minilm_to_ours.weight)
        nn.init.zeros_(self.ours_to_minilm.bias)
        nn.init.zeros_(self.minilm_to_ours.bias)

    def ours_to_minilm_vec(self, x: torch.Tensor) -> torch.Tensor:
        return self.ours_to_minilm(x)

    def minilm_to_ours_vec(self, x: torch.Tensor) -> torch.Tensor:
        return self.minilm_to_ours(x)


def load_entity_embeddings() -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Returns (name -> 64-d tensor), vocab."""
    with open(ENTITY_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(ENTITY_DIR / "entity_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i] for i in range(len(vocab))}, vocab


def get_minilm_embeddings(names: list[str]) -> torch.Tensor:
    """Returns (N, 384) MiniLM embeddings for entity names."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return torch.tensor(model.encode(names, convert_to_numpy=True), dtype=torch.float32)


def train_converter(
    our_embs: dict[str, torch.Tensor],
    minilm_embs: torch.Tensor,
    names: list[str],
) -> EntityMiniLMConverter:
    """Train bidirectional projections to align our embeddings with MiniLM."""
    converter = EntityMiniLMConverter()
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

        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{CONVERTER_EPOCHS}  loss={loss.item():.4f}")

    return converter


def main() -> None:
    print("Loading entity embeddings...")
    our_embs, vocab = load_entity_embeddings()
    names = list(our_embs.keys())
    print(f"  {len(names)} entities")

    print("Computing MiniLM embeddings...")
    minilm_embs = get_minilm_embeddings(names)

    print("Training converter...")
    converter = train_converter(our_embs, minilm_embs, names)

    os.makedirs(ENTITY_DIR, exist_ok=True)
    torch.save(converter.state_dict(), ENTITY_DIR / "converter_ours_minilm.pt")
    print(f"Saved {ENTITY_DIR / 'converter_ours_minilm.pt'}")

    with open(ENTITY_DIR / "converter_config.json", "w") as f:
        json.dump(
            {"entity_dim": ENTITY_DIM, "minilm_dim": MINILM_DIM, "vocab_path": "vocab.json"},
            f,
            indent=2,
        )
    print("Done.")


def load_converter(
    entity_dir: Union[Path, str] = ENTITY_DIR,
) -> tuple[EntityMiniLMConverter, dict[str, torch.Tensor], dict[str, int]]:
    """Load trained converter + entity embeddings + vocab for inference."""
    entity_dir = Path(entity_dir)
    with open(entity_dir / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(entity_dir / "entity_embeddings.pt", weights_only=True)
    emb_weight = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    our_embs = {inv_vocab[i]: emb_weight[i] for i in range(len(vocab))}

    converter = EntityMiniLMConverter()
    converter.load_state_dict(torch.load(entity_dir / "converter_ours_minilm.pt"))
    converter.eval()
    return converter, our_embs, vocab


# Example usage:
#   converter, our_embs, vocab = load_converter()
#   name = "United States of America"
#   ours = our_embs[name]
#   minilm_vec = converter.ours_to_minilm_vec(ours.unsqueeze(0)).squeeze(0)
#   back = converter.minilm_to_ours_vec(minilm_vec.unsqueeze(0)).squeeze(0)
#
# For MiniLM -> ours given a name: first get MiniLM via SentenceTransformer.encode(name), then minilm_to_ours_vec
if __name__ == "__main__":
    main()
