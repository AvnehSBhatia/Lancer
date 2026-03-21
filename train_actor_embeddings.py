"""
train_actor_embeddings.py

Loads a list of actor/entity strings (one per line) from a text file,
assigns each a unique integer ID, trains an embedding table of dim 64,
and saves the model + the id->string vocabulary to ./actor_embeddings/
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# ── Config ────────────────────────────────────────────────────────
INPUT_FILE  = "data/actors.txt"       # one actor string per line
OUTPUT_DIR  = "actor_embeddings"
EMBED_DIM   = 64
EPOCHS      = 200
LR          = 0.01
# ─────────────────────────────────────────────────────────────────


def load_strings(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def build_vocab(strings: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(strings)}


class EmbeddingModel(nn.Module):
    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_entities, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embeddings(idx)

    def get_vector(self, idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.embeddings(torch.tensor(idx))


def dummy_training_loop(model: EmbeddingModel, num_entities: int):
    """
    Placeholder training loop using a self-supervised objective:
    each entity should be distinct from every other (contrastive push).
    Replace this with your real loss once you have labelled event data.
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        all_idx = torch.arange(num_entities)
        vecs = model(all_idx)                        # (N, 64)

        # Normalise
        normed = nn.functional.normalize(vecs, dim=1)

        # Similarity matrix
        sim = normed @ normed.T                      # (N, N)

        # Push all off-diagonal similarities toward 0
        eye = torch.eye(num_entities)
        loss = ((sim - eye) ** 2).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading actors from: {INPUT_FILE}")
    strings = load_strings(INPUT_FILE)
    vocab   = build_vocab(strings)
    n       = len(strings)
    print(f"  Found {n} actors")

    model = EmbeddingModel(num_entities=n, dim=EMBED_DIM)
    print(f"Training embedding table  ({n} x {EMBED_DIM})...")
    dummy_training_loop(model, n)

    # Save model weights
    model_path = os.path.join(OUTPUT_DIR, "actor_embeddings.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

    # Save vocabulary (string -> int id)
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab -> {vocab_path}")

    print("Done.")


if __name__ == "__main__":
    main()