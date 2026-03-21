"""
train_personality_embeddings.py

Each line of personalities.txt is one triplet:
    anchor TAB positive TAB negative
e.g.
    military analyst	defense strategist	pacifist activist

Saves model + vocab to ./personality_embeddings/
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────
INPUT_FILE = "data/personalities.txt"
OUTPUT_DIR = "personality_embeddings"
EMBED_DIM = 64
EPOCHS = 5000
LR = 0.01
# ─────────────────────────────────────────────────────────────────


def load_triplets(path: str) -> tuple[list[tuple], dict[str, int]]:
    raw_triplets = []
    all_strings = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Each line must have exactly 3 tab-separated values.\n"
                    f"Bad line: {repr(line)}"
                )
            anchor, positive, negative = parts
            raw_triplets.append((anchor, positive, negative))
            all_strings.update([anchor, positive, negative])

    vocab = {s: i for i, s in enumerate(sorted(all_strings))}
    id_triplets = [
        (vocab[a], vocab[p], vocab[n])
        for a, p, n in raw_triplets
    ]
    return id_triplets, vocab


class EmbeddingModel(nn.Module):
    def __init__(self, num_personalities: int, dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_personalities, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embeddings(idx)

    def get_vector(self, name: str, vocab: dict) -> torch.Tensor:
        with torch.no_grad():
            return self.embeddings(torch.tensor(vocab[name]))


def ce_loss(model, anchors, positives, negatives):
    """Contrastive CE: classify [positive, negative] with positive as correct class."""
    a = F.normalize(model(anchors), dim=1)
    p = F.normalize(model(positives), dim=1)
    n = F.normalize(model(negatives), dim=1)
    sim_pos = (a * p).sum(dim=1, keepdim=True)
    sim_neg = (a * n).sum(dim=1, keepdim=True)
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    targets = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
    return F.cross_entropy(logits, targets)


def training_loop(model, triplets):
    optimizer = optim.Adam(model.parameters(), lr=LR)

    anchors = torch.tensor([t[0] for t in triplets])
    positives = torch.tensor([t[1] for t in triplets])
    negatives = torch.tensor([t[2] for t in triplets])

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = ce_loss(model, anchors, positives, negatives)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading personality triplets from: {INPUT_FILE}")
    triplets, vocab = load_triplets(INPUT_FILE)
    n = len(vocab)
    print(f"  {len(triplets)} triplets,  {n} unique personalities")

    model = EmbeddingModel(num_personalities=n, dim=EMBED_DIM)
    print(f"Training embedding table  ({n} x {EMBED_DIM})...")
    training_loop(model, triplets)

    model_path = os.path.join(OUTPUT_DIR, "personality_embeddings.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab -> {vocab_path}")

    print("Done.")


if __name__ == "__main__":
    main()