"""
train_entity_embeddings.py

Each line of entities.txt is one triplet:
    anchor TAB positive TAB negative
e.g.
    United States	United Kingdom	North Korea

Both actors and receivers are looked up from this same table.
Saves model + vocab to ./entity_embeddings/
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────
INPUT_FILE = "data/entities.txt"
OUTPUT_DIR = "entity_embeddings"
EMBED_DIM = 64
EPOCHS = 500
LR = 0.02
BATCH_SIZE = 1024
TEMPERATURE = 0.07  # contrastive temperature; lower = sharper
LR_DECAY = 0.995    # per-epoch decay (slower so loss can settle)
# ─────────────────────────────────────────────────────────────────


def load_triplets(path: str) -> tuple[list[tuple], dict[str, int]]:
    """
    Reads the file, builds a vocab from every unique string seen,
    and returns the triplets as (anchor_id, positive_id, negative_id).
    """
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
    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_entities, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embeddings(idx)

    def get_vector(self, name: str, vocab: dict) -> torch.Tensor:
        with torch.no_grad():
            return self.embeddings(torch.tensor(vocab[name]))


def ce_loss(
    model: nn.Module,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Contrastive CE: classify [positive, negative] with positive as correct class."""
    a = F.normalize(model(anchors), dim=1)
    p = F.normalize(model(positives), dim=1)
    n = F.normalize(model(negatives), dim=1)
    sim_pos = (a * p).sum(dim=1, keepdim=True) / temperature
    sim_neg = (a * n).sum(dim=1, keepdim=True) / temperature
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    targets = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
    return F.cross_entropy(logits, targets)


def training_loop(model: nn.Module, triplets: list[tuple]):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    triplets_t = (
        torch.tensor([t[0] for t in triplets]),
        torch.tensor([t[1] for t in triplets]),
        torch.tensor([t[2] for t in triplets]),
    )
    n = len(triplets)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            anchors = triplets_t[0][idx]
            positives = triplets_t[1][idx]
            negatives = triplets_t[2][idx]

            optimizer.zero_grad()
            loss = ce_loss(
                model, anchors, positives, negatives, temperature=TEMPERATURE
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        lr = LR * (LR_DECAY ** epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n:.4f}  lr={lr:.6f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading entity triplets from: {INPUT_FILE}")
    triplets, vocab = load_triplets(INPUT_FILE)
    n = len(vocab)
    print(f"  {len(triplets)} triplets,  {n} unique entities")

    model = EmbeddingModel(num_entities=n, dim=EMBED_DIM)
    print(f"Training embedding table  ({n} x {EMBED_DIM})...")
    training_loop(model, triplets)

    model_path = os.path.join(OUTPUT_DIR, "entity_embeddings.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab -> {vocab_path}")

    print("Done.")


if __name__ == "__main__":
    main()
