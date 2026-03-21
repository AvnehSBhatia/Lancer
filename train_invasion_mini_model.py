"""
train_invasion_mini_model.py

Trains a simple mini-model on invasion training data.
Input: concat(question_emb, affected_group_emb, context_emb, personality_emb) -> 320 dims
Output: [P(invade), P(not invade)] - 2 classes, [1,0]=yes, [0,1]=no
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

DATA_PATH = Path("data/invasion_training.pt")
OUTPUT_PATH = Path("data/invasion_mini_model.pt")
EPOCHS = 100
LR = 0.001
BATCH_SIZE = 64


class InvasionMiniModel(nn.Module):
    """Simple classifier: 4 embeddings -> 2-class logits."""

    def __init__(self, input_dim: int = 320, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def main():
    data = torch.load(DATA_PATH, weights_only=True)
    X = data["X"]
    Y = data["Y"]
    input_dim = data["input_dim"]

    model = InvasionMiniModel(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=LR)
    n = len(X)

    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            x_batch = X[idx]
            y_batch = Y[idx]
            logits = model(x_batch)
            loss = nn.functional.cross_entropy(logits, y_batch.argmax(dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        if (epoch + 1) % 10 == 0:
            acc = (model(X).argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()
            print(f"epoch {epoch+1}/{EPOCHS}  loss={total_loss/n:.4f}  acc={acc:.4f}")

    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
