"""
train_perspective_stages.py

Trains the Stages1to5 perspective model on invasion data.
Uses a, b, c, d_ctx from invasion_training.pt (actor, receiver, personality, context).

Run scripts/build_invasion_training_data.py first.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apollo.paths import INVASION_TRAINING_PT, PERSPECTIVE_STAGES_PT
from apollo.perspective_stages import Config, build_model

DATA_PATH = INVASION_TRAINING_PT
OUTPUT_PATH = PERSPECTIVE_STAGES_PT
EPOCHS = 200
LR = 0.001
BATCH_SIZE = 32

# Config: n=personality slots, d=entity dim, p=personality dim, q=context dim
# Invasion data has 1 personality + 1 context per sample -> n=1
N, D, P, Q = 1, 64, 64, 64


def load_invasion_data():
    """Load and convert invasion data to (a, b, c, d_ctx, y) format."""
    data = torch.load(DATA_PATH, weights_only=True)
    X = data["X"]  # (N_samples, 320): q(64) + affected(128) + ctx(64) + pers(64)
    Y = data["Y"]  # (N_samples, 2)

    # Slice: affected = actor(64) + receiver(64) at 64:192
    a = X[:, 64:128]   # actor
    b = X[:, 128:192]  # receiver
    ctx = X[:, 192:256]   # context
    pers = X[:, 256:320]  # personality

    # c: (batch, n, p), d_ctx: (batch, n, q)
    c = pers.unsqueeze(1)   # (batch, 1, 64)
    d_ctx = ctx.unsqueeze(1)  # (batch, 1, 64)

    return a, b, c, d_ctx, Y


def main():
    a, b, c, d_ctx, y = load_invasion_data()
    n_samples = a.shape[0]
    targets = y.argmax(dim=1)  # 0=invade, 1=not

    cfg = Config(n=N, d=D, p=P, q=Q)
    model = build_model(cfg)
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            a_batch = a[idx]
            b_batch = b[idx]
            c_batch = c[idx]
            d_batch = d_ctx[idx]
            t_batch = targets[idx]

            out = model(a_batch, b_batch, c_batch, d_batch)
            # p_hat: (batch, n, 2) -> take slot 0 -> (batch, 2)
            logits = torch.log(out.p_hat[:, 0, :] + 1e-8)
            loss = nn.functional.nll_loss(logits, t_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                out = model(a, b, c, d_ctx)
                pred = out.p_hat[:, 0, :].argmax(dim=1)
                acc = (pred == targets).float().mean().item()
            print(f"epoch {epoch+1}/{EPOCHS}  loss={total_loss/n_samples:.4f}  acc={acc:.4f}")

    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
