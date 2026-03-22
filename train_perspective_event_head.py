"""
train_perspective_event_head.py

Trains ``PerspectiveEventHead`` (main (24).pdf Part 2) on tensors from
``build_perspective_event_head_training_data.py``.

Expects a .pt with: C_bank (N,p), p_hat (M,N,2), abn (M,p), d_n (M,q), y_class (M,) long.

Usage:
    python build_perspective_event_head_training_data.py --num-samples 50000
    python train_perspective_event_head.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from apollo.perspective_event_head import PerspectiveEventHead

DEFAULT_DATA = Path("data/perspective_event_head_training.pt")
DEFAULT_OUT = Path("data/perspective_event_head.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PerspectiveEventHead on built .pt data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Training .pt path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Save state_dict here.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of data for validation.")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Run: python build_perspective_event_head_training_data.py"
        )

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    pack = torch.load(args.data, weights_only=True)
    required = ("C_bank", "p_hat", "abn", "d_n", "y_class", "n", "p", "q")
    for k in required:
        if k not in pack:
            raise KeyError(f"Dataset missing key {k!r}")

    C_bank: torch.Tensor = pack["C_bank"].float()
    p_hat: torch.Tensor = pack["p_hat"].float()
    abn: torch.Tensor = pack["abn"].float()
    d_n: torch.Tensor = pack["d_n"].float()
    y_class: torch.Tensor = pack["y_class"].long()

    n, p_dim, q_dim = int(pack["n"]), int(pack["p"]), int(pack["q"])
    M = p_hat.shape[0]
    if C_bank.shape != (n, p_dim):
        raise ValueError(f"C_bank shape mismatch: got {tuple(C_bank.shape)}, expect ({n}, {p_dim})")
    if p_hat.shape != (M, n, 2):
        raise ValueError(f"p_hat shape mismatch: {tuple(p_hat.shape)}")
    if abn.shape != (M, p_dim) or d_n.shape != (M, q_dim):
        raise ValueError("abn / d_n shape mismatch with M, p, q")
    if y_class.shape != (M,):
        raise ValueError("y_class must be (M,)")

    # Train/val split
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(M * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    M_train = len(train_idx)

    model = PerspectiveEventHead(n=n, p=p_dim, q=q_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    C_bank_d = C_bank.to(device)

    print(f"Device: {device}")
    print(f"Samples: {M} (train {M_train}, val {n_val}) N={n} p={p_dim} q={q_dim}")

    for epoch in range(args.epochs):
        model.train()
        train_perm = torch.randperm(M_train)
        total_loss = 0.0
        correct = 0
        n_seen = 0
        for start in range(0, M_train, args.batch_size):
            batch_idx = train_perm[start : start + args.batch_size]
            idx = train_idx[batch_idx]
            B = idx.shape[0]
            C_b = C_bank_d.unsqueeze(0).expand(B, -1, -1)
            ph = p_hat[idx].to(device)
            a = abn[idx].to(device)
            d = d_n[idx].to(device)
            y = y_class[idx].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model.forward_logits(C_b, ph, a, d)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            correct += (logits.argmax(dim=-1) == y).sum().item()
            n_seen += B

        train_acc = correct / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_correct = 0
            for start in range(0, n_val, args.batch_size):
                idx = val_idx[start : start + args.batch_size]
                B = idx.shape[0]
                C_b = C_bank_d.unsqueeze(0).expand(B, -1, -1)
                ph = p_hat[idx].to(device)
                a = abn[idx].to(device)
                d = d_n[idx].to(device)
                y = y_class[idx].to(device)
                logits = model.forward_logits(C_b, ph, a, d)
                val_correct += (logits.argmax(dim=-1) == y).sum().item()
        val_acc = val_correct / n_val

        print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / n_seen:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
