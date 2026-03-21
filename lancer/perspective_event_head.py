"""
Tail of the perspective-conditioned event architecture (Stages 5–8; spec: main (21).pdf).

Inputs (what you pass at runtime)
----------------------------------
``C`` — Personality bank from earlier stages. Shape ``(N, p)``. Row ``i`` is the vector
``C_i`` for analytical lens ``i`` (the same ``N`` as the number of mini-models).

``abd`` — Context-enriched actor–target embedding ``ABD_j``. Shape ``(d,)``. This is the
blend of actor ``A``, target ``B``, and one global context slice ``D_j``, produced by
Stages 1–4 in the full model. If ``d != p``, a learned map ``R^d → R^p`` aligns ``abd``
to the personality row space before polling (Stage 6).

Constructor hyperparameters
---------------------------
``n`` — ``N``: number of personalities / mini-models (must match ``C.shape[0]``).

``p`` — Dimension of each ``C_i`` and each row of the ensemble matrix ``E``.

``d`` — Dimension of ``abd`` (actor–target–context space).

Stage summary: ``E`` from ``C`` → poll ``v = E @ align(abd)`` → three bottlenecks → softmax.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MiniModel(nn.Module):
    """
    One personality mini-model: Steps A–C for a single C_i ∈ R^p.
    p̂_i is produced from C*_i via a learned linear head + softmax (spec gap).
    """

    def __init__(self, p: int, n_personalities: int) -> None:
        super().__init__()
        self.p = p
        self.n_personalities = float(n_personalities)
        self.W1 = nn.Linear(p, p)
        self.W_prob = nn.Linear(p, 2)

    def forward(self, C_i: Tensor) -> Tensor:
        if C_i.shape != (self.p,):
            raise ValueError(f"Expected C_i shape ({self.p},), got {tuple(C_i.shape)}")
        c_tilde = F.softmax(self.n_personalities * C_i, dim=-1)
        c_star = F.relu(self.W1(c_tilde))
        p_hat = F.softmax(self.W_prob(c_star), dim=-1)
        c_pp = torch.tanh(p_hat[0] * c_star) + torch.tanh(p_hat[1] * c_star)
        return c_pp


class MiniModelBank(nn.Module):
    """N independent mini-models; forward builds E with rows C''_i."""

    def __init__(self, n: int, p: int) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.p = p
        self.models = nn.ModuleList([MiniModel(p, n) for _ in range(n)])

    def forward(self, C: Tensor) -> Tensor:
        """
        Parameters
        ----------
        C : Tensor, shape (N, p)
            Personality bank; row i is C_i.

        Returns
        -------
        E : Tensor, shape (N, p)
        """
        if C.dim() != 2 or C.shape[0] != self.n or C.shape[1] != self.p:
            raise ValueError(
                f"Expected C shape ({self.n}, {self.p}), got {tuple(C.shape)}"
            )
        rows = [self.models[i](C[i]) for i in range(self.n)]
        return torch.stack(rows, dim=0)


class BottleneckStack(nn.Module):
    """Three independent bottleneck passes: v ← W↓ ReLU(W↑ v)."""

    def __init__(self, n: int, hidden: int = 128) -> None:
        super().__init__()
        self.passes = nn.ModuleList(
            nn.Sequential(
                nn.Linear(n, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n),
            )
            for _ in range(3)
        )

    def forward(self, v: Tensor) -> Tensor:
        if v.dim() != 1:
            raise ValueError(f"Expected v 1D, got shape {tuple(v.shape)}")
        out = v
        for block in self.passes:
            out = block(out)
        return out


def _align_abd(abd: Tensor, align: nn.Linear) -> Tensor:
    """Map abd ∈ R^d to R^p via learned W_align."""
    if abd.dim() != 1:
        raise ValueError(f"Expected abd 1D, got shape {tuple(abd.shape)}")
    return align(abd)


class PerspectiveEventHead(nn.Module):
    """
    Stages 5–8: E from C, poll with abd, bottleneck, softmax to Δ¹.

    When d == p, you can initialize `align` near identity by setting
    `init_align_identity=True` (only valid if d == p).
    """

    def __init__(
        self,
        n: int,
        p: int,
        d: int,
        bottleneck_hidden: int = 128,
        init_align_identity: bool = False,
    ) -> None:
        super().__init__()
        if n < 1 or p < 1 or d < 1:
            raise ValueError("n, p, d must be positive")
        self.n = n
        self.p = p
        self.d = d
        self.mini_bank = MiniModelBank(n, p)
        self.align = nn.Linear(d, p)
        self.bottleneck = BottleneckStack(n, hidden=bottleneck_hidden)
        self.W_out = nn.Linear(n, 2)

        if init_align_identity:
            if d != p:
                raise ValueError("init_align_identity requires d == p")
            with torch.no_grad():
                self.align.weight.zero_()
                self.align.weight.diagonal().fill_(1.0)
                self.align.bias.zero_()

    def forward(self, C: Tensor, abd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        C : (N, p)
            Personality vectors; row ``i`` is ``C_i``.
        abd : (d,)
            Single context-enriched actor–target vector ``ABD_j`` (from Stages 1–4).

        Returns
        -------
        ŷ : (2,) with nonnegative entries summing to 1.
        """
        E = self.mini_bank(C)
        abd_p = _align_abd(abd, self.align)
        v = E @ abd_p
        v = self.bottleneck(v)
        logits = self.W_out(v)
        return F.softmax(logits, dim=-1)

    def forward_batched(self, C: Tensor, abd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        C : (B, N, p)
        abd : (B, d)

        Returns
        -------
        ŷ : (B, 2)
        """
        if C.dim() != 3 or abd.dim() != 2:
            raise ValueError(
                f"Expected C (B,N,p) and abd (B,d); got {tuple(C.shape)}, {tuple(abd.shape)}"
            )
        B = C.shape[0]
        if abd.shape[0] != B:
            raise ValueError("Batch size mismatch between C and abd")
        outs = []
        for b in range(B):
            outs.append(self.forward(C[b], abd[b]))
        return torch.stack(outs, dim=0)

    def forward_all_contexts(
        self,
        C: Tensor,
        abd_stack: Tensor,
        pool: Literal["none", "mean", "max"] = "none",
    ) -> Tensor:
        """
        Parameters
        ----------
        C : (N, p)
        abd_stack : (J, d) — one context-enriched actor–target vector per row.

        pool
            If ``"none"``, returns (J, 2). If ``"mean"`` or ``"max"``, reduces over J to (2,).
        """
        if abd_stack.dim() != 2 or abd_stack.shape[1] != self.d:
            raise ValueError(
                f"Expected abd_stack (J, {self.d}), got {tuple(abd_stack.shape)}"
            )
        J = abd_stack.shape[0]
        ys = []
        for j in range(J):
            ys.append(self.forward(C, abd_stack[j]))
        Y = torch.stack(ys, dim=0)
        if pool == "none":
            return Y
        if pool == "mean":
            return Y.mean(dim=0)
        if pool == "max":
            return Y.max(dim=0).values
        raise ValueError("pool must be 'none', 'mean', or 'max'")
