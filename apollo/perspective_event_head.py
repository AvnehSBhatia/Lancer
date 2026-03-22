"""
Aggregation head from main (24).pdf (Part 2), Steps 6–12 — after all N mini-models
have produced ``p_hat_i`` from Part 1 Step 5.

Step 6 — Sharpen personalities: ``C~_i = softmax(N * C_i)``, then ``C'_i = w ⊙ C~_i + b``
(shared ``w, b ∈ R^p``).

Step 7 — Confidence gating: ``C''_i = tanh(p+_i C'_i) + tanh(p-_i C'_i)`` using each
mini-model vote ``p_hat_i = [p+, p-]``.

Step 8 — ``E`` stacks rows ``C''_i`` → ``(N, p)``.

Step 9 — Situation vector ``ABDn = tanh(wD ⊙ ABn ⊙ Dn + bD)`` (shared ``wD, bD``;
``ABn`` and ``Dn`` are the global actor–receiver blend and context embedding in ``p``-space).

Step 10 — Poll: ``v = E @ ABDn ∈ R^N`` (same as ``ABDn · E^T`` as row-vector convention).

Steps 11–12 — Three bottleneck passes on ``v``, then ``softmax(W_out v^(3) + b)`` → ``y_hat``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BottleneckStack(nn.Module):
    """Step 11: three independent expand–contract passes (``N → hidden → N``)."""

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
        """
        Parameters
        ----------
        v : (N,) or (batch, N)
        """
        single = v.dim() == 1
        if single:
            v = v.unsqueeze(0)
        out = v
        for block in self.passes:
            out = block(out)
        return out.squeeze(0) if single else out


class PerspectiveEventHead(nn.Module):
    """
    Part 2 aggregation (main (24).pdf Steps 6–12). Requires Part 1 outputs ``p_hat``
    and embeddings ``C``, ``ABn``, ``Dn``.
    """

    def __init__(
        self,
        n: int,
        p: int,
        q: int | None = None,
        use_bottleneck: bool = False,
        bottleneck_hidden: int = 128,
    ) -> None:
        super().__init__()
        if n < 1 or p < 1:
            raise ValueError("n, p must be positive")
        self.n = n
        self.p = p
        self.q = q if q is not None else p
        self.use_bottleneck = use_bottleneck
        # Step 6(b): shared across all personalities
        self.w_sharp = nn.Parameter(torch.ones(p))
        self.b_sharp = nn.Parameter(torch.zeros(p))
        # Step 9: shared fusion weights
        self.w_d = nn.Parameter(torch.ones(p))
        self.b_d = nn.Parameter(torch.zeros(p))
        self.context_proj = nn.Linear(self.q, p) if self.q != p else None
        self.bottleneck = BottleneckStack(n, hidden=bottleneck_hidden) if use_bottleneck else None
        self.W_out = nn.Linear(n, 2)

    def _project_d(self, d_n: Tensor) -> Tensor:
        if self.context_proj is not None:
            return self.context_proj(d_n)
        return d_n

    def fuse_abdn(self, abn: Tensor, d_n: Tensor) -> Tensor:
        """Step 9: ``ABDn = tanh(wD ⊙ ABn ⊙ Dn + bD)`` (``d_n`` may be ``q``-dim)."""
        if abn.shape[-1] != self.p:
            raise ValueError(f"abn last dim must be p={self.p}, got {tuple(abn.shape)}")
        d_p = self._project_d(d_n)
        if d_p.shape[-1] != self.p:
            raise ValueError("internal D projection failed")
        return torch.tanh(self.w_d * abn * d_p + self.b_d)

    def build_E(self, C: Tensor, p_hat: Tensor) -> Tensor:
        """Steps 6–8: ``(N, p)`` or ``(batch, N, p)`` ensemble matrix."""
        if C.shape[-2] != self.n or C.shape[-1] != self.p:
            raise ValueError(f"Expected C ... (N={self.n}, p={self.p}), got {tuple(C.shape)}")
        if p_hat.shape[-2] != self.n or p_hat.shape[-1] != 2:
            raise ValueError(f"Expected p_hat ... (N={self.n}, 2), got {tuple(p_hat.shape)}")
        n = self.n
        c_tilde = F.softmax(float(n) * C, dim=-1)
        c_prime = c_tilde * self.w_sharp + self.b_sharp
        pp = p_hat[..., 0:1]
        pn = p_hat[..., 1:2]
        return torch.tanh(pp * c_prime) + torch.tanh(pn * c_prime)

    def forward(
        self,
        C: Tensor,
        p_hat: Tensor,
        abn: Tensor,
        d_n: Tensor,
    ) -> Tensor:
        """
        Full Steps 6–12 for one sample or a batch.

        Unbatched: ``C`` (N, p), ``p_hat`` (N, 2), ``abn`` (p,), ``d_n`` (q,) or (p,).
        Batched: ``C`` (B, N, p), ``p_hat`` (B, N, 2), ``abn`` (B, p), ``d_n`` (B, q).
        """
        if C.dim() == 2:
            return self._forward_batched(
                C.unsqueeze(0),
                p_hat.unsqueeze(0),
                abn.unsqueeze(0),
                d_n.unsqueeze(0),
            ).squeeze(0)
        return self._forward_batched(C, p_hat, abn, d_n)

    def forward_logits(
        self,
        C: Tensor,
        p_hat: Tensor,
        abn: Tensor,
        d_n: Tensor,
    ) -> Tensor:
        """
        Same inputs as ``forward``, returns class logits ``(2,)`` or ``(batch, 2)`` (no softmax).
        Use with ``CrossEntropyLoss`` for training.
        """
        if C.dim() == 2:
            return self._logits_batched(
                C.unsqueeze(0),
                p_hat.unsqueeze(0),
                abn.unsqueeze(0),
                d_n.unsqueeze(0),
            ).squeeze(0)
        return self._logits_batched(C, p_hat, abn, d_n)

    def _logits_batched(
        self,
        C: Tensor,
        p_hat: Tensor,
        abn: Tensor,
        d_n: Tensor,
    ) -> Tensor:
        E = self.build_E(C, p_hat)
        abdn = self.fuse_abdn(abn, d_n)
        v = torch.einsum("bnp,bp->bn", E, abdn)
        v3 = self.bottleneck(v) if self.bottleneck is not None else v
        return self.W_out(v3)

    def _forward_batched(
        self,
        C: Tensor,
        p_hat: Tensor,
        abn: Tensor,
        d_n: Tensor,
    ) -> Tensor:
        return F.softmax(self._logits_batched(C, p_hat, abn, d_n), dim=-1)

    def forward_with_abdn(
        self,
        C: Tensor,
        p_hat: Tensor,
        abdn: Tensor,
    ) -> Tensor:
        """
        Steps 10–12 only: ``abdn`` already fused (Step 9). Same shape rules as ``forward``
        for ``C``, ``p_hat``; ``abdn`` is (p,) or (B, p).
        """
        if C.dim() == 2:
            return self._forward_batched_abdn(
                C.unsqueeze(0),
                p_hat.unsqueeze(0),
                abdn.unsqueeze(0),
            ).squeeze(0)
        return self._forward_batched_abdn(C, p_hat, abdn)

    def _forward_batched_abdn(
        self,
        C: Tensor,
        p_hat: Tensor,
        abdn: Tensor,
    ) -> Tensor:
        E = self.build_E(C, p_hat)
        v = torch.einsum("bnp,bp->bn", E, abdn)
        v3 = self.bottleneck(v) if self.bottleneck is not None else v
        return F.softmax(self.W_out(v3), dim=-1)

    def forward_all_contexts(
        self,
        C: Tensor,
        p_hat: Tensor,
        abn: Tensor,
        d_stack: Tensor,
        pool: Literal["none", "mean", "max"] = "none",
    ) -> Tensor:
        """
        Run Step 9–12 for each context row in ``d_stack`` (J, q) with fixed ``ABn``.

        Returns (J, 2) if pool is ``"none"``, else (2,).
        """
        if C.dim() != 2 or p_hat.dim() != 2 or abn.dim() != 1:
            raise ValueError("forward_all_contexts expects unbatched C, p_hat, abn")
        if d_stack.dim() != 2 or d_stack.shape[1] != self.q:
            raise ValueError(f"Expected d_stack (J, {self.q}), got {tuple(d_stack.shape)}")
        J = d_stack.shape[0]
        ys = []
        for j in range(J):
            ys.append(self.forward(C, p_hat, abn, d_stack[j]))
        Y = torch.stack(ys, dim=0)
        if pool == "none":
            return Y
        if pool == "mean":
            return Y.mean(dim=0)
        if pool == "max":
            return Y.max(dim=0).values
        raise ValueError("pool must be 'none', 'mean', or 'max'")
