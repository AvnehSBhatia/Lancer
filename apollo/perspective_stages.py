"""
stages_1_5.py

Stages 1-5 of the perspective-conditioned geopolitical prediction architecture.
Inputs A, B, C, D are pre-computed embedding tensors passed in from outside.
Output is p_hat: (batch, N, 2) — each mini-model's binary probability vector.

Stage 1  — outer product matrix M:       (batch, N, N, p*q)
Stage 2  — 3x residual refinement M':    (batch, N, N, p*q)
Stage 3  — project M' into d-space:      Q, K each (batch, d, N)
Stage 4  — ABn blend + context fusion:   ABDj (batch, N, d)
Stage 5  — N mini models -> p_hat:       (batch, N, 2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    n: int          # number of personality / context vectors
    d: int          # actor-target embedding dim
    p: int          # personality embedding dim
    q: int          # context embedding dim
    residual_passes: int = 3
    eps: float = 1e-8


# ─────────────────────────────────────────────────────────────────
@dataclass
class Outputs:
    # Stage 1
    m:              torch.Tensor   # (batch, N, N, p*q)  raw outer-product matrix
    # Stage 2
    m_prime:        torch.Tensor   # (batch, N, N, p*q)  after residual refinement
    # Stage 3
    q_proj:         torch.Tensor   # (batch, d, N)
    k_proj:         torch.Tensor   # (batch, d, N)
    # Stage 4
    cos_ab:         torch.Tensor   # (batch,)
    abn:            torch.Tensor   # (batch, d)
    abdj:           torch.Tensor   # (batch, N, d)
    # Stage 5
    c_tilde:        torch.Tensor   # (batch, N, p)  after softmax sharpening
    c_star:         torch.Tensor   # (batch, N, p)  after affine + ReLU
    c_double_prime: torch.Tensor   # (batch, N, p)  after confidence gating
    p_hat:          torch.Tensor   # (batch, N, 2)  each mini-model's [P+, P-]


# ─────────────────────────────────────────────────────────────────
class Stages1to5(nn.Module):
    """
    Args (forward):
        a      : (batch, d)       actor embedding
        b      : (batch, d)       receiver embedding
        c      : (batch, N, p)    personality embeddings
        d_ctx  : (batch, N, q)    context embeddings

    Returns: Outputs dataclass
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        N, d, p, q = cfg.n, cfg.d, cfg.p, cfg.q
        cell = p * q   # flattened outer-product cell size

        # ── Stage 2: residual parameters ─────────────────────────
        # alpha: single shared scalar across all cells and all passes
        self.alpha = nn.Parameter(torch.zeros(1))
        # B: unique learned bias per cell in the N×N matrix
        # shape (N, N, cell) — one bias vector per cell
        self.bias_m = nn.Parameter(torch.zeros(N, N, cell))

        # ── Stage 3: project M' columns into d-space ─────────────
        # M' is (N, N, cell); we treat the N columns (axis 1) as
        # the "key" dimension and project each cell-vector to d.
        # W_Q and W_K are (d, cell) — applied to each cell vector.
        self.w_q = nn.Linear(cell, d, bias=False)
        self.w_k = nn.Linear(cell, d, bias=False)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)

        # ── Stage 4: per-context fusion weights ──────────────────
        # w_j and b_j are (d,) vectors, one per context j
        self.abd_w = nn.Parameter(torch.ones(N, d))
        self.abd_b = nn.Parameter(torch.zeros(N, d))
        # project context vectors from q -> d so they match ABn
        self.context_proj = nn.Linear(q, d, bias=True)

        # ── Stage 5: N independent mini-models ───────────────────
        # Each mini-model has:
        #   W1: (p, p) affine on the personality vector
        #   b1: (p,) bias
        #   W_prob: (2, p+d) — takes [c_star_i || abdj_i] as input
        #   b_prob: (2,) bias
        # We concatenate c_star_i with abdj_i so the probability
        # head sees both the personality AND the current situation.
        self.w1 = nn.ParameterList([
            nn.Parameter(torch.empty(p, p)) for _ in range(N)
        ])
        self.b1 = nn.ParameterList([
            nn.Parameter(torch.zeros(p)) for _ in range(N)
        ])
        self.w_prob = nn.ParameterList([
            nn.Parameter(torch.empty(2, p + d)) for _ in range(N)
        ])
        self.b_prob = nn.ParameterList([
            nn.Parameter(torch.zeros(2)) for _ in range(N)
        ])
        for w in self.w1:
            nn.init.xavier_uniform_(w)
        for w in self.w_prob:
            nn.init.xavier_uniform_(w)

    # ─────────────────────────────────────────────────────────────
    def forward(
        self,
        a:     torch.Tensor,   # (batch, d)
        b:     torch.Tensor,   # (batch, d)
        c:     torch.Tensor,   # (batch, N, p)
        d_ctx: torch.Tensor,   # (batch, N, q)
    ) -> Outputs:

        cfg = self.cfg
        N, d_dim, p, q = cfg.n, cfg.d, cfg.p, cfg.q
        eps = cfg.eps
        batch = a.shape[0]

        # ── Stage 1: outer product matrix ────────────────────────
        # For each pair (i, j): cell = C_i ⊗ D_j flattened to (p*q,)
        # c:     (batch, N, p) -> unsqueeze to (batch, N, 1, p)
        # d_ctx: (batch, N, q) -> unsqueeze to (batch, 1, N, q)
        # outer: (batch, N, N, p, q) -> flatten last two -> (batch, N, N, p*q)
        c_exp  = c.unsqueeze(2)        # (batch, N, 1, p)
        d_exp  = d_ctx.unsqueeze(1)    # (batch, 1, N, q)
        # broadcast multiply gives (batch, N, N, p, q)
        outer  = c_exp.unsqueeze(-1) * d_exp.unsqueeze(-2)
        cell   = p * q
        m      = outer.reshape(batch, N, N, cell)   # (batch, N, N, p*q)

        # ── Stage 2: 3x residual refinement ──────────────────────
        # Per-pass: M^(t) = alpha * M^(t-1) + B + M^(t-1)
        #                  = (1 + alpha) * M^(t-1) + B
        # bias_m is (N, N, cell), broadcast over batch
        m_p = m
        for _ in range(cfg.residual_passes):
            m_p = (1.0 + self.alpha) * m_p + self.bias_m

        # ── Stage 3: project M' into actor-target space ──────────
        # Apply W_Q and W_K (each Linear(cell, d)) to every cell.
        # m_p: (batch, N, N, cell) -> treat last dim as features
        # Result: (batch, N, N, d) then we collapse the N×N grid.
        # We take the mean over axis 2 (context axis) to get
        # one d-vector per personality: (batch, N, d) -> transpose
        # to (batch, d, N) to match the spec's Q,K ∈ R^(d×N).
        q_proj = self.w_q(m_p).mean(dim=2).transpose(1, 2)  # (batch, d, N)
        k_proj = self.w_k(m_p).mean(dim=2).transpose(1, 2)  # (batch, d, N)

        # ── Stage 4: actor-target blend ──────────────────────────
        # Cosine similarity between A and B
        a_norm   = F.normalize(a, dim=-1, eps=eps)            # (batch, d)
        b_norm   = F.normalize(b, dim=-1, eps=eps)            # (batch, d)
        cos_ab   = (a_norm * b_norm).sum(dim=-1)              # (batch,)

        # ABn = tanh(A * (1 - cos)) + tanh(B * cos)
        cos      = cos_ab.unsqueeze(-1)                       # (batch, 1)
        abn      = torch.tanh(a * (1.0 - cos)) + torch.tanh(b * cos)  # (batch, d)

        # Context fusion: ABD_j = tanh(w_j ⊙ ABn ⊙ D_j_projected + b_j)
        # Project context from q -> d
        d_proj   = self.context_proj(d_ctx)                   # (batch, N, d)
        abn_exp  = abn.unsqueeze(1).expand(-1, N, -1)         # (batch, N, d)
        abdj     = torch.tanh(
            self.abd_w * abn_exp * d_proj + self.abd_b
        )                                                      # (batch, N, d)

        # ── Stage 5: N independent mini-models ───────────────────
        # Softmax sharpening: multiply by N before softmax
        c_tilde  = F.softmax(float(N) * c, dim=-1)            # (batch, N, p)

        c_star_list  = []
        p_hat_list   = []
        c_dbl_list   = []

        for i in range(N):
            ci = c_tilde[:, i, :]                              # (batch, p)

            # Step B: affine + ReLU (weights unique to mini-model i)
            c_star_i = F.relu(
                F.linear(ci, self.w1[i], self.b1[i])
            )                                                  # (batch, p)

            # Probability head sees BOTH personality and situation:
            # concatenate c_star_i with abdj[:,i,:] so the vote
            # is conditioned on the current actor-target-context
            situation_i = abdj[:, i, :]                       # (batch, d)
            combined    = torch.cat([c_star_i, situation_i], dim=-1)  # (batch, p+d)

            logits  = F.linear(combined, self.w_prob[i], self.b_prob[i])  # (batch, 2)
            p_hat_i = F.softmax(logits, dim=-1)               # (batch, 2)

            # Confidence gating:
            # C_i'' = tanh(P+ * c_star_i) + tanh(P- * c_star_i)
            p_pos   = p_hat_i[:, 0:1]                         # (batch, 1)
            p_neg   = p_hat_i[:, 1:2]                         # (batch, 1)
            c_dbl_i = (
                torch.tanh(p_pos * c_star_i)
                + torch.tanh(p_neg * c_star_i)
            )                                                  # (batch, p)

            c_star_list.append(c_star_i)
            p_hat_list.append(p_hat_i)
            c_dbl_list.append(c_dbl_i)

        c_star         = torch.stack(c_star_list, dim=1)      # (batch, N, p)
        p_hat          = torch.stack(p_hat_list,  dim=1)      # (batch, N, 2)
        c_double_prime = torch.stack(c_dbl_list,  dim=1)      # (batch, N, p)

        return Outputs(
            m              = m,
            m_prime        = m_p,
            q_proj         = q_proj,
            k_proj         = k_proj,
            cos_ab         = cos_ab,
            abn            = abn,
            abdj           = abdj,
            c_tilde        = c_tilde,
            c_star         = c_star,
            c_double_prime = c_double_prime,
            p_hat          = p_hat,
        )

    def forward_stage4_abdj(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        d_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 4 only (spec / main PDF): ABn from actor–receiver, then per-slot fusion

            ABD_j = tanh(w_j ⊙ ABn ⊙ proj(D_j) + b_j).

        Does not run Stages 1–3 or 5 (no outer-product grid). Use this to build the
        same ``abd`` vectors that Stages 6–8 consume as ``ABD_j`` (one row ``j``).

        Parameters
        ----------
        a, b : (batch, d)
        d_ctx : (batch, N, q)

        Returns
        -------
        abdj : (batch, N, d)
        """
        cfg = self.cfg
        N, eps = cfg.n, cfg.eps
        if d_ctx.dim() != 3 or d_ctx.shape[0] != a.shape[0] or d_ctx.shape[1] != N:
            raise ValueError(
                f"Expected d_ctx (batch, {N}, q), got {tuple(d_ctx.shape)} "
                f"with batch {a.shape[0]}"
            )

        a_norm = F.normalize(a, dim=-1, eps=eps)
        b_norm = F.normalize(b, dim=-1, eps=eps)
        cos_ab = (a_norm * b_norm).sum(dim=-1, keepdim=True)
        abn = torch.tanh(a * (1.0 - cos_ab)) + torch.tanh(b * cos_ab)

        d_proj = self.context_proj(d_ctx)
        abn_exp = abn.unsqueeze(1).expand(-1, N, -1)
        return torch.tanh(self.abd_w * abn_exp * d_proj + self.abd_b)


def compute_abn(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Part 1 Step 2 (main (24).pdf): actor–receiver blend ``ABn`` in ``d``-space.

    Parameters
    ----------
    a, b : (d,) or (batch, d)
    """
    squeeze = False
    if a.dim() == 1:
        squeeze = True
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    a_norm = F.normalize(a, dim=-1, eps=eps)
    b_norm = F.normalize(b, dim=-1, eps=eps)
    cos_ab = (a_norm * b_norm).sum(dim=-1, keepdim=True)
    abn = torch.tanh(a * (1.0 - cos_ab)) + torch.tanh(b * cos_ab)
    return abn.squeeze(0) if squeeze else abn


# ─────────────────────────────────────────────────────────────────
def build_model(cfg: Config) -> Stages1to5:
    return Stages1to5(cfg)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick shape smoke-test
    cfg   = Config(n=10, d=64, p=64, q=64)
    model = build_model(cfg)

    batch = 4
    a     = torch.randn(batch, cfg.d)
    b     = torch.randn(batch, cfg.d)
    c     = torch.randn(batch, cfg.n, cfg.p)
    d_ctx = torch.randn(batch, cfg.n, cfg.q)

    out = model(a, b, c, d_ctx)

    print("m              :", out.m.shape)               # (4, 10, 10, 4096)
    print("m_prime        :", out.m_prime.shape)         # (4, 10, 10, 4096)
    print("q_proj         :", out.q_proj.shape)          # (4, 64, 10)
    print("k_proj         :", out.k_proj.shape)          # (4, 64, 10)
    print("cos_ab         :", out.cos_ab.shape)          # (4,)
    print("abn            :", out.abn.shape)             # (4, 64)
    print("abdj           :", out.abdj.shape)            # (4, 10, 64)
    print("c_tilde        :", out.c_tilde.shape)         # (4, 10, 64)
    print("c_star         :", out.c_star.shape)          # (4, 10, 64)
    print("c_double_prime :", out.c_double_prime.shape)  # (4, 10, 64)
    print("p_hat          :", out.p_hat.shape)           # (4, 10, 2)
    print("All shapes OK.")
