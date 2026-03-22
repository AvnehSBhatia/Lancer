"""
extract_mini_model_params.py

Returns only the ABn and Cn parameters for a specific mini-model index n
from the perspective_stages model (Stages1to5).

ABn: parameters that blend actor (A) and receiver (B) with context for model n:
     - abd_w[n], abd_b[n]  (context fusion for context j=n)
     - context_proj (shared, projects context q->d)

Cn: parameters for personality (C) processing in mini-model n:
     - w1[n], b1[n]  (affine on personality)
     - w_prob[n], b_prob[n]  (classifier head: takes [c_star || abdj] -> 2)

Usage:
    from extract_mini_model_params import get_mini_model_params
    params = get_mini_model_params(model, n=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MiniModelParams:
    """Parameters for a single mini-model n."""

    # ABn: actor-receiver blend + context fusion for this model
    abd_w: torch.Tensor  # (d,)
    abd_b: torch.Tensor  # (d,)

    # Cn: personality processing
    w1: torch.Tensor  # (p, p)
    b1: torch.Tensor  # (p,)

    # Classifier: [c_star || abdj] -> 2
    w_prob: torch.Tensor  # (2, p+d)
    b_prob: torch.Tensor  # (2,)

    # Shared context projection (needed for forward pass)
    context_proj_weight: torch.Tensor  # (d, q)
    context_proj_bias: torch.Tensor  # (d,)


def get_mini_model_params(model: nn.Module, n: int) -> MiniModelParams:
    """
    Extract ABn and Cn parameters for mini-model index n.

    Args:
        model: Stages1to5 model from perspective_stages
        n: mini-model index (0 to N-1)

    Returns:
        MiniModelParams with all parameters for that mini-model
    """
    cfg = model.cfg
    N = cfg.n
    if n < 0 or n >= N:
        raise ValueError(f"n must be in [0, {N-1}], got {n}")

    return MiniModelParams(
        abd_w=model.abd_w[n].detach().clone(),
        abd_b=model.abd_b[n].detach().clone(),
        w1=model.w1[n].detach().clone(),
        b1=model.b1[n].detach().clone(),
        w_prob=model.w_prob[n].detach().clone(),
        b_prob=model.b_prob[n].detach().clone(),
        context_proj_weight=model.context_proj.weight.detach().clone(),
        context_proj_bias=model.context_proj.bias.detach().clone(),
    )


def get_mini_model_params_as_state_dict(model: nn.Module, n: int) -> dict[str, torch.Tensor]:
    """
    Return ABn and Cn as a state_dict for easy saving/loading.

    Keys: abd_w, abd_b, w1, b1, w_prob, b_prob, context_proj_weight, context_proj_bias
    """
    p = get_mini_model_params(model, n)
    return {
        "abd_w": p.abd_w,
        "abd_b": p.abd_b,
        "w1": p.w1,
        "b1": p.b1,
        "w_prob": p.w_prob,
        "b_prob": p.b_prob,
        "context_proj_weight": p.context_proj_weight,
        "context_proj_bias": p.context_proj_bias,
    }


def load_mini_model_params(path: Path) -> MiniModelParams:
    """Load mini-model params from a .pt file saved by get_mini_model_params_as_state_dict."""
    state = torch.load(path, weights_only=True)
    return MiniModelParams(
        abd_w=state["abd_w"],
        abd_b=state["abd_b"],
        w1=state["w1"],
        b1=state["b1"],
        w_prob=state["w_prob"],
        b_prob=state["b_prob"],
        context_proj_weight=state["context_proj_weight"],
        context_proj_bias=state["context_proj_bias"],
    )


def forward_mini_model_n(
    params: MiniModelParams,
    abn: torch.Tensor,  # (batch, d)
    c: torch.Tensor,  # (batch, p)  personality for this model
    d_ctx: torch.Tensor,  # (batch, q)  context for this model
) -> torch.Tensor:
    """
    Run forward pass for a single mini-model n given precomputed abn and its c, d_ctx.

    Returns p_hat: (batch, 2)
    """
    # Project context: (batch, q) -> (batch, d)
    d_proj = torch.nn.functional.linear(d_ctx, params.context_proj_weight, params.context_proj_bias)
    # ABDj for this j=n: tanh(w * abn * d_proj + b)
    abdj = torch.tanh(params.abd_w * abn * d_proj + params.abd_b)
    # Cn: c_star = ReLU(W1 @ c + b1)
    c_star = torch.nn.functional.relu(
        torch.nn.functional.linear(c, params.w1, params.b1)
    )
    # Concatenate and classify
    combined = torch.cat([c_star, abdj], dim=-1)  # (batch, p+d)
    logits = torch.nn.functional.linear(combined, params.w_prob, params.b_prob)
    return torch.nn.functional.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from apollo.perspective_stages import Config, build_model

    cfg = Config(n=10, d=64, p=64, q=64)
    model = build_model(cfg)

    params = get_mini_model_params(model, n=3)
    print("Mini-model 3 params:")
    for k, v in vars(params).items():
        print(f"  {k}: {v.shape}")

    # Save/load roundtrip
    state = get_mini_model_params_as_state_dict(model, 3)
    torch.save(state, "/tmp/mini_model_3.pt")
    loaded = load_mini_model_params(Path("/tmp/mini_model_3.pt"))
    print("Load OK.")
