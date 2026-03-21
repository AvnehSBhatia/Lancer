"""
predict_from_strings.py

Takes strings as input, embeds them, runs through the perspective model,
and returns the prediction vector, ABn vector, and Cn vector.

Usage:
    from predict_from_strings import predict

    result = predict(
        actor="United States of America",
        receiver="Iraq",
        context="Year 2003. United States of America CINC score 0.15, Iraq...",
        personality="military analyst",
    )
    # result.prediction: [P(invade), P(not invade)]
    # result.abn: (64,) actor-receiver blend
    # result.cn: (64,) processed personality (c_star)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Paths
ENTITY_DIR = Path("entity_embeddings")
CONTEXT_DIR = Path("context_embeddings")
PERSONALITY_DIR = Path("personality_embeddings")
MODEL_PATH = Path("data/perspective_stages.pt")

N, D, P, Q = 1, 64, 64, 64  # must match train_perspective_stages


@dataclass
class PredictionResult:
    prediction: torch.Tensor  # (2,) [P(invade), P(not invade)]
    abn: torch.Tensor        # (d,) actor-receiver blend
    cn: torch.Tensor         # (p,) processed personality (c_star)
    invade_prob: float
    not_invade_prob: float


# Lazy-loaded singletons
_st = None
_entity_converter = None
_entity_embs = None
_entity_vocab = None
_context_converter = None
_context_embs = None
_context_vocab = None
_personality_converter = None
_personality_embs = None
_personality_vocab = None


def _get_minilm(text: str) -> torch.Tensor:
    global _st
    if _st is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    x = _st.encode(text, convert_to_numpy=True)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def _embed_entity(text: str) -> torch.Tensor:
    """Embed entity string to 64-d. Vocab lookup or MiniLM->converter."""
    global _entity_converter, _entity_embs, _entity_vocab
    if _entity_embs is None:
        _entity_converter, _entity_embs, _entity_vocab = _load_entity()
    if text in _entity_embs:
        return _entity_embs[text].float().unsqueeze(0)
    minilm = _get_minilm(text)
    with torch.no_grad():
        return _entity_converter.minilm_to_ours_vec(minilm)


def _embed_context(text: str) -> torch.Tensor:
    """Embed context string to 64-d."""
    global _context_converter, _context_embs, _context_vocab
    if _context_embs is None:
        _context_converter, _context_embs, _context_vocab = _load_context()
    if text in _context_embs:
        return _context_embs[text].float().unsqueeze(0)
    minilm = _get_minilm(text)
    with torch.no_grad():
        return _context_converter.minilm_to_ours_vec(minilm)


def _embed_personality(text: str) -> torch.Tensor:
    """Embed personality string to 64-d."""
    global _personality_converter, _personality_embs, _personality_vocab
    if _personality_embs is None:
        _personality_converter, _personality_embs, _personality_vocab = _load_personality()
    if text in _personality_embs:
        return _personality_embs[text].float().unsqueeze(0)
    minilm = _get_minilm(text)
    with torch.no_grad():
        return _personality_converter.minilm_to_ours_vec(minilm)


def _load_entity():
    from entity_minilm_converter import load_converter
    return load_converter(ENTITY_DIR)


def _load_context():
    from context_minilm_converter import load_converter
    return load_converter(CONTEXT_DIR)


def _load_personality():
    from personality_minilm_converter import load_converter
    return load_converter(PERSONALITY_DIR)


def _load_model(path: Path = MODEL_PATH):
    from perspective_stages import Config, build_model
    cfg = Config(n=N, d=D, p=P, q=Q)
    model = build_model(cfg)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def embed_entity(text: str) -> torch.Tensor:
    """Public: embed entity string to (1, 64)."""
    return _embed_entity(text)


def embed_context(text: str) -> torch.Tensor:
    """Public: embed context string to (1, 64)."""
    return _embed_context(text)


def embed_personality(text: str) -> torch.Tensor:
    """Public: embed personality string to (1, 64)."""
    return _embed_personality(text)


def predict_from_embeddings(
    a: torch.Tensor,      # (1, 64) or (64,)
    b: torch.Tensor,      # (1, 64) or (64,)
    ctx: torch.Tensor,    # (1, 64) or (64,)
    pers: torch.Tensor,   # (1, 64) or (64,)
    model_path: Optional[Path] = None,
) -> PredictionResult:
    """Run model with precomputed embeddings. No embedding step."""
    model = _load_model(model_path or MODEL_PATH)
    a = a.flatten().unsqueeze(0) if a.dim() == 1 else a
    b = b.flatten().unsqueeze(0) if b.dim() == 1 else b
    ctx = ctx.flatten().unsqueeze(0) if ctx.dim() == 1 else ctx
    pers = pers.flatten().unsqueeze(0) if pers.dim() == 1 else pers
    c = pers.unsqueeze(1)    # (1, 1, 64)
    d_ctx = ctx.unsqueeze(1)  # (1, 1, 64)
    with torch.no_grad():
        out = model(a, b, c, d_ctx)
    p_hat = out.p_hat[0, 0, :]
    abn = out.abn[0, :]
    cn = out.c_star[0, 0, :]
    return PredictionResult(
        prediction=p_hat,
        abn=abn,
        cn=cn,
        invade_prob=p_hat[0].item(),
        not_invade_prob=p_hat[1].item(),
    )


def predict(
    actor: str,
    receiver: str,
    context: str,
    personality: str,
    model_path: Optional[Path] = None,
) -> PredictionResult:
    """
    Embed strings, run model, return prediction, ABn, and Cn vectors.

    Args:
        actor: e.g. "United States of America"
        receiver: e.g. "Iraq"
        context: e.g. "Year 2003. United States..."
        personality: e.g. "military analyst"
        model_path: optional path to model weights (default: data/perspective_stages.pt)

    Returns:
        PredictionResult with prediction, abn, cn, invade_prob, not_invade_prob
    """
    model = _load_model(model_path or MODEL_PATH)

    a = _embed_entity(actor)               # (1, 64)
    b = _embed_entity(receiver)        # (1, 64)
    ctx = _embed_context(context)      # (1, 64)
    pers = _embed_personality(personality)  # (1, 64)

    c = pers.unsqueeze(1)    # (1, 1, 64)
    d_ctx = ctx.unsqueeze(1)  # (1, 1, 64)

    with torch.no_grad():
        out = model(a, b, c, d_ctx)

    p_hat = out.p_hat[0, 0, :]  # (2,)
    abn = out.abn[0, :]         # (64,)
    cn = out.c_star[0, 0, :]    # (64,)

    return PredictionResult(
        prediction=p_hat,
        abn=abn,
        cn=cn,
        invade_prob=p_hat[0].item(),
        not_invade_prob=p_hat[1].item(),
    )


if __name__ == "__main__":
    import sys

    actor = sys.argv[1] if len(sys.argv) > 1 else "United States of America"
    receiver = sys.argv[2] if len(sys.argv) > 2 else "Iraq"
    context = sys.argv[3] if len(sys.argv) > 3 else (
        "Year 2003. United States of America CINC score 0.1518, "
        "United Arab Emirates CINC score 0.0022. No active dispute."
    )
    personality = sys.argv[4] if len(sys.argv) > 4 else "military analyst"

    r = predict(actor=actor, receiver=receiver, context=context, personality=personality)
    print("Prediction:", r.prediction.tolist())
    print("  P(invade):", f"{r.invade_prob:.4f}")
    print("  P(not invade):", f"{r.not_invade_prob:.4f}")
    print("ABn:", r.abn.tolist()[:8], "...")
    print("Cn:", r.cn.tolist()[:8], "...")
