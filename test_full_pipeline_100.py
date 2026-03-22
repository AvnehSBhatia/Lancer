"""
Test full pipeline: 100 mini-models (Part 1) + aggregate head (Part 2).

Runnable as:
    python test_full_pipeline_100.py
    pytest test_full_pipeline_100.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
P_DIM = 64
Q_DIM = 64
DEFAULT_ACTOR = "Will China invade Taiwan in 2025?"
DEFAULT_RECEIVER = "Taiwan"
DEFAULT_CONTEXT = (
    "Asia 2025"
)


def test_full_pipeline_100_models_plus_aggregate() -> None:
    """Run Part 1 (100 Stages1to5 forwards) + Part 2 (PerspectiveEventHead)."""
    from apollo.perspective_event_head import PerspectiveEventHead
    from apollo.personality_bank import PERSONALITY_BANK
    from apollo.perspective_stages import compute_abn
    from apollo.predict_from_strings import (
        MODEL_PATH,
        _load_model,
        embed_context,
        embed_entity,
        embed_personality,
        predict_from_embeddings,
    )

    personalities = list(PERSONALITY_BANK)
    assert len(personalities) == 100, f"Expected 100 personalities, got {len(personalities)}"

    stages = _load_model(MODEL_PATH)
    a_emb = embed_entity(DEFAULT_ACTOR)
    b_emb = embed_entity(DEFAULT_RECEIVER)
    ctx_emb = embed_context(DEFAULT_CONTEXT)

    pers_embs = [embed_personality(s) for s in personalities]
    C = torch.stack([p.flatten() for p in pers_embs], dim=0)

    t0 = time.perf_counter()
    p_hat_rows = []
    for pers_str, pers_t in zip(personalities, pers_embs):
        r = predict_from_embeddings(a_emb, b_emb, ctx_emb, pers_t, model=stages)
        p_hat_rows.append(r.prediction.detach())
    p_hat = torch.stack(p_hat_rows, dim=0)
    t1 = time.perf_counter()

    abn = compute_abn(a_emb.flatten(), b_emb.flatten())
    d_n = ctx_emb.flatten()

    head = PerspectiveEventHead(n=100, p=P_DIM, q=Q_DIM)
    agg_path = ROOT / "data" / "perspective_event_head.pt"
    if agg_path.is_file():
        head.load_state_dict(torch.load(agg_path, weights_only=True))
    head.eval()

    with torch.no_grad():
        y = head(C, p_hat, abn, d_n)

    t2 = time.perf_counter()

    # Assertions
    assert C.shape == (100, P_DIM), f"C shape {C.shape}"
    assert p_hat.shape == (100, 2), f"p_hat shape {p_hat.shape}"
    assert torch.allclose(p_hat.sum(dim=1), torch.ones(100), atol=1e-5)
    assert y.shape == (2,), f"y shape {y.shape}"
    assert torch.allclose(y.sum(), torch.tensor(1.0), atol=1e-5)
    assert (y >= 0).all() and (y <= 1).all()

    # Optional: print summary when run directly
    if __name__ == "__main__":
        part1_ms = (t1 - t0) * 1000
        part2_ms = (t2 - t1) * 1000
        print(f"Part 1 (100 mini-models): {part1_ms:.1f} ms ({part1_ms/100:.2f} ms/model)")
        print(f"Part 2 (aggregate head):   {part2_ms:.1f} ms")
        print(f"y_plus={y[0].item():.4f}  y_minus={y[1].item():.4f}")
        print("PASS: full pipeline (100 models + aggregate)")


if __name__ == "__main__":
    test_full_pipeline_100_models_plus_aggregate()
