"""
End-to-end inference aligned with main (24).pdf (Part 1 sweep + Part 2 aggregation).

Part 1 — For each vault personality i: embed (sA, sB, sC_i, sD) with the same global
context sD, run ``Stages1to5`` (``N=1`` checkpoint) → ``p_hat_i`` (repo’s Step 5 wiring).

Part 2 — Stack raw personality embeddings ``C_i ∈ R^64`` as ``C`` (N, p), collect
``p_hat`` (N, 2), form ``ABn`` from (A, B) via ``compute_abn``, take ``D_n`` from
context embedding, run ``PerspectiveEventHead`` once → ``[y+, y-]`` (final action
probability vs not).

Weights:
  - ``data/perspective_stages.pt`` — Part 1 (existing).
  - ``data/perspective_event_head.pt`` — Part 2 (optional; random init if missing).

Usage (from repo root):
<<<<<<< Updated upstream:scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --max-personalities 20
    python scripts/run_full_pipeline.py --agg-head path/to/perspective_event_head.pt
=======
    python run_full_pipeline.py
    python run_full_pipeline.py --max-personalities 20
    python run_full_pipeline.py --agg-head path/to/perspective_event_head.pt
    python run_full_pipeline.py --summary  # add Featherless AI summary (needs FEATHERLESS_API_KEY)
>>>>>>> Stashed changes:run_full_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apollo.paths import PERSPECTIVE_EVENT_HEAD_PT, REPO_ROOT

DEFAULT_AGG_HEAD = PERSPECTIVE_EVENT_HEAD_PT

P_DIM = 64
Q_DIM = 64

DEFAULT_ACTOR = "United States of America"
DEFAULT_RECEIVER = "Iraq"
DEFAULT_CONTEXT = (
    "Year 2003. United States of America CINC score 0.1518, "
    "United Arab Emirates CINC score 0.0022. No active dispute between "
    "United States of America and United Arab Emirates. No alliance. Bilateral trade volume: 0.006."
)


def _evenly_sample(lst: list, k: int) -> list:
    """Return k items evenly spaced from lst."""
    n = len(lst)
    if k >= n:
        return list(lst)
    if k < 1:
        return []
    indices = [int(round(i * (n - 1) / (k - 1))) for i in range(k)] if k > 1 else [0]
    return [lst[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="main (24).pdf: Part 1 × N + Part 2 aggregation.")
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument(
        "--max-personalities",
        type=int,
        default=None,
        metavar="K",
        help="Evenly subsample K vault personalities; default = all.",
    )
    parser.add_argument(
        "--agg-head",
        type=Path,
        default=DEFAULT_AGG_HEAD,
        help="PerspectiveEventHead state_dict (.pt); optional.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate natural-language summary via Featherless AI (requires FEATHERLESS_API_KEY).",
    )
    args = parser.parse_args()

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

    all_personalities = list(PERSONALITY_BANK)
    if args.max_personalities is None:
        personality = all_personalities
    else:
        k = max(1, min(args.max_personalities, len(all_personalities)))
        personality = _evenly_sample(all_personalities, k)

    n = len(personality)
    if n == 0:
        raise SystemExit("No personalities selected.")

    stages = _load_model(MODEL_PATH)

    t_start = time.perf_counter()

    a_emb = embed_entity(args.actor)
    b_emb = embed_entity(args.receiver)
    ctx_emb = embed_context(args.context)

    pers_embs = [embed_personality(s) for s in personality]

    # Part 2 Step 6 uses raw personality embeddings C_i (main (24).pdf notation).
    C = torch.stack([p.flatten() for p in pers_embs], dim=0)

    p_hat_rows: list[torch.Tensor] = []
    for pers_str, pers_t in zip(personality, pers_embs):
        try:
            r = predict_from_embeddings(
                a_emb, b_emb, ctx_emb, pers_t, model=stages
            )
            p_hat_rows.append(r.prediction.detach())
        except Exception as e:
            if not args.quiet:
                print(f"FAILED Part 1: {pers_str[:60]}... -> {e}")
            raise

    p_hat = torch.stack(p_hat_rows, dim=0)

    abn = compute_abn(a_emb.flatten(), b_emb.flatten())
    d_n = ctx_emb.flatten()

    head = PerspectiveEventHead(n=n, p=P_DIM, q=Q_DIM)
    if args.agg_head.is_file():
        head.load_state_dict(torch.load(args.agg_head, weights_only=True))
        if not args.quiet:
            print(f"Loaded aggregation head: {args.agg_head}")
    else:
        if not args.quiet:
            print(
                f"Warning: no aggregation checkpoint at {args.agg_head}; "
                "Part 2 uses random weights (train or add perspective_event_head.pt)."
            )
    head.eval()

    with torch.no_grad():
        y = head(C, p_hat, abn, d_n)

    elapsed = time.perf_counter() - t_start

    if not args.quiet:
        print(f"Repository root: {REPO_ROOT}")
        print(f"Part 1: {n} forwards (Stages1to5), Part 2: PerspectiveEventHead (main (24).pdf)")
        print("=" * 70)
        for i, pers_str in enumerate(personality):
            ph = p_hat[i]
            print(
                f"[{i + 1}/{n}] {pers_str[:56]}{'...' if len(pers_str) > 56 else ''}  "
                f"p_hat=({ph[0]:.3f},{ph[1]:.3f})"
            )
        print("=" * 70)

    yp, ym = y[0].item(), y[1].item()
    print(f"Aggregated y (Part 2):  y_plus={yp:.4f}  y_minus={ym:.4f}")
    print(
        f"Timing: {elapsed:.2f} s total, {elapsed / n * 1000:.1f} ms per Part-1 forward "
        f"({n / elapsed:.1f} mini-models/s)"
    )

    if args.summary:
        from generate_summary import generate_summary

        try:
            summary = generate_summary(
                actor=args.actor,
                receiver=args.receiver,
                context=args.context,
                y_plus=yp,
                y_minus=ym,
            )
            print("\n" + "=" * 70)
            print("Summary (Featherless AI):")
            print(summary)
        except Exception as e:
            if not args.quiet:
                print(f"\nSummary generation failed: {e}")


if __name__ == "__main__":
    main()
