"""
Inference-only runner: precomputes all embeddings, then runs model per personality.

Actor, receiver, context embedded once. Personalities evenly sampled and preembedded.

Usage: python run_full_pipeline.py
"""

from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Hardcoded inputs ────────────────────────────────────────────────────────
actor = "United States of America"
iraq = "Iraq"
context = (
    "Year 2003. United States of America CINC score 0.1518, "
    "United Arab Emirates CINC score 0.0022. No active dispute between "
    "United States of America and United Arab Emirates. No alliance. Bilateral trade volume: 0.006."
)
MAX_PERSONALITIES = 100  # evenly sample this many from vault; None = use all 100
QUIET = False


def _evenly_sample(lst: list, k: int) -> list:
    """Return k items evenly spaced from lst."""
    n = len(lst)
    if k >= n:
        return list(lst)
    indices = [int(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [lst[i] for i in indices]


def main() -> None:
    from personality_bank import PERSONALITY_BANK
    from predict_from_strings import (
        embed_context,
        embed_entity,
        embed_personality,
        predict_from_embeddings,
    )

    # ── All 4 inputs at top ─────────────────────────────────────────────────
    all_personalities = list(PERSONALITY_BANK)
    k = max(1, MAX_PERSONALITIES) if MAX_PERSONALITIES else len(all_personalities)
    personality = _evenly_sample(all_personalities, k)

    # ── Precompute all embeddings ───────────────────────────────────────────
    t_start = time.perf_counter()

    a_emb = embed_entity(actor)           # (1, 64)
    b_emb = embed_entity(iraq)            # (1, 64)
    ctx_emb = embed_context(context)      # (1, 64)
    pers_embs = [embed_personality(s) for s in personality]  # list of (1, 64)

    n = len(personality)
    if not QUIET:
        print(f"Working directory: {ROOT}")
        print(f"Precomputed: actor, iraq, context, {n} personalities")
        print(f"Running {n} model forwards")
        print("=" * 70)

    results = []
    for i, (pers_str, pers_emb) in enumerate(zip(personality, pers_embs)):
        try:
            r = predict_from_embeddings(a_emb, b_emb, ctx_emb, pers_emb)
            results.append((pers_str, r))
        except Exception as e:
            if not QUIET:
                print(f"[{i + 1}/{n}] FAILED: {pers_str[:60]}... -> {e}")
            results.append((pers_str, None))

        if not QUIET:
            p = results[-1][1]
            if p is not None:
                inv = f"{p.invade_prob:.3f}"
                noinv = f"{p.not_invade_prob:.3f}"
                print(f"[{i + 1}/{n}] {pers_str[:60]}{'...' if len(pers_str) > 60 else ''}")
                print(f"  P(invade)={inv}  P(not)={noinv}")

    elapsed = time.perf_counter() - t_start
    failures = sum(1 for _, r in results if r is None)

    if not QUIET:
        print()
        print("=" * 70)

    print(f"Full pipeline: precompute + {n} model forwards")
    print(f"  Total: {elapsed:.2f} s")
    print(f"  Per run: {elapsed/n*1000:.2f} ms")
    print(f"  Throughput: {n/elapsed:.0f} predictions/sec")
    if failures:
        print(f"  Failures: {failures}/{n}")

    if failures:
        raise SystemExit(f"Finished with {failures} failed run(s) out of {n}.")


if __name__ == "__main__":
    main()
