"""
build_entity_triplets.py

Builds data/entities.txt (anchor TAB positive TAB negative) from:
  1. Alliance v4.1 — defense/neutrality/nonaggression/entente → similarity scores
  2. Dyadic COW 4.0 — bilateral trade volume → normalized trade score

Combined similarity = (alliance * w_alliance) + (trade * w_trade)
Triplet thresholds: similarity > pos_threshold → positive, < neg_threshold → negative
"""

import csv
from collections import defaultdict
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
ALLIANCE_CSV = Path("data/alliance_v4.1_by_dyad_yearly.csv")
TRADE_CSV = Path("data/Dyadic_COW_4.0.csv")
OUTPUT_FILE = Path("data/entities.txt")

# ── Alliance type scores (user spec) ────────────────────────────────
# defense (Type I) → 1.0, neutrality (IIa) → 0.6, nonaggression (IIb) → 0.5, entente (III) → 0.4
ALLIANCE_SCORES = {
    "defense": 1.0,
    "neutrality": 0.6,
    "nonaggression": 0.5,
    "entente": 0.4,
}

# ── Weights (alliance + trade; diplomacy placeholder = 0) ────────────
W_ALLIANCE = 0.5
W_TRADE = 0.5

# ── Triplet thresholds ──────────────────────────────────────────────
POS_THRESHOLD = 0.5
NEG_THRESHOLD = 0.4
MIN_POSITIVES = 1
MIN_NEGATIVES = 3
MAX_NEGATIVES_PER_ANCHOR_POSITIVE = 25
USE_IMPLICIT_NEGATIVES = True  # dyads with no alliance/trade data → sim=0 (strong negative)


def _sanitize(name: str) -> str:
    """Strip tabs/newlines so output stays strictly anchor\\tpositive\\tnegative."""
    return " ".join(name.replace("\t", " ").replace("\n", " ").split())


def _dyad(c1: int, c2: int) -> tuple[int, int]:
    """Canonical dyad key (smaller ccode first)."""
    return (min(c1, c2), max(c1, c2))


def load_alliance_scores(path: Path) -> tuple[dict[tuple[int, int], float], dict[int, str]]:
    """Returns (dyad -> max alliance score), (ccode -> state_name)."""
    dyad_scores: dict[tuple[int, int], float] = defaultdict(float)
    ccode_to_name: dict[int, str] = {}

    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["ccode1"]), int(row["ccode2"])
            except (ValueError, KeyError):
                continue

            ccode_to_name[c1] = row.get("state_name1", "").strip() or ccode_to_name.get(c1, "")
            ccode_to_name[c2] = row.get("state_name2", "").strip() or ccode_to_name.get(c2, "")

            dyad = _dyad(c1, c2)
            score = 0.0
            for col, val in ALLIANCE_SCORES.items():
                if row.get(col) == "1":
                    score = max(score, val)
            if score > 0:
                dyad_scores[dyad] = max(dyad_scores[dyad], score)

    return dict(dyad_scores), ccode_to_name


def load_trade_scores(path: Path) -> dict[tuple[int, int], float]:
    """Returns dyad -> mean trade volume (excluding missing -9)."""
    dyad_totals: dict[tuple[int, int], list[float]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["ccode1"]), int(row["ccode2"])
                raw = row.get("smoothtotrade", "-9")
                val = float(raw)
            except (ValueError, KeyError):
                continue
            if val < 0:  # -9 = missing
                continue
            dyad = _dyad(c1, c2)
            dyad_totals[dyad].append(val)

    return {d: sum(v) / len(v) if v else 0.0 for d, v in dyad_totals.items()}


def minmax_normalize(scores: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Scale values to [0, 1]; if all same, return 0."""
    if not scores:
        return {}
    vals = [v for v in scores.values() if v is not None]
    if not vals:
        return {k: 0.0 for k in scores}
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def build_triplets(
    dyad_similarity: dict[tuple[int, int], float],
    ccode_to_name: dict[int, str],
    pos_thresh: float,
    neg_thresh: float,
) -> list[tuple[str, str, str]]:
    """Generate (anchor, positive, negative) triplets from dyad similarities."""
    state_positives: dict[int, set[int]] = defaultdict(set)

    all_ccodes = set(ccode_to_name)
    for (c1, c2), sim in dyad_similarity.items():
        if c1 not in all_ccodes or c2 not in all_ccodes:
            continue
        if sim > pos_thresh:
            state_positives[c1].add(c2)
            state_positives[c2].add(c1)

    def get_negatives(anchor: int) -> set[int]:
        neg_set = set()
        for other in all_ccodes:
            if other == anchor or other in state_positives.get(anchor, set()):
                continue
            dyad = _dyad(anchor, other)
            sim = dyad_similarity.get(dyad)
            if USE_IMPLICIT_NEGATIVES:
                sim = sim if sim is not None else 0.0
            if sim is not None and sim < neg_thresh:
                neg_set.add(other)
        return neg_set

    triplets: list[tuple[str, str, str]] = []
    for anchor in state_positives:
        pos_set = state_positives[anchor]
        neg_set = get_negatives(anchor)
        if len(pos_set) < MIN_POSITIVES or len(neg_set) < MIN_NEGATIVES:
            continue

        anchor_name = _sanitize(ccode_to_name.get(anchor, str(anchor)))
        for pos in pos_set:
            pos_name = _sanitize(ccode_to_name.get(pos, str(pos)))
            for neg in list(neg_set)[:MAX_NEGATIVES_PER_ANCHOR_POSITIVE]:
                neg_name = _sanitize(ccode_to_name.get(neg, str(neg)))
                triplets.append((anchor_name, pos_name, neg_name))

    return triplets


def main() -> None:
    print("Loading alliance data...")
    alliance_scores, ccode_to_name = load_alliance_scores(ALLIANCE_CSV)
    print(f"  {len(alliance_scores)} dyads with alliance, {len(ccode_to_name)} states")

    print("Loading trade data...")
    trade_raw = load_trade_scores(TRADE_CSV)
    trade_scores = minmax_normalize(trade_raw)
    print(f"  {len(trade_raw)} dyads with trade")

    # Merge: dyads in either source
    all_dyads = set(alliance_scores) | set(trade_scores)
    alliance_norm = minmax_normalize(alliance_scores) if alliance_scores else {}

    dyad_similarity: dict[tuple[int, int], float] = {}
    for d in all_dyads:
        a = alliance_norm.get(d, 0.0)
        t = trade_scores.get(d, 0.0)
        dyad_similarity[d] = W_ALLIANCE * a + W_TRADE * t

    print("Building triplets...")
    triplets = build_triplets(
        dyad_similarity, ccode_to_name, POS_THRESHOLD, NEG_THRESHOLD
    )
    print(f"  {len(triplets)} triplets")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for a, p, n in triplets:
            f.write(f"{a}\t{p}\t{n}\n")

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
