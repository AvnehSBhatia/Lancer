"""
build_context_triplets.py

Builds data/contexts.txt (anchor TAB positive TAB negative) from rich
context strings assembled from NMC, dyadic MID, alliance, trade data.

Pipeline: build_context_string -> build_all_contexts -> embed_contexts
          -> triplet generation from structural similarity
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apollo.context_data import (
    check_mid_between,
    ccode_to_name,
    get_active_mids,
    get_active_wars,
    get_alliances,
    get_available_pairs_years,
    get_bilateral_trade,
    get_capabilities,
    get_diplomatic_level,
)
from apollo.paths import DATA_DIR
OUTPUT_STRINGS = DATA_DIR / "context_strings.csv"
OUTPUT_TRIPLETS = DATA_DIR / "contexts.txt"
MAX_WORDS = 100  # approximate token truncation

# Triplet similarity: same (has_mid, has_alliance) + same trade tertile
POS_THRESHOLD = 0.6  # structural similarity threshold
NEG_THRESHOLD = 0.3
MIN_POSITIVES = 1
MIN_NEGATIVES = 2
MAX_PAIRS = 5000  # cap for memory; O(n^2) triplet gen


def _dyad(c1: int, c2: int) -> tuple[int, int]:
    return (min(c1, c2), max(c1, c2))


def build_context_string(year: int, country_a: int, country_b: int) -> str:
    """
    Assembles a context string for (country_a, country_b, year).
    country_a, country_b are COW ccodes.
    """
    parts = []
    na = ccode_to_name(country_a)
    nb = ccode_to_name(country_b)

    parts.append(f"Year {year}. ")

    a_caps = get_capabilities(country_a, year)
    b_caps = get_capabilities(country_b, year)
    if a_caps is not None and b_caps is not None:
        parts.append(f"{na} CINC score {a_caps['cinc']:.4f}, {nb} CINC score {b_caps['cinc']:.4f}. ")
    else:
        parts.append(f"{na} and {nb}. ")

    a_b_mid = check_mid_between(country_a, country_b, year)
    if a_b_mid is not None:
        parts.append(f"Active dispute between {na} and {nb}, hostility level {a_b_mid[0]}. ")
    else:
        parts.append(f"No active dispute between {na} and {nb}. ")

    a_all = get_alliances(country_a, year)
    shared_types = [t for p, t in a_all if p == country_b]
    if shared_types:
        parts.append(f"{na} and {nb} share {shared_types[0]} alliance. ")
    else:
        parts.append(f"No alliance between {na} and {nb}. ")

    active_wars = get_active_wars(year)
    if active_wars:
        parts.append(f"Active wars in {year}: {len(active_wars)} ongoing. ")
    # else omit

    trade = get_bilateral_trade(country_a, country_b, year)
    parts.append(f"Bilateral trade volume: {trade:.3f}. ")

    dip = get_diplomatic_level(country_a, country_b, year)
    parts.append(f"Diplomatic relations: {dip}. ")

    s = "".join(parts)
    words = s.split()[:MAX_WORDS]
    return " ".join(words)


def build_all_contexts(pairs: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], str]:
    out = {}
    for c1, c2, yr in pairs:
        out[(c1, c2, yr)] = build_context_string(yr, c1, c2)
    return out


def _structural_features(c1: int, c2: int, yr: int) -> tuple[bool, bool, int]:
    """(has_mid, has_alliance, trade_tertile 0/1/2)."""
    has_mid = check_mid_between(c1, c2, yr) is not None
    a_all = get_alliances(c1, yr)
    b_partners = {p for p, _ in get_alliances(c2, yr)}
    has_alliance = any(p == c2 for p, _ in a_all) or c1 in b_partners
    trade = get_bilateral_trade(c1, c2, yr)
    tertile = 0 if trade < 0.33 else (1 if trade < 0.67 else 2)
    return (has_mid, has_alliance, tertile)


def build_triplets(
    contexts: dict[tuple[int, int, int], str],
) -> list[tuple[str, str, str]]:
    """Generate (anchor, positive, negative) from structural similarity."""
    keys = list(contexts.keys())
    if len(keys) < 10:
        return []

    feats = {k: _structural_features(k[0], k[1], k[2]) for k in keys}
    by_feat: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_feat[feats[k]].append(i)

    triplets = []
    for i, anchor_key in enumerate(keys):
        fa = feats[anchor_key]
        anchor_str = contexts[anchor_key]
        pos_idxs = [j for j in by_feat.get(fa, []) if j != i]
        neg_feats = [f for f in by_feat if f != fa]
        neg_idxs = []
        for f in neg_feats[:5]:
            neg_idxs.extend(by_feat[f][:20])
        neg_idxs = neg_idxs[:100]

        if len(pos_idxs) < MIN_POSITIVES or len(neg_idxs) < MIN_NEGATIVES:
            continue
        for j in pos_idxs[:5]:
            pos_str = contexts[keys[j]]
            for k in neg_idxs[:3]:
                neg_str = contexts[keys[k]]
                triplets.append((anchor_str, pos_str, neg_str))

    return triplets


def main() -> None:
    print("Loading available (country_a, country_b, year) pairs...")
    pairs = list(get_available_pairs_years())
    if len(pairs) > MAX_PAIRS:
        import random
        random.Random(42).shuffle(pairs)
        pairs = pairs[:MAX_PAIRS]
    print(f"  {len(pairs)} pairs")

    print("Building context strings...")
    contexts = build_all_contexts(pairs)
    print(f"  {len(contexts)} context strings")

    print("Generating triplets...")
    triplets = build_triplets(contexts)
    print(f"  {len(triplets)} triplets")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw strings for inspection
    with open(OUTPUT_STRINGS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ccode_a", "ccode_b", "year", "context"])
        for (c1, c2, yr), s in list(contexts.items())[:5000]:
            w.writerow([c1, c2, yr, s[:500]])
    print(f"  Wrote {OUTPUT_STRINGS} (sample)")

    # Save triplets; sanitize to avoid tabs in content
    def _sanitize(s: str) -> str:
        return " ".join(s.replace("\t", " ").replace("\n", " ").split())

    with open(OUTPUT_TRIPLETS, "w", encoding="utf-8", newline="\n") as f:
        seen = set()
        for a, p, n in triplets:
            key = (_sanitize(a)[:80], _sanitize(p)[:80], _sanitize(n)[:80])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{_sanitize(a)}\t{_sanitize(p)}\t{_sanitize(n)}\n")

    print(f"Wrote {OUTPUT_TRIPLETS} ({len(seen)} unique triplets)")


def embed_contexts(
    contexts: dict[tuple[int, int, int], str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[tuple[int, int, int], "np.ndarray"]:
    """
    Embed context strings with a sentence-transformer model.
    Returns dict mapping (ccode_a, ccode_b, year) -> vector.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    keys = list(contexts.keys())
    strings = [contexts[k] for k in keys]
    vecs = model.encode(strings, convert_to_numpy=True)
    return {k: vecs[i] for i, k in enumerate(keys)}


if __name__ == "__main__":
    main()
