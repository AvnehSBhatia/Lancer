"""
build_personality_triplets.py

Builds data/personality_triplets.txt (anchor TAB positive TAB negative) from
Nemotron-Personas arrow files. Uses same-person multi-view: anchor and positive
are different persona aspects (e.g., persona + professional_persona) from the
same row; negative is a persona from a different row.
"""

import random
from pathlib import Path

import pyarrow.ipc as ipc

# ── Paths ───────────────────────────────────────────────────────────
ARROW_FILES = [
    Path("data/nemotron-personas-train-00000-of-00010.arrow"),
    Path("data/nemotron-personas-train-00001-of-00010.arrow"),
]
OUTPUT_FILE = Path("data/personality_triplets.txt")

# ── Persona fields (6 narrative aspects per person) ───────────────────
ANCHOR_FIELD = "persona"  # main description
POSITIVE_FIELDS = [
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
]

# ── Sampling ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
MAX_TRIPLETS_PER_ROW = 5  # one per positive field
MIN_TEXT_LEN = 50  # skip empty or too-short persona texts


def _sanitize(s: str) -> str:
    """Strip tabs/newlines so output stays strictly anchor\\tpositive\\tnegative."""
    return " ".join(s.replace("\t", " ").replace("\n", " ").split())


def load_personas(paths: list[Path]) -> list[dict[str, str]]:
    """Load all persona rows from arrow files into list of dicts."""
    rows = []
    for path in paths:
        if not path.exists():
            print(f"  Skipping missing: {path}")
            continue
        with ipc.open_stream(path) as r:
            table = r.read_all()
            for i in range(len(table)):
                row = {}
                for col in table.column_names:
                    val = table.column(col)[i]
                    if val is None:
                        row[col] = ""
                    else:
                        row[col] = str(val).strip()
                rows.append(row)
    return rows


def build_triplets(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Generate (anchor, positive, negative) triplets."""
    random.seed(RANDOM_SEED)
    n = len(rows)
    triplets = []

    for i, row in enumerate(rows):
        anchor = row.get(ANCHOR_FIELD, "")
        if not anchor or len(anchor) < MIN_TEXT_LEN:
            continue

        # Collect valid positives from same row
        positives = []
        for field in POSITIVE_FIELDS:
            pos = row.get(field, "")
            if pos and len(pos) >= MIN_TEXT_LEN and pos != anchor:
                positives.append(pos)

        if not positives:
            continue

        # Negative: persona from a different row
        j = random.randint(0, n - 1)
        attempts = 0
        while j == i and attempts < n:
            j = random.randint(0, n - 1)
            attempts += 1
        if j == i:
            j = (i + 1) % n

        neg_candidates = [
            rows[j].get(f) for f in [ANCHOR_FIELD] + POSITIVE_FIELDS
        ]
        neg = next((x for x in neg_candidates if x and len(x) >= MIN_TEXT_LEN), None)
        if not neg:
            continue

        for pos in positives[:MAX_TRIPLETS_PER_ROW]:
            triplets.append((anchor, pos, neg))

    return triplets


def main() -> None:
    print("Loading Nemotron-Personas arrow files...")
    rows = load_personas(ARROW_FILES)
    print(f"  {len(rows)} rows")

    print("Building personality triplets...")
    triplets = build_triplets(rows)
    print(f"  {len(triplets)} triplets")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for a, p, n in triplets:
            sa, sp, sn = _sanitize(a), _sanitize(p), _sanitize(n)
            key = (sa[:100], sp[:100], sn[:100])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{sa}\t{sp}\t{sn}\n")
            written += 1

    print(f"Wrote {OUTPUT_FILE} ({written} unique triplets)")


if __name__ == "__main__":
    main()
