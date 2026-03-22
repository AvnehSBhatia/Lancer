"""
Write data/personalities.txt (triplet lines) from personalities_100.json.

Each line: anchor TAB positive TAB negative (ring + half-offset negatives) so
train_personality_embeddings.py learns a 64-d vector for every vault string.

Run after scripts/generate_personalities_100.py and before scripts/train_personality_embeddings.py.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
VAULT_PATH = _REPO_ROOT / "apollo" / "personalities_100.json"
OUT_PATH = _REPO_ROOT / "data" / "personalities.txt"


def _sanitize(s: str) -> str:
    return " ".join(s.replace("\t", " ").replace("\n", " ").split())


def main() -> None:
    if not VAULT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing vault: {VAULT_PATH} — run scripts/generate_personalities_100.py"
        )
    data = json.loads(VAULT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Vault must be a non-empty JSON list of strings")
    names = [_sanitize(str(s)) for s in data]
    n = len(names)
    neg_off = max(1, n // 2)
    lines: list[str] = []
    for i, anchor in enumerate(names):
        pos = names[(i + 1) % n]
        neg = names[(i + neg_off) % n]
        lines.append(f"{anchor}\t{pos}\t{neg}\n")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} triplets to {OUT_PATH}")


if __name__ == "__main__":
    main()
