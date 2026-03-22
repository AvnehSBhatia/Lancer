"""
Shared personality bank for the N-perspective ensemble.

Vault: personalities_100.json next to this module (100 strings). Regenerate with:
    python scripts/generate_personalities_100.py

N equals the vault size: every string is one mini-model slot (Stages 1–5 independent
per persona, then Stages 6–8 fuse all N).

ALL_PERSONALITIES and PERSONALITY_BANK load the JSON lazily on first access so
imports that only need N, D, P, Q stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path

_VAULT_PATH = Path(__file__).resolve().parent / "personalities_100.json"
_EXPECTED_TOTAL = 100

_vault_cache: tuple[str, ...] | None = None


def _load_vault() -> tuple[str, ...]:
    if not _VAULT_PATH.is_file():
        raise FileNotFoundError(
            f"Personality vault missing: {_VAULT_PATH}. "
            "Run: python scripts/generate_personalities_100.py"
        )
    with _VAULT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("personalities JSON must be a list of strings")
    if len(data) != _EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {_EXPECTED_TOTAL} personalities, got {len(data)} in {_VAULT_PATH}"
        )
    out: list[str] = []
    for i, s in enumerate(data):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"Invalid entry at index {i}: must be non-empty string")
        out.append(s.strip())
    return tuple(out)


def _vault() -> tuple[str, ...]:
    global _vault_cache
    if _vault_cache is None:
        _vault_cache = _load_vault()
    return _vault_cache


def __getattr__(name: str):
    if name == "ALL_PERSONALITIES":
        return _vault()
    if name == "PERSONALITY_BANK":
        return _vault()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Panel size = vault size (must match trained FullPerspectiveModel / checkpoints).
N = _EXPECTED_TOTAL

D = 64
P = 64
Q = 64
