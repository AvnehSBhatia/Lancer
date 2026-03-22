"""
Write personalities_100.json in the project root: exactly 100 strings.

The first 9 entries match personality_bank's historical panel so existing
personality embeddings and checkpoints stay aligned. Remaining entries are
deterministic synthetic lenses.
"""

from __future__ import annotations

import json
from pathlib import Path

VAULT_SIZE = 100
_REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = _REPO_ROOT / "apollo" / "personalities_100.json"

MODEL_PANEL = [
    "military analyst",
    "defense strategist",
    "nationalist politician",
    "foreign minister",
    "diplomat",
    "economist",
    "trade policy expert",
    "isolationist",
    "pacifist activist",
]

ADJ = [
    "realist",
    "idealist",
    "hawkish",
    "dovish",
    "pragmatic",
    "normative",
    "institutionalist",
    "constructivist",
    "mercantilist",
    "liberal",
    "postcolonial",
    "feminist",
    "environmental",
    "technocratic",
    "populist",
]

DOMAIN = [
    "trade",
    "security",
    "human rights",
    "alliances",
    "sanctions",
    "energy",
    "migration",
    "cyber",
    "maritime",
    "space",
    "nuclear",
    "regional hegemony",
    "multilateral forums",
    "domestic politics",
    "intelligence",
]

LENS = [
    "game-theoretic",
    "historical",
    "legal-institutional",
    "bargaining",
    "signaling",
    "credibility",
    "reputation",
    "coalition",
    "two-level games",
    "balance-of-power",
    "interdependence",
    "norm diffusion",
]


def synth_line(i: int) -> str:
    """Deterministic persona string for index i >= 9."""
    a = ADJ[i % len(ADJ)]
    d = DOMAIN[(i // 17) % len(DOMAIN)]
    l = LENS[(i // 91) % len(LENS)]
    return f"{a} IR lens on {d} using a {l} framing (vault id {i})"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = list(MODEL_PANEL)
    for i in range(len(MODEL_PANEL), VAULT_SIZE):
        rows.append(synth_line(i))
    assert len(rows) == VAULT_SIZE
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {len(rows)} entries to {OUT}")


if __name__ == "__main__":
    main()
