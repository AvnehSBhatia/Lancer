"""
build_perspective_event_head_training_data.py

Supervised dataset for ``apollo.perspective_event_head.PerspectiveEventHead`` following
main (24).pdf Part 2 (aggregation Steps 6-12 inputs):

  - ``C_bank`` (N, p): personality rows ``C_i`` (same for every sample).
  - ``p_hat`` (M, N, 2): Part 1 mini-model votes ``[p+, p-]`` per slot. This builder
    uses random softmax logits as a **stand-in** unless you replace with real Part 1
    outputs from Step 5.
  - ``abn`` (M, p): actor-receiver blend (Part 1 Step 2 / Part 2 Step 9 input).
  - ``d_n`` (M, q): **global** context embedding ``D_n`` for Step 9 (same string space
    as Part 1 ``sD`` after the context converter).
  - ``y_class`` / ``y``: final head targets; class 0 = ``y+`` (hostile / action), 1 = ``y-``.

``ABDn = tanh(wD * ABn * Dn + bD)`` is **not** precomputed; the head learns ``wD, bD``.

Requires: entity_embeddings/, context_embeddings/, personality_embeddings/; vault
personalities in personality vocab.

Usage:
    python scripts/build_perspective_event_head_training_data.py --num-samples 100000
    python scripts/build_perspective_event_head_training_data.py --label-mode mid --mid-csv data/dyadic_mid_4.02.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apollo.paths import (
    CONTEXT_EMBEDDINGS_DIR as CONTEXT_DIR,
    DATA_DIR,
    ENTITY_EMBEDDINGS_DIR as ENTITY_DIR,
    PERSONALITY_EMBEDDINGS_DIR as PERSONALITY_DIR,
)
from apollo.perspective_stages import compute_abn

DEFAULT_MID_CSV = DATA_DIR / "dyadic_mid_4.02.csv"
OUTPUT_DIR = DATA_DIR
DEFAULT_OUT = OUTPUT_DIR / "perspective_event_head_training.pt"

P = 64
Q = 64

ENTITY_TO_MID: dict[str, str] = {
    "Algeria": "ALG",
    "Angola": "ANG",
    "Burundi": "BDI",
    "Colombia": "COL",
    "Djibouti": "DJI",
    "Egypt": "EGY",
    "Eritrea": "ERI",
    "Ethiopia": "ETH",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Libya": "LBY",
    "Madagascar": "MAG",
    "Mauritius": "MAU",
    "Morocco": "MOR",
    "Mozambique": "MZM",
    "Namibia": "NAM",
    "Rwanda": "RWA",
    "Somalia": "SOM",
    "South Africa": "SAF",
    "South Sudan": "SSD",
    "Sudan": "SUD",
    "Swaziland": "SWA",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Venezuela": "VEN",
    "Zambia": "ZAM",
    "Zimbabwe": "ZIM",
}


def _load_entity_matrix() -> tuple[torch.Tensor, list[str]]:
    with open(ENTITY_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(ENTITY_DIR / "entity_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"].float()
    n_emb = emb.shape[0]
    inv = [None] * n_emb
    for k, ix in vocab.items():
        if 0 <= ix < n_emb:
            inv[ix] = k
    if any(x is None for x in inv):
        raise ValueError("entity vocab indices do not cover 0..num_embeddings-1")
    return emb, inv


def _load_context_strings_and_matrix() -> tuple[list[str], torch.Tensor, list[str]]:
    with open(CONTEXT_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(CONTEXT_DIR / "context_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"].float()
    n_emb = emb.shape[0]
    inv = [None] * n_emb
    for k, ix in vocab.items():
        if 0 <= ix < n_emb:
            inv[ix] = k
    if any(x is None for x in inv):
        raise ValueError("context vocab indices do not cover 0..num_embeddings-1")
    keys = list(inv)
    return keys, emb, inv


def _load_personality_dict() -> dict[str, torch.Tensor]:
    with open(PERSONALITY_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(PERSONALITY_DIR / "personality_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"].float()
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i].clone() for i in range(len(vocab))}


def _build_C_bank(personality_names: tuple[str, ...], pers_embs: dict[str, torch.Tensor]) -> torch.Tensor:
    rows = []
    missing = []
    for name in personality_names:
        if name not in pers_embs:
            missing.append(name)
        else:
            rows.append(pers_embs[name])
    if missing:
        raise KeyError(
            "Personality strings missing from personality_embeddings vocab "
            f"(first 5): {missing[:5]}. Run build_personalities_txt_from_vault.py "
            "and scripts/train_personality_embeddings.py."
        )
    return torch.stack(rows, dim=0)


def _build_mid_positives(csv_path: Path, min_hihost: int) -> set[tuple[frozenset[str], int]]:
    out: set[tuple[frozenset[str], int]] = set()
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hi = int(row["hihost"])
                war = int(row["war"])
                year = int(row["year"])
                ca = row["namea"].strip()
                cb = row["nameb"].strip()
            except (KeyError, ValueError):
                continue
            if war == 1 or hi >= min_hihost:
                out.add((frozenset({ca, cb}), year))
    return out


def _context_keys_for_year(ctx_keys: list[str], year: int) -> list[str]:
    needle = f"Year {year}"
    return [k for k in ctx_keys if needle in k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PerspectiveEventHead training tensors (main (24).pdf Part 2 inputs).",
    )
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--label-mode",
        choices=("mid", "synthetic"),
        default="mid",
    )
    parser.add_argument("--mid-csv", type=Path, default=DEFAULT_MID_CSV)
    parser.add_argument("--mid-min-hihost", type=int, default=4)
    parser.add_argument("--pos-fraction", type=float, default=0.5)
    parser.add_argument("--year-min", type=int, default=1946)
    parser.add_argument("--year-max", type=int, default=2014)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.year_min > args.year_max:
        raise SystemExit("--year-min must be <= --year-max")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from apollo.personality_bank import PERSONALITY_BANK

    personality_names = tuple(PERSONALITY_BANK)
    n = len(personality_names)

    mid_pos: set[tuple[frozenset[str], int]] | None = None
    if args.label_mode == "mid":
        if not args.mid_csv.is_file():
            raise SystemExit(f"--label-mode mid requires --mid-csv file: {args.mid_csv}")
        mid_pos = _build_mid_positives(args.mid_csv, min_hihost=args.mid_min_hihost)
        print(
            f"MID positive dyad-years (war==1 or hihost>={args.mid_min_hihost}): {len(mid_pos)}"
        )

    synthetic_pos_pairs: set[tuple[str, str]] = {
        ("Iran", "Iraq"),
        ("Iraq", "Iran"),
    }

    print("Loading embedding tables...")
    E_ent, ent_names = _load_entity_matrix()
    ctx_keys, E_ctx, _ = _load_context_strings_and_matrix()
    ctx_key_to_ix = {k: i for i, k in enumerate(ctx_keys)}
    pers_embs = _load_personality_dict()

    Ve, Vc = E_ent.shape[0], E_ctx.shape[0]
    if Ve < 2 or Vc < 1:
        raise RuntimeError("Entity or context embedding table is empty.")

    C_bank = _build_C_bank(personality_names, pers_embs)

    mapped_entities = [nm for nm in ent_names if nm in ENTITY_TO_MID]
    if args.label_mode == "mid" and len(mapped_entities) < 2:
        print(
            "Warning: fewer than 2 entities map to MID codes; consider extending ENTITY_TO_MID."
        )

    M = args.num_samples
    abn_out = torch.empty(M, P)
    d_n_out = torch.empty(M, Q)
    y_class = torch.empty(M, dtype=torch.long)

    all_pairs = [(i, j) for i in range(Ve) for j in range(Ve) if i != j]

    # Part 1 Step 5 stand-in: i.i.d. votes (replace with real ``M' @ ABn`` heads when wired).
    p_hat_logits = torch.randn(M, n, 2, generator=g)
    p_hat_out = torch.softmax(p_hat_logits, dim=-1)

    print(
        f"Generating {M} samples (abn, d_n per main (24).pdf Step 9; p_hat random stand-in)..."
    )
    offset = 0
    while offset < M:
        B = min(args.batch_size, M - offset)
        a_b = []
        b_b = []
        d_rows = []
        labels = []

        for _ in range(B):
            if args.label_mode == "mid" and mid_pos is not None and mapped_entities:
                want_hostile = rng.random() < args.pos_fraction
                ia, ib, year, lab = 0, 1, args.year_min, 1
                for _try in range(400):
                    ia, ib = rng.choice(all_pairs)
                    na, nb = ent_names[ia], ent_names[ib]
                    ca, cb = ENTITY_TO_MID.get(na), ENTITY_TO_MID.get(nb)
                    if ca is None or cb is None:
                        continue
                    year = rng.randint(args.year_min, args.year_max)
                    is_hostile = (frozenset({ca, cb}), year) in mid_pos
                    lab = 0 if is_hostile else 1
                    if want_hostile == is_hostile:
                        break
                else:
                    ia, ib = rng.choice(all_pairs)
                    na, nb = ent_names[ia], ent_names[ib]
                    ca, cb = ENTITY_TO_MID.get(na), ENTITY_TO_MID.get(nb)
                    year = rng.randint(args.year_min, args.year_max)
                    if ca is None or cb is None:
                        lab = 1
                    else:
                        is_h = (frozenset({ca, cb}), year) in mid_pos
                        lab = 0 if is_h else 1
            else:
                ia, ib = rng.choice(all_pairs)
                na, nb = ent_names[ia], ent_names[ib]
                year = rng.randint(args.year_min, args.year_max)
                lab = 0 if (na, nb) in synthetic_pos_pairs else 1

            keys = _context_keys_for_year(ctx_keys, year)
            if keys:
                ck = rng.choice(keys)
                ic = ctx_key_to_ix[ck]
            else:
                ic = rng.randrange(Vc)

            a_b.append(E_ent[ia])
            b_b.append(E_ent[ib])
            d_rows.append(E_ctx[ic])
            labels.append(lab)

        a_t = torch.stack(a_b, dim=0)
        b_t = torch.stack(b_b, dim=0)
        with torch.no_grad():
            abn_b = compute_abn(a_t, b_t)
        d_t = torch.stack(d_rows, dim=0)

        abn_out[offset : offset + B] = abn_b
        d_n_out[offset : offset + B] = d_t
        y_class[offset : offset + B] = torch.tensor(labels, dtype=torch.long)
        offset += B

        if offset % max(args.batch_size * 50, 1) == 0 or offset >= M:
            print(f"  ... {offset}/{M}")

    n_plus = int((y_class == 0).sum().item())
    print(f"  Class 0 (y_plus / hostile): {n_plus} ({100.0 * n_plus / M:.1f}%)")

    y_one_hot = torch.zeros(M, 2)
    y_one_hot[torch.arange(M), y_class] = 1.0

    payload = {
        "C_bank": C_bank,
        "p_hat": p_hat_out,
        "abn": abn_out,
        "d_n": d_n_out,
        "y_class": y_class,
        "y": y_one_hot,
        "n": n,
        "p": P,
        "q": Q,
        "num_samples": M,
        "label_mode": args.label_mode,
        "spec_reference": "main (24).pdf Part 2 Steps 6-12; p_hat is random stand-in for Part 1 Step 5.",
        "y_semantics": "class 0 = y_plus (hostile/action); class 1 = y_minus.",
        "personality_names": list(personality_names),
    }
    if args.label_mode == "mid":
        payload["mid_csv"] = str(args.mid_csv)
        payload["mid_min_hihost"] = args.mid_min_hihost
        payload["pos_fraction"] = args.pos_fraction

    torch.save(payload, args.output)
    print(f"Saved {args.output}")
    print(
        f"  C_bank: {tuple(C_bank.shape)}, p_hat: {tuple(p_hat_out.shape)}, "
        f"abn: {tuple(abn_out.shape)}, d_n: {tuple(d_n_out.shape)}, y_class: {tuple(y_class.shape)}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
