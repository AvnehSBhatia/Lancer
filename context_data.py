"""
context_data.py

Data loaders for build_context_string: NMC capabilities, dyadic MIDs,
alliances, trade. Uses COW ccode for country identification.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
NMC_CSV = DATA_DIR / "NMC-60-abridged.csv"
DYADIC_MID_CSV = DATA_DIR / "dyadic_mid_4.02.csv"
ALLIANCE_CSV = DATA_DIR / "alliance_v4.1_by_dyad_yearly.csv"
TRADE_CSV = DATA_DIR / "Dyadic_COW_4.0.csv"

ALLIANCE_TYPES = {"defense": 1, "neutrality": 2, "nonaggression": 3, "entente": 4}


def _dyad(c1: int, c2: int) -> tuple[int, int]:
    return (min(c1, c2), max(c1, c2))


# ── In-memory caches (loaded once) ───────────────────────────────────
_nmc: dict[tuple[int, int], dict] = {}
_dyadic_mid: dict[tuple[int, int], list[dict]] = defaultdict(list)
_alliance: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
_trade: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
_ccode_to_name: dict[int, str] = {}


def _load_nmc() -> None:
    global _nmc, _ccode_to_name
    with open(NMC_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ccode = int(row["ccode"])
                year = int(row["year"])
                cinc = float(row.get("cinc", 0) or 0)
                milex = float(row.get("milex", 0) or 0)
                milper = float(row.get("milper", 0) or 0)
                irst = float(row.get("irst", 0) or 0)
                pec = float(row.get("pec", 0) or 0)
            except (ValueError, KeyError):
                continue
            _nmc[(ccode, year)] = {
                "cinc": cinc,
                "military_expenditure": milex,
                "military_personnel": milper,
                "energy_consumption": irst,
                "pec": pec,
            }
            abb = row.get("stateabb", "").strip()
            if ccode not in _ccode_to_name and abb:
                _ccode_to_name[ccode] = abb

    # ccode->name from alliance (full names preferred)
    with open(ALLIANCE_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["ccode1"]), int(row["ccode2"])
                _ccode_to_name[c1] = row.get("state_name1", "").strip() or _ccode_to_name.get(c1, "")
                _ccode_to_name[c2] = row.get("state_name2", "").strip() or _ccode_to_name.get(c2, "")
            except (ValueError, KeyError):
                continue

    # Fallback: dyadic_mid namea/nameb for any still missing
    if DYADIC_MID_CSV.exists():
        with open(DYADIC_MID_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    c1, c2 = int(row["statea"]), int(row["stateb"])
                    n1, n2 = row.get("namea", "").strip(), row.get("nameb", "").strip()
                    if c1 not in _ccode_to_name and n1:
                        _ccode_to_name[c1] = n1
                    if c2 not in _ccode_to_name and n2:
                        _ccode_to_name[c2] = n2
                except (ValueError, KeyError):
                    continue


def _load_dyadic_mid() -> None:
    global _dyadic_mid
    seen = set()
    with open(DYADIC_MID_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["statea"]), int(row["stateb"])
                yr = int(row.get("year", 0) or row.get("strtyr", 0))
                strtyr = int(row.get("strtyr", yr))
                endyear = int(row.get("endyear", yr))
                hihost = int(row.get("hihost", 0)) if row.get("hihost", "") != "" else 0
            except (ValueError, KeyError):
                continue
            key = (_dyad(c1, c2), yr)
            if key in seen:
                continue
            seen.add(key)
            _dyadic_mid[_dyad(c1, c2)].append({
                "strtyr": strtyr, "endyear": endyear, "year": yr,
                "hostility_level": hihost, "statea": c1, "stateb": c2,
            })


def _load_alliance() -> None:
    global _alliance
    with open(ALLIANCE_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["ccode1"]), int(row["ccode2"])
                yr = int(row["year"])
                for col in ALLIANCE_TYPES:
                    if row.get(col) == "1":
                        _alliance[c1][yr].append((c2, col))
                        _alliance[c2][yr].append((c1, col))
                        break
            except (ValueError, KeyError):
                continue


def _load_trade() -> None:
    global _trade
    totals: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    with open(TRADE_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c1, c2 = int(row["ccode1"]), int(row["ccode2"])
                yr = int(row["year"])
                val = float(row.get("smoothtotrade", "-9"))
            except (ValueError, KeyError):
                continue
            if val < 0:
                continue
            totals[(*_dyad(c1, c2), yr)].append(val)
    for (c1, c2, yr), vals in totals.items():
        _trade[_dyad(c1, c2)][yr] = sum(vals) / len(vals)
    _trade_raw_max = max(v for d in _trade.values() for v in d.values()) if _trade else 1.0
    for d in _trade.values():
        for yr in d:
            d[yr] = d[yr] / _trade_raw_max if _trade_raw_max > 0 else 0.0


def _ensure_loaded() -> None:
    if not _nmc and NMC_CSV.exists():
        _load_nmc()
    if not _dyadic_mid and DYADIC_MID_CSV.exists():
        _load_dyadic_mid()
    if not _alliance and ALLIANCE_CSV.exists():
        _load_alliance()
    if not _trade and TRADE_CSV.exists():
        _load_trade()


def get_capabilities(ccode: int, year: int) -> dict | None:
    _ensure_loaded()
    return _nmc.get((ccode, year))


def get_active_mids(year: int) -> list[tuple[int, int, int, str | None]]:
    """Returns list of (statea, stateb, hostility_level, None) for MIDs active in year."""
    _ensure_loaded()
    out = []
    for (c1, c2), recs in _dyadic_mid.items():
        for r in recs:
            if r["strtyr"] <= year <= r["endyear"]:
                out.append((c1, c2, r["hostility_level"], None))
                break
    return out


def check_mid_between(ccode_a: int, ccode_b: int, year: int) -> tuple[int, str] | None:
    _ensure_loaded()
    dyad = _dyad(ccode_a, ccode_b)
    for r in _dyadic_mid.get(dyad, []):
        if r["strtyr"] <= year <= r["endyear"]:
            return (r["hostility_level"], "militarized")
    return None


def get_alliances(ccode: int, year: int) -> list[tuple[int, str]]:
    """Returns list of (partner_ccode, alliance_type)."""
    _ensure_loaded()
    return _alliance.get(ccode, {}).get(year, [])


def get_active_wars(year: int) -> list:
    """Placeholder: no war dataset loaded."""
    return []


def get_bilateral_trade(ccode_a: int, ccode_b: int, year: int) -> float:
    _ensure_loaded()
    return _trade.get(_dyad(ccode_a, ccode_b), {}).get(year, 0.0)


def get_diplomatic_level(ccode_a: int, ccode_b: int, year: int) -> str:
    """Placeholder: no diplomatic exchange dataset."""
    return "unknown"


def ccode_to_name(ccode: int) -> str:
    _ensure_loaded()
    return _ccode_to_name.get(ccode, str(ccode))


def get_available_pairs_years() -> set[tuple[int, int, int]]:
    """Returns (ccode_a, ccode_b, year) with NMC + trade or MID coverage."""
    _ensure_loaded()
    pairs = set()
    for (c1, c2), by_year in _trade.items():
        for yr in by_year:
            if (c1, yr) in _nmc and (c2, yr) in _nmc:
                pairs.add((c1, c2, yr))
    for (c1, c2), recs in _dyadic_mid.items():
        for r in recs:
            yr = r["year"]
            if (c1, yr) in _nmc and (c2, yr) in _nmc:
                pairs.add((c1, c2, yr))
    return pairs
