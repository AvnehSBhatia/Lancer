"""
Simple Flask API for APOLLO — runs test_full_pipeline_100 logic.

POST /predict   - actor, receiver, context  -> y_plus, y_minus
POST /predict/region - same + returns oppressor vs region countries

Run: python scripts/sim_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RECEIVER_COUNTRIES = [
    {"name": "United States", "region": "Americas", "lat": 38, "lng": -97},
    {"name": "China", "region": "East Asia", "lat": 35, "lng": 105},
    {"name": "India", "region": "South Asia", "lat": 20, "lng": 77},
    {"name": "Russia", "region": "Eurasia", "lat": 60, "lng": 100},
    {"name": "Brazil", "region": "South America", "lat": -10, "lng": -55},
    {"name": "Australia", "region": "Oceania", "lat": -25, "lng": 135},
    {"name": "Canada", "region": "Americas", "lat": 60, "lng": -95},
    {"name": "United Kingdom", "region": "Europe", "lat": 55, "lng": -3},
    {"name": "Germany", "region": "Europe", "lat": 51, "lng": 9},
    {"name": "France", "region": "Europe", "lat": 46, "lng": 2},
    {"name": "Japan", "region": "East Asia", "lat": 36, "lng": 138},
    {"name": "South Africa", "region": "Africa", "lat": -30, "lng": 22},
    {"name": "Egypt", "region": "North Africa", "lat": 26, "lng": 30},
    {"name": "Saudi Arabia", "region": "Middle East", "lat": 23, "lng": 45},
    {"name": "Mexico", "region": "Americas", "lat": 23, "lng": -102},
    {"name": "Indonesia", "region": "Southeast Asia", "lat": -0.5, "lng": 118},
    {"name": "Nigeria", "region": "Africa", "lat": 10, "lng": 8},
    {"name": "Iran", "region": "Middle East", "lat": 32, "lng": 53},
    {"name": "Turkey", "region": "Eurasia", "lat": 39, "lng": 35},
    {"name": "Argentina", "region": "South America", "lat": -38, "lng": -63},
    {"name": "South Korea", "region": "East Asia", "lat": 36, "lng": 128},
    {"name": "Italy", "region": "Europe", "lat": 42, "lng": 12},
    {"name": "Spain", "region": "Europe", "lat": 40, "lng": -4},
    {"name": "Thailand", "region": "Southeast Asia", "lat": 15, "lng": 101},
    {"name": "Kenya", "region": "Africa", "lat": -1, "lng": 37},
    {"name": "Sweden", "region": "Northern Europe", "lat": 62, "lng": 15},
    {"name": "Poland", "region": "Europe", "lat": 52, "lng": 20},
    {"name": "Colombia", "region": "South America", "lat": 4, "lng": -72},
    {"name": "Pakistan", "region": "South Asia", "lat": 30, "lng": 69},
    {"name": "Ukraine", "region": "Europe", "lat": 49, "lng": 32},
    {"name": "Taiwan", "region": "East Asia", "lat": 25, "lng": 121},
    {"name": "Iraq", "region": "Middle East", "lat": 33, "lng": 44},
    {"name": "Cuba", "region": "Americas", "lat": 22, "lng": -80},
]

NAME_TO_COUNTRY = {c["name"]: c for c in RECEIVER_COUNTRIES}


def _run_pipeline(actor: str, receiver: str, context: str) -> tuple[float, float]:
    """Same logic as test_full_pipeline_100.py"""
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
    from apollo.paths import PERSPECTIVE_EVENT_HEAD_PT
    import torch

    personalities = list(PERSONALITY_BANK)
    stages = _load_model(MODEL_PATH)
    a_emb = embed_entity(actor)
    b_emb = embed_entity(receiver)
    ctx_emb = embed_context(context)

    pers_embs = [embed_personality(s) for s in personalities]
    C = torch.stack([p.flatten() for p in pers_embs], dim=0)
    p_hat_rows = []
    for pers_t in pers_embs:
        r = predict_from_embeddings(a_emb, b_emb, ctx_emb, pers_t, model=stages)
        p_hat_rows.append(r.prediction.detach())
    p_hat = torch.stack(p_hat_rows, dim=0)
    abn = compute_abn(a_emb.flatten(), b_emb.flatten())
    d_n = ctx_emb.flatten()

    head = PerspectiveEventHead(n=100, p=64, q=64)
    if PERSPECTIVE_EVENT_HEAD_PT.is_file():
        head.load_state_dict(torch.load(PERSPECTIVE_EVENT_HEAD_PT, weights_only=True))
    head.eval()

    with torch.no_grad():
        y = head(C, p_hat, abn, d_n)
    return float(y[0].item()), float(y[1].item())


def _coords(name: str) -> dict | None:
    return NAME_TO_COUNTRY.get(name)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    actor = (data.get("actor") or "China").strip()
    receiver = (data.get("receiver") or "Taiwan").strip()
    context = (data.get("context") or "Asia 2025").strip()

    yp, ym = _run_pipeline(actor, receiver, context)
    actor_coords = _coords(actor)
    receiver_coords = _coords(receiver)
    return jsonify({
        "actor": actor,
        "receiver": receiver,
        "y_plus": yp,
        "y_minus": ym,
        "actor_coords": actor_coords,
        "receiver_coords": receiver_coords,
    })


@app.route("/predict/region", methods=["POST"])
def predict_region():
    data = request.get_json() or {}
    actor = (data.get("actor") or "China").strip()
    receiver = (data.get("receiver") or "Taiwan").strip()
    context = (data.get("context") or "Asia 2025").strip()

    receiver_info = _coords(receiver)
    if not receiver_info:
        return jsonify({"error": f"Unknown receiver: {receiver}"}), 400

    region = receiver_info["region"]
    region_countries = [c for c in RECEIVER_COUNTRIES if c["region"] == region and c["name"] != actor and c["name"] != receiver]

    yp, ym = _run_pipeline(actor, receiver, context)
    actor_coords = _coords(actor)
    receiver_coords = _coords(receiver)

    primary = {
        "receiver": receiver,
        "y_plus": yp,
        "y_minus": ym,
        "lat": receiver_coords["lat"] if receiver_coords else 0,
        "lng": receiver_coords["lng"] if receiver_coords else 0,
    }

    region_results = []
    for c in region_countries:
        try:
            ryp, rym = _run_pipeline(actor, c["name"], context)
            region_results.append({
                "receiver": c["name"],
                "y_plus": ryp,
                "y_minus": rym,
                "lat": c["lat"],
                "lng": c["lng"],
                "region": c["region"],
            })
        except Exception:
            continue

    return jsonify({
        "actor": actor,
        "actor_coords": actor_coords,
        "primary": primary,
        "region": region_results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
