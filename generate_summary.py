"""
Generate a natural-language summary from pipeline inputs and aggregate predictions
using Featherless AI (OpenAI-compatible API).

Usage:
    from generate_summary import generate_summary
    summary = generate_summary(
        actor="United States of America",
        receiver="Iraq",
        context="Year 2003. United States...",
        y_plus=0.72,
        y_minus=0.28,
    )
"""

from __future__ import annotations

import os
from typing import Optional

# Featherless is OpenAI-compatible; use openai client with base_url override
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"

SYSTEM_PROMPT = """You are an analyst summarizing geopolitical event predictions. Given an actor (who might act), a receiver (who the action targets), context (situation data), and model outputs, write a concise 2–4 sentence summary. State the predicted probability of the event occurring and briefly interpret what it implies given the context. Be factual and neutral."""

USER_PROMPT_TEMPLATE = """Actor (A): {actor}
Receiver (B): {receiver}
Context (D): {context}

Aggregate model predictions:
- Probability event occurs (invade / action): {y_plus:.2%}
- Probability event does not occur: {y_minus:.2%}

Write a brief summary."""


def generate_summary(
    actor: str,
    receiver: str,
    context: str,
    y_plus: float,
    y_minus: float,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Call Featherless AI to generate a summary from A string, B string, D string,
    and aggregate model predictions (y_plus, y_minus).

    Args:
        actor: A string — who might act (e.g. "United States of America").
        receiver: B string — who the action targets (e.g. "Iraq").
        context: D string — situation/context description.
        y_plus: Probability the event occurs (invade/action).
        y_minus: Probability the event does not occur.
        api_key: Featherless API key. Defaults to FEATHERLESS_API_KEY env var.
        model: Model ID for Featherless (default: deepseek-ai/DeepSeek-R1-0528).

    Returns:
        Generated summary string.

    Raises:
        RuntimeError: If openai is not installed or API key is missing.
    """
    if OpenAI is None:
        raise RuntimeError(
            "openai package required for Featherless. Install with: pip install openai"
        )

    key = api_key or os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError(
            "Featherless API key required. Set FEATHERLESS_API_KEY env var or pass api_key=..."
        )

    client = OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=key)

    user_content = USER_PROMPT_TEMPLATE.format(
        actor=actor,
        receiver=receiver,
        context=context,
        y_plus=y_plus,
        y_minus=y_minus,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.model_dump()["choices"][0]["message"]["content"]
    return (content or "").strip()


def generate_simulation_setup(
    theory: str,
    start_year: int,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    personality_candidates: Optional[list[str]] = None,
) -> dict:
    """
    Call Featherless AI to generate simulation setup from a theory string.

    Returns dict with:
      - aggressors: list of country/actor names (e.g. ["United States", "China"])
      - context: rich context string for the scenario
      - personalities: list of personality strings to use (subset of candidates)
    """
    if OpenAI is None:
        raise RuntimeError("openai package required. Install with: pip install openai")

    key = api_key or os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError("Set FEATHERLESS_API_KEY or pass api_key=...")

    client = OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=key)

    personalities_text = ""
    if personality_candidates:
        personalities_text = "\n".join(f"- {p}" for p in personality_candidates[:50])

    system = """You are a geopolitical simulation designer. Given a scenario theory and year, produce a JSON object with:
1. "aggressors": array of 1-3 country names (who might act) - use standard names like "United States", "China", "Russia"
2. "context": 2-4 sentence rich context string describing the geopolitical situation for that year
3. "personalities": array of 15-25 personality strings - pick the MOST RELEVANT from the candidate list for analyzing this scenario. Use the exact strings from the list."""

    user = f"""Theory: {theory}
Start year: {start_year}

Available personalities (pick 12-20 most relevant):
{personalities_text or "military analyst, defense strategist, nationalist politician, diplomat, economist, realist IR lens on security, liberal IR lens on trade, hawkish IR lens on security, dovish IR lens on security"}

Respond with ONLY a JSON object, no markdown, no explanation:
{{"aggressors": [...], "context": "...", "personalities": [...]}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = (response.model_dump()["choices"][0]["message"]["content"] or "").strip()

    # Parse JSON (handle markdown code blocks if present)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    import json as _json
    data = _json.loads(content)

    aggressors = data.get("aggressors") or ["United States"]
    if isinstance(aggressors, str):
        aggressors = [aggressors]
    context_str = data.get("context") or theory
    personalities = data.get("personalities") or []

    # Match returned personalities to our candidates; pad to 100 for model compatibility
    target_n = 100
    if personality_candidates:
        allowed = {p.strip().lower(): p for p in personality_candidates}
        matched = [allowed[p.strip().lower()] for p in personalities if p.strip().lower() in allowed]
        picked = list(dict.fromkeys(matched)) if matched else list(personality_candidates[:20])
        # Fill to target_n with remaining candidates (prioritize Featherless picks first)
        remaining = [p for p in personality_candidates if p not in picked]
        personalities = picked + remaining[: target_n - len(picked)]
    if len(personalities) < target_n and personality_candidates:
        personalities = list(personality_candidates[:target_n])
    personalities = personalities[:target_n]

    return {
        "aggressors": aggressors,
        "context": context_str,
        "personalities": personalities,
    }


def extrapolate_elaboration(
    actor: str,
    receiver: str,
    context: str,
    y_plus: float,
    y_minus: float,
    region_results: list[dict],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Call Featherless AI to extrapolate downstream impacts and elaborate on the prediction.

    region_results: list of {receiver, y_plus, y_minus} for other countries in region.
    """
    if OpenAI is None:
        raise RuntimeError("openai package required. Install with: pip install openai")

    key = api_key or os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError("Set FEATHERLESS_API_KEY or pass api_key=...")

    client = OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=key)

    region_text = ""
    if region_results:
        region_text = "\n".join(
            f"- {r.get('receiver', '?')}: {r.get('y_plus', 0)*100:.1f}% probability"
            for r in region_results
        )

    system = """You are a geopolitical analyst extrapolating downstream impacts from a risk prediction model.
Given an actor, primary receiver, context, probability outputs, and regional probabilities for neighboring countries,
write a 3–5 paragraph strategic assessment that:
1. Interprets the primary prediction and what it implies
2. Extrapolates likely downstream effects (economic, diplomatic, humanitarian, military)
3. Identifies regional ripple effects based on the country-level probabilities
4. Suggests potential inflection points and escalation pathways
Be analytical, specific, and cite the probability data where relevant."""

    user = f"""Actor: {actor}
Primary receiver: {receiver}
Context: {context}

Primary prediction: {y_plus:.1%} probability of action, {y_minus:.1%} probability no action.

Regional downstream probabilities (actor vs other countries in same region):
{region_text or 'No regional data.'}

Extrapolate the strategic implications and downstream impacts."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = (response.model_dump()["choices"][0]["message"]["content"] or "").strip()
    return content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate summary via Featherless AI")
    parser.add_argument("--actor", default="United States of America")
    parser.add_argument("--receiver", default="Iraq")
    parser.add_argument(
        "--context",
        default="Year 2003. United States CINC score 0.15. No active dispute.",
    )
    parser.add_argument("--y-plus", type=float, default=0.72)
    parser.add_argument("--y-minus", type=float, default=0.28)
    args = parser.parse_args()

    summary = generate_summary(
        actor=args.actor,
        receiver=args.receiver,
        context=args.context,
        y_plus=args.y_plus,
        y_minus=args.y_minus,
    )
    print(summary)
