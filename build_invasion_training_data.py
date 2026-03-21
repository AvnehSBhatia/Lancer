"""
build_invasion_training_data.py

Builds training data for mini-models that predict invade/not-invade based on:
- question (embedded via entity: MiniLM -> entity space)
- affected group (actor + receiver, e.g. US and Iraq) - entity embeddings
- context (e.g. Year 2003 Iraq war context) - context embedding
- personality - personality embedding

Output: [1, 0] for invade (yes), [0, 1] for not invade (no).
Labels are assigned based on personality (militaristic -> invade, pacifist -> not).
Model input: concat(question_emb, affected_group_emb, context_emb, personality_emb).
"""

import json
import os
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────
ENTITY_DIR = Path("entity_embeddings")
CONTEXT_DIR = Path("context_embeddings")
PERSONALITY_DIR = Path("personality_embeddings")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "invasion_training.pt"
# ─────────────────────────────────────────────────────────────────

# Personality -> invade (1) or not (0). More hawkish -> invade, more dovish -> not.
PERSONALITY_TO_INVADE = {
    "military analyst": 1,
    "defense strategist": 1,
    "nationalist politician": 1,
    "foreign minister": 0,  # diplomatic
    "diplomat": 0,
    "economist": 0,
    "trade policy expert": 0,
    "isolationist": 0,  # isolationist might avoid
    "pacifist activist": 0,
}

# Entity pairs for affected groups (actor, receiver) - must exist in entity vocab
ENTITY_PAIRS = [
    ("United States of America", "Iraq"),
    ("United States of America", "Afghanistan"),
    ("Russia", "Ukraine"),
    ("China", "Taiwan"),
    ("United Kingdom", "Argentina"),
    ("France", "Libya"),
    ("Iran", "Iraq"),
    ("North Korea", "South Korea"),
    ("India", "Pakistan"),
]

# Question templates
QUESTION_TEMPLATES = [
    "Will {actor} invade {receiver}?",
    "Would {actor} invade {receiver}?",
    "Should {actor} invade {receiver}?",
    "Will {actor} launch a military invasion of {receiver}?",
    "Would {actor} use military force against {receiver}?",
]


def load_entity_embeddings():
    """Load entity embeddings and vocab."""
    with open(ENTITY_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(ENTITY_DIR / "entity_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i].float() for i in range(len(vocab))}, vocab


def load_context_embeddings():
    """Load context embeddings and vocab."""
    with open(CONTEXT_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(CONTEXT_DIR / "context_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i].float() for i in range(len(vocab))}, vocab


def load_personality_embeddings():
    """Load personality embeddings and vocab."""
    with open(PERSONALITY_DIR / "vocab.json") as f:
        vocab = json.load(f)
    state = torch.load(PERSONALITY_DIR / "personality_embeddings.pt", weights_only=True)
    emb = state["embeddings.weight"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return {inv_vocab[i]: emb[i].float() for i in range(len(vocab))}, vocab


def load_question_embedder():
    """Load MiniLM->entity converter for embedding questions."""
    from entity_minilm_converter import load_converter
    from sentence_transformers import SentenceTransformer

    converter, _, _ = load_converter(ENTITY_DIR)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return converter, st


def embed_question(question: str, converter, st) -> torch.Tensor:
    """Embed question via MiniLM -> entity space (64-d)."""
    minilm = torch.tensor(st.encode(question, convert_to_numpy=True), dtype=torch.float32)
    with torch.no_grad():
        entity_vec = converter.minilm_to_ours_vec(minilm.unsqueeze(0)).squeeze(0)
    return entity_vec


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings...")
    entity_embs, entity_vocab = load_entity_embeddings()
    context_embs, context_vocab = load_context_embeddings()
    personality_embs, personality_vocab = load_personality_embeddings()

    print("Loading question embedder (MiniLM + entity converter)...")
    try:
        converter, st = load_question_embedder()
        use_question_embedder = True
    except Exception as e:
        print(f"  Warning: Could not load converter: {e}")
        print("  Using mean of actor+receiver entity embeddings as question proxy.")
        use_question_embedder = False

    # Filter entity pairs to those in vocab
    valid_pairs = []
    for actor, receiver in ENTITY_PAIRS:
        if actor in entity_vocab and receiver in entity_vocab:
            valid_pairs.append((actor, receiver))
        else:
            print(f"  Skipping pair ({actor}, {receiver}) - not in entity vocab")

    # Pick context strings; prefer 2003 for Iraq war era
    context_keys = list(context_vocab.keys())
    year_2003_contexts = [k for k in context_keys if "Year 2003" in k]
    if year_2003_contexts:
        us_any = [k for k in year_2003_contexts if "United States" in k]
        context_samples = (us_any or year_2003_contexts)[:20]
    else:
        context_samples = context_keys[:30]

    personality_names = list(personality_vocab.keys())
    personality_invade_map = PERSONALITY_TO_INVADE

    print("Building dataset...")
    samples = []
    for actor, receiver in valid_pairs:
        actor_emb = entity_embs[actor]
        receiver_emb = entity_embs[receiver]
        affected_emb = torch.cat([actor_emb, receiver_emb], dim=0)  # (128,)

        for ctx_key in context_samples:
            if ctx_key not in context_embs:
                continue
            ctx_emb = context_embs[ctx_key]

            for pers_name in personality_names:
                pers_emb = personality_embs[pers_name]
                invade = personality_invade_map.get(pers_name, 0)

                for template in QUESTION_TEMPLATES:
                    question = template.format(actor=actor, receiver=receiver)

                    if use_question_embedder:
                        q_emb = embed_question(question, converter, st)
                    else:
                        q_emb = (actor_emb + receiver_emb) / 2

                    # Input: concat(question_emb, affected_emb, context_emb, personality_emb)
                    # 64 + 128 + 64 + 64 = 320
                    x = torch.cat([q_emb, affected_emb, ctx_emb, pers_emb], dim=0)
                    y = torch.tensor([1.0, 0.0] if invade else [0.0, 1.0])

                    samples.append({
                        "x": x,
                        "y": y,
                        "question": question,
                        "actor": actor,
                        "receiver": receiver,
                        "context": ctx_key[:80] + "..." if len(ctx_key) > 80 else ctx_key,
                        "personality": pers_name,
                        "invade": bool(invade),
                    })

    if not samples:
        raise RuntimeError("No samples generated. Check entity/context/personality vocabs.")

    X = torch.stack([s["x"] for s in samples])
    Y = torch.stack([s["y"] for s in samples])

    # Save as .pt for easy loading
    torch.save({
        "X": X,
        "Y": Y,
        "samples": samples,
        "input_dim": X.shape[1],
        "output_dim": 2,
    }, OUTPUT_FILE)

    print(f"Saved {len(samples)} samples to {OUTPUT_FILE}")
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  Input dim: {X.shape[1]} (q:64 + affected:128 + ctx:64 + pers:64)")
    print("Done.")


if __name__ == "__main__":
    main()
