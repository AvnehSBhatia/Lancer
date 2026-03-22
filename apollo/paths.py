"""
Repository-root-relative paths. All training/inference should use these so code works
when run from any cwd (as long as ``REPO_ROOT`` is correct).
"""

from __future__ import annotations

from pathlib import Path

# apollo/paths.py -> parent is apollo/, grandparent is repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

ENTITY_EMBEDDINGS_DIR = REPO_ROOT / "entity_embeddings"
CONTEXT_EMBEDDINGS_DIR = REPO_ROOT / "context_embeddings"
PERSONALITY_EMBEDDINGS_DIR = REPO_ROOT / "personality_embeddings"
DATA_DIR = REPO_ROOT / "data"

PERSPECTIVE_STAGES_PT = DATA_DIR / "perspective_stages.pt"
PERSPECTIVE_EVENT_HEAD_PT = DATA_DIR / "perspective_event_head.pt"
PERSPECTIVE_EVENT_HEAD_TRAINING_PT = DATA_DIR / "perspective_event_head_training.pt"
INVASION_TRAINING_PT = DATA_DIR / "invasion_training.pt"
INVASION_MINI_MODEL_PT = DATA_DIR / "invasion_mini_model.pt"
