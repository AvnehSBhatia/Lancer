# Inference pipeline audit (`scripts/run_full_pipeline.py`)

When [`scripts/run_full_pipeline.py`](scripts/run_full_pipeline.py) runs: **Part 1** — one [`Stages1to5`](apollo/perspective_stages.py) forward per vault personality (`N=1` checkpoint); **Part 2** — [`PerspectiveEventHead`](apollo/perspective_event_head.py) once. **No training** in the runner.

## Entrypoints

| File | Role |
|------|------|
| [`scripts/run_full_pipeline.py`](scripts/run_full_pipeline.py) | Pre-embeds strings; loops personalities; [`apollo.predict_from_strings.predict_from_embeddings`](apollo/predict_from_strings.py); [`apollo.perspective_stages.compute_abn`](apollo/perspective_stages.py); [`PerspectiveEventHead`](apollo/perspective_event_head.py). |
| [`apollo/predict_from_strings.py`](apollo/predict_from_strings.py) | One `Stages1to5` forward, **N = 1**. |
| [`apollo/personality_bank.py`](apollo/personality_bank.py) | Vault list (`apollo/personalities_100.json`). |

## Library modules

| File | Role |
|------|------|
| [`apollo/entity_minilm_converter.py`](apollo/entity_minilm_converter.py) | Entity MiniLM ↔ 64-d |
| [`apollo/context_minilm_converter.py`](apollo/context_minilm_converter.py) | Context converter |
| [`apollo/personality_minilm_converter.py`](apollo/personality_minilm_converter.py) | Personality converter |
| [`apollo/perspective_stages.py`](apollo/perspective_stages.py) | `Stages1to5`, `compute_abn` |
| [`apollo/paths.py`](apollo/paths.py) | `REPO_ROOT`, embedding dirs, `data/*.pt` paths |

## Artifacts (repo root)

- `entity_embeddings/`, `context_embeddings/`, `personality_embeddings/` — vocabs, `.pt` tables, converters
- `data/perspective_stages.pt`, optional `data/perspective_event_head.pt`

## External

- PyTorch, **sentence_transformers** (`all-MiniLM-L6-v2`) for OOV strings.

## `predict()` only (no aggregation)

[`apollo.predict_from_strings.predict`](apollo/predict_from_strings.py) does **not** run Part 2; it returns slot `p_hat` only. Use `run_full_pipeline` for **[ŷ⁺, ŷ⁻]**.

## Spec gaps

- **100 independent weight sets:** repo often uses one shared `perspective_stages.pt` and varies **C** only.
- **Step 5** in code ≠ PDF **M′·ABn** (see `perspective_stages.py`).
