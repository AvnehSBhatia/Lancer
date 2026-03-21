# Inference pipeline audit (`run_full_pipeline.py` → Part 1 + Part 2)

This document lists **every project file** involved when [`run_full_pipeline.py`](run_full_pipeline.py) runs: **Part 1** — one [`Stages1to5`](perspective_stages.py) forward per vault personality (same `N=1` checkpoint); **Part 2** — [`PerspectiveEventHead`](apollo/perspective_event_head.py) once (main (24).pdf aggregation). **No training** in the runner.

## 1. Entrypoints

| File | Role |
|------|------|
| [`run_full_pipeline.py`](run_full_pipeline.py) | In-process: pre-embeds actor/receiver/context; for each vault personality (or `--max-personalities` subsample), calls [`predict_from_embeddings`](predict_from_strings.py) with a **shared** loaded `Stages1to5`; stacks **raw** `C_i`, collects `p_hat`, builds `ABn` via [`compute_abn`](perspective_stages.py), runs [`PerspectiveEventHead`](apollo/perspective_event_head.py). Optional weights: `data/perspective_event_head.pt`. |
| [`predict_from_strings.py`](predict_from_strings.py) | Loads embedders + checkpoint, runs **one** forward of [`perspective_stages.Stages1to5`](perspective_stages.py) with **N = 1**, returns `p_hat` for that single slot. |
| [`personality_bank.py`](personality_bank.py) | Supplies the ordered list of 100 personality strings (vault JSON). |

## 2. Python modules (transitive, inference-only)

| File | Role |
|------|------|
| [`entity_minilm_converter.py`](entity_minilm_converter.py) | `load_converter(ENTITY_DIR)` → trained linear maps + embedding table for entities. |
| [`context_minilm_converter.py`](context_minilm_converter.py) | Same pattern for contexts. |
| [`personality_minilm_converter.py`](personality_minilm_converter.py) | Same pattern for personalities. |
| [`perspective_stages.py`](perspective_stages.py) | **`Stages1to5` only**: docstring Stages 1–5 (outer product grid, residual, Q/K projection, ABn/ABDj fusion, per-slot softmax + heads). **`build_model` = `Stages1to5`**, not Part 2. Also **`compute_abn`** for PDF **ABn**. |
| [`apollo/perspective_event_head.py`](apollo/perspective_event_head.py) | **Part 2** (main (24).pdf Steps 6–12): `PerspectiveEventHead`. |

## 3. On-disk artifacts (must exist from prior training)

| Path | Used for |
|------|-----------|
| `entity_embeddings/vocab.json` | String → id |
| `entity_embeddings/entity_embeddings.pt` | In-vocab 64-d vectors |
| `entity_embeddings/converter_ours_minilm.pt` | MiniLM (384) ↔ 64-d |
| `context_embeddings/vocab.json` | Same pattern |
| `context_embeddings/context_embeddings.pt` | |
| `context_embeddings/converter_ours_minilm.pt` | |
| `personality_embeddings/vocab.json` | |
| `personality_embeddings/personality_embeddings.pt` | |
| `personality_embeddings/converter_ours_minilm.pt` | |
| `data/perspective_stages.pt` | Weights for `Stages1to5` (`Config(n=1,…)` must match [`predict_from_strings`](predict_from_strings.py) and [`train_perspective_stages`](train_perspective_stages.py)). |
| `data/perspective_event_head.pt` | Optional. Weights for [`PerspectiveEventHead`](apollo/perspective_event_head.py) (Part 2). If missing, `run_full_pipeline.py` uses a random-init head. |

## 4. External packages (runtime)

- **PyTorch** (`torch`)
- **sentence_transformers** — `all-MiniLM-L6-v2` when a string is **not** in the local embedding vocab (then MiniLM → converter → 64-d).

## 5. What actually runs inside `predict()` (per subprocess)

1. **MiniLM encode** (conditional): only if entity/context/personality string missing from respective `*_embeddings` vocab.
2. **64-d embeddings**: actor `a`, receiver `b`, context `ctx`, personality `pers` → shaped to `(1,1,64)` for `c` and `d_ctx`.
3. **`Stages1to5.forward`**: Stages **1–5** as implemented in [`perspective_stages.py`](perspective_stages.py) with **N = 1** (one personality slot, one context slot; same context replicated in that slot).
4. **Readout**: `out.p_hat[0,0,:]`, `out.abn[0,:]`, `out.c_star[0,0,:]` — see [`predict_from_strings.py`](predict_from_strings.py).

## 6. Not executed on this path (important)

| Item | Detail |
|------|--------|
| **[`apollo/perspective_event_head.py`](apollo/perspective_event_head.py)** (`Stages6to8`, `FullPerspectiveModel`) | **Not imported** by `predict_from_strings.py`. There is **no** aggregation head, no `v ∈ ℝ^100`, no final softmax over two classes from pooled votes in this script. |
| **100 parallel mini-models in one forward** | The checkpoint is trained with **N = 1** ([`train_perspective_stages.py`](train_perspective_stages.py)). Running `predict_from_strings` **100 times** with different personality strings is **100 separate forwards**, not one tensor with 100 slots sharing main (23)’s Part 2. |
| **`q_proj` / `k_proj`** | Computed in Stage 3 and returned on `Outputs`, but **nothing in `predict_from_strings` consumes them** for the printed probabilities. They are not “skipped” as dead code in the forward (the ops run), but they **do not affect** the selected `p_hat` path. |

## 7. Comparison to **main (23).pdf**

- **PDF Part 1:** 100 **independent** mini-models, each with **its own weights**, each outputting **ŷp_i**.
- **This repo (inference):** One **`Stages1to5`** with **N = 1**; weights are **shared** across the 100 subprocess calls (same checkpoint), only the **input personality embedding** changes. That is **not** the same as 100 independently parameterized mini-models.
- **PDF Part 2:** Aggregation over **100** votes and **E ∈ ℝ^{100×64}**.
- **This repo:** **No** Part 2 in `predict_from_strings.py`; final printed probabilities are **per-call `p_hat` from Stage 5 only** (single slot).

## 8. Conclusion

**End-to-end for *this codebase’s* inference:** embedding pipeline (vocab or MiniLM+converter) → **`Stages1to5`** → **`p_hat`** is **fully executed** for each of the 100 subprocess runs, with **no silent shortcut** inside that forward that bypasses Stages 1–5.

**Relative to main (23).pdf’s full architecture:** the **aggregation / second-half head is absent** from `predict_from_strings.py`, and the **100-way independent mini-model** design is **not** implemented as a single **N = 100** forward; instead you get **100 × (N = 1)** forwards.
