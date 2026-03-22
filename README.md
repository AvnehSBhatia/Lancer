# Apollo / Lancer — Perspective-Conditioned Geopolitical Event Prediction

This repository implements components of the architecture in **main (24).pdf** (*Perspective-Conditioned Geopolitical Event Prediction — Complete Architecture Specification*). The confirmed spec describes **100 independent mini-models** (each with its own weights), followed by **one aggregation head** that outputs a single binary distribution **[ŷ⁺, ŷ⁻]** (action toward B vs. not).

Place **main (24).pdf** alongside this README for full detail; the sections below mirror its structure.

---

## Abstract (spec)

- Each mini-model takes **four English strings**, embeds them with **MiniLM-v2** and learned **64-dimensional converters**, and outputs **[p̂⁺, p̂⁻]**.
- The **aggregation head** runs **once** after all mini-models, combining **personality vectors** and **votes** into **[ŷ⁺, ŷ⁻]**.

---

## Part 1 — The mini-model (runs N times; N = 100 in the PDF)

Each run uses **independent parameters** in the full spec (**no weight sharing** between mini-models). Mini-model **i** consumes **(sA, sB, sCi, sDi)**.

### Step 1 — Strings and embedding

| Input | Role |
|--------|------|
| **sA**, **sB** | Actor and target entities (shared **entity** converter). |
| **sCi** | Personality string for lens **i** (**personality** converter). |
| **sDi** | Context string for lens **i** (**context** converter). |

Pipeline per string: **s → MiniLM-v2 → z ∈ ℝ³⁸⁴ → converter_k → e ∈ ℝ⁶⁴**. MiniLM is frozen; each converter is a learned two-layer MLP (triplet-trained). **Entity** space is shared by **A** and **B** so **cosine similarity between A and B is meaningful**.

### Step 2 — Actor–receiver blend **ABn**

- **cos_AB = (A·B) / (‖A‖‖B‖)** ∈ [−1, 1]
- **ABn = tanh(A ⊙ (1 − cos_AB)) + tanh(B ⊙ cos_AB)** ∈ ℝ⁶⁴  
  (actor-weighted when A and B differ; receiver-weighted when they align.)

### Step 3 — Outer product **M**

**M = C ⊗ D ∈ ℝ⁶⁴ˣ⁶⁴** (one **64×64** block per mini-model from its **Cᵢ** and **Dᵢ**).

### Step 4 — Residual refinement (×3)

**M⁽ᵗ⁾ = (1 + α) M⁽ᵗ⁻¹⁾ + B**, **t = 1,2,3**  
**α ∈ ℝ** shared; **B ∈ ℝ⁶⁴ˣ⁶⁴** per-cell learned bias. Result **M′**.

### Step 5 — Collapse to **p̂ᵢ**

**h = M′ · ABn ∈ ℝ⁶⁴** (matrix–vector product treating **ABn** as a column).  
**p̂ᵢ = softmax(W_out⁽ⁱ⁾ h + b_out⁽ⁱ⁾) ∈ ℝ²** with **W_out⁽ⁱ⁾ ∈ ℝ²ˣ⁶⁴**.  
Output **p̂ᵢ = [p̂⁺, p̂⁻]**, **p̂⁺ + p̂⁻ = 1**.

---

## Part 2 — Aggregation head (runs once)

Inputs: all **{Cᵢ, p̂ᵢ}**, plus **global ABn** and **global Dn** for the situation (spec Step 9).

### Step 6 — Sharpen personalities

- **C̃ᵢ = softmax(N · Cᵢ)** (PDF uses **N = 100** before softmax.)
- **C′ᵢ = w ⊙ C̃ᵢ + b** with **shared** **w, b ∈ ℝ⁶⁴**.

### Step 7 — Confidence gating

**C″ᵢ = tanh(p̂⁺ᵢ · C′ᵢ) + tanh(p̂⁻ᵢ · C′ᵢ)** ∈ ℝ⁶⁴.

### Step 8 — Ensemble matrix **E**

Stack rows **C″ᵢ** → **E ∈ ℝᴺˣ⁶⁴** (PDF: **100×64**).

### Step 9 — Situation vector **ABDn**

**ABDn = tanh(wD ⊙ ABn ⊙ Dn + bD)** ∈ ℝ⁶⁴, **shared** **wD, bD ∈ ℝ⁶⁴**.

### Step 10 — Poll

**v = E @ ABDn ∈ ℝᴺ** (equivalent to **ABDn · Eᵀ** as row vectors).

### Step 11 — Bottlenecks (×3)

Each pass: **ℝᴺ → ℝ¹²⁸ → ℝᴺ** with **ReLU**; **independent** **W↑**, **W↓** per pass (no weight sharing between passes). Produces **v⁽³⁾**.

### Step 12 — Final output

**ŷ = softmax(W_out v⁽³⁾ + b_out)**, **ŷ = [ŷ⁺, ŷ⁻]**, **W_out ∈ ℝ²ˣᴺ**.  
**ŷ⁺**: probability **A** takes the specified action toward **B**; **ŷ⁻** the complement.

---

## End-to-end flow (spec)

1. For each **i**: embed strings → **A, B, Cᵢ, Dᵢ** → **ABn** → **M → M′** → **h** → **p̂ᵢ**.  
2. Once: **{Cᵢ, p̂ᵢ}** → Steps 6–8 → **E**; **ABn, Dn** → **ABDn** → **v** → bottlenecks → **ŷ**.

---

## Repository layout

| Location | Purpose |
|----------|---------|
| **`apollo/`** | Importable package: `perspective_stages`, `predict_from_strings`, `perspective_event_head`, converters, `personality_bank`, `paths` (repo-root paths). |
| **`scripts/`** | Runnable entry points (training, data builds, `run_full_pipeline.py`). Run from repo root: `python scripts/<name>.py`. |
| **`tests/`** | `pytest` |
| **`data/`**, **`entity_embeddings/`**, … | Datasets and trained embedding tables (unchanged locations). |

---

## What this repository implements

| Spec piece | Code | Notes |
|------------|------|--------|
| Step 1 embeddings + converters | Artifact dirs at repo root; `apollo/*_minilm_converter.py`; `scripts/train_*_embeddings.py`, `scripts/train_all_converters.py` | MiniLM → 64-d as in the spec. |
| Step 2 **ABn** | `apollo.perspective_stages.compute_abn` | Matches PDF blend. |
| Steps 3–5 (per mini-model) | `apollo.perspective_stages.Stages1to5` | **N×N** grid when **N>1**; **Step 5** differs from PDF (**concat head**). Train Part 1 with `scripts/train_perspective_stages.py`. |
| Part 2 Steps 6–12 | `apollo.perspective_event_head.PerspectiveEventHead` | Matches PDF aggregation; `scripts/train_perspective_event_head.py`, `scripts/build_perspective_event_head_training_data.py`. |
| **N = 100** personalities | `apollo/personality_bank.py`, `apollo/personalities_100.json` | |
| Full inference (Part 1 loop + Part 2) | `scripts/run_full_pipeline.py` | Optional `data/perspective_event_head.pt`. |
| Single-lens API | `apollo.predict_from_strings` | One **N=1** forward. CLI: `python -m apollo.predict_from_strings …`. |

**Gaps vs main (24).pdf**

- **Independent mini-model weights:** The spec requires **100 separate** parameter sets. Training/inference often uses **one** `perspective_stages.pt` with **N=1**, repeated with different **C** — an approximation unless you train **N=100** or per-index checkpoints.
- **Step 5:** Implementation does not strictly use **M′·ABn** → **softmax**; see `perspective_stages.py`.
- **sDi per i:** The spec allows a **different context string per mini-model**; `run_full_pipeline.py` typically uses **one** shared context embedding **Dn** for all **i** (valid as a special case).

---

## Setup

```bash
pip install -r requirements.txt
```

## Tests

From the **repository root**, `pyproject.toml` sets `pythonpath = ["."]` for pytest:

```bash
python -m pytest tests/ -v
```

## Training / data (Part 2)

```bash
python scripts/build_perspective_event_head_training_data.py --num-samples 100000
python scripts/train_perspective_event_head.py
```

Saves `data/perspective_event_head.pt`.

## Inference (full stack)

```bash
python scripts/run_full_pipeline.py
python scripts/run_full_pipeline.py --max-personalities 20 --agg-head data/perspective_event_head.pt
```

The aggregation checkpoint in `data/perspective_event_head.pt` is built for **the same N** as in training (default vault **N = 100**). If you pass `--max-personalities` **K** with **K ≠ N**, either omit a checkpoint (point `--agg-head` at a non-existent path) so Part 2 uses random weights, or use **all** vault personalities so **n** matches the saved head.

## Code: aggregation head only

```python
import torch
from apollo import PerspectiveEventHead

n, p, q = 100, 64, 64
head = PerspectiveEventHead(n=n, p=p, q=q)
C = torch.randn(n, p)
p_hat = torch.softmax(torch.randn(n, 2), dim=-1)
abn = torch.randn(p)
d_n = torch.randn(q)
y = head(C, p_hat, abn, d_n)  # (2,), nonnegative, sums to 1
# Training: logits = head.forward_logits(C, p_hat, abn, d_n)
```

---

## Reference

**main (24).pdf** — complete confirmed specification (notation, intuition, diagrams).  
Earlier drafts (e.g. main (21).pdf) may differ; this README follows **main (24).pdf**.
