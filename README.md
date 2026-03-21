# Lancer — Perspective-conditioned geopolitical event prediction

This repository implements pieces of the neural architecture described in **main (21).pdf** (*Perspective-Conditioned Geopolitical Event Prediction — A Full Architecture Specification*). The model predicts whether one real-world entity will take a meaningful action toward another (invasion, election, trade deal, etc.) as a **two-way probability**: chance the action happens, and chance it does not.

## Core ideas

1. **No single predictor.** The network runs **N independent sub-models**, one per learned **personality perspective** (e.g. military vs. diplomatic vs. economic “lens”). Each casts a vote; votes are aggregated.
2. **Global context.** Every prediction is conditioned on **N context vectors** that summarize slices of the current world state. The same actor–target pair can look different under famine, elections, or crisis.
3. **Three separate embedding spaces.** Actor/target share one space; personality vectors live in another; context vectors in a third. Distances in actor–target space carry geometric meaning; personalities and contexts are not confused with “looking like a country.”

## The four inputs (each forward pass)

| Symbol | Space | Meaning |
|--------|--------|---------|
| **A** | ℝ^d | Actor — who might act (e.g. a government, a public figure). |
| **B** | ℝ^d | Target — who the action would be directed at. |
| **Cᵢ** | ℝ^p | One of **N** personality vectors (analytical perspectives). |
| **Dⱼ** | ℝ^q | One of **N** context vectors (global situation slices). |

**N** is a hyperparameter (same count for personalities and contexts in the spec). Dimensions **d**, **p**, **q** are fixed by design.

## Pipeline (stages in the PDF)

### Stage 1 — Personality–context matrix **M**

All pairs (personality **i**, context **j**) are combined as an outer product:

**Mᵢⱼ = Cᵢ ⊗ Dⱼ**

Stacked, **M** is **N×N** (each cell is a matrix block; the spec treats the structure as an **N×N** grid of such combinations).

### Stage 2 — Residual refinement

Three passes refine **M** without destroying the original signal:

**M⁽ᵗ⁾ = α·M⁽ᵗ⁻¹⁾ + B + M⁽ᵗ⁻¹⁾**  →  **M′**

**α** is one shared scalar; **B** is a learned **N×N** bias (one bias per cell). After three passes: **M′**.

### Stage 3 — Projection into actor–target space

**Q = W_Q M′**, **K = W_K M′**, with **W_Q**, **W_K ∈ ℝᵈˣᴺ**, yielding **Q**, **K ∈ ℝᵈˣᴺ** (columns live in the same **d**-space as **A** and **B**).

### Stage 4 — Blend actor and target, then fuse context

- **cos_AB = (A·B) / (‖A‖‖B‖)** in **[-1, 1]**.
- Blended vector **ABₙ** mixes **A** and **B** using **tanh** and **cos_AB** (when actors are similar, **B** dominates; when dissimilar, **A** dominates).
- For each context **j**: **ABDⱼ = tanh(wⱼ ⊙ ABₙ ⊙ Dⱼ + bⱼ)** (element-wise; learned **wⱼ**, **bⱼ** per context).

### Stage 5 — N mini-models → ensemble matrix **E**

For each personality **i**:

1. **C̃ᵢ = softmax(N · Cᵢ)**
2. **C*ᵢ = ReLU(Wᵢ⁽¹⁾ C̃ᵢ + bᵢ⁽¹⁾)** (weights unique per **i**)
3. Confidence **p̂ᵢ** (two probabilities) gates **C*ᵢ** via **tanh** terms → row **C″ᵢ**
4. Rows stack to **E ∈ ℝᴺˣᵖ**

*(The PDF leaves the map to **p̂ᵢ** open; the implementation uses a small softmax head on **C*ᵢ**.)*

### Stage 6 — Polling

**v = ABDⱼ · Eᵀ** → **v ∈ ℝᴺ** (one score per personality for this situation).

### Stage 7 — Bottleneck refinement

Three passes: **ℝᴺ → ℝ¹²⁸ → ℝᴺ** with **ReLU**, independent weights per pass → **v⁽³⁾**.

### Stage 8 — Output

**ŷ = softmax(W_out v⁽³⁾)** with **ŷ = (ŷ⁺, ŷ⁻)**, **ŷ⁺ + ŷ⁻ = 1**.

## What this repo implements

| Piece | Status |
|-------|--------|
| Stages 1–4 (M, residual, Q/K, **ABDⱼ**) | Not in code yet — supply **C** and **abd** (your **ABDⱼ**) externally. |
| Stages 5–8 | [`lancer/perspective_event_head.py`](lancer/perspective_event_head.py) — `PerspectiveEventHead`, `MiniModelBank`, `BottleneckStack`. |

The head expects a personality bank **C** with shape **(N, p)** and a single fused vector **abd** with shape **(d,)**. If **d ≠ p**, a learned linear map aligns **abd** into ℝᵖ before the poll (see module docstring).

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
set PYTHONPATH=.
python -m pytest tests/ -v
```

(On PowerShell: `$env:PYTHONPATH="."`.)

## Minimal usage

```python
import torch
from lancer import PerspectiveEventHead

n, p, d = 8, 64, 32
head = PerspectiveEventHead(n=n, p=p, d=d)
C = torch.randn(n, p)
abd = torch.randn(d)
y = head(C, abd)  # shape (2,), nonnegative, sums to 1
```

## Reference

Full notation and intuition: **main (21).pdf** in the project (architecture specification).
