# Apollo — Actor-target Prediction through Personality-conditioned Large Likelihood Optimization

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

### Part 2 — Aggregation head (main (24).pdf)

After Part 1, each mini-model has produced **p̂ᵢ**. Steps 6–8 build **E** from **Cᵢ** and **p̂ᵢ** (shared **w**, **b** sharpen; tanh confidence gate). Step 9 forms **ABDn = tanh(wD ⊙ ABn ⊙ Dn + bD)**. Step 10 polls **v = E @ ABDn**. Steps 11–12: three bottlenecks on **v**, then **ŷ = softmax(W_out v⁽³⁾)**.

## What this repo implements

| Piece | Status |
|-------|--------|
| Part 1 (per mini-model, incl. **ABn**, **M′**, **p̂ᵢ**) | [`perspective_stages.py`](perspective_stages.py) — `Stages1to5` (differs slightly from main (24) on Step 5 wiring; see that module). |
| Part 2 aggregation (Steps 6–12) | [`apollo/perspective_event_head.py`](apollo/perspective_event_head.py) — `PerspectiveEventHead`, `BottleneckStack`. |

The aggregation head expects **C** (N, p), Part 1 votes **p_hat** (N, 2), actor–receiver blend **ABn** (p,), and global context **Dn** (q,) with optional **Linear(q → p)** inside Step 9 when **q ≠ p**.

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
from apollo import PerspectiveEventHead

n, p, q = 8, 64, 64
head = PerspectiveEventHead(n=n, p=p, q=q)
C = torch.randn(n, p)
p_hat = torch.softmax(torch.randn(n, 2), dim=-1)
abn = torch.randn(p)
d_n = torch.randn(q)
y = head(C, p_hat, abn, d_n)  # shape (2,), nonnegative, sums to 1
```

## Reference

Full notation and intuition: **main (21).pdf** in the project (architecture specification).
