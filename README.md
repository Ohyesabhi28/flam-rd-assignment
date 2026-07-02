# Flam AI R&D Assignment — Solution

> **Candidate:** Abhinav Nair  
> **Role Applied:** Research & Development / AI  

---

## Problem

Find the unknown parameters **θ (theta)**, **M**, and **X** for the parametric curve:

$$x = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X$$

$$y = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)$$

**Constraints:**
| Parameter | Range |
|-----------|-------|
| θ | 0° < θ < 50° |
| M | −0.05 < M < 0.05 |
| X | 0 < X < 100 |
| t | 6 < t < 60 |

**Dataset:** `xy_data.csv` — 1500 observed (x, y) points lying on the curve.

---

## Final Answer

| Parameter | Value |
|-----------|-------|
| **θ (radians)** | `0.5235983019` |
| **θ (degrees)** | `29.99997°` ≈ **exactly 30°** |
| **M** | `0.0300000002` ≈ **exactly 0.03** |
| **X** | `54.9999951` ≈ **exactly 55** |
| **L1 Loss** | `0.00006` (nearest-point) |

> **The parameters converge to clean exact values — θ = π/6, M = 3/100, X = 55 — strongly suggesting these are the true ground-truth parameters.**

### Desmos Verification

Verify the curve visually: [Desmos Parametric Calculator](https://www.desmos.com/calculator/fl6ozkr1xv)

---

## Approach & Methodology

### Step 1 — Understanding the Curve Structure

The parametric curve has three free parameters: **θ** (rotation angle), **M** (exponential growth rate), **X** (horizontal offset). The parameter **t** runs from 6 to 60 and is not directly observed — it must be inferred.

**Key insight:** For every set of (θ, M, X), we can independently find the optimal `t_i` for each observed point. This **decouples** the 3D parameter search from the 1500D t-assignment problem.

### Step 2 — Analytical Bootstrap

Before any numerical search, I derived rough bounds analytically:

- At `t = 6`, the curve is at minimum y. The data shows `y_min ≈ 46.03`.
- Ignoring the exponential correction: `y - 42 ≈ t·sin(θ)` → `sin(θ) ≈ (46-42)/6 ≈ 0.5` → `θ ≈ 30°`
- `X` can be estimated from `x_mean - t_mean·cos(θ) ≈ 83.7 - 33·cos(30°) ≈ 55`

This gave us an excellent starting point before any optimization.

### Step 3 — Vectorised Per-Point t Assignment

For a fixed (θ, M, X), finding the best `t_i` for each point:

```python
T_GRID = np.linspace(6.0, 60.0, 2000)  # coarse grid
x_grid, y_grid = curve(T_GRID[:, None], theta, M, X)  # (2000, N) broadcast
distances = |x_grid - x_obs| + |y_grid - y_obs|       # L1 per point
t_est = T_GRID[argmin(distances, axis=0)]              # (N,) best t per point
```

Then iteratively refined with a shrinking window search (5 iterations, starting Δ=0.027).

### Step 4 — Global Optimisation (Differential Evolution)

Used `scipy.optimize.differential_evolution` with:
- **Population size:** 20
- **Max iterations:** 400
- **Initialisation:** Sobol sequence (quasi-random, better coverage than random)
- **Polish:** True (L-BFGS-B polish after convergence)
- **Tolerance:** 1e-10

Differential Evolution is gradient-free and escapes local minima — critical here since the objective surface is non-convex due to the exponential term.

### Step 5 — Multi-Start Nelder-Mead

Ran Nelder-Mead from 6 different starting points (including the DE result and analytically motivated starts) to refine the solution further. Nelder-Mead is fast and handles non-differentiable objectives well.

### Step 6 — L-BFGS-B Fine-Tuning

Final high-precision gradient-based refinement using L-BFGS-B with tolerances of 1e-15 to squeeze out maximum precision.

---

## Code Structure

```
Flam/
├── elite_solver.py        ← Main solution (primary script)
├── visualize.py           ← Plot fitted curve + residuals → curve_fit.png
├── xy_data.csv            ← Given dataset (1500 points)
├── results_elite.txt      ← Final parameter values (auto-generated)
├── requirements.txt       ← Python dependencies
├── exploratory/
│   └── initial_solver.py  ← Early rough solver (superseded)
└── README.md              ← This file
```

### Quick Start

```bash
pip install -r requirements.txt
python elite_solver.py    # runs optimization, saves results_elite.txt
python visualize.py       # generates curve_fit.png
```

**Output:**
- Printed final θ, M, X values with 10 decimal precision
- Desmos-ready LaTeX string
- `results_elite.txt` with all values

---

## Verification

![Fitted curve vs observed data — θ=30°, M=0.03, X=55](curve_fit.png)

The curve `x ∈ [59.61, 109.23]` and `y ∈ [46.01, 69.69]` matches the data range `x ∈ [59.66, 109.23]`, `y ∈ [46.03, 69.69]` almost perfectly — confirming the parameters are correct.

The **nearest-point L1 loss = 0.00006** (mean absolute deviation per point after 877s of global optimization) represents an extremely tight fit. The parameters converge to suspiciously clean values:

| Parameter | Optimized | Clean Form | Match |
|-----------|-----------|------------|-------|
| θ | 0.5235983 rad | π/6 = 0.5235988 rad | ✅ |
| M | 0.0300000002 | 3/100 | ✅ |
| X | 54.9999951 | 55 | ✅ |

This strongly suggests the **true ground-truth values are θ = π/6 (30°), M = 0.03, X = 55**.

---

*Submitted by Abhinav Nair — Backend Developer & AI Engineer*
