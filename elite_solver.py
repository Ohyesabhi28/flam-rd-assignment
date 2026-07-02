import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('xy_data.csv')
x_obs = df['x'].values
y_obs = df['y'].values
N = len(x_obs)
print(f"Loaded {N} points  |  x: [{x_obs.min():.4f}, {x_obs.max():.4f}]  |  y: [{y_obs.min():.4f}, {y_obs.max():.4f}]")


def curve(t, theta, M, X):
    e = np.exp(M * np.abs(t))
    s = np.sin(0.3 * t)
    cx, sx = np.cos(theta), np.sin(theta)
    x = t * cx - e * s * sx + X
    y = 42.0 + t * sx + e * s * cx
    return x, y


T_GRID = np.linspace(6.0, 60.0, 2000)

def best_t(theta, M, X, refine_iters=5, delta=0.027):
    xg, yg = curve(T_GRID[:, None], theta, M, X)
    d = np.abs(xg - x_obs) + np.abs(yg - y_obs)
    t_est = T_GRID[np.argmin(d, axis=0)]

    for _ in range(refine_iters):
        offsets = np.linspace(-delta, delta, 9)
        candidates = np.clip(t_est + offsets[:, None], 6.0, 60.0)
        xc, yc = curve(candidates, theta, M, X)
        dc = np.abs(xc - x_obs) + np.abs(yc - y_obs)
        t_est = candidates[np.argmin(dc, axis=0), np.arange(N)]
        delta *= 0.4

    return t_est


def objective(params):
    theta, M, X = params
    t = best_t(theta, M, X)
    xp, yp = curve(t, theta, M, X)
    return np.mean(np.abs(xp - x_obs) + np.abs(yp - y_obs))


# analytical estimate: y_min ~ 46 at t=6 -> sin(theta) ~ 4/6 -> theta ~ 30 deg after correction
theta_boot = 30.0 * np.pi / 180.0
print(f"\nStarting estimate: theta={np.degrees(theta_boot):.1f} deg, M=0.03, X=55")

print("\n[Stage 1] Differential Evolution (global)...")
t0 = time.time()

bounds_de = [
    (0.0, 50.0 * np.pi / 180.0),
    (-0.05, 0.05),
    (0.0, 100.0),
]

result_de = differential_evolution(
    objective,
    bounds_de,
    maxiter=400,
    popsize=20,
    tol=1e-10,
    mutation=(0.5, 1.5),
    recombination=0.9,
    seed=42,
    workers=1,
    polish=True,
    init='sobol',
    disp=True,
)

print(f"[Stage 1] Done in {time.time()-t0:.1f}s  |  L1={result_de.fun:.8f}")
p1 = result_de.x

print("\n[Stage 2] Multi-start Nelder-Mead...")
best_params = p1
best_loss = result_de.fun

starts = [
    p1,
    [30.0 * np.pi/180, 0.030, 55.0],
    [28.0 * np.pi/180, 0.032, 56.0],
    [32.0 * np.pi/180, 0.028, 54.0],
    [30.0 * np.pi/180, 0.030, 54.5],
    [30.0 * np.pi/180, 0.030, 55.5],
]

for i, s in enumerate(starts):
    r = minimize(objective, s,
                 method='Nelder-Mead',
                 options={'xatol': 1e-11, 'fatol': 1e-11,
                          'maxiter': 100000, 'maxfev': 500000})
    if r.fun < best_loss:
        best_loss = r.fun
        best_params = r.x
        print(f"  [start {i}] New best: L1={best_loss:.8f}  "
              f"theta={np.degrees(best_params[0]):.5f} deg  "
              f"M={best_params[1]:.6f}  X={best_params[2]:.5f}")

print("\n[Stage 3] L-BFGS-B fine-tuning...")
r3 = minimize(objective, best_params,
              method='L-BFGS-B',
              bounds=[(0, 50*np.pi/180), (-0.05, 0.05), (0, 100)],
              options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 10000, 'maxfun': 100000})
if r3.fun < best_loss:
    best_loss = r3.fun
    best_params = r3.x
    print(f"  Improved: L1={best_loss:.8f}")
else:
    print(f"  No improvement (best stays {best_loss:.8f})")

theta_opt, M_opt, X_opt = best_params

print("\n" + "="*60)
print("  FINAL RESULTS")
print("="*60)
print(f"  theta (radians) = {theta_opt:.10f}")
print(f"  theta (degrees) = {np.degrees(theta_opt):.8f}")
print(f"  M               = {M_opt:.10f}")
print(f"  X               = {X_opt:.10f}")
print(f"  L1 Loss         = {best_loss:.10f}")
print("="*60)

theta_s = f"{theta_opt:.6f}"
M_s     = f"{abs(M_opt):.6f}"
M_sign  = "" if M_opt >= 0 else "-"

latex = (
    r"\left(t\cdot\cos(" + theta_s + r")"
    r"-e^{" + M_sign + M_s + r"\left|t\right|}\cdot\sin(0.3t)\cdot\sin(" + theta_s + r")"
    r"+" + f"{X_opt:.6f}" + r","
    r"42+t\cdot\sin(" + theta_s + r")"
    r"+e^{" + M_sign + M_s + r"\left|t\right|}\cdot\sin(0.3t)\cdot\cos(" + theta_s + r")"
    r"\right)"
)

print(f"\n  Desmos: https://www.desmos.com/calculator/fl6ozkr1xv")
print(f'\n  "{latex}"\n')

with open('results_elite.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("FLAM AI R&D ASSIGNMENT - FINAL RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"theta_rad: {theta_opt}\n")
    f.write(f"theta_deg: {np.degrees(theta_opt)}\n")
    f.write(f"M:         {M_opt}\n")
    f.write(f"X:         {X_opt}\n")
    f.write(f"L1_loss:   {best_loss}\n\n")
    f.write("Desmos LaTeX:\n")
    f.write(f'"{latex}"\n')

print("[+] Saved to results_elite.txt")

t_uniform = np.linspace(6, 60, N)
x_u, y_u = curve(t_uniform, theta_opt, M_opt, X_opt)
idx_obs  = np.argsort(x_obs)
idx_pred = np.argsort(x_u)
l1_uniform = np.mean(
    np.abs(x_u[idx_pred] - x_obs[idx_obs]) +
    np.abs(y_u[idx_pred] - y_obs[idx_obs])
)
print(f"\n  Uniform-t L1 (sorted match):   {l1_uniform:.6f}")
print(f"  Nearest-point L1 (optimized):  {best_loss:.8f}")
