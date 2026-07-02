import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('xy_data.csv')
x_obs = df['x'].values
y_obs = df['y'].values
N = len(x_obs)

print(f"Loaded {N} points.")


def parametric_curve(t, theta, M, X):
    exp_term = np.exp(M * np.abs(t))
    sin_term = np.sin(0.3 * t)
    x = t * np.cos(theta) - exp_term * sin_term * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + exp_term * sin_term * np.cos(theta)
    return x, y


def find_best_t_for_points(theta, M, X):
    t_grid = np.linspace(6.0, 60.0, 1000)
    x_grid, y_grid = parametric_curve(t_grid[:, None], theta, M, X)
    dists = np.abs(x_grid - x_obs) + np.abs(y_grid - y_obs)
    t_est = t_grid[np.argmin(dists, axis=0)]

    for _ in range(3):
        delta = 0.05
        t_candidates = np.array([t_est - delta, t_est, t_est + delta])
        x_c, y_c = parametric_curve(t_candidates, theta, M, X)
        d_c = np.abs(x_c - x_obs) + np.abs(y_c - y_obs)
        t_est = t_candidates[np.argmin(d_c, axis=0), np.arange(N)]

    return t_est


def objective_global(params):
    theta, M, X = params
    t_est = find_best_t_for_points(theta, M, X)
    x_pred, y_pred = parametric_curve(t_est, theta, M, X)
    return np.mean(np.abs(x_pred - x_obs) + np.abs(y_pred - y_obs))


bounds = [
    (0.0, 50.0 * np.pi / 180.0),
    (-0.05, 0.05),
    (0.0, 100.0)
]

x0 = [26.4 * np.pi / 180.0, 0.013, 50.3]

res = minimize(objective_global, x0, bounds=bounds, method='Nelder-Mead')

theta_opt, M_opt, X_opt = res.x
print(f"\ntheta: {theta_opt:.8f} rad  ({np.degrees(theta_opt):.5f} deg)")
print(f"M:     {M_opt:.8f}")
print(f"X:     {X_opt:.8f}")
print(f"L1:    {res.fun:.8f}")

with open('results.txt', 'w') as f:
    f.write(f"theta_rad: {theta_opt}\n")
    f.write(f"theta_deg: {np.degrees(theta_opt)}\n")
    f.write(f"M: {M_opt}\n")
    f.write(f"X: {X_opt}\n")
    f.write(f"L1_loss: {res.fun}\n")
