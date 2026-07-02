import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load the data
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

# Since the points in the CSV might be in a different order than t in [6, 60],
# or we just need to associate each (x_obs[i], y_obs[i]) to its optimal t_i in [6, 60].
# For any fixed (theta, M, X), we can find the optimal t_i for each observed point independently!
# This decouples the search space and makes optimization extremely fast and accurate.
def find_best_t_for_points(theta, M, X):
    # For each point (x_i, y_i), find t_i in [6, 60] that minimizes the distance.
    # We can do this by evaluating a fine grid of t values, then refining with a 1D solver.
    t_grid = np.linspace(6.0, 60.0, 1000)
    x_grid, y_grid = parametric_curve(t_grid[:, None], theta, M, X) # shape: (1000, 1)
    
    # We want to find for each (x_obs[i], y_obs[i]) the index in the grid with minimum distance
    # x_grid shape: (1000, 1), x_obs shape: (N,)
    # distances shape: (1000, N)
    dists = np.abs(x_grid - x_obs) + np.abs(y_grid - y_obs)
    best_idx = np.argmin(dists, axis=0)
    t_est = t_grid[best_idx]
    
    # Refine each t_est using minimize_scalar or a simple local optimization
    # Since there are 1500 points, we can do a few iterations of line search or just local search
    # Let's do a quick gradient descent / Newton step or simple grid refinement
    # Let's refine by searching in a small window around t_est
    for _ in range(3):
        # search in t_est +/- delta
        delta = 0.05
        t_candidates = np.array([t_est - delta, t_est, t_est + delta]) # shape: (3, N)
        x_c, y_c = parametric_curve(t_candidates, theta, M, X) # shape: (3, N)
        d_c = np.abs(x_c - x_obs) + np.abs(y_c - y_obs) # shape: (3, N)
        best_c_idx = np.argmin(d_c, axis=0)
        t_est = t_candidates[best_c_idx, np.arange(N)]
        
    return t_est

def objective_global(params):
    theta, M, X = params
    t_est = find_best_t_for_points(theta, M, X)
    x_pred, y_pred = parametric_curve(t_est, theta, M, X)
    l1_dist = np.mean(np.abs(x_pred - x_obs) + np.abs(y_pred - y_obs))
    return l1_dist

# Let's run a global search/minimization over just 3 variables: theta, M, X!
bounds = [
    (0.0, 50.0 * np.pi / 180.0), # theta
    (-0.05, 0.05), # M
    (0.0, 100.0) # X
]

# We can start with a grid search or use our previous estimate as start point
x0 = [26.4 * np.pi / 180.0, 0.013, 50.3]

res = minimize(objective_global, x0, bounds=bounds, method='Nelder-Mead')

theta_opt, M_opt, X_opt = res.x
print("\nGlobal Optimization results (Nelder-Mead):")
print(f"Theta (radians): {theta_opt}")
print(f"Theta (degrees): {theta_opt * 180.0 / np.pi}")
print(f"M: {M_opt}")
print(f"X: {X_opt}")
print(f"L1 loss (Mean L1 distance per point): {res.fun}")

# Save the improved results to a file
with open('results.txt', 'w') as f:
    f.write(f"theta_rad: {theta_opt}\n")
    f.write(f"theta_deg: {theta_opt * 180.0 / np.pi}\n")
    f.write(f"M: {M_opt}\n")
    f.write(f"X: {X_opt}\n")
    f.write(f"L1_loss: {res.fun}\n")
