import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('xy_data.csv')
x_obs, y_obs = df['x'].values, df['y'].values

results_file = 'results_elite.txt' if os.path.exists('results_elite.txt') else 'results.txt'
params = {}
with open(results_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if ':' in line and not line.startswith('=') and not line.startswith('"') \
                and not line.startswith('Desmos') and not line.startswith('FLAM'):
            k, v = line.split(':', 1)
            try:
                params[k.strip()] = float(v.strip())
            except ValueError:
                pass

theta = params.get('theta_rad', 0.5235983019)
M     = params.get('M', 0.03)
X     = params.get('X', 55.0)
L1    = params.get('L1_loss', None)

print(f"theta={np.degrees(theta):.5f} deg  M={M:.6f}  X={X:.5f}")


def curve(t, theta, M, X):
    e = np.exp(M * np.abs(t))
    s = np.sin(0.3 * t)
    x = t * np.cos(theta) - e * s * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + e * s * np.cos(theta)
    return x, y


t_dense = np.linspace(6, 60, 5000)
x_curve, y_curve = curve(t_dense, theta, M, X)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0d0b1a')

for ax in axes:
    ax.set_facecolor('#07060f')
    ax.tick_params(colors='#8b89b0')
    for spine in ax.spines.values():
        spine.set_color('#1e1c3a')

ax = axes[0]
ax.scatter(x_obs, y_obs, s=4, alpha=0.35, color='#5b4fcf', label='Observed data (1500 pts)', zorder=2)
ax.plot(x_curve, y_curve, color='#00e5ff', linewidth=2.5, label='Fitted curve', zorder=3)

for t_mark, label in [(6, 't = 6'), (60, 't = 60')]:
    xm, ym = curve(np.array([t_mark]), theta, M, X)
    ax.scatter(xm, ym, s=120, color='#f0168a', zorder=5)
    ax.annotate(label, (xm[0], ym[0]), textcoords='offset points',
                xytext=(8, 4), color='#f0168a', fontsize=9, fontweight='bold')

ax.set_title('Fitted Parametric Curve vs Observed Data',
             color='#e8e6ff', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('x', color='#8b89b0', fontsize=11)
ax.set_ylabel('y', color='#8b89b0', fontsize=11)
ax.legend(facecolor='#12102b', edgecolor='#1e1c3a', labelcolor='#e8e6ff', fontsize=9)
ax.grid(True, color='#1e1c3a', linewidth=0.5, alpha=0.5)

ax2 = axes[1]

T_GRID_VIS = np.linspace(6, 60, 2000)
xg, yg = curve(T_GRID_VIS[:, None], theta, M, X)
d = np.abs(xg - x_obs) + np.abs(yg - y_obs)
t_best = T_GRID_VIS[np.argmin(d, axis=0)]

delta = 0.027
for _ in range(5):
    offsets = np.linspace(-delta, delta, 9)
    candidates = np.clip(t_best + offsets[:, None], 6.0, 60.0)
    xc, yc = curve(candidates, theta, M, X)
    dc = np.abs(xc - x_obs) + np.abs(yc - y_obs)
    t_best = candidates[np.argmin(dc, axis=0), np.arange(len(x_obs))]
    delta *= 0.4

xp, yp = curve(t_best, theta, M, X)
residuals = np.abs(xp - x_obs) + np.abs(yp - y_obs)

ax2.hist(residuals, bins=60, color='#5b4fcf', alpha=0.75, edgecolor='#3d2fa0', linewidth=0.4)
ax2.axvline(residuals.mean(), color='#00e5ff', linewidth=2, linestyle='--',
            label=f'Mean L1 = {residuals.mean():.4f}')
ax2.axvline(np.median(residuals), color='#f0168a', linewidth=2, linestyle=':',
            label=f'Median  = {np.median(residuals):.4f}')

ax2.set_title('Per-Point L1 Residual Distribution',
              color='#e8e6ff', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('L1 Distance (|dx| + |dy|)', color='#8b89b0', fontsize=11)
ax2.set_ylabel('Count', color='#8b89b0', fontsize=11)
ax2.legend(facecolor='#12102b', edgecolor='#1e1c3a', labelcolor='#e8e6ff', fontsize=9)
ax2.grid(True, color='#1e1c3a', linewidth=0.5, alpha=0.5)

title_str = (
    f'theta = {np.degrees(theta):.4f} deg   |   M = {M:.5f}   |   X = {X:.4f}'
    + (f'   |   Mean L1 = {L1:.5f}' if L1 else '')
)
fig.suptitle(title_str, color='#ffb300', fontsize=11, fontweight='bold', y=0.01)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('curve_fit.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: curve_fit.png")
plt.show()
