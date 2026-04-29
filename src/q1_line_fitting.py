import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('output/q1', exist_ok=True)

D = np.genfromtxt('lines.csv', delimiter=',', skip_header=1)


# Q1(a) - Total Least Squares on Line 1 (x1, y1 columns only)

x1 = D[:, 0]
y1 = D[:, 3]

# Center the data, then SVD - the last right singular vector gives the
# line normal [a, b] that minimises perpendicular (orthogonal) distances
mx, my = x1.mean(), y1.mean()
_, _, Vt = np.linalg.svd(np.column_stack([x1 - mx, y1 - my]))
a, b = Vt[-1]
c = -(a * mx + b * my)

slope = -a / b
intercept = -c / b

print("Q1(a) Total Least Squares - Line 1")
print(f"  Normal form : {a:.6f}*x + {b:.6f}*y + {c:.6f} = 0")
print(f"  y = {slope:.6f}*x + {intercept:.6f}")

x_range = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x1, y1, color='steelblue', s=30, label='Line 1 data', zorder=3)
ax.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2,
        label=f'TLS fit: y = {slope:.4f}x + {intercept:.4f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Q1(a): Total Least Squares Fit — Line 1')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/q1/q1a_tls_line1.png', dpi=150)
plt.close()


# Q1(b) - RANSAC to recover all three lines from the combined point set

X_all = D[:, :3].flatten()
Y_all = D[:, 3:].flatten()


def tls_fit(x, y):
    mx, my = x.mean(), y.mean()
    _, _, Vt = np.linalg.svd(np.column_stack([x - mx, y - my]))
    a, b = Vt[-1]
    c = -(a * mx + b * my)
    return a, b, c


def line_distances(x, y, a, b, c):
    return np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


def ransac_line(x, y, n_iter=2000, threshold=0.5):
    rng = np.random.default_rng(42)
    best_inliers = None
    best_count = 0

    for _ in range(n_iter):
        idx = rng.choice(len(x), 2, replace=False)
        a, b, c = tls_fit(x[idx], y[idx])
        inliers = np.where(line_distances(x, y, a, b, c) < threshold)[0]
        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers

    # Refit on the full inlier set for a more accurate line
    a, b, c = tls_fit(x[best_inliers], y[best_inliers])
    return a, b, c, best_inliers


print("\nQ1(b) RANSAC - Three Lines")

rem_x = X_all.copy()
rem_y = Y_all.copy()
rem_idx = np.arange(len(X_all))

lines = []
inlier_groups = []
colors = ['tomato', 'seagreen', 'mediumpurple']

for i in range(3):
    a, b, c, local_idx = ransac_line(rem_x, rem_y)
    global_idx = rem_idx[local_idx]
    lines.append((a, b, c))
    inlier_groups.append(global_idx)

    s = -a / b
    ic = -c / b
    print(f"  Line {i+1}: {a:.6f}*x + {b:.6f}*y + {c:.6f} = 0")
    print(f"          y = {s:.6f}*x + {ic:.6f}  ({len(global_idx)} inliers)")

    # Remove inliers before searching for the next line
    mask = np.ones(len(rem_x), dtype=bool)
    mask[local_idx] = False
    rem_x = rem_x[mask]
    rem_y = rem_y[mask]
    rem_idx = rem_idx[mask]

fig, ax = plt.subplots(figsize=(9, 6))
used = np.zeros(len(X_all), dtype=bool)

for i, (global_idx, (a, b, c)) in enumerate(zip(inlier_groups, lines)):
    ax.scatter(X_all[global_idx], Y_all[global_idx],
               color=colors[i], s=25, label=f'Line {i+1} inliers', zorder=3)
    used[global_idx] = True
    xi = X_all[global_idx]
    xr = np.linspace(xi.min() - 0.5, xi.max() + 0.5, 200)
    s, ic = -a / b, -c / b
    ax.plot(xr, s * xr + ic, color=colors[i], linewidth=2,
            label=f'Fit {i+1}: y={s:.3f}x+{ic:.3f}')

if (~used).any():
    ax.scatter(X_all[~used], Y_all[~used], color='gray', s=15,
               alpha=0.5, label='Outliers')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Q1(b): RANSAC — Three Line Fits')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/q1/q1b_ransac_3lines.png', dpi=150)
plt.close()
