import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import chi2

# ------------------------------------------------
# 1) Test intensity matrix (replace with your own)
# ------------------------------------------------
# Define grid for the matrix
ny, nx = 120, 150
y = np.linspace(-5, 5, ny)
x = np.linspace(-6, 6, nx)
X, Y = np.meshgrid(x, y, indexing="xy")  # X,Y shape: (ny, nx)

# Create a synthetic Gaussian surface + noise as test data
def gaussian2d(xy, A, x0, y0, sx, sy, theta, C):
    Xg, Yg = xy
    ct, st = np.cos(theta), np.sin(theta)
    Xc, Yc = Xg - x0, Yg - y0
    xr =  ct*Xc + st*Yc
    yr = -st*Xc + ct*Yc
    return (A * np.exp(-0.5*((xr/sx)**2 + (yr/sy)**2)) + C).ravel()

true = dict(A=300.0, x0=0.6, y0=-0.8, sx=1.3, sy=2.1, theta=np.deg2rad(25), C=5.0)
Z = gaussian2d((X, Y), **true).reshape(ny, nx)
rng = np.random.default_rng(7)
Z = Z + 10.0 * rng.standard_normal(Z.shape)  # add some noise

# ---- If you already have an intensity matrix, use this instead: ----
# Z = your_intensity_matrix  # shape (ny, nx)
# y = np.arange(Z.shape[0])  # or your known y-axis array
# x = np.arange(Z.shape[1])  # or your known x-axis array
# X, Y = np.meshgrid(x, y, indexing="xy")

# ------------------------------------------------
# 2) Fit a general 2D Gaussian (with rotation) to the matrix
# ------------------------------------------------
A0 = float(np.nanmax(Z) - np.nanmin(Z))
y0_idx, x0_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
x0_0, y0_0 = X[y0_idx, x0_idx], Y[y0_idx, x0_idx]
sx0, sy0 = (np.ptp(x)/6 if np.ptp(x) > 0 else 1.0,
            np.ptp(y)/6 if np.ptp(y) > 0 else 1.0)
theta0, C0 = 0.0, float(np.nanmedian(Z))
p0 = (A0, x0_0, y0_0, sx0, sy0, theta0, C0)
bounds = ((0, x.min(), y.min(), 1e-6, 1e-6, -np.pi/2, -np.inf),
          (np.inf, x.max(), y.max(),  np.inf,  np.inf,  np.pi/2,  np.inf))

popt, _ = curve_fit(gaussian2d, (X, Y), Z.ravel(), p0=p0, bounds=bounds, maxfev=40000)
A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit = popt

# Evaluate fitted surface on the grid
Z_fit = gaussian2d((X, Y), *popt).reshape(Z.shape)

# ------------------------------------------------
# 3) 3D plot: XY heat map + full Gaussian surface + XZ/YZ cross-sections
# ------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Base plane offset for the heat map
z0 = np.nanmin(Z) - 0.1 * (np.nanmax(Z) - np.nanmin(Z))
norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

# (a) Heat map on the XY plane (base)
ax.contourf(X, Y, Z, zdir='z', offset=z0, levels=60, cmap=cm.plasma)
mappable = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
mappable.set_array([])
fig.colorbar(mappable, ax=ax, shrink=0.75, label='Measured intensity')

# (b) Full 3D fitted Gaussian surface
ax.plot_surface(X, Y, Z_fit, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.7, cmap=cm.inferno)

# (c) Cross-section on the XZ plane (y = y0_fit)
x_line = np.linspace(x.min(), x.max(), 400)
y_const = np.full_like(x_line, y0_fit)
z_x_section = gaussian2d((x_line, y_const), A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)

# guide plane at y = y0_fit (transparent)
Xp_xz, Zp_xz = np.meshgrid(np.linspace(x.min(), x.max(), 2),
                           np.linspace(z0, np.nanmax(Z_fit), 2))
Yp_xz = np.full_like(Xp_xz, y0_fit)
ax.plot_surface(Xp_xz, Yp_xz - np.min(y), Zp_xz, alpha=0.00, linewidth=0, shade=False)

# the cross-section curve
ax.plot(x_line, y_const - np.min(y), z_x_section, linewidth=2.5)

# (d) Cross-section on the YZ plane (x = x0_fit)
y_line = np.linspace(y.min(), y.max(), 400)
x_const = np.full_like(y_line, x0_fit)
z_y_section = gaussian2d((x_const, y_line), A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)

# guide plane at x = x0_fit (transparent)
Yp_yz, Zp_yz = np.meshgrid(np.linspace(y.min(), y.max(), 2),
                           np.linspace(z0, np.nanmax(Z_fit), 2))
Xp_yz = np.full_like(Yp_yz, x0_fit)
ax.plot_surface(Xp_yz + (np.min(x)), Yp_yz, Zp_yz, alpha=0.00, linewidth=0, shade=False)

# the cross-section curve
ax.plot(x_const + (np.min(x)), y_line, z_y_section, linewidth=2.5)

# Labels & view
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (intensity)')
ax.set_title('3D: XY heat map (matrix data) + fitted Gaussian surface with XZ/YZ cross-sections')

# Optional: center/zoom around the peak
# ax.set_xlim(x0_fit - 3*sx_fit, x0_fit + 3*sx_fit)
# ax.set_ylim(y0_fit - 3*sy_fit, y0_fit + 3*sy_fit)

ax.set_zlim(z0, np.nanmax(Z_fit))
ax.view_init(elev=30, azim=-60)
ax.grid(False)
# plt.tight_layout()
# plt.show()

print("Fitted parameters:")
print(f"A={A_fit:.3f}, x0={x0_fit:.3f}, y0={y0_fit:.3f}, "
      f"sx={sx_fit:.3f}, sy={sy_fit:.3f}, theta={np.rad2deg(theta_fit):.1f}°, C={C_fit:.3f}")





# Add Poisson-like noise: draw from Poisson around Z_true clipped to >=0
Z_true = gaussian2d((X, Y), **true).reshape(ny, nx)
rng = np.random.default_rng(7)
Z = rng.poisson(np.clip(Z_true, a_min=0, a_max=None)).astype(float)

# ---- If you already have an intensity matrix, use this instead: ----
# Z = your_intensity_matrix.astype(float)  # shape (ny, nx)
# y = your_y_axis  # len ny
# x = your_x_axis  # len nx
# X, Y = np.meshgrid(x, y, indexing="xy")

# ------------------------------------------------
# 2) Fit a general 2D Gaussian (with rotation) to the matrix
# ------------------------------------------------
A0 = float(np.nanmax(Z) - np.nanmin(Z))
y0_idx0, x0_idx0 = np.unravel_index(np.nanargmax(Z), Z.shape)
x0_0, y0_0 = X[y0_idx0, x0_idx0], Y[y0_idx0, x0_idx0]
sx0 = np.ptp(x)/6 if np.ptp(x) > 0 else 1.0
sy0 = np.ptp(y)/6 if np.ptp(y) > 0 else 1.0
theta0, C0 = 0.0, float(np.nanmedian(Z))
p0 = (A0, x0_0, y0_0, sx0, sy0, theta0, C0)
bounds = ((0, x.min(), y.min(), 1e-6, 1e-6, -np.pi/2, -np.inf),
          (np.inf, x.max(), y.max(),  np.inf,  np.inf,  np.pi/2,  np.inf))

popt2d, pcov2d = curve_fit(gaussian2d, (X, Y), Z.ravel(), p0=p0, bounds=bounds, maxfev=60000)
A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit = popt2d

# ------------------------------------------------
# 3) Extract cross-sections from the DATA near fitted center
# ------------------------------------------------
# Find nearest grid indices to the fitted center
ix0 = int(np.argmin(np.abs(x - x0_fit)))
iy0 = int(np.argmin(np.abs(y - y0_fit)))

# XZ slice at y = y[iy0]: vary x across row iy0
x_slice = x.copy()
z_x_data = Z[iy0, :].copy()
sigma_x = np.sqrt(np.clip(z_x_data, 1.0, None))  # Poisson errors; avoid zeros with floor=1

# YZ slice at x = x[ix0]: vary y across column ix0
y_slice = y.copy()
z_y_data = Z[:, ix0].copy()
sigma_y = np.sqrt(np.clip(z_y_data, 1.0, None))

# ------------------------------------------------
# 4) Parameterize each cross-section with a 1D Gaussian + offset
# ------------------------------------------------
def gaussian1d(u, A, mu, s, C):
    return A * np.exp(-0.5*((u - mu)/s)**2) + C

# Initial guesses from 2D fit, projected
p0_x = (max(z_x_data) - min(z_x_data), x0_fit, max(sx_fit, (x[1]-x[0])*1.5), np.median(z_x_data))
p0_y = (max(z_y_data) - min(z_y_data), y0_fit, max(sy_fit, (y[1]-y[0])*1.5), np.median(z_y_data))

# Fit with weights (sigma) and absolute_sigma=True to get proper covariances
popt_x, pcov_x = curve_fit(gaussian1d, x_slice, z_x_data, p0=p0_x,
                           sigma=sigma_x, absolute_sigma=True, maxfev=20000)
popt_y, pcov_y = curve_fit(gaussian1d, y_slice, z_y_data, p0=p0_y,
                           sigma=sigma_y, absolute_sigma=True, maxfev=20000)

Ax, mux, sxg, Cx = popt_x
Ay, muy, syg, Cy = popt_y
perr_x = np.sqrt(np.diag(pcov_x))
perr_y = np.sqrt(np.diag(pcov_y))

# ------------------------------------------------
# 5) Chi-squared goodness-of-fit for each cross-section
# ------------------------------------------------
z_x_model = gaussian1d(x_slice, *popt_x)
z_y_model = gaussian1d(y_slice, *popt_y)

chi2_x = np.sum(((z_x_data - z_x_model)/sigma_x)**2)
chi2_y = np.sum(((z_y_data - z_y_model)/sigma_y)**2)

dof_x = len(x_slice) - len(popt_x)   # N - p
dof_y = len(y_slice) - len(popt_y)

red_chi2_x = chi2_x / dof_x
red_chi2_y = chi2_y / dof_y

# p-value: probability of observing chi2 >= observed under H0 (model correct)
p_x = chi2.sf(chi2_x, dof_x)
p_y = chi2.sf(chi2_y, dof_y)

print("\n=== 1D Gaussian fit to XZ cross-section (y = y0) ===")
print(f"A = {Ax:.3g} ± {perr_x[0]:.3g}")
print(f"mu = {mux:.4g} ± {perr_x[1]:.3g}")
print(f"sigma = {sxg:.4g} ± {perr_x[2]:.3g}")
print(f"C = {Cx:.3g} ± {perr_x[3]:.3g}")
print(f"chi2 = {chi2_x:.2f}, dof = {dof_x}, reduced chi2 = {red_chi2_x:.3f}, p = {p_x:.3f}")

print("\n=== 1D Gaussian fit to YZ cross-section (x = x0) ===")
print(f"A = {Ay:.3g} ± {perr_y[0]:.3g}")
print(f"mu = {muy:.4g} ± {perr_y[1]:.3g}")
print(f"sigma = {syg:.4g} ± {perr_y[2]:.3g}")
print(f"C = {Cy:.3g} ± {perr_y[3]:.3g}")
print(f"chi2 = {chi2_y:.2f}, dof = {dof_y}, reduced chi2 = {red_chi2_y:.3f}, p = {p_y:.3f}")

# ------------------------------------------------
# 6) Quick diagnostics: data vs fit and residuals for both slices
# ------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col', gridspec_kw={'height_ratios':[2,1]})

# XZ slice plots
ax1, ax2 = axes[:, 0]
ax1.errorbar(x_slice, z_x_data, yerr=sigma_x, fmt='.', label='data', alpha=0.7, color='black')
ax1.plot(x_slice, z_x_model, label='Gaussian fit', color='red')
ax1.set_title(f'XZ slice at y≈{y[iy0]:.3g}')
ax1.set_ylabel('Intensity')
ax1.legend()
res_x = (z_x_data - z_x_model)/sigma_x
ax2.axhline(0, lw=1, color='grey', ls='dotted')
ax2.axhline(-np.std(res_x), lw=2, color='red', ls='dashed')
ax2.axhline(np.std(res_x), lw=2, color='red', ls='dashed')
ax2.errorbar(x_slice, res_x, fmt='.', alpha=0.7, color='black')
ax2.set_xlabel('x')
ax2.set_ylabel('Normalised Residuals')

# YZ slice plots
ax3, ax4 = axes[:, 1]
ax3.errorbar(y_slice, z_y_data, yerr=sigma_y, fmt='.', label='data', alpha=0.7, color='black')
ax3.plot(y_slice, z_y_model, label='Gaussian fit', color='red')
ax3.set_title(f'YZ slice at x≈{x[ix0]:.3g}')
ax3.set_ylabel('Intensity')
ax3.legend()
res_y = (z_y_data - z_y_model)/sigma_y
ax4.axhline(0, lw=2, color='grey', ls='dotted')
ax4.axhline(-np.std(res_y), lw=2, color='red', ls='dashed')
ax4.axhline(np.std(res_y), lw=2, color='red', ls='dashed')
ax4.errorbar(y_slice, res_y, fmt='.', alpha=0.7, color='black')
ax4.set_xlabel('y')
ax4.set_ylabel('Normalised Residuals')



# ------------------------------------------------
# 7) (Optional) Heuristic interpretation
# ------------------------------------------------
def interpret(reduced_chi2, pval):
    if 0.5 <= reduced_chi2 <= 1.5 and pval > 0.05:
        return "Fit looks statistically consistent with a Gaussian."
    if reduced_chi2 > 2 or pval < 0.01:
        return "Poor agreement; Gaussian may be unsuitable or errors underestimated."
    return "Borderline; check residuals/systematics and error model."

print("\nInterpretation XZ:", interpret(red_chi2_x, p_x))
print("Interpretation YZ:", interpret(red_chi2_y, p_y))

plt.tight_layout()
plt.show()