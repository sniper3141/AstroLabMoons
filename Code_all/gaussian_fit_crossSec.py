import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import Normalize

# -----------------------------
# 0) Your data (two 1D arrays)
# -----------------------------
Intensity = [np.random.randn(100000), np.random.randn(100000)]
x_samples, y_samples = Intensity  # x and y point clouds

# -----------------------------------------------
# 1) Build a 2D intensity matrix via histogram2d
# -----------------------------------------------
xbins, ybins = 80, 80
H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=[xbins, ybins])
# Bin centers → coordinate grid
xc = 0.5 * (xedges[:-1] + xedges[1:])
yc = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(xc, yc, indexing="xy")    # shape (ybins, xbins)
Z = H.T  # transpose so Z[y, x] aligns with X,Y from meshgrid

# ------------------------------------------------
# 2) Fit a general 2D Gaussian (with rotation)
# ------------------------------------------------
def gaussian2d(xy, A, x0, y0, sx, sy, theta, C):
    Xg, Yg = xy
    ct, st = np.cos(theta), np.sin(theta)
    Xc, Yc = Xg - x0, Yg - y0
    xr =  ct*Xc + st*Yc
    yr = -st*Xc + ct*Yc
    G = A * np.exp(-0.5 * ((xr/sx)**2 + (yr/sy)**2)) + C
    return G.ravel()

# Initial guesses
A0 = float(np.nanmax(Z) - np.nanmin(Z))
y0_idx, x0_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
x0_0, y0_0 = X[y0_idx, x0_idx], Y[y0_idx, x0_idx]
sx0, sy0 = (np.ptp(xc)/6, np.ptp(yc)/6)
theta0, C0 = 0.0, float(np.nanmedian(Z))
p0 = (A0, x0_0, y0_0, sx0, sy0, theta0, C0)
bounds = ((0, xc.min(), yc.min(), 1e-6, 1e-6, -np.pi/2, -np.inf),
          (np.inf, xc.max(), yc.max(),  np.inf,  np.inf,  np.pi/2,  np.inf))

popt, _ = curve_fit(gaussian2d, (X, Y), Z.ravel(), p0=p0, bounds=bounds, maxfev=30000)
A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit = popt

# Evaluate fitted surface
Z_fit = gaussian2d((X, Y), *popt).reshape(Z.shape)

# ------------------------------------------------
# 3) 3D plot: heat map on XY base + full Gaussian surface + cross-sections
# ------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Base plane offset
z0 = Z.min() - 0.1 * (Z.max() - Z.min())
norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

# Heat map on the XY plane (as filled contours)
ax.contourf(X, Y, Z, zdir='z', offset=z0, levels=60, cmap=cm.viridis)
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array([])
fig.colorbar(mappable, ax=ax, shrink=0.75, label='Measured intensity')

# Full 3D fitted Gaussian surface
ax.plot_surface(X, Y, Z_fit, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.5, cmap=cm.inferno)

# ------------------------------------------------
# 3a) Cross-section on the XZ plane at y = y0_fit
# ------------------------------------------------
x_line = np.linspace(xc.min(), xc.max(), 400)
y_const = np.full_like(x_line, y0_fit)
z_x_section = gaussian2d((x_line, y_const), A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)

# guide plane at y = y0_fit (transparent)
Xp_xz, Zp_xz = np.meshgrid(np.linspace(xc.min(), xc.max(), 2),
                           np.linspace(z0, np.nanmax(Z_fit), 2))
Yp_xz = np.full_like(Xp_xz, y0_fit)
ax.plot_surface(Xp_xz, Yp_xz-yedges[0], Zp_xz, alpha=0.08, linewidth=0, shade=False)

# the actual cross-section curve
ax.plot(x_line, y_const-yedges[0], z_x_section, linewidth=2.5)

# ------------------------------------------------
# 3b) Cross-section on the YZ plane at x = x0_fit
# ------------------------------------------------
y_line = np.linspace(yc.min(), yc.max(), 400)
x_const = np.full_like(y_line, x0_fit)
z_y_section = gaussian2d((x_const, y_line), A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)

# guide plane at x = x0_fit (transparent)
Yp_yz, Zp_yz = np.meshgrid(np.linspace(yc.min(), yc.max(), 2),
                           np.linspace(z0, np.nanmax(Z_fit), 2))
Xp_yz = np.full_like(Yp_yz, x0_fit)
ax.plot_surface(Xp_yz+xedges[0], Yp_yz, Zp_yz, alpha=0.08, linewidth=0, shade=False)

# the actual cross-section curve
ax.plot(x_const+xedges[0], y_line, z_y_section, linewidth=2.5)

# Decorations
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (intensity)')
ax.set_title('3D: Heat map (data) + fitted Gaussian surface with XZ/YZ cross-sections')

# Optional: center/zoom around the peak
# ax.set_xlim(x0_fit - 3*sx_fit, x0_fit + 3*sx_fit)
# ax.set_ylim(y0_fit - 3*sy_fit, y0_fit + 3*sy_fit)

ax.set_zlim(z0, np.nanmax(Z_fit))
ax.view_init(elev=30, azim=-60)
ax.grid(False)
plt.tight_layout()
plt.show()

print("Fitted parameters:")
print(f"A={A_fit:.3f}, x0={x0_fit:.3f}, y0={y0_fit:.3f}, "
      f"sx={sx_fit:.3f}, sy={sy_fit:.3f}, theta={np.rad2deg(theta_fit):.1f}°, C={C_fit:.3f}")