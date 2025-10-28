import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import chi2
from astropy.io import fits

# ------------------------------------------------
# 1) Test intensity matrix (replace with your own)
# ------------------------------------------------
# Define grid for the matrix
def Gaussian_model_position(img_url, x_pos_start, x_pos_end, y_pos_start, y_pos_end):
    hdul = fits.open(img_url)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    Z = data[y_pos_start:y_pos_end, x_pos_start:x_pos_end].astype(float)     #372:396, 534:558

    # ny, nx = 120, 150
    ny, nx = Z.shape
    # y = np.linspace(-5, 5, ny)
    # x = np.linspace(-6, 6, nx)
    y = np.arange(ny)
    x = np.arange(nx)
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
    # Z = gaussian2d((X, Y), **true).reshape(ny, nx)
    # rng = np.random.default_rng(7)
    # Z = Z + 10.0 * rng.standard_normal(Z.shape)  # add some noise

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
    # print("ZX section")
    # print(z_x_section)
    # print(x_line)
    # guide plane at y = y0_fit (transparent)
    Xp_xz, Zp_xz = np.meshgrid(np.linspace(x.min(), x.max(), 2),
                            np.linspace(z0, np.nanmax(Z_fit), 2))
    Yp_xz = np.full_like(Xp_xz, y0_fit)
    ax.plot_surface(Xp_xz, np.max(y), Zp_xz, alpha=0.00, linewidth=0, shade=False)

    # the cross-section curve
    ax.plot(x_line, np.max(y), z_x_section, linewidth=2.5)

    # (d) Cross-section on the YZ plane (x = x0_fit)
    y_line = np.linspace(y.min(), y.max(), 400)
    x_const = np.full_like(y_line, x0_fit)
    z_y_section = gaussian2d((x_const, y_line), A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)





    # guide plane at x = x0_fit (transparent)
    Yp_yz, Zp_yz = np.meshgrid(np.linspace(y.min(), y.max(), 2),
                            np.linspace(z0, np.nanmax(Z_fit), 2))
    Xp_yz = np.full_like(Yp_yz, x0_fit)
    ax.plot_surface(np.min(x), Yp_yz, Zp_yz, alpha=0.00, linewidth=0, shade=False)

    # the cross-section curve
    ax.plot(np.min(x), y_line, z_y_section, linewidth=2.5)

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
    # rng = np.random.default_rng(7)
    # Z = rng.poisson(np.clip(Z_true, a_min=0, a_max=None)).astype(float)

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
    ax1.plot(x_line, z_x_section, label='Gaussian fit', color='red')
    ax1.set_title(f'XZ slice at y≈{y[iy0]:.3g}')
    ax1.set_ylabel('Intensity')
    ax1.legend()

    counter = 0
    previous_diff = 1000

    location_of_smallest_diff = []
    for b in range(len(x_slice)):
        for i in range(len(x_line)):
            # print(previous_diff, (x_slice[b] - x_line[i]))
            if previous_diff < abs(x_slice[b] - x_line[i]):
                location_of_smallest_diff.append(i-1)
                break
            previous_diff = x_slice[b] - x_line[i] 
            # print(x_line[i])
        previous_diff = 1000

    # print(x_slice)
    # print(z_x_section)
    # print(location_of_smallest_diff)

    model_x = []
    for i in range(len(location_of_smallest_diff)):
        model_x.append(z_x_section[location_of_smallest_diff[i]])
    model_x.append(model_x[-1])
    # print(model)
    # print(z_x_data)



    res_x = (z_x_data - model_x)/sigma_x            #we need to compare the residual to the model at that specific pixal
    ax2.axhline(np.mean(res_x), lw=1, color='grey', ls='dotted')
    ax2.axhline(-np.std(res_x), lw=2, color='red', ls='dashed')
    ax2.axhline(np.std(res_x), lw=2, color='red', ls='dashed')
    ax2.errorbar(x_slice, res_x, fmt='.', alpha=0.7, color='black')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Normalised Residuals')

    # YZ slice plots
    ax3, ax4 = axes[:, 1]
    ax3.errorbar(y_slice, abs(z_y_data), yerr=sigma_y, fmt='.', label='data', alpha=0.7, color='black')
    ax3.plot(y_line, z_y_section, label='Gaussian fit', color='red')
    ax3.set_title(f'YZ slice at x≈{x[ix0]:.3g}')
    ax3.set_ylabel('Intensity')
    ax3.legend()


    counter = 0
    previous_diff = 1000

    location_of_smallest_diff = []
    for b in range(len(y_slice)):
        for i in range(len(y_line)):
            # print(previous_diff, (x_slice[b] - x_line[i]))
            if previous_diff < abs(y_slice[b] - y_line[i]):
                location_of_smallest_diff.append(i-1)
                break
            previous_diff = y_slice[b] - y_line[i] 
            # print(x_line[i])
        previous_diff = 1000

    # print(x_slice)
    # print(z_x_section)
    # print(location_of_smallest_diff)

    model_y = []
    for i in range(len(location_of_smallest_diff)):
        model_y.append(z_y_section[location_of_smallest_diff[i]])
    model_y.append(model_y[-1])
    # print(model)
    # print(z_x_data)


    # ------------------------------------------------
    # 3.5) Chi-squared analysis for XZ and YZ profiles
    # ------------------------------------------------
    def robust_sigma(a):
        """Robust background sigma via MAD (scaled)."""
        med = np.nanmedian(a)
        return 1.4826 * np.nanmedian(np.abs(a - med))

    def chi2_stats(obs, model, sigma, k_params_used=0):
        """
        Return (chi2, dof, chi2_red, p_value).
        k_params_used: number of fitted params considered 'used' for this dataset.
                    If parameters were NOT obtained from *this* 1D profile, you can set 0.
                    If you want to be conservative and subtract the full 2D- fit params, set 7.
        """
        obs = np.asarray(obs, dtype=float)
        model = np.asarray(model, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # avoid zero/NaN uncertainties
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        mask = np.isfinite(obs) & np.isfinite(model) & np.isfinite(sigma)
        obs, model, sigma = obs[mask], model[mask], sigma[mask]
        N = obs.size
        dof = max(N - k_params_used, 1)

        chi2_val = np.sum(((obs - model) / sigma) ** 2)
        chi2_red = chi2_val / dof
        p = chi2.sf(chi2_val, dof)  # goodness-of-fit (prob to exceed)
        return chi2_val, dof, chi2_red, p, N

    # --- Choose noise model for σ ---
    NOISE_MODEL = 'poisson'  # options: 'poisson' or 'constant_bg'
    K_PARAMS_FOR_SLICE = 0   # set to 7 if you want to subtract the full 2D fit params

    # Build discrete pixel profiles at the nearest fitted centers
    row_y0 = int(np.clip(np.rint(y0_fit), 0, ny - 1))
    col_x0 = int(np.clip(np.rint(x0_fit), 0, nx - 1))

    x_pix = x  # discrete pixel x positions along the row
    y_pix = y  # discrete pixel y positions along the column

    # Observed profiles from the data
    obs_xz = Z[row_y0, :]          # at y ≈ y0_fit, vary x
    obs_yz = Z[:, col_x0]          # at x ≈ x0_fit, vary y

    # # Model profiles evaluated on the same pixel grid, using the 2D fitted params
    # model_xz = gaussian2d((x_pix, np.full_like(x_pix, row_y0, dtype=float)),
    #                       A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)
    # model_yz = gaussian2d((np.full_like(y_pix, col_x0, dtype=float), y_pix),
    #                       A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, C_fit)

    # Uncertainty estimates
    if NOISE_MODEL.lower() == 'poisson':
        # Poisson variance ~ value; guard against negatives and zeros
        sigma_xz = np.sqrt(np.clip(obs_xz, a_min=1.0, a_max=None))
        sigma_yz = np.sqrt(np.clip(obs_yz, a_min=1.0, a_max=None))
    elif NOISE_MODEL.lower() == 'constant_bg':
        # Estimate a constant noise from the frame edges as "background"
        # Use a 3-pixel border region; adjust if your data needs different
        border = 3
        edges = np.concatenate([
            Z[:border, :].ravel(), Z[-border:, :].ravel(),
            Z[:, :border].ravel(), Z[:, -border:].ravel()
        ])
        sig_const = robust_sigma(edges)
        if not np.isfinite(sig_const) or sig_const <= 0:
            sig_const = np.nanstd(edges)
            if not np.isfinite(sig_const) or sig_const <= 0:
                sig_const = 1.0  # final fallback
        sigma_xz = np.full_like(obs_xz, sig_const, dtype=float)
        sigma_yz = np.full_like(obs_yz, sig_const, dtype=float)
    else:
        raise ValueError("Unknown NOISE_MODEL. Use 'poisson' or 'constant_bg'.")

    # Compute chi-squared stats
    chi2_xz, dof_xz, chi2r_xz, p_xz, N_xz = chi2_stats(
        obs_xz, model_x, sigma_xz, k_params_used=K_PARAMS_FOR_SLICE
    )
    chi2_yz, dof_yz, chi2r_yz, p_yz, N_yz = chi2_stats(
        obs_yz, model_y, sigma_yz, k_params_used=K_PARAMS_FOR_SLICE
    )

    print("\nChi-squared analysis (profiles at nearest pixel to fitted center):")
    print(f"XZ @ y≈{y0_fit:.3f} (row {row_y0}): N={N_xz}, dof={dof_xz}, "
        f"chi2={chi2_xz:.2f}, chi2_red={chi2r_xz:.3f}, p={p_xz:.3g}")
    print(f"YZ @ x≈{x0_fit:.3f} (col {col_x0}): N={N_yz}, dof={dof_yz}, "
        f"chi2={chi2_yz:.2f}, chi2_red={chi2r_yz:.3f}, p={p_yz:.3g}")

    # (Optional) overlay the discrete data and model used for chi2 on your 3D figure edges
    # so you can *see* what was tested.
    # ax.plot(x_pix, np.full_like(x_pix, row_y0), model_xz, linewidth=2, alpha=0.9)
    ax.scatter(x_pix, np.full_like(x_pix, row_y0), obs_xz, s=9, depthshade=False)

    # ax.plot(np.full_like(y_pix, col_x0), y_pix, model_yz, linewidth=2, alpha=0.9)
    ax.scatter(np.full_like(y_pix, col_x0), y_pix, obs_yz, s=9, depthshade=False)










    res_y = (abs(z_y_data) - model_y)/sigma_y
    ax4.axhline(np.mean(res_y), lw=2, color='grey', ls='dotted')
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

    print(f"\n\nThe global position of the centre of the moon is ({x_pos_start + x0_fit:.3f} +/- {round(sx_fit/((x_pos_end-x_pos_start)**0.5),3)}, {y_pos_start+y0_fit:3f} +/- {round(sy_fit/((y_pos_end-y_pos_start)**0.5),3)})px")

    plt.tight_layout()
    plt.show()


# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/mosaic.fits', 534, 558, 372, 396) #372:396, 534:558
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/mosaic_N.fits', 568, 592, 412, 436) #Neptune 9th Oct
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/mosaic_N.fits', 586, 596, 408, 416) #Triton 9th Oct



#SATURN - Titan data 25_10_25

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/S_stack_25_10_25/et0p05/mosaic.fits', 772, 786, 418, 429) #25th oct et0p05
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/S_stack_25_10_25/et0p1/mosaic.fits', 774, 785, 421, 431) #25th oct et0p1
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/S_stack_25_10_25/et0p2/mosaic.fits', 771, 783, 419, 432) #25th oct et0p2
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/S_stack_25_10_25/et0p5/mosaic.fits', 771, 789, 418, 435) #25th oct et0p5
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/S_stack_25_10_25/et1p0/d0050.fits', 773, 788, 422, 434) #25th oct et1p0


#NEPTUNE - Tritan data 25_10_25

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p05/mosaic.fits', 565, 572, 399, 410) #25th oct et0p05 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p05/mosaic.fits', 561, 584, 408, 426) #25th oct et0p05 planet

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p2/mosaic.fits', 563, 573, 398, 410) #25th oct et0p2 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p2/mosaic.fits', 562, 586, 407, 427) #25th oct et0p2 planet

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p5/mosaic.fits', 565, 576, 400, 409) #25th oct et0p5 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et0p5/mosaic.fits', 563, 586, 408, 429) #25th oct et0p5 planet

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et1p0/mosaic.fits', 566, 572, 400, 408) #25th oct et1p0 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et1p0/mosaic.fits', 564, 583, 408, 426) #25th oct et1p0 planet

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et2p0/mosaic.fits', 556, 574, 399, 407) #25th oct et2p0 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et2p0/mosaic.fits', 562, 584, 408, 428) #25th oct et2p0 planet

# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et4p0/mosaic.fits', 560, 568, 399, 407) #25th oct et4p0 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et4p0/mosaic.fits', 558, 577, 409, 425) #25th oct et4p0 planet

Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et8p0/mosaic.fits', 560, 569, 399, 407) #25th oct et8p0 moon
# Gaussian_model_position('/Users/jameshartman/Desktop/AstroLabMoons/Stacked_Images/N_stack_25_10_25/et8p0/mosaic.fits', 558, 578, 408, 425) #25th oct et8p0 planet