# ================== EXTRA CODE FOR FIG-3 STYLE ENERGY MAPS ==================
import matplotlib.pyplot as plt
import numpy as np
from parameters import *

# EPM position (from Fig. 3 caption, in m)
p_epm = np.array([0.0, 0.100, 0.290])   # [0 100 260] mm

# ---------- 2) Helper: dipole field from EPM at any point p ----------
def B_from_epm(p):
    """
    Magnetic field B at point p (3,) due to EPM dipole at p_epm.
    Dipole model: B = mu0/(4π r^3) (3 r_hat r_hat^T - I) μ
    """
    r_vec = p - p_epm
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-6:
        return np.zeros(3)
    r_hat = r_vec / r_norm
    A = 3.0 * np.outer(r_hat, r_hat) - np.eye(3)
    B = (mu0 / (4.0 * np.pi * r_norm**3)) * (A @ mu_epm)
    return B

# ---------- 3) Energy models (simplified) for a given tip point (y,z) ----------
def energies_for_tip_yz(y, z):
    """
    y,z in meters (y vertical, z along catheter axis as in the paper).
    Assume catheter centerline is a planar circular arc from (0,0)
    to (z,y) with constant curvature and total length L_cat.
    Compute:
      - Elastic bending energy of both catheters
      - Gravitational potential energy of both catheters
      - Magnetic potential energy of IPM in EPM field
    """
    # Geometric angle of the tip relative to +z axis
    # (approximate; for large deflections this is not exact but ok for energy scale)
    theta = np.arctan2(y, z if z > 1e-6 else 1e-6)  # rad
    kappa = theta / L_cat                            # curvature [1/m]

    # Elastic energy (sum of two catheters)
    U_el = 0.5 * (EI1 + EI2) * (kappa**2) * L_cat

    # Gravitational energy: approximate COM at halfway to tip (z/2, y/2)
    # Only vertical displacement (y) matters: ΔU = m g (y_COM)
    y_com = 0.5 * y
    U_g_cat = m_cat_total * (-1*gvec[1]) * y_com

    # Magnetic energy: IPM at tip, dipole μ_ipm in field B_from_epm
    p_ipm = np.array([0.0, y, z])
    B_here = B_from_epm(p_ipm)
    U_mag = - (mu_ipm @ B_here)

    return U_el, U_g_cat, U_mag

# ---------- 4) Build a y-z grid and evaluate energies ----------
# Grid similar to paper: y from -150 to 150 mm, z from 0 to 200 mm
y_vals = np.linspace(-0.15, 0.15, 150)   # m
z_vals = np.linspace(0.02, 0.20, 150)    # m (avoid exactly 0)

Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
U_el_grid   = np.zeros_like(Y)
U_gcat_grid = np.zeros_like(Y)
U_mag_grid  = np.zeros_like(Y)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        U_el, U_gc, U_mag = energies_for_tip_yz(Y[i, j], Z[i, j])
        U_el_grid[i, j]   = U_el
        U_gcat_grid[i, j] = U_gc
        U_mag_grid[i, j]  = U_mag

# ---------- 5) Plotting: elastic, catheter grav, magnetic grav (like Fig. 3) ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

# Convert to mm for plot axes
Ymm = Y * 1e3
Zmm = Z * 1e3

# (a) Elastic potential energy
im0 = axes[0].pcolormesh(Zmm, Ymm, U_el_grid,
                         shading='auto', cmap='jet')
axes[0].set_title('Elastic potential energy')
axes[0].set_xlabel('z (mm)')
axes[0].set_ylabel('y (mm)')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label('U_el (J)')

# (b) Catheter gravitational potential energy
im1 = axes[1].pcolormesh(Zmm, Ymm, U_gcat_grid,
                         shading='auto', cmap='jet')
axes[1].set_title('Catheter gravitational energy')
axes[1].set_xlabel('z (mm)')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label('U_g, catheter (J)')

# (c) Magnetic potential energy
im2 = axes[2].pcolormesh(Zmm, Ymm, U_mag_grid,
                         shading='auto', cmap='jet')
axes[2].set_title('Magnetic potential energy')
axes[2].set_xlabel('z (mm)')
cbar2 = fig.colorbar(im2, ax=axes[2])
cbar2.set_label('U_mag (J)')

for ax in axes:
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
# ================== END EXTRA CODE =# ========= EXTRA: TOTAL POTENTIAL ENERGY + EQUILIBRIUM POINT ============

# 1) Total potential energy
U_tot_grid = U_el_grid + U_gcat_grid + U_mag_grid

# 2) Find the (y,z) point of minimum total energy (equilibrium)
min_idx = np.unravel_index(np.argmin(U_tot_grid), U_tot_grid.shape)
y_eq_m  = Y[min_idx]
z_eq_m  = Z[min_idx]
y_eq_mm = Ymm[min_idx]
z_eq_mm = Zmm[min_idx]
U_min   = U_tot_grid[min_idx]

print("Equilibrium (approx) from total energy:")
print(f"  y_eq = {y_eq_mm:.2f} mm,  z_eq = {z_eq_mm:.2f} mm,  U_min = {U_min:.3e} J")

# 3) Plot total energy with equilibrium point marked
fig_tot, ax_tot = plt.subplots(figsize=(6, 5))

im_tot = ax_tot.pcolormesh(Zmm, Ymm, U_tot_grid, shading='auto', cmap='jet')
ax_tot.set_aspect('equal')
ax_tot.set_xlabel('z (mm)')
ax_tot.set_ylabel('y (mm)')
ax_tot.set_title('Total potential energy')

cbar_tot = fig_tot.colorbar(im_tot, ax=ax_tot)
cbar_tot.set_label('U_total (J)')

# Mark equilibrium point
ax_tot.plot(z_eq_mm, y_eq_mm, 'wo', markersize=8, markeredgecolor='k')
ax_tot.text(z_eq_mm + 5, y_eq_mm + 5,
            f"eq\n({z_eq_mm:.1f}, {y_eq_mm:.1f}) mm",
            color='w', fontsize=9, ha='left', va='bottom',
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

ax_tot.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
# ===================== END EXTRA TOTAL-ENERGY CODE ======================
