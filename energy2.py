import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from energy_minimisation import energy_and_grad
lengths = np.array([L1], dtype=float)
n = len(lengths)

r_arr  = np.array([r1, r1], dtype=float)
E_arr  = np.array([E_niti, E_niti], dtype=float)
nu_arr = np.array([0.49, 0.49], dtype=float)

# masses per segment (same density/area)
lam = rho_niti * A_cs
masses = lam * lengths


p_epm = np.array([0.0, 0.100, 0.260])   # [0 100 260] mm
  
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

def thetas_from_yz(y, z, lengths, L_cat):
    """
    Map a tip (y,z) to segment bending angles thetas
    assuming a single constant-curvature arc of total length L_cat.

    Total tip angle θ_tot is atan2(y, z), then each segment i
    gets θ_i = θ_tot * (L_i / L_cat) so that κ = θ_i / L_i is constant.
    """
    z_safe = z if z > 1e-6 else 1e-6
    theta_tot = np.arctan2(y, z_safe)   # tip angle wrt +z
    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas
def energies_for_tip_yz(y, z,
                        axes=axes,
                        lengths=lengths,
                        masses=masses,
                        gvec=gvec,
                        E=E_arr,
                        r=r_arr,
                        nu=nu_arr,
                        L_cat=L_cat):
    """
    y,z in meters (y vertical, z along catheter axis).
    Use your full PCC energy model (energy_and_grad) for:
      - Elastic bending energy of both catheters
      - Gravitational potential energy of both catheters
    and keep the original:
      - Magnetic potential of a tip IPM in the EPM dipole field.
    Returns (U_el, U_g_cat, U_mag) to match original interface.
    """

    # --- 1) Map (y,z) -> segment angles thetas, assuming constant curvature
    thetas = thetas_from_yz(y, z, lengths, L_cat)

    # --- 2) Elastic + gravity energy from your general code
    # energy_and_grad internally computes:
    #   gamma = [ (θ_i/L_i) * n_hat_i ] stacked
    #   K = build_k(E, r, nu, lengths)
    #   U_stiff = 0.5 * gamma^T K gamma
    #   U_grav  = ∫ -m g·p ds
    #   U = U_stiff + U_grav
    U_total, grad, K = energy_and_grad(
        axes, thetas, lengths, masses, gvec, E, r, nu,
        R_base=None, p_base=None, quad_n=12
    )

    # --- 3) Recover U_el = U_stiff using the same gamma & K
    gamma = np.concatenate([
        (theta_i / L_i) * np.asarray(n_hat, float)
        for n_hat, theta_i, L_i in zip(axes, thetas, lengths)
    ])
    U_el = 0.5 * gamma @ (K @ gamma)

    # Gravity energy is the remainder
    U_g_cat = U_total - U_el

    # --- 4) Magnetic energy: keep your original point-dipole model
    # Tip position in world coords (as in your Fig.3 approximation)
    p_ipm = np.array([0.0, y, z])
    B_here = B_from_epm(p_ipm)     # uses p_epm, mu_epm, mu0, etc.
    U_mag = - (mu_ipm @ B_here)    # IPM dipole at the tip

    return U_el, U_g_cat, U_mag
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
fig, ax_arr = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

# Convert to mm for plot axes
Ymm = Y * 1e3
Zmm = Z * 1e3

# (a) Elastic potential energy
im0 = ax_arr[0].pcolormesh(Zmm, Ymm, U_el_grid,
                         shading='auto', cmap='jet')
ax_arr[0].set_title('Elastic potential energy')
ax_arr[0].set_xlabel('z (mm)')
ax_arr[0].set_ylabel('y (mm)')
cbar0 = fig.colorbar(im0, ax=ax_arr[0])
cbar0.set_label('U_el (J)')

# (b) Catheter gravitational potential energy
im1 = ax_arr[1].pcolormesh(Zmm, Ymm, U_gcat_grid,
                         shading='auto', cmap='jet')
ax_arr[1].set_title('Catheter gravitational energy')
ax_arr[1].set_xlabel('z (mm)')
cbar1 = fig.colorbar(im1, ax=ax_arr[1])
cbar1.set_label('U_g, catheter (J)')

# (c) Magnetic potential energy
im2 = ax_arr[2].pcolormesh(Zmm, Ymm, U_mag_grid,
                         shading='auto', cmap='jet')
ax_arr[2].set_title('Magnetic potential energy')
ax_arr[2].set_xlabel('z (mm)')
cbar2 = fig.colorbar(im2, ax=ax_arr[2])
cbar2.set_label('U_mag (J)')

for ax in ax_arr:
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
