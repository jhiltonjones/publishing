

import numpy as np
import matplotlib.pyplot as plt
from run_solver import magnetic_field_for_theta, single_epm_pose_from_B
import numpy as np
from parameters import *
# parameters.py (sketch)
L_cat = 0.0400
L1 = 0.04
mu_mag = 342
e_r   = np.array([1, 0, 0.0])  
# # catheter radii
r1 = 0.0001
r_epm = np.array([0.08, 0.04, 0.0])

# # NiTi material
E_niti = 83e9
rho_niti = 6700
nu_niti = 0.33 
I1 = np.pi * r1**4 / 4.0
EI1 = E_niti * I1


# # magnetization of inner tube / IPM
Br  = 1.4
mu0 = 4e-7*np.pi
D_ipm = 3e-3
L_ipm = 4e-2
V_ipm = np.pi*(0.5*D_ipm)**2*L_ipm
mu_ipm_mag = Br*V_ipm/mu0
mu_ipm_hat = np.array([0.0, 0.0,1])
mu_ipm = mu_ipm_mag * mu_ipm_hat
# EPM parameters + nominal pose
D_epm = 80e-3
L_epm = 90e-3
V_epm = np.pi*(0.5*D_epm)**2*L_epm
mu_epm_mag = Br*V_epm/mu0
mu_epm_hat = np.array([0.0, 0,1])
mu_epm = mu_epm_mag * mu_epm_hat
# p_epm_nom = np.array([0.0, 0.100, 0.260])
gvec = np.array([0.0, -9.81, 0.0])
A_cs = np.pi * r1**2
# Section properties
A1 = np.pi * r1**2
I1 = np.pi * r1**4 / 4.0
EI1 = E_niti * I1


m1 = rho_niti * A1 * L_cat

axes = [np.array([-1.0, 0.0, 0.0], float)]
        # np.array([-1.0, 0.0, 0.0], float)]
lengths = np.array([L1], dtype=float)
n = len(lengths)

r_arr  = np.array([r1, r1], dtype=float)
E_arr  = np.array([E_niti, E_niti], dtype=float)
nu_arr = np.array([0.49, 0.49], dtype=float)

# # masses per segment (same density/area)
lam = rho_niti * A_cs
# masses = np.array([lam, lam], dtype=float)
masses = lam * lengths      # mass = density × length

# 6n wrench vector
f_e = np.zeros(6*n, dtype=float)

  

m_dir_local = np.array([0.0, 0.0, -1.0], float)
eta_mag = (mu_ipm_mag / L_ipm) * m_dir_local   # or another physically correct linear density

eta0_list = [
    1*eta_mag.copy()  
    # eta_mag.copy(),     
]

thetas0 = np.zeros(n, float)
def rotate_epm_around_global_x(R_epm0, phi):
    """
    Rotate the EPM around the GLOBAL x-axis by angle phi (radians).

    R_epm0 : 3x3 rotation matrix (EPM body axes in world frame) at phi = 0
    phi    : rotation about world x-axis
    """
    Rx_global = np.array([
        [1.0,        0.0,         0.0],
        [0.0,  np.cos(phi), -np.sin(phi)],
        [0.0,  np.sin(phi),  np.cos(phi)]
    ])

    # GLOBAL rotation: rotate world, then inherit by EPM
    return Rx_global @ R_epm0
from energy_minimisation import residual_theta, central_difference_jacobian

def my_gauss_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e,
                       gvec, E, r, nu,
                       R_base=None, p_base=None, quad_n=12,
                       max_iter=80,
                       tol_grad=1e-10,
                       tol_step=1e-12,
                       tol_r=1e-10,
                       lambda_init=1e-4,
                       h_floor=1e-5):
    theta = np.asarray(thetas0, float).copy()
    lam = float(lambda_init)

    for _ in range(max_iter):
        # residual and Jacobian of *residual_theta*, not eq20_residual
        res, J = central_difference_jacobian(
            residual_theta, theta,
            rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
            axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
            B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_norm = np.linalg.norm(res)
        if res_norm < tol_r:
            break

        grad = J.T @ res
        if np.linalg.norm(grad, np.inf) < tol_grad:
            break

        JTJ = J.T @ J
        scale = max(np.max(np.diag(JTJ)), 1e-12)
        A = JTJ + lam * scale * np.eye(len(theta))

        try:
            h = -np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(A) @ grad

        if np.linalg.norm(h) < tol_step * (np.linalg.norm(theta) + 1e-12):
            break

        theta_new = theta + h
        # evaluate new residual
        res_new = residual_theta(
            theta_new, axes, lengths, masses, eta0_list,
            B, f_e, gvec, E, r, nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_new_norm = np.linalg.norm(res_new)

        act = 0.5 * (res_norm**2 - res_new_norm**2)
        pred = np.abs(h @ (lam * scale * h + grad))
        rho = act / (pred + 1e-30)

        if rho > 0:
            theta = theta_new
            res = res_new
            res_norm = res_new_norm
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            lam = max(lam, 1e-15)
        else:
            lam = lam * min(10, 1/(1+rho))

    return theta



def B_from_dipole(r_eval, r_epm, mu_hat, mu_mag, mu0=4e-7*np.pi):
    """
    Magnetic field at point r_eval due to a dipole located at r_epm.

    B(r) = (μ0 / (4π |R|^3)) ( 3 R̂ R̂ᵀ - I ) (μ_mag * μ_hat )

    R = r_eval - r_epm
    """
    r_eval = np.asarray(r_eval, float)
    r_epm  = np.asarray(r_epm, float)
    mu_hat = np.asarray(mu_hat, float)
    mu_hat = mu_hat / np.linalg.norm(mu_hat)

    R = r_eval - r_epm
    dist = np.linalg.norm(R)
    if dist < 1e-12:
        raise ValueError("Evaluation point too close to magnet location.")

    R_hat = R / dist
    A = 3.0 * np.outer(R_hat, R_hat) - np.eye(3)
    B = (mu0 / (4.0 * np.pi * dist**3)) * A @ (mu_mag * mu_hat)
    return B
# thetas_target = np.array([np.deg2rad(2)])
# B_req2, info = magnetic_field_for_theta(
#     thetas_target, axes, lengths, masses, eta0_list,
#     f_e=f_e, gvec=gvec, E=r_arr*0 + E_arr, r=r_arr, nu=nu_arr,
#     R_base=None, p_base=None, quad_n=12,
#     prefer_dir=None, 
#     reg=0.0
# )
# r_vec, mu_hat, R_epm2 = single_epm_pose_from_B(B_req2, mu_mag, e_r, mu0)
# print(f"R_epm: {R_epm2}")
# print(f"R_vec: {r_vec}")

# # use finer resolution
# phis_deg = np.linspace(0.0, 180.0, 361)   # 0.5° steps
# phis_rad = np.deg2rad(phis_deg)
# thetas_all = np.zeros((len(phis_rad), 1))
# thetas_init = thetas0.copy()
# n_seg = len(lengths)
# thetas_all = np.zeros((len(phis_rad), n_seg))  # store bending per segment


# for i, phi in enumerate(phis_rad):
#     R_epm_phi  = np.array([
#         [1.0,         0.0,          0.0],
#         [0.0,  np.cos(phi), -np.sin(phi)],
#         [0.0,  np.sin(phi),  np.cos(phi)]
#     ]) @ R_epm2

#     mu_hat_phi = R_epm_phi[:, 2]
#     B_phi      = B_from_dipole(np.zeros(3), r_epm, mu_hat_phi, mu_mag)

#     thetas_phi = my_gauss_newton_lm(
#         thetas_init,
#         axes, lengths, masses, eta0_list,
#         B=B_phi,
#         f_e=np.zeros(6),
#         gvec=gvec,
#         E=E_arr,
#         r=r_arr,
#         nu=nu_arr
#     )


#     thetas_all[i, :] = thetas_phi
#     thetas_init = thetas_phi.copy()

# thetas_all_deg = np.rad2deg(thetas_all[:, 0])

# # simple smoothing: moving average over ±2 steps
# window = 5
# pad = window // 2
# smoothed = np.convolve(thetas_all_deg, np.ones(window)/window, mode='same')

# plt.figure(figsize=(8, 5))
# plt.plot(phis_deg, thetas_all_deg, 'o', label="raw θ")
# plt.plot(phis_deg, smoothed, '-', label="smoothed θ (moving avg)")
# plt.axhline(0, color='k', linewidth=0.8)
# plt.xlabel("Magnet rotation about global x-axis [deg]")
# plt.ylabel("Segment bending angle θ [deg]")
# plt.title("Beam bending vs magnet rotation")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# print("Bending angles (deg) at each magnet rotation angle:")
# for phi_deg, th_deg in zip(phis_deg, thetas_all_deg):
#     print(f"phi = {phi_deg:6.1f} deg -> thetas = {th_deg}")

# print("Last angle:", phi_deg, "deg, B =", B_phi)