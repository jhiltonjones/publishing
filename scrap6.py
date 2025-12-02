import numpy as np
from energy_minimisation import (
    magnetic_field_for_theta,
    central_difference_jacobian,
    residual_theta,
    my_gauss_newton_lm,
)
from beam_pcc import stack_blocks
prefer_dir = None
# ======================
#  Geometry & Materials
# ======================

L_cat = 0.1        # total catheter length [m]
L1    = L_cat      # single-segment length [m]

r1 = 0.001         # catheter radius [m]

E_niti   = 3e6
rho_niti = 6700
nu_niti  = 0.33

# z is UP (+z) / DOWN (-z); gravity acts along -z
gvec     = np.array([0.0, 0.0, -9.8])   # <<< CHANGED

A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0
EI1  = E_niti * I1

lam    = rho_niti * A_cs
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths

# Bending axis:
#   Beam lies along +y, bending is rotation about +x → deflection in (y,z) plane.
axes = [np.array([0.0, 0.0, 1.0], float)]   # bend about local/world +z

r_arr  = np.array([r1],       dtype=float)
E_arr  = np.array([E_niti],   dtype=float)
nu_arr = np.array([0.49],     dtype=float)
f_e    = np.zeros(6 * n, float)

# ======================
#  Internal Magnet (IPM)
# ======================

Br   = 1.2
mu0  = 4e-7 * np.pi

D_ipm = 2e-3
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm

mu_ipm_scalar = Br * V_ipm / mu0

# Magnetization direction in local/base frame:
# still along +z in beam frame (this is internal; OK to keep)
m_dir_local = np.array([0.0, 1.0, 0.0], float)

eta_ipm   = (mu_ipm_scalar / L_ipm) * m_dir_local
eta0_list = [eta_ipm]

print("Linear dipole density eta_ipm [A·m]:", eta_ipm)

# ======================
#  Target bend & solve for required B
# ======================

theta_target_deg = 10.0
thetas_target    = np.array([np.deg2rad(theta_target_deg)])
thetas0          = np.array([np.deg2rad(5.0)])

B_req, info = magnetic_field_for_theta(
    thetas_target,
    axes, lengths, masses, eta0_list,
    f_e=f_e,
    gvec=gvec,          # <<< uses new gravity
    E=E_arr,
    r=r_arr,
    nu=nu_arr,
    R_base=None,
    p_base=None,
    quad_n=12,
    prefer_dir=None,
    reg=0.0
)

theta_total = np.sum(thetas_target)

print("\n=== Field needed for target bend ===")
print("Overall target (deg):", np.rad2deg(theta_total))
print("Per-segment thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req)
print("Required |B| (mT):", np.linalg.norm(B_req) * 1e3)
print("||H B - rhs||:", info["residual_norm"], "cond(H):", info["H_cond_est"])

# ======================
#  Check equilibrium
# ======================

r0, J0 = central_difference_jacobian(
    residual_theta,
    thetas0,
    axes=axes,
    lengths=lengths,
    masses=masses,
    eta0_list=eta0_list,
    B=B_req,
    f_e=f_e,
    gvec=gvec,          # <<< new gravity
    E=E_arr,
    r=r_arr,
    nu=nu_arr
)

print("\n=== Newton initial state diagnostics ===")
print("Initial guess thetas0 (deg):", np.rad2deg(thetas0))
print("||r(thetas0)|| =", np.linalg.norm(r0))
print("Column norms(J(thetas0)) =", np.linalg.norm(J0, axis=0))

thetas_sol = my_gauss_newton_lm(
    thetas0,
    axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec,      # <<< new gravity
    E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12,
    max_iter=80,
    tol_grad=1e-10,
    tol_step=1e-12,
    tol_r=1e-10,
    lambda_init=1e-4,
    h_floor=1e-5
)

print("\n=== Newton solution ===")
print("Solved thetas (deg):", np.rad2deg(thetas_sol))
print("Sum(theta) achieved (deg):", np.rad2deg(np.sum(thetas_sol)))
print("Error vs target (deg):", np.rad2deg(np.sum(thetas_sol) - theta_total))

# ======================
#  Helper: y,z -> thetas  (beam along +y, bending about +x)
# ======================
def thetas_from_xy(x, y, lengths, L_cat):
    """
    Map desired tip (x,y) to segment bending angles under
    a planar constant-curvature assumption in the x–y plane.

    Assumes:
      - beam base at origin,
      - initial beam axis along +y,
      - bending about +z (so deflection is in x–y).
    """

    y_safe = y if abs(y) > 1e-6 else 1e-6
    theta_tot = 2.0 * np.arctan2(x, y_safe)  # <<< note: atan2(x, y)

    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas
# def thetas_from_yz(y, z, lengths, L_cat):
#     """
#     Map desired tip (y,z) to segment bending angles under
#     a planar constant-curvature assumption in the y–z plane.

#     Assumes:
#       - beam base at origin,
#       - initial beam axis along +y,
#       - bending about +x (so deflection is in y–z).

#     For a constant-curvature arc (beam along +y):
#         y = R sin(theta_tot)
#         z = R (1 - cos(theta_tot))
#       => z / y = tan(theta_tot / 2)
#          theta_tot = 2 * atan2(z, y)
#     """
#     y_safe = y if abs(y) > 1e-6 else 1e-6
#     theta_tot = 2.0 * np.arctan2(z, y_safe)   # <<< CHANGED (was atan2(y, z))

#     lengths = np.asarray(lengths, float)
#     thetas = theta_tot * (lengths / float(L_cat))
#     return thetas

def tip_position_from_theta(axes, thetas, lengths,
                            R_base=None, p_base=None):
    Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(
        axes, thetas, lengths,
        s_list=lengths,
        R_base=R_base, p_base=p_base
    )
    p_tip = p_nodes[-1]
    return p_tip

# ======================
#  Single EPM from B
# ======================
import numpy as np

def single_epm_pose_from_B_unrestricted(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Given required B at origin, return:
      - r_vec  : EPM position (in world frame)
      - mu_hat : world dipole direction (unit 3D vector, unconstrained)
      - R_epm  : rotation matrix with body y-axis = mu_hat
                 (no restriction to pure yaw; full 3D rotation).
    """

    # --- basic setup ---
    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    # --- dipole model matrix A: B ∝ A * mu_hat / r^3 ---
    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    # --- unconstrained dipole direction from inverse of A ---
    B_hat = B_req / Bmag
    mu_hat_raw = np.linalg.solve(A, B_hat)   # some direction that maps to B_hat
    mu_hat = mu_hat_raw / np.linalg.norm(mu_hat_raw)  # unit vector, full 3D

    # --- solve for r to match |B| ---
    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)               # ||A * mu_hat||
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # place magnet so vector (magnet -> origin) = r_mag * e_r
    r_vec = -r_mag * e_r

    y_body = mu_hat

    # pick an "up" vector not parallel to y_body
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, y_body)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    # x_body = normalized (up × y_body)
    x_body = np.cross(up, y_body)
    x_body /= np.linalg.norm(x_body)

    # z_body = x_body × y_body
    z_body = np.cross(x_body, y_body)

    # # Enforce that body z-axis roughly points "up" in world frame
    # if z_body[2] < 0:
    #     x_body = -x_body
    #     z_body = -z_body

    R_epm = np.column_stack((x_body, y_body, z_body))


    # columns of R_epm are body axes expressed in world frame
    R_epm = np.column_stack((x_body, y_body, z_body))

    return r_vec, mu_hat, R_epm

def single_epm_pose_from_B(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Given required B at origin, return:
      - r_vec  : EPM position
      - mu_hat : world dipole direction (we enforce it in x–y plane)
      - R_epm  : rotation matrix with body y-axis = mu_hat,
                 and R_epm is a pure yaw about world z.
    """

    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    # Dipole model matrix
    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    # First, unconstrained μ̃ from the dipole inverse
    B_hat = B_req / Bmag
    mu_hat_raw = np.linalg.solve(A, B_hat)
    mu_hat_raw /= np.linalg.norm(mu_hat_raw)

    # ---- NEW: force dipole into x–y plane and renormalise ----
    mu_eff = mu_hat_raw.copy()
    mu_eff[2] = 0.0  # kill z-component
    if np.linalg.norm(mu_eff[:2]) < 1e-8:
        # degenerate case: fall back to +y
        mu_eff = np.array([0.0, 1.0, 0.0])
    else:
        mu_eff /= np.linalg.norm(mu_eff)

    # Use this as our final world dipole direction
    mu_hat = mu_eff

    # Magnitude condition using this μ̂
    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # Place magnet so that vector magnet→origin = r_mag * e_r
    r_vec = -r_mag * e_r

    # ---- NEW: build R_epm as *pure yaw about z* with y_body = μ_hat ----
    mx, my, _ = mu_hat
    psi = np.arctan2(-mx, my)   # yaw angle

    c, s = np.cos(psi), np.sin(psi)

    R_epm = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return r_vec, mu_hat, R_epm


# ======================
#  EPM pose for desired tip in y–z plane
# ======================

# def epm_pose_for_tip_yz(y, z,
#                         lengths, axes, masses,
#                         eta0_list, gvec, E, r, nu,
#                         L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
#                         quad_n=12,
#                         prefer_B_dir=None):
#     # thetas_target = thetas_from_yz(y, z, lengths, L_cat)

#     B_req, info = magnetic_field_for_theta(
#         thetas_target, axes, lengths, masses, eta0_list,
#         f_e=np.zeros(6*len(lengths)),
#         gvec=gvec, E=E, r=r, nu=nu,
#         R_base=None, p_base=None, quad_n=quad_n,
#         prefer_dir=None,
#         reg=0.0
#     )

#     r_vec, mu_hat, R_epm = single_epm_pose_from_B(B_req, mu_mag, e_r, mu0)
#     return thetas_target, B_req, r_vec, mu_hat, R_epm, info
def solve_theta_given_B_lengths(thetas_init,
                                axes, lengths, masses, eta0_list,
                                B, f_e, gvec, E, r, nu,
                                R_base=None, p_base=None, quad_n=12):
    """
    Solve Eq. (20) for theta given:
      - B: magnetic field,
      - lengths, masses, etc.
    Uses your Gauss-Newton-LM solver.
    """
    theta_sol = my_gauss_newton_lm(
        thetas_init, axes, lengths, masses, eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
        R_base=R_base, p_base=p_base, quad_n=quad_n
    )
    return theta_sol
def epm_pose_for_tip_xy(x, y,
                        lengths, axes, masses,
                        eta0_list, gvec, E, r, nu,
                        L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
                        quad_n=12,
                        prefer_B_dir=None):
    """
    Given a desired planar tip position (x,y) of the beam,
    solve (approximately) for the EPM pose that should produce
    that configuration under static equilibrium.

    Beam:
      - base at origin
      - initial axis along +y
      - bends in x–y plane about +z
    """

    # 1) x,y -> thetas  (constant curvature in x–y plane)
    # thetas_target = thetas_from_xy(x, y, lengths, L_cat)
    thetas_target = np.array([np.deg2rad(-15)])
    # 2) thetas -> required B
    B_req, info = magnetic_field_for_theta(
        thetas_target, axes, lengths, masses, eta0_list,
        f_e=np.zeros(6*len(lengths)),
        gvec=gvec, E=E, r=r, nu=nu,
        R_base=None, p_base=None, quad_n=quad_n,
        prefer_dir=None,
        reg=0.0
    )

    # 3) B -> EPM pose from dipole model
    r_vec, mu_hat, R_epm = single_epm_pose_from_B_unrestricted(B_req, mu_mag, e_r, mu0)

    return thetas_target, B_req, r_vec, mu_hat, R_epm, info

# ======================
#  Beam & EPM parameters for your final block
# ======================

L_cat = 0.06
L1    = L_cat
r1    = 0.001

E_niti   = 3e6
rho_niti = 6700
nu_niti  = 0.33
gvec     = np.array([0.0, 0.0, -9.8])   # <<< CHANGED here too

A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0

lam    = rho_niti * A_cs
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths

r_arr   = np.array([r1],       float)
E_arr   = np.array([E_niti],   float)
nu_arr  = np.array([0.49],     float)
f_e     = np.zeros(6*n, float)

Br   = 1.3
mu0  = 4e-7 * np.pi
D_ipm = 15e-4
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_scalar = Br * V_ipm / mu0

# m_dir_local = np.array([0.0, 0.0, 1.0], float)
eta_ipm = (mu_ipm_scalar / L_ipm) * m_dir_local
eta0_list = [eta_ipm]

mu_mag = 220.0

# Magnet above the beam: beam along +y, origin at base. 
# "Above" = +z. e_r is direction FROM magnet TO origin.
# If magnet is at (0,0,+h) and origin at (0,0,0), vector from magnet to origin is (0,0,-1).
e_r    = np.array([0.0, 0.0, -1.0])   # <<< CHANGED: magnet above (+z)

# desired tip position in (y,z) plane
# desired tip position in (x,y) plane
x_target = 20e-3    # sideways deflection
y_target = 35e-3    # along the beam

thetas_target, B_req, r_epm, mu_hat_epm, R_epm, info = epm_pose_for_tip_xy(
    x_target, y_target,
    lengths=lengths, axes=axes, masses=masses,
    eta0_list=eta0_list, gvec=gvec,
    E=E_arr, r=r_arr, nu=nu_arr,
    L_cat=L_cat,
    mu_mag=mu_mag, e_r=e_r,
    mu0=mu0, prefer_B_dir = prefer_dir
)


print("thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req, " |B| (mT):", np.linalg.norm(B_req)*1e3)
print("EPM position r_epm:", r_epm, " |r|:", np.linalg.norm(r_epm))
print("EPM dipole direction μ̂:", mu_hat_epm)
def tip_pos_from_B_s(x,               # x = [Bx, By, Bz, s]
                     thetas_init,
                     axes, lengths0, lam,          # lam = rho_niti * A_cs
                     eta0_list, gvec, E_arr, r_arr, nu_arr,
                     f_e=None,
                     R_base=None, p_base=None,
                     quad_n=12):
    """
    Map control variables x = [B_x, B_y, B_z, s] to tip position p_tip.

    lengths0 : nominal lengths array [L1, L2, ...]
    lam      : linear mass density (rho * area)
    """

    x = np.asarray(x, float)
    B = x[:3]
    s = float(x[3])

    # 1) Modify lengths: L1 -> L1 + s
    lengths = np.asarray(lengths0, float).copy()
    lengths[0] = lengths0[0] + s

    # 2) Update masses for each segment
    masses = lam * lengths

    # 3) Solve equilibrium for theta with these lengths and B
    if f_e is None:
        f_e = np.zeros(6*len(lengths), float)

    theta_sol = solve_theta_given_B_lengths(
        thetas_init=thetas_init,
        axes=axes, lengths=lengths, masses=masses,
        eta0_list=eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
        R_base=R_base, p_base=p_base, quad_n=quad_n
    )

    # 4) Compute tip position from theta_sol
    p_tip = tip_position_from_theta(
        axes, theta_sol, lengths,
        R_base=R_base, p_base=p_base
    )
    return p_tip


print(repr(R_epm))
print(thetas_target)
# thetas_target2 = np.array([np.deg2rad(1)])
thetas_eq = my_gauss_newton_lm(
    thetas_target, axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)
s0 = 0.0 
x0 = np.hstack([B_req, s0]) 
print(f"Bending is {np.rad2deg(thetas_eq)}")
p_tip = tip_pos_from_B_s( 
    x0, thetas_init=thetas_eq, axes=axes, lengths0=lengths, lam=lam, 
    eta0_list=eta0_list, gvec=gvec, E_arr=E_arr, r_arr=r_arr, 
    nu_arr=nu_arr, f_e=f_e, R_base=None, p_base=None, quad_n=12 ) 
print("Tip position p_tip:", p_tip)


mu_world_from_R = R_epm @ np.array([0.0, 1.0, 0.0])
# mu_world_from_R == mu_hat  (up to numerical noise)
mu_check = R_epm @ np.array([0.0, 1.0, 0.0])
print("μ_hat from solver:", mu_world_from_R)
print("μ from R_epm @ e_y:", mu_check)



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
# import numpy as np

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
# import numpy as np

# def plot_beam_and_epm(axes_list, thetas, lengths,
#                       r_epm, mu_hat_epm, R_epm,
#                       B_req=None,
#                       R_base=None, p_base=None,
#                       cyl_radius=0.15,
#                       cyl_length=0.9):
#     """
#     Visualise:
#       - beam centerline,
#       - external cylindrical magnet:
#           * axis along WORLD Z,
#           * r_epm = magnet CENTRE (same as dipole position),
#           * half red / half blue on circular cross-section,
#             split rotated to match the dipole direction
#             (projection of μ̂ onto x–y plane),
#       - EPM dipole direction (μ̂),
#       - (optionally) B-field at origin,
#       - (optionally) EPM local axes from R_epm.
#     """

#     # Get beam nodes from stack_blocks
#     Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(
#         axes_list, thetas, lengths,
#         s_list=lengths,
#         R_base=R_base, p_base=p_base
#     )
#     p_nodes = np.asarray(p_nodes)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # --- Plot beam centerline ---
#     ax.plot(p_nodes[:, 0], p_nodes[:, 1], p_nodes[:, 2],
#             '-o', label='Beam', linewidth=2)

#     # Mark base and tip
#     ax.scatter(p_nodes[0, 0], p_nodes[0, 1], p_nodes[0, 2],
#                color='k', s=40, label='Base')
#     ax.scatter(p_nodes[-1, 0], p_nodes[-1, 1], p_nodes[-1, 2],
#                color='r', s=40, label='Tip')

#     # --- EPM position ---
#     r_epm = np.asarray(r_epm, float)
#     mu_hat_epm = np.asarray(mu_hat_epm, float)
#     R_epm = np.asarray(R_epm, float)

#     ax.scatter(r_epm[0], r_epm[1], r_epm[2],
#                color='magenta', s=60, label='Magnet center / dipole')

#     # ---------- External magnet as cylinder aligned with WORLD Z ----------
#     # r_epm is the *center* of the cylinder.
#     # Cylinder axis: world z
#     n_theta = 80
#     n_h = 10

#     theta = np.linspace(0.0, 2*np.pi, n_theta)
#     h = np.linspace(-cyl_length/2.0, cyl_length/2.0, n_h)  # centered at r_epm.z
#     theta_grid, h_grid = np.meshgrid(theta, h)

#     # Cylinder geometry in world frame
#     Xc = r_epm[0] + cyl_radius * np.cos(theta_grid)
#     Yc = r_epm[1] + cyl_radius * np.sin(theta_grid)
#     Zc = r_epm[2] + h_grid

#     # --- Color pattern: half red / half blue aligned with μ direction ---
#     # Project μ onto x–y plane:
#     mu_xy = np.array([mu_hat_epm[0], mu_hat_epm[1]])
#     mu_xy_norm = np.linalg.norm(mu_xy)

#     if mu_xy_norm < 1e-8:
#         # If μ is almost vertical, choose some default split (e.g. along +x/-x)
#         mu_xy_hat = np.array([1.0, 0.0])
#     else:
#         mu_xy_hat = mu_xy / mu_xy_norm

#     # Radial direction for each (theta) on the cylinder surface
#     r_x = np.cos(theta_grid)
#     r_y = np.sin(theta_grid)

#     # Dot product between projected μ and radial direction
#     # > 0 -> "north" side (red), < 0 -> "south" side (blue)
#     dots = mu_xy_hat[0] * r_x + mu_xy_hat[1] * r_y

#     colors = np.empty(theta_grid.shape + (4,), dtype=float)
#     red_rgba  = np.array([1.0, 0.0, 0.0, 0.5])
#     blue_rgba = np.array([0.0, 0.0, 1.0, 0.5])

#     colors[dots >= 0] = red_rgba
#     colors[dots <  0] = blue_rgba

#     ax.plot_surface(Xc, Yc, Zc,
#                     facecolors=colors,
#                     linewidth=0.2,
#                     edgecolor='k')

#     # --- EPM dipole direction as arrow ---
#     L_mu = cyl_length * 0.7
#     mu_end = r_epm + L_mu * mu_hat_epm
#     ax.plot([r_epm[0], mu_end[0]],
#             [r_epm[1], mu_end[1]],
#             [r_epm[2], mu_end[2]],
#             '-', linewidth=3, label='μ̂ (EPM dipole)')

#     # --- EPM local axes from R_epm (this is where that matrix you printed shows up) ---
#     L_axis = cyl_length * 0.5
#     origin = r_epm
#     x_body = origin + L_axis * R_epm[:, 0]
#     y_body = origin + L_axis * R_epm[:, 1]
#     z_body = origin + L_axis * R_epm[:, 2]

#     ax.plot([origin[0], x_body[0]],
#             [origin[1], x_body[1]],
#             [origin[2], x_body[2]],
#             '-', label='EPM x_body')
#     ax.plot([origin[0], y_body[0]],
#             [origin[1], y_body[1]],
#             [origin[2], y_body[2]],
#             '-', label='EPM y_body')
#     ax.plot([origin[0], z_body[0]],
#             [origin[1], z_body[1]],
#             [origin[2], z_body[2]],
#             '-', label='EPM z_body')

#     # --- B field at origin (optional) ---
#     if B_req is not None:
#         B_req = np.asarray(B_req, float)
#         L_B = cyl_length * 0.7
#         B_hat = B_req / (np.linalg.norm(B_req) + 1e-12)
#         B_end = L_B * B_hat

#         ax.plot([0.0, B_end[0]],
#                 [0.0, B_end[1]],
#                 [0.0, B_end[2]],
#                 '-', linewidth=2, label='B direction @ origin')

#     # Axes labels
#     ax.set_xlabel('x [m]')
#     ax.set_ylabel('y [m]')
#     ax.set_zlabel('z [m]')

#     # Equal aspect ratio (include cylinder in bounds)
#     xs = np.concatenate([p_nodes[:, 0], Xc.ravel(), [r_epm[0]]])
#     ys = np.concatenate([p_nodes[:, 1], Yc.ravel(), [r_epm[1]]])
#     zs = np.concatenate([p_nodes[:, 2], Zc.ravel(), [r_epm[2]]])

#     x_range = xs.max() - xs.min()
#     y_range = ys.max() - ys.min()
#     z_range = zs.max() - zs.min()
#     max_range = max(x_range, y_range, z_range) * 0.6

#     x_mid = 0.5 * (xs.max() + xs.min())
#     y_mid = 0.5 * (ys.max() + ys.min())
#     z_mid = 0.5 * (zs.max() + zs.min())

#     ax.set_xlim(x_mid - max_range, x_mid + max_range)
#     ax.set_ylim(y_mid - max_range, y_mid + max_range)
#     ax.set_zlim(z_mid - max_range, z_mid + max_range)

#     ax.legend()
#     ax.set_title('Beam + External Cylindrical Magnet (μ-aligned split, axis = z)')

#     plt.tight_layout()
#     plt.show()

# plot_beam_and_epm(
#     axes_list=axes,
#     thetas=thetas_eq,
#     lengths=lengths,
#     r_epm=r_epm,    
#     mu_hat_epm=mu_hat_epm,
#     R_epm=R_epm,
#     B_req=B_req,
#     R_base=None,
#     p_base=None,
#     cyl_radius=0.15,
#     cyl_length=0.2,
# )
