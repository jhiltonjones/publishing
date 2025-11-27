from energy_minimisation import *
import numpy as np

L_cat = 0.04         # total catheter length [m]
L1    = L_cat        # single-segment length [m]
L_epm = 90e-3
D_epm=30e-3
# Catheter radius
r1 = 0.001          # [m]

# NiTi (or soft equivalent) material
E_niti   = 3e6       # Young's modulus [Pa]
rho_niti = 6700      # density [kg/m^3]
nu_niti  = 0.33      # Poisson ratio (for mechanics)
gvec     = np.array([0.0, 0.0, 9.8])  # gravity [m/s^2]

# Cross-sectional properties (circular)
A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0
EI1  = E_niti * I1

# Mass properties
lam    = rho_niti * A_cs           # linear mass density [kg/m]
m1     = lam * L_cat               # total mass of segment
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths            # per-segment mass array

# Bending axis:
#   We want to bend in the x–z plane, i.e. rotation about +x.
axes = [np.array([1.0, 0.0, 0.0], float)]    # bending about +x

# Section arrays (one segment)
r_arr  = np.array([r1],       dtype=float)
E_arr  = np.array([E_niti],   dtype=float)
nu_arr = np.array([0.49],     dtype=float)   # nearly incompressible

# 6n external wrench vector (no point loads/moments in this example)
f_e = np.zeros(6 * n, float)

# ======================
#  Internal Magnet (IPM)
# ======================

Br   = 1.2
mu0  = 4e-7 * np.pi

D_ipm = 2e-3      # IPM diameter [m]
L_ipm = 4e-2      # IPM length [m]
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm    # IPM volume [m^3]

# Total dipole moment of the IPM (scalar magnitude)
mu_ipm_scalar = Br * V_ipm / mu0            # [A·m^2]

# Magnetization direction in local/base frame:
#   Choose along +z so that eta x B (with B along +y) gives torque about +x.
m_dir_local = np.array([0.0, 0.0, 1.0], float)

# Magnetic dipole moment per unit length (linear density) [A·m]
eta_ipm = (mu_ipm_scalar / L_ipm) * m_dir_local
eta0_list = [eta_ipm]

print("Linear dipole density eta_ipm [A·m]:", eta_ipm)

# ======================
#  Target bend & solve for required B
# ======================

# Single-segment target rotation (about +x)
theta_target_deg = 15.0
thetas_target    = np.array([np.deg2rad(theta_target_deg)])

# Initial guess for Newton solve (used later)
thetas0 = np.array([np.deg2rad(5.0)])   # any small nonzero is fine

# We tell magnetic_field_for_theta that B must lie along +y:
prefer_dir = np.array([0.0, 1.0, 0.0])  # B = alpha * +y

B_req, info = magnetic_field_for_theta(
    thetas_target,
    axes, lengths, masses, eta0_list,
    f_e=f_e,
    gvec=gvec,
    E=E_arr,
    r=r_arr,
    nu=nu_arr,
    R_base=None,
    p_base=None,
    quad_n=12,
    prefer_dir=prefer_dir,   # <--- constrain B direction
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
#  Check equilibrium by solving for theta under this B
# ======================

# Diagnostics at initial guess
r0, J0 = central_difference_jacobian(
    residual_theta,
    thetas0,
    axes=axes,
    lengths=lengths,
    masses=masses,
    eta0_list=eta0_list,
    B=B_req,
    f_e=f_e,
    gvec=gvec,
    E=E_arr,
    r=r_arr,
    nu=nu_arr
)

print("\n=== Newton initial state diagnostics ===")
print("Initial guess thetas0 (deg):", np.rad2deg(thetas0))
print("||r(thetas0)|| =", np.linalg.norm(r0))
print("Column norms(J(thetas0)) =", np.linalg.norm(J0, axis=0))

# Run Gauss–Newton LM to find equilibrium under B_req
thetas_sol = gauss_newton_lm(
    thetas0,
    axes, lengths, masses, eta0_list,
    B=B_req,
    f_e=f_e,
    gvec=gvec,
    E=E_arr,
    r=r_arr,
    nu=nu_arr,
    R_base=None,
    p_base=None,
    quad_n=12,
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

print("||r(0)|| =", np.linalg.norm(r0))
print("col norms(J(0)) =", np.linalg.norm(J0, axis=0))
B = np.array([0.0, B_req[2], 0.0])   # pure y-field, B0 > 0
thetas_sol = gauss_newton_lm(
    thetas0, axes, lengths, masses, eta0_list,
    B=B, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12,
    max_iter=50, tol_r=1e-8, tol_step=1e-8
)
print("Solved thetas (deg):", np.rad2deg(thetas_sol))
print("Sum(theta) achieved (deg):", np.rad2deg(np.sum(thetas_sol)))
def thetas_from_yz(y, z, lengths, L_cat):
    """
    Map desired tip (y,z) to segment bending angles under
    a planar constant-curvature assumption.

    y,z  : desired tip position in the base frame (y–z plane)
    lengths : array of segment lengths
    L_cat   : total length (sum(lengths))
    """
    z_safe = z if z > 1e-6 else 1e-6

    # For a constant-curvature arc:
    #   y/z = tan(theta_tot / 2)
    # => theta_tot = 2 * atan2(y, z)
    theta_tot = 2.0 * np.arctan2(y, z_safe)

    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas


def epm_pose_from_required_B(B_req, mu_mag, e2=np.array([0,1,0.0]), mu0=4e-7*np.pi):
    """
    Given desired uniform field B_req (3,), dipole magnitude |mu|,
    and the common axis e2 used in the paper, return:
      r_vec  : position of each EPM (place one at +r_vec and one at -r_vec)
      mu_hat : unit dipole direction for each EPM
      R_epm  : rotation matrix whose z-axis = mu_hat   (EE orientation)
    """
    B_req = np.asarray(B_req, float)
    e2    = np.asarray(e2, float) / np.linalg.norm(e2)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B_req is (near) zero.")

    # --- magnitude of r from Eq.(24) -> rearranged correctly
    r_mag = (mu0 * mu_mag / (2.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # --- mu_hat from (3 e2 e2^T - I) mu_hat = B_hat
    A = 3.0 * np.outer(e2, e2) - np.eye(3)
    B_hat = B_req / Bmag
    mu_hat = np.linalg.solve(A, B_hat)
    mu_hat = mu_hat / np.linalg.norm(mu_hat)

    # --- choose to place the two EPMs symmetrically along ±e2
    r_vec = r_mag * e2

    # --- build a rotation with z-axis aligned to mu_hat (EE orientation)
    z = mu_hat
    # pick a 'not parallel' vector to build a basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_epm = np.column_stack([x, y, z])  # columns are body x,y,z

    return r_vec, mu_hat, R_epm
def tip_position_from_theta(axes, thetas, lengths,
                            R_base=None, p_base=None):
    """
    Compute tip position p_tip given thetas and lengths using stack_blocks.
    Tip = last node position in p_nodes.
    """
    Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(
        axes, thetas, lengths,
        s_list=lengths,
        R_base=R_base, p_base=p_base
    )
    # p_nodes has shape (n+1, 3); last one is the tip
    p_tip = p_nodes[-1]
    return p_tip
def single_epm_pose_from_B(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Compute the pose (r, mu_hat, R_epm) of a SINGLE external permanent magnet (EPM)
    that generates the required field B_req at the origin, using the dipole model.

    Equation used:
        B = (μ0 / (4π|r|³)) (3 ê_r ê_rᵀ - I₃) (|μ| μ̂)

    Inputs:
      B_req : desired field at the origin [T], shape (3,)
      mu_mag: dipole magnitude [A·m²]
      e_r   : unit vector from magnet to workspace center
      mu0   : vacuum permeability [T·m/A]

    Returns:
      r_vec  : position of magnet (3,)
      mu_hat : unit dipole direction
      R_epm  : rotation matrix (columns are body axes, z = μ̂)
    """
    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    B_hat = B_req / Bmag
    mu_hat = np.linalg.solve(A, B_hat)
    mu_hat /= np.linalg.norm(mu_hat)

    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    r_vec = -r_mag * e_r   


    z = mu_hat
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_epm = np.column_stack([x, y, z])

    return r_vec, mu_hat, R_epm


def epm_dipole_from_br_dims(Br, D, L, mu0=4e-7*np.pi):
    M = Br / mu0
    V = np.pi * (0.5*D)**2 * L
    return M * V

# mu_mag = epm_dipole_from_br_dims(Br, Dm, Lm, mu0)
mu_mag = 283
e_r   = np.array([1, 0, 0.0])     

r_vec, mu_hat, R_epm = single_epm_pose_from_B(B_req, mu_mag, e_r, mu0)
R_epm *=1
# r_vec, mu_hat, R_epm = epm_pose_from_required_B(B_req, mu_mag, e_r, mu0)

print(f"|μ| = {mu_mag:.3f} A·m²")
print("Magnet position r (m):", r_vec, " |r| =", np.linalg.norm(r_vec))
print("μ̂ =", mu_hat)
print("Rotation matrix:\n", R_epm)
def epm_pose_for_tip_yz(y, z,
                        lengths, axes, masses,
                        eta0_list, gvec, E, r, nu,
                        L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
                        quad_n=12):
    """
    Given a desired planar tip position (y,z) of the beam,
    solve (approximately) for the EPM pose that should produce
    that configuration under static equilibrium.

    Steps:
      (1) map (y,z) -> thetas (constant curvature assumption)
      (2) solve Eq.(20) for B_req via magnetic_field_for_theta
      (3) solve dipole inverse problem for EPM pose via single_epm_pose_from_B
    Returns:
      thetas_target, B_req, r_vec, mu_hat, R_epm
    """
    # 1) y,z -> thetas
    thetas_target = thetas_from_yz(y, z, lengths, L_cat)

    # 2) thetas -> required field B
    B_req, info = magnetic_field_for_theta(
        thetas_target, axes, lengths, masses, eta0_list,
        f_e=np.zeros(6*len(lengths)),
        gvec=gvec, E=E, r=r, nu=nu,
        R_base=None, p_base=None, quad_n=quad_n,
        prefer_dir=None, reg=0.0
    )

    # 3) B -> EPM pose
    r_vec, mu_hat, R_epm = single_epm_pose_from_B(B_req, mu_mag, e_r, mu0)

    return thetas_target, B_req, r_vec, mu_hat, R_epm, info
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
    theta_sol = gauss_newton_lm(
        thetas_init, axes, lengths, masses, eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
        R_base=R_base, p_base=p_base, quad_n=quad_n
    )
    return theta_sol
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
y_target = 0.01   # 30 mm
z_target = 0.0  # 150 mm

thetas_target, B_req, r_epm, mu_hat_epm, R_epm, info = epm_pose_for_tip_yz(
    y_target, z_target,
    lengths=lengths, axes=axes, masses=masses,
    eta0_list=eta0_list, gvec=gvec,
    E=E_arr, r=r_arr, nu=nu_arr,
    L_cat=L_cat,
    mu_mag=mu_mag, e_r=np.array([1,0,0.0]),
    mu0=mu0
)

print("thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B:", B_req)
print("EPM position:", r_epm)
print("EPM dipole dir:", mu_hat_epm)
thetas_eq = my_gauss_newton_lm(
    thetas_target, axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)
print(f"Bending is: {np.rad2deg(thetas_eq)}")
s0 = 0.0 

x0 = np.hstack([B_req, s0])
x1 = x0 
p1 = tip_pos_from_B_s(
    x1,
    thetas_init=thetas_eq,    # warm-start with same theta
    axes=axes, lengths0=lengths, lam=lam,
    eta0_list=eta0_list, gvec=gvec,
    E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
    f_e=f_e,
    R_base=None, p_base=None,
    quad_n=12)
print(f"Tip position is: {p1}")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- utility: equal 3D scaling ---
def set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = abs(limits[:, 1] - limits[:, 0])
    centers = limits.mean(axis=1)
    radius = 0.5 * spans.max()
    for ctr, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim([ctr - radius, ctr + radius])

# --- utility: draw coordinate frame triad ---
def draw_triad(ax, origin, R, scale=0.02, lw=2.0):
    """Draw a coordinate triad with columns of R as axes."""
    o = np.asarray(origin).ravel()
    colors = ['r', 'g', 'b']
    for i, c in enumerate(colors):
        ax.quiver(o[0], o[1], o[2], *(scale * R[:, i]),
                  color=c, arrow_length_ratio=0.15, linewidth=lw)

# --- beam centerline sampling ---
def sample_centerline(axes, thetas, lengths, n_per_seg=50, R_base=None, p_base=None):
    axes   = list(axes)
    thetas = np.asarray(thetas, float).reshape(-1)
    lengths= np.asarray(lengths, float).reshape(-1)
    if not (len(axes) == len(thetas) == len(lengths)):
        raise ValueError(f"Segment count mismatch: len(axes)={len(axes)}, "
                         f"len(thetas)={len(thetas)}, len(lengths)={len(lengths)}")

    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()
    P = [p_up.copy()]
    for n_hat, theta, L in zip(axes, thetas, lengths):
        n_hat = np.asarray(n_hat, float)
        gamma = (theta / L) * n_hat
        s_vals = np.linspace(0.0, L, n_per_seg+1)[1:]
        for s in s_vals:
            R_s, p_s = pose_from_gamma(R_up, p_up, gamma, s)
            P.append(p_s.copy())
        R_up, p_up = pose_from_gamma(R_up, p_up, gamma, L)
    return np.vstack(P), R_up, p_up

# --- EPM cylinder with red/blue poles ---
def draw_epm_cylinder(ax, center, R_epm, length, diameter,
                      n_theta=80, n_z=40, alpha=0.9):
    center = np.asarray(center, float).ravel()
    R = np.asarray(R_epm, float).reshape(3, 3)

    z_local = np.linspace(-0.5*length, 0.5*length, n_z)
    th = np.linspace(0, 2*np.pi, n_theta)
    Th, Z = np.meshgrid(th, z_local)
    R_cyl = 0.5 * diameter
    X, Y = R_cyl*np.cos(Th), R_cyl*np.sin(Th)

    P_local = np.stack([X, Y, Z], axis=0)
    P_world = (R @ P_local.reshape(3, -1)).reshape(3, *X.shape)
    Xw, Yw, Zw = P_world[0]+center[0], P_world[1]+center[1], P_world[2]+center[2]

    fc = np.empty(X.shape + (4,), float)
    mask = (Z >= 0)
    fc[mask]  = (1.0, 0.1, 0.1, alpha)   # red pole
    fc[~mask] = (0.1, 0.1, 1.0, alpha)   # blue pole
    ax.plot_surface(Xw, Yw, Zw, facecolors=fc, rstride=1, cstride=1, linewidth=0)

    mu_hat = R[:, 2]
    ax.quiver(*center, *(0.6*length*mu_hat),
              color='k', linewidth=3, arrow_length_ratio=0.15)

# --- check and pick the correct thetas (length matches #segments) ---
def _check_and_pick_thetas_for_viz(thetas_sol=None, thetas_target=None, n=None):
    if thetas_sol is not None and np.size(thetas_sol) == n:
        return np.asarray(thetas_sol, float)
    elif thetas_target is not None and np.size(thetas_target) == n:
        return np.asarray(thetas_target, float)
    else:
        raise ValueError(f"No theta array of length {n} available.")

# --- visualization setup ---
n = len(lengths)
thetas_plot = _check_and_pick_thetas_for_viz(
    thetas_sol=thetas_eq if 'thetas_sol' in globals() else None,
    thetas_target=thetas_target if 'thetas_target' in globals() else None,
    n=n
)

assert len(axes) == n, "axes must have same length as beam segments"

# compute beam centerline
P, R_end, p_end = sample_centerline(axes, thetas_plot, lengths, n_per_seg=100)

# pick field to show
B_show = B_req if 'B_req' in globals() else np.array([0, 0, 1])

# --- PLOT ---
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

# beam centerline
ax.plot(P[:,0], P[:,1], P[:,2], '-', color='gray', linewidth=3, label='beam')

# base & end triads
draw_triad(ax, np.zeros(3), np.eye(3), scale=0.05*sum(lengths))
draw_triad(ax, p_end, R_end, scale=0.05*sum(lengths))

# draw magnet(s)
if 'r_vec' in globals() and 'R_epm' in globals():
    use_double_epm = False

    if use_double_epm:
        # 180° about local x-axis (proper rotation, det=+1)
        R_x_pi = np.array([[1.0, 0.0,  0.0],
                           [0.0,-1.0,  0.0],
                           [0.0, 0.0, -1.0]])

        R_epm_opposite = R_epm @ R_x_pi  # μ̂ flips sign

        # +r_vec uses original orientation, -r_vec uses opposite
        draw_epm_cylinder(ax, center= r_epm,  R_epm=R_epm,
                          length=L_epm, diameter=D_epm, n_theta=100, n_z=60)
        draw_epm_cylinder(ax, center=-r_epm,  R_epm=R_epm_opposite,
                          length=L_epm, diameter=D_epm, n_theta=100, n_z=60)
    else:
        draw_epm_cylinder(ax, center=r_epm, R_epm=R_epm,
                          length=L_epm, diameter=D_epm, n_theta=100, n_z=60)

# magnetic field vector
B_unit = B_show / (np.linalg.norm(B_show) + 1e-16)
ax.quiver(0, 0, 0, *(0.03 * B_unit),
          color='k', linewidth=2.5, arrow_length_ratio=0.25, label='B field')

# labels and cosmetics
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_title('Beam bending and EPM orientation')
ax.legend()
ax.grid(True)
set_axes_equal(ax)
plt.tight_layout()
plt.show()
# # ==========================================================
# # RESIDUAL-BASED ENERGY IN θ–SPACE (CONSISTENT WITH SOLVER)
# # ==========================================================

# def q_total_from_residual(theta, B):
#     """
#     Generalised torque/force from the same residual used by
#     Gauss–Newton. For n=1 segment, residual_theta returns a 1D
#     array [R], a scalar torque-like quantity that the solver drives to zero.
#     """
#     thetas = np.array([theta], dtype=float)
#     # residual_theta returns only the residual vector, NOT (residual, J)
#     R = residual_theta(
#         thetas,
#         axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
#         B=B, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
#         R_base=None, p_base=None, quad_n=12
#     )
#     return float(R[0])


# def U_from_residual(theta, B, n_steps=400):
#     """
#     Build a scalar potential U(θ) such that dU/dθ = q_total_from_residual(θ,B)
#     by numerical integration from θ0=0 to θ.

#     Any additive constant doesn't matter for the location of minima,
#     only the shape and stationary points.
#     """
#     theta0 = 0.0
#     if abs(theta - theta0) < 1e-12:
#         return 0.0

#     thetas = np.linspace(theta0, theta, n_steps)
#     q_vals = np.array([q_total_from_residual(th, B) for th in thetas])
#     U = np.trapz(q_vals, thetas)
#     return U


# # ----------------------------------------------------------
# # 1D CHECK: DOES θ_eq MINIMISE THIS U(θ)?
# # ----------------------------------------------------------

# theta_grid_deg = np.linspace(-40.0, 40.0, 161)
# theta_grid = np.deg2rad(theta_grid_deg)

# # energy along θ using the *same* B_req as the solver
# U_vals = np.array([U_from_residual(th, B_req) for th in theta_grid])

# # equilibrium from Gauss–Newton
# theta_eq = float(thetas_eq[0])
# U_eq = U_from_residual(theta_eq, B_req)

# # minimum from sampled energy
# idx_min = np.argmin(U_vals)
# theta_min = theta_grid[idx_min]
# U_min = U_vals[idx_min]

# print("\n=== residual-based θ–space energy check ===")
# print("θ_eq   from solver (deg):", np.rad2deg(theta_eq))
# print("θ_min from U(θ)      (deg):", np.rad2deg(theta_min))
# print("U(θ_eq)  =", U_eq)
# print("U(θ_min) =", U_min)

# plt.figure(figsize=(6, 4))
# plt.plot(theta_grid_deg, U_vals, label="Ū(θ) from residual", linewidth=2)
# plt.axvline(np.rad2deg(theta_eq), color='r', linestyle='--',
#             label='θ_eq (Gauss–Newton)')
# plt.axvline(np.rad2deg(theta_min), color='g', linestyle=':',
#             label='θ_min of Ū(θ)")')

# plt.xlabel("θ [deg]")
# plt.ylabel("Ū(θ) [arbitrary units]")
# plt.title("Residual-based energy-like potential along θ")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # ----------------------------------------------------------
# # 2D ENERGY MAP: U(θ, k) WITH B = k * B_req
# # ----------------------------------------------------------

# # scale factors for B_req
# k_vals = np.linspace(0.0, 1.5, 81)   # 0 → 1.5 × B_req
# Theta_grid, K_grid = np.meshgrid(theta_grid, k_vals, indexing='ij')

# U_map = np.zeros_like(Theta_grid)

# for i in range(Theta_grid.shape[0]):
#     for j in range(Theta_grid.shape[1]):
#         th = Theta_grid[i, j]
#         k  = K_grid[i, j]
#         Bk = k * B_req
#         U_map[i, j] = U_from_residual(th, Bk, n_steps=300)

# # convert θ axis to degrees for plotting
# Theta_deg_grid = np.rad2deg(Theta_grid)

# plt.figure(figsize=(7, 5))
# im = plt.pcolormesh(Theta_deg_grid, K_grid, U_map,
#                     shading='auto', cmap='jet')
# plt.colorbar(im, label='Ū(θ, k) [arbitrary units]')
# plt.xlabel('θ [deg]')
# plt.ylabel('Field scale k  (B = k · B_req)')
# plt.title('Residual-based energy map Ū(θ, k)')
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
