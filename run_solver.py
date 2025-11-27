from parameters import *
from energy_minimisation import *
import numpy as np
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

thetas_target = np.array([np.deg2rad(25),
                         np.deg2rad(35)])
B_req, info = magnetic_field_for_theta(
    thetas_target, axes, lengths, masses, eta0_list,
    f_e=f_e, gvec=gvec, E=r_arr*0 + E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12,
    prefer_dir=None, 
    reg=0.0
)
theta_total=0
for theta_val in thetas_target: theta_total+=theta_val
print("Overall target (deg):", np.rad2deg(theta_total))
print("Per-segment thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req)
print("||H B - rhs||:", info["residual_norm"], " cond(H):", info["H_cond_est"])

r0, J0 = central_difference_jacobian(
    residual_theta, thetas0,  
    axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr
)
print("||r(0)|| =", np.linalg.norm(r0))
print("col norms(J(0)) =", np.linalg.norm(J0, axis=0))

thetas_sol = gauss_newton_lm(
    thetas0, axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
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
mu_mag = 342
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
y_target = 0.03   # 30 mm
z_target = 0.1  # 150 mm

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
thetas_eq = gauss_newton_lm(
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
import numpy as np
import matplotlib.pyplot as plt

# ---------- rotation & field helpers ----------

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
if __name__ == '__main__':
    thetas_target = np.array([np.deg2rad(5)])
    B_req2, info = magnetic_field_for_theta(
        thetas_target, axes, lengths, masses, eta0_list,
        f_e=f_e, gvec=gvec, E=r_arr*0 + E_arr, r=r_arr, nu=nu_arr,
        R_base=None, p_base=None, quad_n=12,
        prefer_dir=None, 
        reg=0.0
    )
    r_vec, mu_hat, R_epm2 = single_epm_pose_from_B(B_req2, mu_mag, e_r, mu0)
    print(f"R_epm: {R_epm2}")
    print(f"R_vec: {r_vec}")
    # ---------- SWEEP MAGNET ROTATION AROUND GLOBAL Y ----------

    # Angle sweep, e.g. from 0 to 180 degrees
    phis_deg = np.linspace(0.0, 180.0, 90)   # ~2° steps
    phis_rad = np.deg2rad(phis_deg)

    n_seg = len(lengths)
    thetas_all = np.zeros((len(phis_rad), n_seg))  # store bending per segment

    # Choose an initial guess for theta for all solves
    # (if you already computed an equilibrium, use that instead)
    try:
        thetas_init0 = thetas0.copy()
    except NameError:
        thetas_init0 = thetas0.copy()

    # Magnet position (fixed) in world
    r_epm = np.array([0.15, 0.0, 0.0])

    for i, phi in enumerate(phis_rad):
        # 1) Rotate magnet around GLOBAL y-axis
        #    IMPORTANT: use the *orientation matrix*, not the position.
        #    R_epm must come from your earlier epm_pose_for_tip_yz / single_epm_pose_from_B.
        R_epm_phi = rotate_epm_around_global_x(R_epm2, phi)

        # 2) Dipole direction in world (magnet's local z-axis)
        mu_hat_phi = R_epm_phi[:, 2]

        # 3) Compute B at origin from rotated magnet
        B_phi = B_from_dipole(
            r_eval=np.zeros(3),
            r_epm=r_epm,
            mu_hat=mu_hat_phi,
            mu_mag=mu_mag,
            mu0=mu0
        )

        # 4) Solve equilibrium with THIS B_phi
        thetas_phi = gauss_newton_lm(
            thetas_init0,            # fixed initial guess for all angles
            axes, lengths, masses, eta0_list,
            B=B_phi,
            f_e=f_e,
            gvec=gvec,
            E=E_arr,
            r=r_arr,
            nu=nu_arr,
            R_base=None,
            p_base=None,
            quad_n=12,
            max_iter=50,
            tol_r=1e-10,
            tol_step=1e-12
        )

        thetas_all[i, :] = thetas_phi

    # Convert to degrees for plotting
    thetas_all_deg = np.rad2deg(thetas_all)

    # ---------- PLOT BENDING VS ANGLE ----------

    plt.figure(figsize=(8, 5))
    for seg_idx in range(n_seg):
        plt.plot(phis_deg, thetas_all_deg[:, seg_idx],
                marker='o',
                label=f"Segment {seg_idx+1}")

    plt.axhline(0, color='k', linewidth=0.8)
    plt.xlabel("Magnet rotation about global y-axis [deg]")
    plt.ylabel("Segment bending angle θ [deg]")
    plt.title("Beam bending vs. magnet rotation about global y-axis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Bending angles (deg) at each magnet rotation angle:")
    for phi_deg, th_deg in zip(phis_deg, thetas_all_deg):
        print(f"phi = {phi_deg:6.1f} deg -> thetas = {th_deg}")

    print("Last angle:", phi_deg, "deg, B =", B_phi)
