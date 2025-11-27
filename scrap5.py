import numpy as np
from energy_minimisation import (
    magnetic_field_for_theta,
    central_difference_jacobian,
    residual_theta,
    my_gauss_newton_lm,
)
from beam_pcc import stack_blocks

# ======================
#  Geometry & Materials
# ======================

L_cat = 0.1        # total catheter length [m]
L1    = L_cat        # single-segment length [m]

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
theta_target_deg = 10.0
thetas_target    = np.array([np.deg2rad(theta_target_deg)])

# Initial guess for Newton solve (used later)
thetas0 = np.array([np.deg2rad(5.0)])   # any small nonzero is fine

# We tell magnetic_field_for_theta that B must lie along +y:
prefer_dir = None  # B = alpha * +y

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
thetas_sol = my_gauss_newton_lm(
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
def thetas_from_yz(y, z, lengths, L_cat):
    """
    Map desired tip (y,z) to segment bending angles under
    a planar constant-curvature assumption in the y–z plane.

    Assumes:
      - beam base at origin,
      - initial beam axis along +z,
      - bending about +x (so deflection is in y–z).

    For a constant-curvature arc:
        y / z = tan(theta_tot / 2)
        => theta_tot = 2 * atan2(y, z)

    y, z    : desired tip position in base frame
    lengths : array of segment lengths
    L_cat   : total length (sum(lengths))
    """
    z_safe = z if abs(z) > 1e-6 else 1e-6
    theta_tot = 2.0 * np.arctan2(y, z_safe)

    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas
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
    p_tip = p_nodes[-1]  # last node
    return p_tip
def single_epm_pose_from_B(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Compute the pose (r_vec, mu_hat, R_epm) of a SINGLE external permanent magnet (EPM)
    that generates the required field B_req at the origin, using the dipole model:

        B = (μ0 / (4π|r|³)) (3 ê_r ê_rᵀ - I₃) (|μ| μ̂)

    Inputs:
      B_req : desired field at the origin [T], shape (3,)
      mu_mag: dipole magnitude [A·m²]
      e_r   : unit vector from magnet to workspace center (origin)
      mu0   : vacuum permeability [T·m/A]

    Returns:
      r_vec  : position of magnet in world frame (3,)
      mu_hat : unit dipole direction (3,)
      R_epm  : rotation matrix (columns are body x,y,z; z = μ̂)
    """
    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    # A = (3 ê_r ê_rᵀ - I)
    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    # Solve A μ̂ = B̂  for μ̂
    B_hat = B_req / Bmag
    mu_hat = np.linalg.solve(A, B_hat)
    mu_hat /= np.linalg.norm(mu_hat)

    # Magnitude condition: Bmag = μ0 |μ| ||A μ̂|| / (4π r^3)
    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # Place magnet so that vector from magnet to origin is +r_mag * e_r.
    # If r_vec is magnet position, origin - r_vec = r_mag * e_r -> r_vec = -r_mag * e_r.
    r_vec = -r_mag * e_r

    # Build rotation with z-axis aligned to μ̂
    z = mu_hat
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_epm = np.column_stack([x, y, z])

    return r_vec, mu_hat, R_epm
def epm_dipole_from_br_dims(Br, D, L, mu0=4e-7*np.pi):
    """
    Given EPM remanence Br [T], diameter D [m], length L [m],
    return its magnetic dipole magnitude |μ| [A·m²] under
    a uniformly magnetized cylinder assumption.
    """
    M = Br / mu0
    V = np.pi * (0.5*D)**2 * L
    return M * V
def epm_pose_for_tip_yz(y, z,
                        lengths, axes, masses,
                        eta0_list, gvec, E, r, nu,
                        L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
                        quad_n=12,
                        prefer_B_dir=None):
    """
    Given a desired planar tip position (y,z) of the beam,
    solve (approximately) for the EPM pose that should produce
    that configuration under static equilibrium.

    Steps:
      (1) map (y,z) -> thetas (constant curvature assumption)
      (2) solve Eq.(20) for B_req via magnetic_field_for_theta
          with B constrained along prefer_B_dir
      (3) solve dipole inverse problem for EPM pose via single_epm_pose_from_B

    Returns:
      thetas_target, B_req, r_vec, mu_hat, R_epm, info
    """
    # 1) y,z -> thetas  (constant curvature)
    thetas_target = thetas_from_yz(y, z, lengths, L_cat)

    # 2) thetas -> required field B, constrained along prefer_B_dir
    B_req, info = magnetic_field_for_theta(
        thetas_target, axes, lengths, masses, eta0_list,
        f_e=np.zeros(6*len(lengths)),
        gvec=gvec, E=E, r=r, nu=nu,
        R_base=None, p_base=None, quad_n=quad_n,
        prefer_dir=None,
        reg=0.0
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
    theta_sol = my_gauss_newton_lm(
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
# --- beam & material parameters as in your working example ---
L_cat = 0.047
L1    = L_cat
r1    = 0.001

E_niti   = 3e6
rho_niti = 6700
nu_niti  = 0.33
gvec     = np.array([0.0, 0.0, 9.8])

A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0

lam    = rho_niti * A_cs
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths

axes    = [np.array([1.0, 0.0, 0.0], float)]  # bend about +x
r_arr   = np.array([r1],       float)
E_arr   = np.array([E_niti],   float)
nu_arr  = np.array([0.49],     float)
f_e     = np.zeros(6*n, float)

# --- internal IPM (same as before) ---
Br   = 1
mu0  = 4e-7 * np.pi
D_ipm = 15e-4
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_scalar = Br * V_ipm / mu0

m_dir_local = np.array([0.0, 0.0, 1.0], float)   # along +z
eta_ipm = (mu_ipm_scalar / L_ipm) * m_dir_local  # [A·m]
eta0_list = [eta_ipm]

# --- EPM dipole magnitude (given) & e_r direction ---
mu_mag = 220.0               # [A·m²]
e_r    = np.array([1.0, 0.0, 0.0])   # magnet lies on -x side

# --- desired tip position in y–z plane ---
y_target = -10e-3    # 4 cm sideways
z_target = 42e-3    # near "horizontal" tip

thetas_target, B_req, r_epm, mu_hat_epm, R_epm, info = epm_pose_for_tip_yz(
    y_target, z_target,
    lengths=lengths, axes=axes, masses=masses,
    eta0_list=eta0_list, gvec=gvec,
    E=E_arr, r=r_arr, nu=nu_arr,
    L_cat=L_cat,
    mu_mag=mu_mag, e_r=e_r,
    mu0=mu0
)

print("thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req, " |B| (mT):", np.linalg.norm(B_req)*1e3)
print("EPM position r_epm:", r_epm, " |r|:", np.linalg.norm(r_epm))
print("EPM dipole direction μ̂:", mu_hat_epm)

# --- check equilibrium with that B_req ---
thetas_eq = my_gauss_newton_lm(
    thetas_target, axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)
print("Equilibrium thetas (deg):", np.rad2deg(thetas_eq))

# --- tip position from this B and geometry ---
s0 = 0.0
x0 = np.hstack([B_req, s0])

p_tip = tip_pos_from_B_s(
    x0,
    thetas_init=thetas_eq,
    axes=axes, lengths0=lengths, lam=lam,
    eta0_list=eta0_list, gvec=gvec,
    E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
    f_e=f_e,
    R_base=None, p_base=None,
    quad_n=12
)
print("Tip position p_tip:", p_tip)
import numpy as np

def Rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c,-s],
        [0, s, c]
    ])

def Ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c,-s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def R_from_rxyz(rx, ry, rz):
    """
    R = Rz(rz) * Ry(ry) * Rx(rx)
    """
    return Rz(rz) @ Ry(ry) @ Rx(rx)

def rxyz_from_R(R):
    """
    Inverse of R_from_rxyz for R = Rz(rz) * Ry(ry) * Rx(rx)

    Returns:
        rx, ry, rz (radians)
    """
    # sy = sqrt(R00^2 + R10^2)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        rz = np.arctan2(R[1, 0], R[0, 0])
        ry = np.arctan2(-R[2, 0], sy)
        rx = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock: sy ~ 0
        rz = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rx = 0.0

    return rx, ry, rz
rx0, ry0, rz0 = 3.198, 0.508, 0.075
R_neutral = R_from_rxyz(rx0, ry0, rz0)

# Where does the tool's +z axis point in world?
ez_tool = np.array([0.0, 0.0, 1.0])
mu_hat_neutral = R_neutral @ ez_tool
print("mu_hat at neutral:", mu_hat_neutral)
R_new = Rz(np.deg2rad(30)) @ R_neutral
rx2, ry2, rz2 = rxyz_from_R(R_new)
print("new" ,[rx2, ry2, rz2])

R_des = R_epm              # rotation of magnet in world

rx_des, ry_des, rz_des = rxyz_from_R(R_des)

print("Desired robot orientation:")
print("R_epm ", R_epm)
print("  rx =", rx_des, "rad")
print("  ry =", ry_des, "rad")
print("  rz =", rz_des, "rad")
print("  (deg):", np.degrees([rx_des, ry_des, rz_des]))
R_neutral = R_from_rxyz(0.0, -3.116, 0.0)
mu_hat_neutral = R_neutral @ np.array([0.0, 0.0, 1.0])
print(mu_hat_neutral)
import numpy as np
import math

# --- Existing helper: rotation vector -> rotation matrix (Rodrigues) ---
def rotvec_to_rotmat(r):
    r = np.array(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

# --- Your function: rotation matrix -> rotation vector (axis * angle) ---
def rot_to_axis_angle(R):
    eps = 1e-12
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # clamp for numerical stability
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)  # rotation vector

def apply_z_rotation_deg_to_tcp(tcp_pose, delta_deg):
    """
    tcp_pose: [x, y, z, rx, ry, rz]  (UR-style axis-angle)
    delta_deg: rotation about *base* Z axis in degrees (right-hand rule)

    Returns a new TCP pose [x, y, z, rx', ry', rz'].
    """
    tcp_pose = list(tcp_pose)
    pos = np.array(tcp_pose[0:3], dtype=float)
    r   = np.array(tcp_pose[3:6], dtype=float)

    # current orientation as matrix
    R_current = rotvec_to_rotmat(r)

    # rotation about base Z axis
    theta = math.radians(delta_deg)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # New orientation: first rotate around base Z, then apply old orientation
    R_new = Rz @ R_current

    # back to rotation vector
    r_new = np.array(rot_to_axis_angle(R_new))

    return np.concatenate([pos, r_new])

# -----------------------------------------------------------------
# Example with your pose:
tcp = [0.6405172952674639, -0.2882901437037632, 0.7215228405182936,
       -3.0052053711273765, -0.47764426456326603, -0.07041225137046711]

tcp_new = apply_z_rotation_deg_to_tcp(tcp, 30.0)

print("Original TCP:", tcp)
print("New TCP (+30° about base Z):", tcp_new.tolist())

import numpy as np
import math

# R_epm, r_epm from your optimisation:
# R_epm: (3,3) rotation matrix
# r_epm: (3,) position vector [x, y, z] in meters (UR base frame)

rx, ry, rz = rot_to_axis_angle(R_epm)   # radians

x, y, z = tcp[:3]  # assuming r_epm already in UR base frame [m]

TCP_TARGET = [float(x), float(y), float(z),
              float(rx), float(ry), float(rz)]

print("TCP_TARGET to send to UR:", TCP_TARGET)
