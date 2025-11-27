import numpy as np
import matplotlib.pyplot as plt   # <-- fix this import
from run_solver2 import *

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def project_tip_to_yz_plane(p_target):
    """
    Given a 3D target point p_target in world frame, construct a rotation Rz
    about the z-axis such that in the rotated frame the point lies in the y'-z'
    plane (x'=0). Returns (Rz, y', z').
    """
    p = np.asarray(p_target, float).reshape(3)
    x, y, z = p
    rho = np.hypot(x, y)      # radial distance in x-y plane

    if rho < 1e-9:
        # Already on z-axis: no azimuth needed
        Rz = np.eye(3)
        return Rz, y, z

    # We want Rz^T @ p = [0, rho, z] (i.e., x' = 0, y' = rho).
    # For rotation about z: Rz^T p = rot_z(-phi) p.
    # A convenient choice: phi = atan2(-x, y)
    phi = np.arctan2(-x, y)
    Rz = rot_z(phi)

    p_rot = Rz.T @ p
    _, y_prime, z_prime = p_rot
    return Rz, y_prime, z_prime

def epm_pose_for_tip_3d(p_target,
                        lengths, axes, masses,
                        eta0_list, gvec, E, r, nu,
                        L_cat, mu_mag, mu0=4e-7*np.pi,
                        quad_n=12):
    """
    Option B: treat the beam model as planar in its own 'beam frame'
    (bends only in y–z with given axes, gvec, etc.), then rotate the
    entire solution around the world z-axis to hit a 3D target.

    Steps:
      1) Compute a rotation Rz such that the plane through the world z-axis
         and p_target is mapped to the beam's y–z plane.
      2) Rotate the target into the beam frame -> get (y', z').
      3) Solve the planar IK (epm_pose_for_tip_yz) in that beam frame
         using the *original* axes, gvec, etc. (no rotation of physics).
      4) Rotate B, EPM pose, and beam geometry back into the world frame
         with the same Rz.

    Returns dict with:
      thetas_target   : segment angles (in beam frame)
      B_req           : required field in world frame
      r_epm           : EPM position in world frame
      mu_hat_epm      : EPM dipole direction in world frame
      R_epm           : EPM rotation matrix in world frame
      Rz              : rotation from beam frame -> world
      info            : debug info from magnetic_field_for_theta
    """
    p_target = np.asarray(p_target, float).reshape(3)

    # 1) Choose the bending plane and get rotated planar (y', z')
    Rz, y_2d, z_2d = project_tip_to_yz_plane(p_target)
    # IMPORTANT: we DO NOT rotate axes or gvec. The beam model lives in beam frame.

    # 2) Planar IK in the beam frame (y', z')
    # e_r is defined in the beam frame; e.g. magnet line along +x' axis
    e_r_beam = np.array([1.0, 0.0, 0.0])

    thetas_target, B_req_beam, r_epm_beam, mu_hat_beam, R_epm_beam, info = epm_pose_for_tip_yz(
        y_2d, z_2d,
        lengths=lengths, axes=axes, masses=masses,
        eta0_list=eta0_list, gvec=gvec,   # <- notice: original gvec, axes
        E=E, r=r, nu=nu,
        L_cat=L_cat,
        mu_mag=mu_mag, e_r=e_r_beam,
        mu0=mu0
    )

    # 3) Map solution from beam frame to world frame
    B_req_world      = Rz @ B_req_beam
    r_epm_world      = Rz @ r_epm_beam
    mu_hat_epm_world = Rz @ mu_hat_beam
    R_epm_world      = Rz @ R_epm_beam

    return {
        "thetas_target": thetas_target,
        "B_req": B_req_world,
        "r_epm": r_epm_world,
        "mu_hat_epm": mu_hat_epm_world,
        "R_epm": R_epm_world,
        "Rz": Rz,
        "info": info
    }


# Example 3D target (in meters)
p_target = np.array([0.02, -0.01, 0.035])  # (x,y,z)

result = epm_pose_for_tip_3d(
    p_target,
    lengths=lengths, axes=axes, masses=masses,
    eta0_list=eta0_list, gvec=gvec,
    E=E_arr, r=r_arr, nu=nu_arr,
    L_cat=L_cat,
    mu_mag=mu_mag, mu0=mu0
)


thetas_target = result["thetas_target"]
B_req         = result["B_req"]
r_epm         = result["r_epm"]
R_epm         = result["R_epm"]
Rz            = result["Rz"]
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

# --- compute centerline in BEAM frame (planar) ---
P_beam, R_end_beam, p_end_beam = sample_centerline(
    axes, thetas_target, lengths, n_per_seg=100
)

# --- map to world frame ---
P     = (Rz @ P_beam.T).T
R_end = Rz @ R_end_beam
p_end = Rz @ p_end_beam

print("Beam tip position (world):", p_end)
print("Target tip position      :", p_target)
print("Tip error:", p_end - p_target, "   ||error|| =", np.linalg.norm(p_end - p_target))

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# beam centerline
ax.plot(P[:,0], P[:,1], P[:,2],
        '-', color='gray', linewidth=3, label='beam')

# base and tip frames
draw_triad(ax, origin=np.zeros(3), R=np.eye(3), scale=0.05*np.sum(lengths))
draw_triad(ax, origin=p_end,      R=R_end,     scale=0.05*np.sum(lengths))

# EPM in world
draw_epm_cylinder(ax, center=r_epm, R_epm=R_epm,
                  length=L_epm, diameter=D_epm,
                  n_theta=80, n_z=40, alpha=0.9)

# B-field arrow
B_unit = B_req / (np.linalg.norm(B_req) + 1e-16)
ax.quiver(0, 0, 0, *(0.03*B_unit),
          color='k', linewidth=2.5, arrow_length_ratio=0.25,
          label='B field')

# target point marker
ax.scatter(*p_target, color='m', s=40, label='target tip')

ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_title('3D bending + EPM pose from 3D tip target ')
ax.legend()
ax.grid(True)
set_axes_equal(ax)
plt.tight_layout()
plt.show()
