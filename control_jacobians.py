
from beam_pcc import *
from energy_minimisation import *
from run_solver2 import *
import matplotlib.pyplot as plt
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
    p_tip = p_nodes[-1]
    return p_tip
def solve_theta_given_B_lengths(thetas_init,
                                axes, lengths, masses, eta0_list,
                                B, f_e, gvec, E, r, nu,
                                R_base=None, p_base=None, quad_n=12):
    """
    Solve Eq. (20) for theta given:
      - B: magnetic field,
      - lengths, masses, etc.
    Uses Gauss-Newton-LM solver.
    """
    theta_sol = my_gauss_newton_lm(
        thetas_init, axes, lengths, masses, eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
        R_base=R_base, p_base=p_base, quad_n=quad_n,
        tol_grad=1e-12, tol_r=1e-12
    )

    return theta_sol
def tip_pos_from_B_s(x,               
                     thetas_init,
                     axes, lengths0, lam,          
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
    r_test = residual_theta(theta_sol, axes, lengths, masses, eta0_list,
                        B, f_e, gvec, E_arr, r_arr, nu_arr,
                        R_base=R_base, p_base=p_base, quad_n=quad_n)
    print("res_norm after solve:", np.linalg.norm(r_test))


    # 4) Compute tip position from theta_sol
    p_tip = tip_position_from_theta(
        axes, theta_sol, lengths,
        R_base=R_base, p_base=p_base
    )
    return p_tip
def control_jacobian_B_s(B0, s0,
                         thetas_init,
                         axes, lengths0, lam,
                         eta0_list, gvec, E_arr, r_arr, nu_arr,
                         f_e=None,
                         R_base=None, p_base=None,
                         quad_n=12,
                         rel_eps=1e-6, abs_eps=1e-8, h_floor=1e-4):
    """
    Compute the control Jacobian J_ctrl = ∂p_tip / ∂[B_x, B_y, B_z, s]
    at the operating point (B0, s0).

    Returns:
      p_tip0 : tip position at (B0, s0)
      J_ctrl : 3 × 4 Jacobian
    """

    if f_e is None:
        f_e = np.zeros(6*len(lengths0), float)

    x0 = np.hstack([np.asarray(B0, float).reshape(3), float(s0)])

    # wrapper so central_difference_jacobian sees a function f(x)
    def f_controls(x):
        x = np.asarray(x, float)
        print("f_controls called with x:", x)  # DEBUG
        out = tip_pos_from_B_s(
            x,
            thetas_init=thetas_init,
            axes=axes, lengths0=lengths0, lam=lam,
            eta0_list=eta0_list, gvec=gvec,
            E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
            f_e=f_e,
            R_base=R_base, p_base=p_base,
            quad_n=quad_n
        )
        print("  -> p_tip:", out)
        return out


    p0, J_ctrl = central_difference_jacobian(
        f_controls, x0,
        rel_eps=rel_eps, abs_eps=abs_eps, h_floor=h_floor
    )
    # p0 is the tip position at (B0, s0)
    # J_ctrl is 3 × 4: [∂p/∂Bx, ∂p/∂By, ∂p/∂Bz, ∂p/∂s]
    return p0, J_ctrl
import numpy as np


B0 = B_req.copy()        # from your earlier equilibrium solve
s0 = 0.0                 # no extension at nominal point
lengths0 = lengths.copy()
lam = rho_niti * A_cs    # you already defined this earlier

# Get an equilibrium theta at (B0, s0)
# (if you already have thetas_sol at this B0 and lengths, reuse it)
thetas_eq = my_gauss_newton_lm(
    thetas0, axes, lengths0, masses, eta0_list,
    B=B0, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)

# --- 2) Compute control Jacobian at this operating point ---

p0, J_ctrl = control_jacobian_B_s(
    B0, s0,
    thetas_init=thetas_eq,
    axes=axes, lengths0=lengths0, lam=lam,
    eta0_list=eta0_list, gvec=gvec,
    E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
    f_e=f_e,
    R_base=None, p_base=None,
    quad_n=12
)
print(f"First Magnetic Field: {B0}")
print("Tip position at operating point p0:", p0)
print("Control Jacobian J_ctrl shape:", J_ctrl.shape)
print("J_ctrl =\n", J_ctrl)

# --- 3) Finite test: compare linear prediction vs actual change ---

def test_direction(delta_u):
    """
    delta_u is a 4-vector [dBx, dBy, dBz, ds].
    This tests p(B0+δB, s0+δs) against p0 + J_ctrl @ delta_u.
    """
    x0 = np.hstack([B0, s0])
    x1 = x0 + delta_u

    p1 = tip_pos_from_B_s(
        x1,
        thetas_init=thetas_eq,    # warm-start with same theta
        axes=axes, lengths0=lengths0, lam=lam,
        eta0_list=eta0_list, gvec=gvec,
        E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
        f_e=f_e,
        R_base=None, p_base=None,
        quad_n=12
    )

    p_pred = p0 + J_ctrl @ delta_u

    err = np.linalg.norm(p1 - p_pred)
    print("\n--- Test direction ---")
    print("delta_u:", delta_u)
    print("actual tip:", p1)
    print("predicted tip:", p_pred)
    print("error norm:", err)
    print("relative error (vs |p1-p0|):",
          err / max(1e-12, np.linalg.norm(p1 - p0)))
    np.set_printoptions(precision=12, suppress=False)
    print("J_ctrl =\n", J_ctrl)

h = 1e-4  # same order as your h_floor

x0 = np.hstack([B0, s0])

# unit vector in Bx direction
e_Bx = np.array([1.0, 0.0, 0.0, 0.0])



eps_B = 5e-3     # or 1e-3
eps_s = 5e-2     # or something small vs L1

test_direction(np.array([eps_B, 0.0,   0.0,   0.0]))
test_direction(np.array([0.0,   eps_B, 0.0,   0.0]))
test_direction(np.array([0.0,   0.0,   eps_B, 0.0]))
test_direction(np.array([0.0,   0.0,   0.0,   eps_s]))

max_iters = 100
tol = 1e-2  

p_target = np.array([0.0, 0.04, 0.08])

u = np.hstack([B0, s0])  

for k in range(max_iters):
    p, J_ctrl = control_jacobian_B_s(
        B0=u[:3], s0=u[3],
        thetas_init=thetas_eq,   
        axes=axes, lengths0=lengths0, lam=lam,
        eta0_list=eta0_list, gvec=gvec,
        E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
        f_e=f_e
    )

    e = p_target - p  

    print(f"iter {k}: |e| = {np.linalg.norm(e):.6e}, u = {u}")

    if np.linalg.norm(e) < tol:
        print("Converged.")
        break

    lambda_reg = 1e-3
    J = J_ctrl
    JT = J.T
    A = JT @ J + lambda_reg * np.eye(4)
    b = JT @ e
    delta_u = np.linalg.solve(A, b)

    max_step = np.array([5e-3, 5e-3, 5e-3, 5e-2])
    delta_u = np.clip(delta_u, -max_step, max_step)

    u = u + delta_u

B_final = u[:3]
s_final = u[3]
print("Final B:", B_final, "Final s:", s_final)
x0 = np.hstack([B_final, s_final])
p1 = tip_pos_from_B_s(
    x0,
    thetas_init=thetas_eq,    # warm-start with same theta
    axes=axes, lengths0=lengths0, lam=lam,
    eta0_list=eta0_list, gvec=gvec,
    E_arr=E_arr, r_arr=r_arr, nu_arr=nu_arr,
    f_e=f_e,
    R_base=None, p_base=None,
    quad_n=12
)
print(f"Final reconstruction is: {p1}")
r_vec, mu_hat, R_epm = single_epm_pose_from_B(B_final, mu_mag, e_r, mu0)
R_epm *=1

print(f"|μ| = {mu_mag:.3f} A·m²")
print("Magnet position r (m):", r_vec, " |r| =", np.linalg.norm(r_vec))
print("μ̂ =", mu_hat)
print("Rotation matrix:\n", R_epm)
# ================================
# VISUALISE FINAL RECONSTRUCTION
# ================================
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

# 1) Final segment lengths and masses (apply s_final to first segment)
#    (lengths0 should be your nominal lengths used in the control loop;
#     if you don't already have it, set lengths0 = lengths.copy() *before* the loop)
lengths_final = lengths0.copy()
lengths_final[0] = lengths0[0] + s_final
masses_final = lam * lengths_final

# 2) Solve for final equilibrium theta for (B_final, lengths_final)
thetas_final = gauss_newton_lm(
    thetas_eq,                     # warm-start from previous equilibrium
    axes, lengths_final, masses_final, eta0_list,
    B=B_final, f_e=np.zeros(6*len(lengths_final)),
    gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)

print("Final bending (deg):", np.rad2deg(thetas_final))

# 3) Sample centerline for final configuration
P_final, R_end_final, p_end_final = sample_centerline(
    axes, thetas_final, lengths_final, n_per_seg=100
)

# 4) Plot: target point, reconstructed beam, and final magnet pose
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# beam centerline
ax.plot(P_final[:, 0], P_final[:, 1], P_final[:, 2],
        '-', color='tab:blue', linewidth=3, label='final beam')

# base & end triads
draw_triad(ax, np.zeros(3), np.eye(3), scale=0.05*np.sum(lengths_final))
draw_triad(ax, p_end_final, R_end_final, scale=0.05*np.sum(lengths_final))

# target tip position
ax.scatter(p_target[0], p_target[1], p_target[2],
           color='tab:green', s=60, marker='^', label='target tip')

# reconstructed tip position (from p1 you already computed)
ax.scatter(p1[0], p1[1], p1[2],
           color='tab:red', s=60, marker='o', label='reconstructed tip')

# draw final EPM corresponding to B_final (r_vec, R_epm already from single_epm_pose_from_B)
draw_epm_cylinder(ax, center=r_vec, R_epm=R_epm,
                  length=L_epm, diameter=D_epm,
                  n_theta=100, n_z=60)

# show B_final direction at origin
B_unit_final = B_final / (np.linalg.norm(B_final) + 1e-16)
ax.quiver(0, 0, 0, *(0.03 * B_unit_final),
          color='k', linewidth=2.5, arrow_length_ratio=0.25,
          label='B_final')

# labels and cosmetics
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Final reconstruction for B_final and s_final')
ax.legend()
ax.grid(True)
set_axes_equal(ax)
plt.tight_layout()
plt.show()
