import numpy as np
from modelling_phi_angle import constant, root_theta
from magnetic_field_magnitude import magnetic_field, magnetic_moment
from modelling_alpha import tip_angle_from_theta
def theta_angle_solved(B, phi, mag, A_cs, L, E, I):
    rhs_eq = constant(B, mag, A_cs, L, E, I)
    theta_L = root_theta(rhs_eq, phi)
    return theta_L

def dtheta_dB(B, phi, mag, A_cs, L, E, I, dB=1e-4):
    theta_plus = theta_angle_solved(B+dB, phi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B-dB, phi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dB)

def dtheta_dphi(B, phi, mag, A_cs, L, E, I, dphi=1e-4):
    theta_plus = theta_angle_solved(B, phi+dphi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B, phi-dphi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dphi)

def dtheta_dL(B, phi, mag, A_cs, L, E, I, dL=1e-4):
    theta_plus = theta_angle_solved(B, phi, mag, A_cs, L+dL, E, I)
    theta_minus = theta_angle_solved(B, phi, mag, A_cs, L-dL, E, I)
    return (theta_plus - theta_minus) / (2*dL)

def jacobian_controller(theta_des,B_init, phi_init, L_init,
                        mag, A_cs, E, I, L_min = 0.04, L_max=0.06, B_min=0.008, B_max = 0.02,
                        phi_min = np.deg2rad(-80), phi_max = np.deg2rad(80), dt=0.05,
                        Kp = 5.0, Ki = 0.0, Kd = 0.5, 
                        max_iter = 300, damping = 1e-3):


    B = B_init
    phi = phi_init
    L = L_init

    e_init = 0
    e_prev = 0.0
    for k in range(max_iter):
        # print(B, phi, mag, A_cs, L, E, I)
        act_theta = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - act_theta
        e_dot = (e-e_prev)/dt
        e_init += e*dt

        theta_do_cmd = Kp*e + Ki*e_init + Kd * e_dot

        J_B = dtheta_dB(B, phi, mag, A_cs, L, E, I)
        J_L = dtheta_dL(B, phi, mag, A_cs, L, E, I)
        J_phi = dtheta_dphi(B, phi, mag, A_cs, L, E, I)

        if (B <= B_min and theta_do_cmd* J_B < 0):
            J_B = 0.0
        if (B>= B_max and theta_do_cmd*J_B > 0):
            J_B = 0.0

        if (L <= L_min and theta_do_cmd* J_L < 0):
            J_L = 0.0

        if (L>= L_max and theta_do_cmd*J_L > 0):
            J_L = 0.0
        
        if (phi <= phi_min and theta_do_cmd* J_phi < 0):
            J_phi = 0.0
        if (phi>= phi_max and theta_do_cmd*J_phi > 0):
            J_phi = 0.0
        
        JJt = J_B**2 + J_L**2 + J_phi**2
        gain = theta_do_cmd/ (JJt+damping**2)
        dB = J_B * gain
        dL = J_L * gain
        dphi = J_phi * gain 

        B += dB*dt
        L += dL*dt
        phi += dphi*dt

        B = min(max(B, B_min), B_max)
        L = min(max(L, L_min), L_max)
        phi = min(max(phi, phi_min), phi_max)

        theta_update = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - theta_update
        if abs(e) <= np.deg2rad(.5):
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return B, phi, L, np.rad2deg(theta_update)

def dB_dx(x, mu_0, mu, mu_hat, h=1e-4):
    p_plus = np.array([x+h, 0, 0])
    p_minus = np.array([x-h, 0, 0])

    Bp = magnetic_field(mu_0, mu, p_plus, mu_hat)
    Bm = magnetic_field(mu_0, mu, p_minus, mu_hat)
    return (Bp[0] - Bm[0])/ (2*h)
def solve_mag_pose(field_des, x_init, mu_0, mu, mu_hat, dt = 0.05, Kp = 1, Ki = 0.0, Kd = 0.01, damping = 1e-3, max_iter = 500):
    e_init = 0.0
    e_prev = 0.0
    x = x_init

    for k in range(max_iter):
        p_vec = np.array([x, 0, 0])
        # print(f"parameters are x: {x}, mu: {mu}, mu_hat: {mu_hat}")
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        # print(f"Current field is {field_curr[0]}, desired field is {field_des}")
        e = field_des - field_curr[0]
        e_init +=e*dt
        e_dot = (e-e_prev)*dt
        theta_dot_cmd = Kp*e + Ki*e_init + Kd*e_dot
        J_x = dB_dx(x, mu_0, mu, mu_hat)
        JJt = J_x**2
        gain = theta_dot_cmd / (JJt + damping**2)
        dx = J_x *gain
        x += dx * dt
        p_vec = np.array([x, 0, 0])
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        e = field_des - field_curr[0]
        if abs(e) <= 1e-5:
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return x, field_curr
import numpy as np

def build_danger_intervals(phi_values, alphas, all_dtheta_dalpha,
                           eps_grad=1e-3,
                           buffer_deg=5.0):
    """
    For each phi, find alpha zones where dθ/dα changes sign.
    Around each sign change, define a buffered 'danger' interval.

    Returns
    -------
    danger_intervals_by_phi : dict
        {phi_value: [(alpha_min, alpha_max), ...]} in radians
    """
    buffer = np.deg2rad(buffer_deg)

    danger_intervals_by_phi = {}

    for i, phi in enumerate(phi_values):
        grad = np.array(all_dtheta_dalpha[i])
        alpha_curve = alphas

        # signed gradient, ignore tiny values
        sign_grad = np.sign(grad)
        sign_grad[np.abs(grad) < eps_grad] = 0

        sign_change_indices = []
        for j in range(1, len(sign_grad)):
            if sign_grad[j-1] == 0 or sign_grad[j] == 0:
                continue
            if sign_grad[j] * sign_grad[j-1] < 0:
                sign_change_indices.append(j)

        snap_alphas = alpha_curve[sign_change_indices]

        intervals = []
        for a_snap in snap_alphas:
            a_min = a_snap - buffer
            a_max = a_snap + buffer
            intervals.append((a_min, a_max))

        danger_intervals_by_phi[phi] = intervals

    return danger_intervals_by_phi


def alpha_in_intervals(alpha, intervals):
    """Return True if alpha lies in any (min,max) interval."""
    for a_min, a_max in intervals:
        if a_min <= alpha <= a_max:
            return True
    return False
def dtheta_dalpha(phi, alpha, R, m0, mu0, mag, A_cs, L, E, I, dalpha=1e-4):
    """
    Numerical Jacobian dθ/dα at fixed φ and beam parameters.
    """
    theta_plus, _ = tip_angle_from_theta(phi, alpha + dalpha, R, m0, mu0,
                                         mag, A_cs, L, E, I)
    theta_minus, _ = tip_angle_from_theta(phi, alpha - dalpha, R, m0, mu0,
                                          mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2 * dalpha)

def alpha_controller(theta_des,
                     phi_fixed,
                     alpha_init,
                     R, m0, mu0,
                     mag, A_cs, L, E, I,
                     danger_intervals_by_phi,
                     phi_values,
                     alpha_min = -np.pi,
                     alpha_max = np.pi,
                     dt = 0.05,
                     Kp = 5.0, Ki = 0.0, Kd = 0.5,
                     damping = 1e-3,
                     max_iter = 300,
                     grad_runtime_thresh = 20.0,
                     theta_zero_thresh = np.deg2rad(3.0)):
    """
    Online alpha-controller:
    - keeps phi_fixed and L fixed
    - uses local dθ/dα + PID to drive θ -> theta_des
    - refuses to enter precomputed 'danger' regions in alpha
      (based on gradient sign changes)
    - also watches for |θ| ~ 0 (about to flip bending side).
    """

    # pick the nearest phi grid point for which we have intervals
    phi_arr = np.array(phi_values)
    idx_phi = np.argmin(np.abs(phi_arr - phi_fixed))
    phi_key = phi_arr[idx_phi]
    danger_intervals = danger_intervals_by_phi.get(phi_key, [])

    alpha = alpha_init
    e_int = 0.0
    e_prev = 0.0

    for k in range(max_iter):
        # current beam angle for this alpha
        theta_curr, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0,
                                                  mag, A_cs, L, E, I)

        # safety check: near zero bending angle (about to flip direction)
        if abs(theta_curr) < theta_zero_thresh:
            print(f"[alpha controller] |theta| ~ 0 (|θ|={np.rad2deg(theta_curr):.2f} deg). "
                  "Stopping to avoid flip.")
            return alpha, theta_curr, B_curr, "theta_near_zero"

        # error
        e = theta_des - theta_curr
        e_int += e * dt
        e_dot = (e - e_prev) / dt

        # PID target "theta rate"
        theta_dot_cmd = Kp * e + Ki * e_int + Kd * e_dot

        # local Jacobian wrt alpha
        J_alpha = dtheta_dalpha(phi_fixed, alpha, R, m0, mu0,
                                mag, A_cs, L, E, I)

        # safety: huge gradient (we're near something nasty)
        if abs(J_alpha) > grad_runtime_thresh:
            print(f"[alpha controller] |dθ/dα| too large ({J_alpha:.2e}). "
                  "Stopping near snap region.")
            return alpha, theta_curr, B_curr, "grad_too_large"

        # pseudo-inverse gain
        JJt = J_alpha**2
        gain = theta_dot_cmd / (JJt + damping**2)
        dalpha = J_alpha * gain

        # candidate new alpha
        alpha_new = alpha + dalpha * dt

        # clamp to allowed bounds
        alpha_new = min(max(alpha_new, alpha_min), alpha_max)

        # check if alpha_new would enter precomputed danger intervals
        if alpha_in_intervals(alpha_new, danger_intervals):
            print("[alpha controller] Proposed alpha enters danger zone (snap region). "
                  "Not proceeding further.")
            return alpha, theta_curr, B_curr, "danger_zone"

        # accept update
        alpha = alpha_new

        # evaluate updated theta
        theta_new, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0,
                                                 mag, A_cs, L, E, I)
        e_new = theta_des - theta_new

        if abs(e_new) <= np.deg2rad(0.5):
            print(f"[alpha controller] Converged in {k} steps")
            return alpha, theta_new, B_curr, "converged"

        e_prev = e_new

    print("[alpha controller] Max iterations reached without full convergence")
    return alpha, theta_new, B_curr, "max_iter"
if __name__ == '__main__':
    theta_target = np.deg2rad(50.049)
    theta_target2 = np.deg2rad(65)
    mag = 128e3
    r = 0.0015
    E = 4.5e6
    A_cs = np.pi * r**2
    I = np.pi * r**4/4
    B_init = 0.01; phi_init = np.deg2rad(25); L_init = 0.04
    mu_0 = 4e-7*np.pi
    B_r = 1.2
    r = 0.03
    p = 0.09
    m_hat = np.array([1,0,0])
    x_init = 0.09
    mag_epm = magnetic_moment(B_r, mu_0, r, p)
    print(f"Magnetic moment of EPM is {mag_epm}")
    B_sol, phi_sol, L_sol, theta_sol = jacobian_controller(theta_target, B_init, phi_init, L_init, mag, A_cs, E, I, L_max=0.06)
    print(f"B solution: {B_sol*1000}mT, phi_sol: {np.rad2deg(phi_sol)}, L_sol: {L_sol}, with angle of: {theta_sol}")
    mag_pose, field = solve_mag_pose(B_sol, x_init, mu_0, mag_epm, m_hat)
    print(f"Magnetic distance is {mag_pose} which prodices a field of {field}")
    # offline (once):
    phi_values = np.deg2rad(np.linspace(10, 90, 5))
    alphas = np.deg2rad(np.linspace(-180, 180, 91))
    allBmag = []
    all_bend = []
    all_dtheta_dalpha = []
    for phi in phi_values:
        bend_values = []
        B_values  = []
        for alpha in alphas:
            theta_L, B = tip_angle_from_theta(phi, alpha, mag_pose, mag_epm, mu_0, mag, A_cs, L_sol, E, I)
            bend_values.append(theta_L)
            B_values.append(B)
        all_bend.append(bend_values)
        allBmag.append(B_values)

    for bending in all_bend:
        bending = np.array(bending)
        dtheta_dalpha_curve = np.gradient(bending, alphas)  # use 'alphas' here
        all_dtheta_dalpha.append(dtheta_dalpha_curve)
    # all_bend[i][j], all_dtheta_dalpha[i][j] computed as before
    danger_intervals_by_phi = build_danger_intervals(phi_values, alphas, all_dtheta_dalpha,
                                                    eps_grad=1e-3, buffer_deg=5.0)
    # after stage 1:
    phi_fixed = phi_sol  # from jacobian_controller
    L_fixed = L_sol      # if you want to freeze length; or your chosen L
    alpha_init = 0.0

    alpha_sol, theta_final, B_final, status = alpha_controller(
        theta_des = theta_target2,      # final desired angle
        phi_fixed = phi_fixed,
        alpha_init = alpha_init,
        R = mag_pose, m0 = mag_epm, mu0 = mu_0,
        mag = mag, A_cs = A_cs, L = L_fixed, E = E, I = I,
        danger_intervals_by_phi = danger_intervals_by_phi,
        phi_values = phi_values,
        alpha_min = np.deg2rad(-180),
        alpha_max = np.deg2rad(180),
        dt = 0.05,
        Kp = 5.0, Ki = 0.0, Kd = 0.5
    )

    print("alpha status:", status)
    print("alpha solution:", np.rad2deg(alpha_sol), "deg")
    print("theta final:", np.rad2deg(theta_final), "deg")