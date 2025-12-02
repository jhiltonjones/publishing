import numpy as np 
from scipy.integrate import quad
from scipy.optimize import root_scalar

# ---------- existing elastica functions ----------

def lambda_rhs(mag, B_mag, a_cs, leng, E, I):
    return (mag*B_mag*a_cs*leng**2) / (E*I)

def xi_integral(phi, theta_l, eps = 1e-6):
    def integrand(theta):
        w = np.cos(phi-theta_l) - np.cos(phi-theta)
        return 1/np.sqrt(w)
    upper = theta_l - eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    xi_val = 0.5*val**2
    return xi_val

def solve_for_lamba(lambda_target, phi, tol=1e-6):
    eps = 1e-4
    theta_min = eps
    theta_max = max(phi - eps, theta_min)
    def f(theta_l):
        return xi_integral(phi, theta_l) - lambda_target
    sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
    return sol.root

def tip_angle_from_B_phi_L(B, phi, mag, A_cs, L, E, I):
    lam = lambda_rhs(mag, B, A_cs, L, E, I)
    theta = solve_for_lamba(lam, phi)
    return theta  # radians

# ---------- numerical partial derivatives ----------

def dtheta_dB(B, phi, mag, A_cs, L, E, I, h_rel=1e-3):
    h = h_rel * max(1.0, abs(B))
    theta_p = tip_angle_from_B_phi_L(B + h, phi, mag, A_cs, L, E, I)
    theta_m = tip_angle_from_B_phi_L(B - h, phi, mag, A_cs, L, E, I)
    return (theta_p - theta_m) / (2*h)

def dtheta_dphi(B, phi, mag, A_cs, L, E, I, h=1e-4):
    theta_p = tip_angle_from_B_phi_L(B, phi + h, mag, A_cs, L, E, I)
    theta_m = tip_angle_from_B_phi_L(B, phi - h, mag, A_cs, L, E, I)
    return (theta_p - theta_m) / (2*h)

def dtheta_dL(B, phi, mag, A_cs, L, E, I, h_rel=1e-3):
    # relative step to keep it scale-aware
    h = h_rel * max(1.0, abs(L))
    theta_p = tip_angle_from_B_phi_L(B, phi, mag, A_cs, L + h, E, I)
    theta_m = tip_angle_from_B_phi_L(B, phi, mag, A_cs, L - h, E, I)
    return (theta_p - theta_m) / (2*h)

# ---------- Jacobian-damped PID for B, phi, L ----------

def jacobian_pid_controller_B_phi_L(theta_des_deg,
                                    B_init, phi_init, L_init,
                                    mag, A_cs, E, I,
                                    L_min=0.04, L_max=0.06,
                                    B_min=0.008, B_max=0.08,   # 8–80 mT
                                    Kp=5.0, Ki=0.0, Kd=0.5,
                                    dt=0.05,
                                    max_iter=200,
                                    damping=1e-3):

    theta_des = np.deg2rad(theta_des_deg)

    B   = B_init
    phi = phi_init
    L   = L_init

    e_int  = 0.0
    e_prev = 0.0

    for k in range(max_iter):
        theta = tip_angle_from_B_phi_L(B, phi, mag, A_cs, L, E, I)
        e = theta_des - theta
        e_dot = (e - e_prev) / dt
        e_int += e * dt

        # PID on angle (desired angle rate)
        theta_dot_cmd = Kp*e + Ki*e_int + Kd*e_dot

        # Jacobian entries
        J_B   = dtheta_dB(  B, phi, mag, A_cs, L, E, I)
        J_phi = dtheta_dphi(B, phi, mag, A_cs, L, E, I)
        J_L   = dtheta_dL(  B, phi, mag, A_cs, L, E, I)

        JJt  = J_B**2 + J_phi**2 + J_L**2
        gain = theta_dot_cmd / (JJt + damping**2)

        dB   = J_B   * gain
        dphi = J_phi * gain
        dL   = J_L   * gain

        # integrate controls
        B   += dB * dt
        phi += dphi * dt
        L   += dL * dt

        # ---- bounds / saturation ----
        # B in [8 mT, 80 mT]
        B = min(max(B, B_min), B_max)

        # phi in [0, π]
        if phi < 0.0:
            phi += np.pi
        if phi > np.pi:
            phi -= np.pi

        # L in [L_min, L_max]
        L = min(max(L, L_min), L_max)

        e_prev = e

        if abs(e) < np.deg2rad(0.5):
            break

    return B, phi, L, np.rad2deg(theta), k



import numpy as np

def magnetic_field_bar(mu0, mag, p_vec, m_hat):
    r = np.linalg.norm(p_vec)
    p_hat = p_vec / r
    constant = (mu0 * mag) / (4 * np.pi * r**3)
    term1 = (3 * np.outer(p_hat, p_hat)) - np.eye(3)
    return (constant * term1) @ m_hat

def magnetic_moment(B_r, r, length):
    term1 = B_r / mu0
    term2 = np.pi * (r**2) * length
    return term1 * term2

def dBx_dx(mu0, mag, x, m_hat, h=1e-4):
    """Derivative of the x-component of B wrt x."""
    p_plus  = np.array([x + h, 0.0, 0.0])
    p_minus = np.array([x - h, 0.0, 0.0])
    Bp = magnetic_field_bar(mu0, mag, p_plus,  m_hat)
    Bm = magnetic_field_bar(mu0, mag, p_minus, m_hat)
    return (Bp[0] - Bm[0]) / (2*h)

def solve_x_for_Bmag_des(B_des_mag,
                         x_init,
                         mu0, mag, m_hat,
                         Kp=1.0,
                         dt=0.05,
                         max_iter=5000,
                         damping=1e-3,
                         tol=1e-5):

    x = float(x_init)

    for k in range(max_iter):
        p_vec = np.array([x, 0, 0])
        B = magnetic_field_bar(mu0, mag, p_vec, m_hat)

        B_mag = abs(B[0])        # field magnitude on dipole axis
        e = B_des_mag - B_mag    # scalar magnitude error

        if abs(e) < tol:
            break

        # derivative of magnitude wrt x
        Jx = dBx_dx(mu0, mag, x, m_hat)  # derivative of Bx wrt x
        Jmag = np.sign(B[0]) * Jx        # d|B|/dx = sign(B) * dBx/dx

        dx = (Kp * e * Jmag) / (Jmag**2 + damping**2)
        x += dx * dt

    return np.array([x, 0, 0]), magnetic_field_bar(mu0, mag, np.array([x,0,0]), m_hat), k

if __name__ == '__main__':
    mag   = 128e3       
    E_mag = 3.6e6     
    r     = 0.0015      
    A_cs  = np.pi * r**2
    I_mag = np.pi * r**4 / 4.0

    theta_desired = 40.0          # deg
    B0   = 0.01                   # T
    phi0 = np.deg2rad(15.0)       # rad
    L0   = 0.05                   # m (initial free length)

    B_out, phi_out, L_out, theta_out_deg, iters = jacobian_pid_controller_B_phi_L(
        theta_desired,
        B0, phi0, L0,
        mag, A_cs, E_mag, I_mag,
        L_min=0.04, L_max=0.04
    )

    print(f"Desired angle: {theta_desired:.1f} deg")
    print(f"Achieved angle: {theta_out_deg:.2f} deg in {iters} iterations")
    print(f"B ≈ {B_out:.4f} T")
    print(f"phi ≈ {np.rad2deg(phi_out):.2f} deg")
    print(f"L  ≈ {L_out:.4f} m")

    mu0 = 4e-7 * np.pi
    B_r = 1.2
    r_m = 0.03
    L_m = 0.09
    mag = magnetic_moment(1.2, 0.03, 0.09)

    m_hat = np.array([-1.0, 0.0, 0.0])   # dipole moment along -x
    x_init = 0.09                        # initial guess (9 cm)

    p_vec, B_final, iters = solve_x_for_Bmag_des(
        B_out,
        x_init,
        mu0, mag, m_hat
    )

    print("p_vec (x-axis only):", p_vec)
    print("B at that p_vec:", B_final)
    print("Bx target:", B_out, "Bx achieved:", B_final[0])
    print("iterations:", iters)